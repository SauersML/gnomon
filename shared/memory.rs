//! Host and scheduler/container memory headroom. Unknown limits fail closed.

use sysinfo::System;

/// `(total, available)` bytes, intersecting host RAM with every visible memory
/// cgroup ancestor. Availability is additional headroom, not reclaimable cache:
/// treating a job's charged page cache as free can exceed its hard limit.
pub fn memory_bytes() -> (u64, u64) {
    let mut system = System::new();
    system.refresh_memory();
    let host = (system.total_memory(), system.available_memory());
    #[cfg(target_os = "linux")]
    {
        let limits = std::fs::read_to_string("/proc/self/cgroup").and_then(|groups| {
            std::fs::read_to_string("/proc/self/mountinfo").and_then(|mounts| {
                cgroup_bounds(&groups, &mounts, |path| std::fs::read_to_string(path))
            })
        });
        match limits {
            Ok((total, available)) => (host.0.min(total), host.1.min(available)),
            Err(error) => {
                eprintln!(
                    "> Cannot establish the process memory limit: {error}. Memory budget is zero."
                );
                (0, 0)
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    host
}

#[cfg(target_os = "linux")]
fn cgroup_bounds(
    groups: &str,
    mounts: &str,
    mut read: impl FnMut(&std::path::Path) -> std::io::Result<String>,
) -> std::io::Result<(u64, u64)> {
    use std::io::{Error, ErrorKind};
    use std::path::{Component, Path};
    let invalid = || Error::new(ErrorKind::InvalidData, "Malformed memory cgroup metadata");
    let mut bounds = (u64::MAX, u64::MAX);
    for group in groups.lines() {
        let mut fields = group.splitn(3, ':');
        let _id = fields.next().ok_or_else(invalid)?;
        let controllers = fields.next().ok_or_else(invalid)?;
        let group_path = fields.next().ok_or_else(invalid)?;
        let v2 = controllers.is_empty();
        if !v2 && !controllers.split(',').any(|c| c == "memory") {
            continue;
        }
        let group_path = Path::new(group_path);
        if !group_path.is_absolute()
            || group_path
                .components()
                .any(|c| matches!(c, Component::ParentDir))
        {
            return Err(invalid());
        }
        let mut found = false;
        for mount in mounts.lines() {
            let (left, right) = mount.split_once(" - ").ok_or_else(invalid)?;
            let left: Vec<_> = left.split_whitespace().collect();
            let right: Vec<_> = right.split_whitespace().collect();
            if left.len() < 6 || right.len() < 3 {
                return Err(invalid());
            }
            if if v2 {
                right[0] != "cgroup2"
            } else {
                right[0] != "cgroup" || !right[2].split(',').any(|c| c == "memory")
            } {
                continue;
            }
            let mount_root = unescape_mount_path(left[3])?;
            let mount_point = unescape_mount_path(left[4])?;
            let Ok(relative) = group_path.strip_prefix(&mount_root) else {
                continue;
            };
            let mut directory = mount_point.join(relative);
            found = true;
            loop {
                let limit_path = directory.join(if v2 {
                    "memory.max"
                } else {
                    "memory.limit_in_bytes"
                });
                match read(&limit_path) {
                    Ok(value) => {
                        let value = value.trim();
                        if value != "max" || !v2 {
                            let limit = value.parse::<u64>().map_err(|_| invalid())?;
                            let usage = read(&directory.join(if v2 {
                                "memory.current"
                            } else {
                                "memory.usage_in_bytes"
                            }))?;
                            let usage = usage.trim().parse::<u64>().map_err(|_| invalid())?;
                            bounds.0 = bounds.0.min(limit);
                            bounds.1 = bounds.1.min(limit.saturating_sub(usage));
                        }
                    }
                    // A v2 hierarchy can exist without its memory controller
                    // enabled, including the root which has no memory.max.
                    Err(error) if v2 && error.kind() == ErrorKind::NotFound => {}
                    Err(error) => return Err(error),
                }
                if directory == mount_point {
                    break;
                }
                if !directory.pop() || !directory.starts_with(&mount_point) {
                    return Err(invalid());
                }
            }
        }
        if !found {
            return Err(invalid());
        }
    }
    Ok(bounds)
}

#[cfg(target_os = "linux")]
fn unescape_mount_path(value: &str) -> std::io::Result<std::path::PathBuf> {
    use std::os::unix::ffi::OsStringExt;
    let mut bytes = Vec::with_capacity(value.len());
    let mut input = value.as_bytes().iter().copied();
    while let Some(byte) = input.next() {
        if byte == b'\\' {
            let mut decoded = 0u16;
            for _ in 0..3 {
                let digit = input
                    .next()
                    .filter(|b| (b'0'..=b'7').contains(b))
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Malformed mount path escape",
                        )
                    })?;
                decoded = decoded * 8 + u16::from(digit - b'0');
            }
            bytes.push(u8::try_from(decoded).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid mount path byte")
            })?);
        } else {
            bytes.push(byte);
        }
    }
    Ok(std::ffi::OsString::from_vec(bytes).into())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    fn fixture(groups: &str, mounts: &str, files: &[(&str, &str)]) -> std::io::Result<(u64, u64)> {
        cgroup_bounds(groups, mounts, |path| {
            files
                .iter()
                .find(|(name, _)| path == std::path::Path::new(name))
                .map(|(_, contents)| contents.to_string())
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
        })
    }

    #[test]
    fn ancestor_headroom_caps_unlimited_child() {
        let bounds = fixture(
            "0::/job/step\n",
            "1 0 0:1 / /cg rw - cgroup2 cgroup rw\n",
            &[
                ("/cg/job/step/memory.max", "max"),
                ("/cg/job/memory.max", "1000"),
                ("/cg/job/memory.current", "900"),
            ],
        )
        .unwrap();
        assert_eq!(bounds, (1000, 100));
    }

    #[test]
    fn nested_limits_intersect_capacity_and_headroom_separately() {
        let bounds = fixture(
            "0::/job/step",
            "1 0 0:1 /job /cg rw - cgroup2 cgroup rw",
            &[
                ("/cg/step/memory.max", "400"),
                ("/cg/step/memory.current", "50"),
                ("/cg/memory.max", "1000"),
                ("/cg/memory.current", "900"),
            ],
        )
        .unwrap();
        assert_eq!(bounds, (400, 100));
    }

    #[test]
    fn v1_and_escaped_mounts_respect_exhausted_limit() {
        let bounds = fixture(
            "10:memory:/job",
            "1 0 0:1 / /cg\\040mem rw - cgroup cgroup rw,memory",
            &[
                ("/cg mem/job/memory.limit_in_bytes", "1000"),
                ("/cg mem/job/memory.usage_in_bytes", "1001"),
                ("/cg mem/memory.limit_in_bytes", "2000"),
                ("/cg mem/memory.usage_in_bytes", "500"),
            ],
        )
        .unwrap();
        assert_eq!(bounds, (1000, 0));
    }

    #[test]
    fn unknown_usage_and_bad_metadata_are_errors() {
        let mount = "1 0 0:1 / /cg rw - cgroup2 cgroup rw";
        assert!(fixture("0::/job", mount, &[("/cg/job/memory.max", "1000")]).is_err());
        assert!(fixture("0::/job", mount, &[("/cg/job/memory.max", "oops")]).is_err());
        assert!(fixture("0::/../job", mount, &[]).is_err());
        assert!(fixture("0::/job", "", &[]).is_err());
    }

    #[test]
    fn disabled_memory_controller_does_not_invent_a_limit() {
        assert_eq!(
            fixture("0::/", "1 0 0:1 / /cg rw - cgroup2 cgroup rw", &[]).unwrap(),
            (u64::MAX, u64::MAX)
        );
        assert_eq!(
            fixture("4:cpu:/job", "", &[]).unwrap(),
            (u64::MAX, u64::MAX)
        );
    }
}
