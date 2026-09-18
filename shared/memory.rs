//! Host and scheduler/container memory headroom.

use sysinfo::System;

/// `(total, available)` bytes, intersecting host RAM with every visible memory
/// cgroup ancestor.
///
/// Under a limit, availability is the limit less the cgroup's working set: its
/// usage minus inactive file pages. Page cache charged to a job, such as a .bed it
/// has just read, is reclaimed before the limit kills anything, so counting it as
/// used shrinks every plan as the job does I/O and hands a second run in the same
/// job a fraction of the first run's budget. Active file pages still count as used.
///
/// When this process cannot be matched to its limits (unreadable /proc files, a
/// hierarchy not mounted in this mount namespace, a path outside the cgroup
/// namespace, an unreadable or unparseable limit), the host figures stand, with a
/// warning. Planning against zero instead turns a detection gap on an unfamiliar
/// container or scheduler into a refusal to score any input. Only a known limit
/// whose usage cannot be read plans against no free memory.
pub fn memory_bytes() -> (u64, u64) {
    let mut system = System::new();
    system.refresh_memory();
    let host = (system.total_memory(), system.available_memory());
    #[cfg(target_os = "linux")]
    {
        let bounds = match (
            std::fs::read_to_string("/proc/self/cgroup"),
            std::fs::read_to_string("/proc/self/mountinfo"),
        ) {
            (Ok(groups), Ok(mounts)) => {
                cgroup_bounds(&groups, &mounts, |path| std::fs::read_to_string(path))
            }
            (Err(error), _) | (_, Err(error)) => Err(CgroupError::Unmatched(format!(
                "cannot read this process's cgroup metadata: {error}"
            ))),
        };
        within_cgroup_bounds(host, bounds)
    }
    #[cfg(not(target_os = "linux"))]
    host
}

/// This process's resident memory in bytes, or 0 when it cannot be read.
///
/// A budget read after preparation plans only what comes next, but the memory preparation left
/// resident still counts toward the process's peak; a plan that fills its budget charges it.
pub fn resident_bytes() -> u64 {
    let pid = sysinfo::Pid::from_u32(std::process::id());
    let mut system = System::new();
    if !system.refresh_process(pid) {
        return 0;
    }
    system.process(pid).map_or(0, sysinfo::Process::memory)
}

/// Returns the pages the allocator holds free to the system (glibc `malloc_trim`).
///
/// glibc keeps freed memory resident, in amounts that depend on allocation order and on how many
/// threads allocated: after the same preparation of 50,000 people and 512 scores, one run held
/// 172 MiB at the budget line and another 324 MiB. A resident reading taken after this counts
/// what the process holds.
pub fn release_free_heap() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        unsafe {
            malloc_trim(0);
        }
    }
}

/// Why [`cgroup_bounds`] could not bound this process.
#[cfg(target_os = "linux")]
#[derive(Debug)]
enum CgroupError {
    /// The process's memory cgroups could not be matched to limit files it can read,
    /// so no limit is known.
    Unmatched(String),
    /// A limit applies, but the usage under it cannot be read.
    UsageUnreadable { limit: u64, error: String },
}

#[cfg(target_os = "linux")]
fn within_cgroup_bounds(host: (u64, u64), bounds: Result<(u64, u64), CgroupError>) -> (u64, u64) {
    match bounds {
        Ok((total, available)) => (host.0.min(total), host.1.min(available)),
        Err(CgroupError::Unmatched(reason)) => {
            eprintln!(
                "> Cannot match this process to its memory cgroup limits ({reason}); planning against host memory."
            );
            host
        }
        Err(CgroupError::UsageUnreadable { limit, error }) => {
            eprintln!(
                "> A memory cgroup limit of {limit} bytes applies, but its usage cannot be read ({error}); planning against no free memory."
            );
            (host.0.min(limit), 0)
        }
    }
}

#[cfg(target_os = "linux")]
fn cgroup_bounds(
    groups: &str,
    mounts: &str,
    mut read: impl FnMut(&std::path::Path) -> std::io::Result<String>,
) -> Result<(u64, u64), CgroupError> {
    use std::io::ErrorKind;
    use std::path::{Component, Path};
    let mut bounds = (u64::MAX, u64::MAX);
    let mut matched = false;
    // Why a unified-hierarchy line could not be matched, when one could not.
    let mut unmatched_v2 = None;
    for group in groups.lines().filter(|line| !line.trim().is_empty()) {
        let mut fields = group.splitn(3, ':');
        let (Some(_), Some(controllers), Some(group_path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            return Err(CgroupError::Unmatched(format!(
                "malformed /proc/self/cgroup line '{group}'"
            )));
        };
        let v2 = controllers.is_empty();
        if !v2 && !controllers.split(',').any(|c| c == "memory") {
            continue;
        }
        let group_path = Path::new(group_path);
        // A process outside its cgroup namespace sees `..` components.
        let inside_namespace = group_path.is_absolute()
            && !group_path
                .components()
                .any(|c| matches!(c, Component::ParentDir));
        let mut found = false;
        for mount in mounts.lines().filter(|_| inside_namespace) {
            // A line this parser cannot read describes some other mount.
            let Some((left, right)) = mount.split_once(" - ") else {
                continue;
            };
            let left: Vec<_> = left.split_whitespace().collect();
            let right: Vec<_> = right.split_whitespace().collect();
            if left.len() < 6 || right.len() < 3 {
                continue;
            }
            let memory_mount = if v2 {
                right[0] == "cgroup2"
            } else {
                right[0] == "cgroup" && right[2].split(',').any(|c| c == "memory")
            };
            if !memory_mount {
                continue;
            }
            let (Ok(mount_root), Ok(mount_point)) =
                (unescape_mount_path(left[3]), unescape_mount_path(left[4]))
            else {
                continue;
            };
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
                    Ok(value) if v2 && value.trim() == "max" => {}
                    Ok(value) => {
                        let limit = value.trim().parse::<u64>().map_err(|_| {
                            CgroupError::Unmatched(format!(
                                "unparseable limit '{}' in {}",
                                value.trim(),
                                limit_path.display()
                            ))
                        })?;
                        let usage_path = directory.join(if v2 {
                            "memory.current"
                        } else {
                            "memory.usage_in_bytes"
                        });
                        let usage = match read(&usage_path) {
                            Ok(text) => text
                                .trim()
                                .parse::<u64>()
                                .map_err(|_| format!("unparseable value '{}'", text.trim())),
                            Err(error) => Err(error.to_string()),
                        }
                        .map_err(|error| CgroupError::UsageUnreadable {
                            limit: bounds.0.min(limit),
                            error: format!("{}: {error}", usage_path.display()),
                        })?;
                        // Hierarchical inactive file pages. Without a readable
                        // memory.stat, all usage counts as held.
                        let inactive_key = if v2 {
                            "inactive_file "
                        } else {
                            "total_inactive_file "
                        };
                        let reclaimable = read(&directory.join("memory.stat"))
                            .ok()
                            .and_then(|stat| {
                                stat.lines().find_map(|line| {
                                    line.strip_prefix(inactive_key)?.trim().parse::<u64>().ok()
                                })
                            })
                            .unwrap_or(0);
                        let working_set = usage.saturating_sub(reclaimable);
                        bounds.0 = bounds.0.min(limit);
                        bounds.1 = bounds.1.min(limit.saturating_sub(working_set));
                    }
                    // A v2 hierarchy can exist without its memory controller
                    // enabled, including the root which has no memory.max.
                    Err(error) if v2 && error.kind() == ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(CgroupError::Unmatched(format!(
                            "{}: {error}",
                            limit_path.display()
                        )));
                    }
                }
                if directory == mount_point
                    || !directory.pop()
                    || !directory.starts_with(&mount_point)
                {
                    break;
                }
            }
        }
        if found {
            matched = true;
        } else if v2 {
            // Once cgroup2 is mounted anywhere, every process lists a unified-hierarchy
            // line, but a container on a hybrid host often mounts only the v1
            // controllers. Any visible limit then lives on another line.
            if unmatched_v2.is_none() {
                unmatched_v2 = Some(format!("no cgroup2 mount covers {}", group_path.display()));
            }
        } else {
            return Err(CgroupError::Unmatched(format!(
                "no memory cgroup mount covers {}",
                group_path.display()
            )));
        }
    }
    match unmatched_v2 {
        Some(reason) if !matched => Err(CgroupError::Unmatched(reason)),
        _ => Ok(bounds),
    }
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

    fn fixture(
        groups: &str,
        mounts: &str,
        files: &[(&str, &str)],
    ) -> Result<(u64, u64), CgroupError> {
        cgroup_bounds(groups, mounts, |path| {
            files
                .iter()
                .find(|(name, _)| path == std::path::Path::new(name))
                .map(|(_, contents)| contents.to_string())
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
        })
    }

    fn unmatched(result: Result<(u64, u64), CgroupError>) -> bool {
        matches!(result, Err(CgroupError::Unmatched(_)))
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
    fn page_cache_filling_the_limit_still_counts_as_headroom() {
        let v2 = fixture(
            "0::/job/step",
            "1 0 0:1 / /cg rw - cgroup2 cgroup rw",
            &[
                ("/cg/job/step/memory.max", "max"),
                ("/cg/job/memory.max", "1000"),
                ("/cg/job/memory.current", "1000"),
                (
                    "/cg/job/memory.stat",
                    "anon 300\nfile 700\nactive_file 100\ninactive_file 600\n",
                ),
            ],
        )
        .unwrap();
        assert_eq!(v2, (1000, 600));
        let v1 = fixture(
            "10:memory:/job",
            "1 0 0:1 / /cg rw - cgroup cgroup rw,memory",
            &[
                ("/cg/job/memory.limit_in_bytes", "1000"),
                ("/cg/job/memory.usage_in_bytes", "900"),
                (
                    "/cg/job/memory.stat",
                    "inactive_file 5\ntotal_rss 250\ntotal_inactive_file 550\n",
                ),
                ("/cg/memory.limit_in_bytes", "9223372036854771712"),
                ("/cg/memory.usage_in_bytes", "2000"),
            ],
        )
        .unwrap();
        assert_eq!(v1, (1000, 650));
    }

    #[test]
    fn unmounted_unified_hierarchy_defers_to_the_v1_memory_controller() {
        let bounds = fixture(
            "12:memory:/docker/abc\n1:name=systemd:/docker/abc\n0::/system.slice/containerd.service",
            "1 0 0:1 /docker/abc /sys/fs/cgroup/memory ro - cgroup cgroup rw,memory\n\
             2 0 0:2 /docker/abc /sys/fs/cgroup/systemd ro - cgroup cgroup rw,name=systemd",
            &[
                ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "4000"),
                ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "1000"),
            ],
        )
        .unwrap();
        assert_eq!(bounds, (4000, 3000));
        assert!(unmatched(fixture(
            "0::/job",
            "1 0 0:1 / /proc rw - proc proc rw",
            &[]
        )));
    }

    #[test]
    fn unmatched_metadata_is_told_apart_from_unreadable_usage() {
        let mount = "1 0 0:1 / /cg rw - cgroup2 cgroup rw";
        assert!(unmatched(fixture("0::/", "", &[])));
        assert!(unmatched(fixture("0::/../job", mount, &[])));
        assert!(unmatched(fixture(
            "0::/job",
            mount,
            &[("/cg/job/memory.max", "oops")]
        )));
        assert!(unmatched(fixture("10:memory:/job", "", &[])));
        assert!(unmatched(fixture(
            "10:memory:/job",
            "1 0 0:1 / /cg rw - cgroup cgroup rw,memory",
            &[]
        )));
        assert!(unmatched(fixture("garbage", mount, &[])));
        assert!(matches!(
            fixture("0::/job", mount, &[("/cg/job/memory.max", "1000")]),
            Err(CgroupError::UsageUnreadable { limit: 1000, .. })
        ));
    }

    #[test]
    fn malformed_unrelated_mount_lines_are_skipped() {
        let bounds = fixture(
            "0::/job",
            "not a mountinfo line\n\
             1 0 0:1 / /cg rw - cgroup2 cgroup rw\n\
             2 0 0:2 / /x - short",
            &[
                ("/cg/job/memory.max", "1000"),
                ("/cg/job/memory.current", "100"),
            ],
        )
        .unwrap();
        assert_eq!(bounds, (1000, 900));
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

    #[test]
    fn unmatched_metadata_plans_against_the_host_and_unreadable_usage_fails_closed() {
        let host = (64_000, 32_000);
        assert_eq!(
            within_cgroup_bounds(host, Ok((16_000, 4_000))),
            (16_000, 4_000)
        );
        assert_eq!(within_cgroup_bounds(host, Ok((u64::MAX, u64::MAX))), host);
        assert_eq!(
            within_cgroup_bounds(host, Err(CgroupError::Unmatched("test".into()))),
            host
        );
        assert_eq!(
            within_cgroup_bounds(
                host,
                Err(CgroupError::UsageUnreadable {
                    limit: 16_000,
                    error: "test".into(),
                })
            ),
            (16_000, 0)
        );
    }
}
