//! CPUs this process can actually run on.
//!
//! Two limits bound a process. Its affinity mask is the set of CPUs it may be
//! scheduled on, which is how Slurm and `taskset` confine a job. A CPU-time
//! quota (cgroup `cpu.max`, or `cpu.cfs_quota_us` over `cpu.cfs_period_us`) is
//! how Docker `--cpus` and Kubernetes limits confine a container. A pool wider
//! than either only queues its threads behind one another.
//!
//! `std::thread::available_parallelism` folds both into one number. This module
//! keeps them apart, beside the NUMA nodes the allowed CPUs sit on, so a caller
//! can both choose a thread count and say why it chose it.
//!
//! Metadata that cannot be read or parsed counts as no limit, never as zero: a
//! detection gap on an unfamiliar container must not become a one-thread run.

use std::num::NonZeroUsize;

/// What bounds this process's CPU use. Every figure is `None` or empty when it
/// is not visible on this platform.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuLimits {
    /// CPUs in this process's affinity mask.
    pub affinity: Option<usize>,
    /// The tightest CPU quota over this process's cgroup and every ancestor, in
    /// whole CPUs rounded down and at least one.
    pub quota: Option<usize>,
    /// Allowed CPUs on each NUMA node that holds any, largest first.
    pub numa_nodes: Vec<usize>,
}

impl CpuLimits {
    /// CPUs a pool can use: the affinity count capped by the quota, with
    /// `available_parallelism` standing in where no affinity is visible, and
    /// never below one.
    pub fn usable(&self) -> usize {
        let fallback = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        let cpus = self.affinity.unwrap_or(fallback);
        self.quota.map_or(cpus, |quota| cpus.min(quota)).max(1)
    }
}

/// This process's CPU limits, read fresh.
pub fn cpu_limits() -> CpuLimits {
    #[cfg(target_os = "linux")]
    {
        let read = |path: &std::path::Path| std::fs::read_to_string(path);
        let affinity = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| allowed_cpu_list(&status))
            .and_then(|list| parse_cpu_list(&list));
        let quota = match (
            std::fs::read_to_string("/proc/self/cgroup"),
            std::fs::read_to_string("/proc/self/mountinfo"),
        ) {
            (Ok(groups), Ok(mounts)) => cgroup_cpu_quota(&groups, &mounts, read),
            _ => None,
        };
        let numa_nodes = affinity
            .as_ref()
            .map(|allowed| numa_node_counts(allowed, read))
            .unwrap_or_default();
        CpuLimits {
            affinity: affinity.map(|allowed| allowed.len()),
            quota,
            numa_nodes,
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        CpuLimits::default()
    }
}

/// The `Cpus_allowed_list` value of a `/proc/<pid>/status` file.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn allowed_cpu_list(status: &str) -> Option<String> {
    status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .map(|value| value.trim().to_string())
}

/// The CPUs of a kernel CPU list such as `0-3,8,10-11`, sorted; `None` for an
/// empty or malformed list.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_cpu_list(list: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in list.trim().split(',').filter(|part| !part.is_empty()) {
        match part.split_once('-') {
            Some((first, last)) => {
                let (first, last) = (first.parse::<usize>().ok()?, last.parse::<usize>().ok()?);
                if last < first {
                    return None;
                }
                cpus.extend(first..=last);
            }
            None => cpus.push(part.parse::<usize>().ok()?),
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    (!cpus.is_empty()).then_some(cpus)
}

/// Allowed CPUs per NUMA node, largest first, from `/sys/devices/system/node`.
#[cfg(target_os = "linux")]
fn numa_node_counts(
    allowed: &[usize],
    mut read: impl FnMut(&std::path::Path) -> std::io::Result<String>,
) -> Vec<usize> {
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") else {
        return Vec::new();
    };
    let mut counts: Vec<usize> = entries
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.strip_prefix("node")
                .is_some_and(|index| index.parse::<usize>().is_ok())
        })
        .filter_map(|entry| read(&entry.path().join("cpulist")).ok())
        .filter_map(|list| parse_cpu_list(&list))
        .map(|node| {
            node.iter()
                .filter(|cpu| allowed.binary_search(cpu).is_ok())
                .count()
        })
        .filter(|&count| count > 0)
        .collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));
    counts
}

/// The tightest CPU quota over this process's cgroups and their ancestors, in
/// whole CPUs rounded down and at least one; `None` when no quota applies or
/// when the metadata cannot be matched or parsed. A thread for the fractional
/// remainder would spend its periods throttled, stalling every join that waits
/// on it.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn cgroup_cpu_quota(
    groups: &str,
    mounts: &str,
    mut read: impl FnMut(&std::path::Path) -> std::io::Result<String>,
) -> Option<usize> {
    use std::path::{Component, Path};
    let mut tightest: Option<f64> = None;
    for group in groups.lines().filter(|line| !line.trim().is_empty()) {
        let mut fields = group.splitn(3, ':');
        let (Some(_), Some(controllers), Some(group_path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            return None;
        };
        let v2 = controllers.is_empty();
        if !v2 && !controllers.split(',').any(|controller| controller == "cpu") {
            continue;
        }
        let group_path = Path::new(group_path);
        if !group_path.is_absolute()
            || group_path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            continue;
        }
        for mount in mounts.lines() {
            let Some((left, right)) = mount.split_once(" - ") else {
                continue;
            };
            let left: Vec<_> = left.split_whitespace().collect();
            let right: Vec<_> = right.split_whitespace().collect();
            if left.len() < 5 || right.len() < 3 {
                continue;
            }
            let cpu_mount = if v2 {
                right[0] == "cgroup2"
            } else {
                right[0] == "cgroup" && right[2].split(',').any(|option| option == "cpu")
            };
            if !cpu_mount {
                continue;
            }
            let (Some(mount_root), Some(mount_point)) =
                (unescape_mount_path(left[3]), unescape_mount_path(left[4]))
            else {
                continue;
            };
            let Ok(relative) = group_path.strip_prefix(&mount_root) else {
                continue;
            };
            let mut directory = mount_point.join(relative);
            loop {
                // A level without the limit file (the root, or a v1 hierarchy
                // without CFS) sets no limit; only unparseable content hides one.
                let quota = if v2 {
                    match read(&directory.join("cpu.max")) {
                        Ok(text) => parse_cpu_max(&text),
                        Err(_) => Some(None),
                    }
                } else {
                    match (
                        read(&directory.join("cpu.cfs_quota_us")),
                        read(&directory.join("cpu.cfs_period_us")),
                    ) {
                        (Ok(quota), Ok(period)) => parse_cfs_quota(&quota, &period),
                        _ => Some(None),
                    }
                };
                match quota {
                    // Unparseable metadata: no limit is visible at all.
                    None => return None,
                    Some(Some(cpus)) => {
                        tightest = Some(tightest.map_or(cpus, |known| known.min(cpus)));
                    }
                    Some(None) => {}
                }
                if directory == mount_point || !directory.pop() {
                    break;
                }
            }
        }
    }
    tightest.map(|cpus| (cpus.floor() as usize).max(1))
}

/// A v2 `cpu.max` file: `Some(None)` for `max`, `Some(Some(cpus))` for a quota,
/// `None` when it cannot be parsed.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_cpu_max(text: &str) -> Option<Option<f64>> {
    let mut fields = text.split_whitespace();
    let quota = fields.next()?;
    let period = fields
        .next()
        .map_or(Some(100_000.0), |value| value.parse::<f64>().ok())?;
    if quota == "max" {
        return Some(None);
    }
    let quota = quota.parse::<f64>().ok()?;
    (quota > 0.0 && period > 0.0).then_some(Some(quota / period))
}

/// A v1 `cpu.cfs_quota_us` and `cpu.cfs_period_us` pair, with the same meaning
/// as [`parse_cpu_max`]; a quota of `-1` means unlimited.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_cfs_quota(quota: &str, period: &str) -> Option<Option<f64>> {
    let quota = quota.trim().parse::<i64>().ok()?;
    let period = period.trim().parse::<i64>().ok()?;
    if quota < 0 {
        return Some(None);
    }
    (quota > 0 && period > 0).then(|| Some(quota as f64 / period as f64))
}

/// A mountinfo path field, with the kernel's octal escapes (`\040` for a space)
/// decoded.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn unescape_mount_path(field: &str) -> Option<std::path::PathBuf> {
    let bytes = field.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            let digits = bytes.get(index + 1..index + 4)?;
            let text = std::str::from_utf8(digits).ok()?;
            decoded.push(u8::from_str_radix(text, 8).ok()?);
            index += 4;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    Some(std::path::PathBuf::from(String::from_utf8(decoded).ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{Error, ErrorKind};
    use std::path::{Path, PathBuf};

    const V2_MOUNT: &str = "30 25 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:4 - cgroup2 cgroup2 rw,nsdelegate\n";
    const V1_MOUNT: &str = "40 25 0:35 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid shared:13 - cgroup cgroup rw,cpu,cpuacct\n";

    fn files(entries: &[(&str, &str)]) -> impl FnMut(&Path) -> std::io::Result<String> + use<> {
        let map: HashMap<PathBuf, String> = entries
            .iter()
            .map(|(path, text)| (PathBuf::from(path), text.to_string()))
            .collect();
        move |path: &Path| {
            map.get(path)
                .cloned()
                .ok_or_else(|| Error::new(ErrorKind::NotFound, path.display().to_string()))
        }
    }

    #[test]
    fn cpu_lists_parse_ranges_singletons_and_refuse_garbage() {
        assert_eq!(
            parse_cpu_list("0-3,8,10-11\n"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_cpu_list("5"), Some(vec![5]));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("4-2"), None);
        assert_eq!(parse_cpu_list("x"), None);
        let status =
            "Name:\tgnomon\nCpus_allowed:\tff\nCpus_allowed_list:\t0-7\nMems_allowed:\t1\n";
        assert_eq!(allowed_cpu_list(status).as_deref(), Some("0-7"));
    }

    #[test]
    fn a_v2_quota_is_the_tightest_ancestor_rounded_down() {
        let groups = "0::/kubepods/pod1/container\n";
        let read = files(&[
            (
                "/sys/fs/cgroup/kubepods/pod1/container/cpu.max",
                "300000 100000\n",
            ),
            ("/sys/fs/cgroup/kubepods/pod1/cpu.max", "150000 100000\n"),
            ("/sys/fs/cgroup/kubepods/cpu.max", "max 100000\n"),
        ]);
        assert_eq!(cgroup_cpu_quota(groups, V2_MOUNT, read), Some(1));

        let unlimited = files(&[("/sys/fs/cgroup/job/cpu.max", "max 100000\n")]);
        assert_eq!(cgroup_cpu_quota("0::/job\n", V2_MOUNT, unlimited), None);

        let two = files(&[("/sys/fs/cgroup/job/cpu.max", "200000 100000\n")]);
        assert_eq!(cgroup_cpu_quota("0::/job\n", V2_MOUNT, two), Some(2));
    }

    #[test]
    fn a_v1_quota_reads_the_cfs_pair_and_minus_one_is_unlimited() {
        let groups = "11:memory:/docker/abc\n4:cpu,cpuacct:/docker/abc\n";
        let limited = files(&[
            (
                "/sys/fs/cgroup/cpu,cpuacct/docker/abc/cpu.cfs_quota_us",
                "250000\n",
            ),
            (
                "/sys/fs/cgroup/cpu,cpuacct/docker/abc/cpu.cfs_period_us",
                "100000\n",
            ),
            ("/sys/fs/cgroup/cpu,cpuacct/docker/cpu.cfs_quota_us", "-1\n"),
            (
                "/sys/fs/cgroup/cpu,cpuacct/docker/cpu.cfs_period_us",
                "100000\n",
            ),
        ]);
        assert_eq!(cgroup_cpu_quota(groups, V1_MOUNT, limited), Some(2));

        let slurm = files(&[
            (
                "/sys/fs/cgroup/cpu,cpuacct/slurm/job_1/cpu.cfs_quota_us",
                "-1\n",
            ),
            (
                "/sys/fs/cgroup/cpu,cpuacct/slurm/job_1/cpu.cfs_period_us",
                "100000\n",
            ),
        ]);
        assert_eq!(
            cgroup_cpu_quota("4:cpu,cpuacct:/slurm/job_1\n", V1_MOUNT, slurm),
            None
        );
    }

    #[test]
    fn a_mount_rooted_below_the_hierarchy_resolves_relative_paths() {
        let mounts =
            "50 40 0:35 /docker/abc /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n";
        let read = files(&[
            ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "50000\n"),
            ("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "100000\n"),
        ]);
        assert_eq!(
            cgroup_cpu_quota("4:cpu,cpuacct:/docker/abc\n", mounts, read),
            Some(1)
        );
    }

    #[test]
    fn odd_metadata_means_no_visible_limit_never_zero() {
        let garbage = files(&[("/sys/fs/cgroup/job/cpu.max", "lots\n")]);
        assert_eq!(cgroup_cpu_quota("0::/job\n", V2_MOUNT, garbage), None);
        assert_eq!(
            cgroup_cpu_quota("not a cgroup line\n", V2_MOUNT, files(&[])),
            None
        );
        assert_eq!(
            cgroup_cpu_quota("0::/../outside\n", V2_MOUNT, files(&[])),
            None
        );
        assert_eq!(parse_cpu_max("0 100000"), None);
        assert_eq!(parse_cfs_quota("0", "100000"), None);

        let limits = CpuLimits {
            affinity: Some(8),
            quota: Some(3),
            numa_nodes: vec![8],
        };
        assert_eq!(limits.usable(), 3);
        let unbounded = CpuLimits {
            affinity: Some(6),
            quota: None,
            numa_nodes: Vec::new(),
        };
        assert_eq!(unbounded.usable(), 6);
        assert!(CpuLimits::default().usable() >= 1);
    }

    #[test]
    fn escaped_mount_points_decode() {
        assert_eq!(
            unescape_mount_path("/mnt/with\\040space"),
            Some(PathBuf::from("/mnt/with space"))
        );
        assert_eq!(unescape_mount_path("/bad\\04"), None);
    }
}
