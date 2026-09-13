use std::time::Instant;
use sysinfo::{ProcessRefreshKind, System};

fn main() {
    let release = std::sync::Arc::new(std::sync::Barrier::new(3));
    let workers: Vec<_> = (0..2).map(|_| {
        let release = release.clone();
        std::thread::spawn(move || { release.wait(); })
    }).collect();
    for rep in 0..3 {
        let start = Instant::now();
        let mut system = System::new();
        system.refresh_memory();
        system.refresh_processes_specifics(ProcessRefreshKind::new());
        let mut reference: Vec<_> = system.processes().values()
            .filter(|p| p.name().starts_with("gnomon"))
            .map(|p| p.name().to_owned()).collect();
        reference.sort();
        println!("rep={rep} sysinfo={:?} count={}", start.elapsed(), reference.len());
        let start = Instant::now();
        let mut names = Vec::new();
        for entry in std::fs::read_dir("/proc").unwrap().flatten() {
            if !entry.file_name().as_encoded_bytes().iter().all(u8::is_ascii_digit) { continue; }
            if let Ok(name) = std::fs::read_to_string(entry.path().join("comm")) {
                if name.starts_with("gnomon") { names.push(name.trim_end().to_owned()); }
            }
        }
        names.sort();
        println!("rep={rep} proc={:?} count={} equal={}", start.elapsed(), names.len(), names == reference);
        if rep == 0 {
            let mut leaders = std::collections::BTreeSet::new();
            for (pid, _) in system.processes().iter().filter(|(_,p)| p.name().starts_with("gnomon")) {
                let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap();
                let leader = status.lines().find_map(|line| line.strip_prefix("Tgid:")).unwrap().trim().to_owned();
                leaders.insert(leader);
            }
            println!("sysinfo unique processes={} proc count={}", leaders.len(), names.len());
            assert_eq!(leaders.len(), names.len());
        }
    }
    release.wait();
    for worker in workers { worker.join().unwrap(); }
}
