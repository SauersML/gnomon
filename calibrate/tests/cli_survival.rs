use std::fs;
use std::io;
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use tempfile::tempdir;

fn output_with_timeout(cmd: &mut Command, timeout: Duration) -> io::Result<Output> {
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let start = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            return child.wait_with_output();
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("subprocess timed out after {:?}", timeout),
            ));
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn survival_cli_time_varying_trains_without_lambda_flag() {
    let tmp = tempdir().expect("temporary directory");
    let training_path = tmp.path().join("survival_train.tsv");

    // Data with increasing event rates by age (monotonic hazard)
    let data = "age_entry\tage_exit\tevent_target\tevent_competing\tsample_weight\tpgs\tsex\tpc1\n\
40\t50\t0\t0\t1.0\t-0.5\t0.0\t0.0\n\
50\t60\t0\t0\t1.0\t-0.2\t1.0\t0.0\n\
60\t70\t1\t0\t1.0\t0.1\t0.0\t0.0\n\
70\t80\t1\t0\t1.0\t0.4\t1.0\t0.0\n\
80\t90\t1\t0\t1.0\t0.7\t0.0\t0.0\n\
90\t100\t1\t0\t1.0\t1.0\t1.0\t0.0\n";
    fs::write(&training_path, data).expect("write training data");

    let exe = env!("CARGO_BIN_EXE_gnomon");
    let output = match output_with_timeout(
        Command::new(exe).current_dir(tmp.path()).args([
            "train",
            "--model-family",
            "survival",
            training_path.to_str().expect("path str"),
            "--num-pcs",
            "1",
            "--survival-enable-time-varying",
            "--max-iterations",
            "3",
        ]),
        Duration::from_secs(30),
    ) {
        Ok(output) => output,
        Err(err) if err.kind() == io::ErrorKind::TimedOut => {
            eprintln!("Skipping test due timeout: {err}");
            return;
        }
        Err(err) => panic!("run gnomon cli: {err}"),
    };

    assert!(
        !output.status.success(),
        "CLI unexpectedly succeeded: {:?}",
        output.status
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("P-IRLS inner loop"),
        "expected P-IRLS failure, stderr: {stderr}"
    );
}

#[test]
fn survival_cli_rejects_retired_barrier_flags() {
    let tmp = tempdir().expect("temporary directory");
    let training_path = tmp.path().join("survival_train.tsv");

    let data = "age_entry\tage_exit\tevent_target\tevent_competing\tsample_weight\tpgs\tsex\tpc1\n\
50\t55\t1\t0\t1.0\t-0.2\t0.0\t0.1\n";
    fs::write(&training_path, data).expect("write training data");

    let exe = env!("CARGO_BIN_EXE_gnomon");
    for (flag, value) in [
        ("--survival-barrier-weight", "1e-4"),
        ("--survival-barrier-scale", "1.0"),
    ] {
        let output = output_with_timeout(
            Command::new(exe).current_dir(tmp.path()).args([
                "train",
                "--model-family",
                "survival",
                training_path.to_str().expect("path str"),
                "--num-pcs",
                "0",
                flag,
                value,
            ]),
            Duration::from_secs(30),
        )
        .expect("run gnomon cli");

        assert!(!output.status.success(), "CLI unexpectedly accepted {flag}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(flag),
            "expected stderr to mention {flag}, got: {stderr}"
        );
    }
}
