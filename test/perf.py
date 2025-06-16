import argparse
import subprocess
import sys
from pathlib import Path
# Import the data generator directly from the existing bench.py script.
from bench import RealisticDataGenerator


# --- Configuration ---
WORKDIR = Path("./perf_workdir")
# The binary is built with the 'coz' profile, so it's in the 'coz' subdirectory.
GNOMON_BINARY = Path("./target/coz/gnomon").resolve()

# A small, fast-to-generate configuration tailored for CI profiling runs.
# It should be complex enough to exercise the core logic but fast enough for CI.
PERF_WORKLOAD = {
    "test_name": "ci_perf_small",
    "n_individuals": 500,
    "genome_variants": 50_000,
    "target_variants": 20_000,
    "score_files": [
        {
            "name": "perf_score",
            "n_scores": 5,
            "gwas_source_variants": 15_000,
            "overlap_pct": 0.90,
            "flip_pct": 0.10,
            "missing_weight_pct": 0.0,
            "score_sparsity": 0.3,
        }
    ],
}

def print_header(title: str, char: str = "="):
    """Prints a formatted header to the console for better log readability."""
    width = 80
    print("\n" + char * width, flush=True)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}", flush=True)
    print(char * width, flush=True)

def run_command(cmd, title, **kwargs):
    """
    Runs a command, streams its output in real-time, and checks for success.
    This is ideal for CI environments where seeing live output is important.
    """
    print_header(title, char="-")
    print(f"Executing: {' '.join(map(str, cmd))}", flush=True)
    try:
        # Popen allows real-time streaming of stdout/stderr.
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
        )
        with proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                sys.stdout.write(f"  > {line}")
                sys.stdout.flush()

        retcode = proc.wait()
        if retcode != 0:
            print(f"\n❌ Command failed with exit code {retcode}", flush=True)
            return False
        print(f"\n✅ Command successful.", flush=True)
        return True
    except Exception as e:
        print(f"❌ Failed to execute command: {e}", flush=True)
        return False

def main(args):
    """Main function to orchestrate data generation, profiling, and reporting."""
    print_header("Setting up synthetic data for profiling run")
    # Use the imported data generator class.
    generator = RealisticDataGenerator(workload_params=PERF_WORKLOAD, workdir=WORKDIR)
    data_prefix, score_files = generator.generate_all_files()

    # Define the core arguments for the gnomon binary.
    gnomon_args = ["--score", str(score_files[0]), str(data_prefix)]

    # --- Causal Profiling (coz) ---
    if args.tool in ['coz', 'all']:
        coz_cmd = ["coz", "run", "---", str(GNOMON_BINARY)] + gnomon_args
        run_command(coz_cmd, title="Running Causal Profiler (coz)")

        print_header("Causal Profile Report (coz)", char="*")
        print("# Answers: 'If I speed this line up, what's the throughput impact?'")
        profile_file = Path("profile.coz")
        if profile_file.exists():
            # Use a shell command to sort the text output by the 4th column (speedup %).
            # This is robust and avoids reading a large file into Python memory.
            subprocess.run(
                f"(head -n 2 {profile_file} && tail -n +3 {profile_file} | sort -t$'\\t' -k4,4nr) || cat {profile_file}",
                shell=True, check=False, executable='/bin/bash'
            )
        else:
            print("profile.coz not found. The profiling run may have failed.")

    # --- Traditional Profiling (perf) ---
    if args.tool in ['perf', 'all']:
        # Define where the raw profiling data will be stored.
        perf_data_file = Path("./perf.data")
        perf_record_cmd = ["perf", "record", "-g", "-o", str(perf_data_file), "--", str(GNOMON_BINARY)] + gnomon_args
        run_command(perf_record_cmd, title="Running Traditional Profiler (perf record)")

        print_header("Traditional Profile Report (perf)", char="*")
        print("# Answers: 'Where did my program spend the most CPU time?'")
        if perf_data_file.exists():
            # The report command reads the data file and generates a text summary.
            perf_report_cmd = ["perf", "report", "--stdio", "--no-children", "-i", str(perf_data_file)]
            run_command(perf_report_cmd, title="Generating perf report")
        else:
            print("perf.data not found. The profiling run may have failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance Profiling Harness for Gnomon")
    parser.add_argument('--tool', choices=['coz', 'perf', 'all'], default='all', help="Which profiler(s) to run.")
    args = parser.parse_args()

    # Pre-flight check to ensure the binary exists before starting.
    if not GNOMON_BINARY.exists():
        print(f"❌ Error: Compiled binary not found at {GNOMON_BINARY}")
        print("Please ensure you have built the project with: cargo build --profile coz")
        sys.exit(1)

    main(args)
