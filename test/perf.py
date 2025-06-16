import subprocess
import sys
import os
from pathlib import Path
from bench import RealisticDataGenerator

# --- Configuration ---
WORKDIR = Path("./perf_workdir")
# The binary should have profiling symbols. The CI workflow correctly uses a binary
# from `target/profiling/gnomon`.
GNOMON_BINARY = Path("./target/profiling/gnomon").resolve()

# Define a variety of workloads to profile the program under different conditions.
WORKLOADS = {
    # A small, clean workload.
    "small_clean": {
        "test_name": "small_clean",
        "n_individuals": 1_000,
        "genome_variants": 100_000,
        "target_variants": 40_000,
        "score_files": [
            {
                "name": "high_quality_score",
                "n_scores": 5,
                "gwas_source_variants": 40_000,
                "overlap_pct": 0.95,
                "flip_pct": 0.01,
                "missing_weight_pct": 0.0,
                "score_sparsity": 1.0,
            }
        ],
    },
    # A larger, more complex workload with multiple, heterogeneous score files.
    "large_complex": {
        "test_name": "large_complex",
        "n_individuals": 5_000,
        "genome_variants": 1_000_000,
        "target_variants": 200_000,
        "score_files": [
            {
                "name": "large_score_1",
                "n_scores": 10,
                "gwas_source_variants": 150_000,
                "overlap_pct": 0.90,
                "flip_pct": 0.05,
                "missing_weight_pct": 0.01,
                "score_sparsity": 1.0,
            },
            {
                "name": "large_score_2",
                "n_scores": 8,
                "gwas_source_variants": 120_000,
                "overlap_pct": 0.80,
                "flip_pct": 0.02,
                "missing_weight_pct": 0.05,
                "score_sparsity": 0.9,
            },
        ],
    },
    # A workload specifically designed to test performance on "messy" data.
    "messy_data": {
        "test_name": "messy_data",
        "n_individuals": 2_000,
        "genome_variants": 200_000,
        "target_variants": 100_000,
        "score_files": [
            {
                "name": "messy_score",
                "n_scores": 4,
                "gwas_source_variants": 50_000,
                "overlap_pct": 0.50, # Low overlap
                "flip_pct": 0.15,      # High error rate
                "missing_weight_pct": 0.20, # 20% of weights missing
                "score_sparsity": 0.7,
            }
        ],
    },
}

def print_header(title: str, char: str = "="):
    """Prints a formatted header to the console for better log readability."""
    width = 80
    print("\n" + char * width, flush=True)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}", flush=True)
    print(char * width, flush=True)

def run_command(cmd, title, **kwargs):
    """
    Runs a command, streams its output, and on failure, prints the last 20 lines
    of output for easier debugging. Returns True on success, False on failure.
    """
    print_header(title, char="-")
    cmd_str_list = [str(c) for c in cmd]
    print(f"Executing: {' '.join(cmd_str_list)}", flush=True)
    
    output_lines = []
    try:
        # Use Popen to stream output in real-time
        proc = subprocess.Popen(
            cmd_str_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', **kwargs
        )
        with proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                sys.stdout.write(f"  > {line.rstrip()}\n")
                sys.stdout.flush()
                output_lines.append(line)
        retcode = proc.wait()
        
        if retcode != 0:
            print(f"\n❌ Command failed with exit code {retcode}", flush=True)
            print("--- Last 20 lines of output for debugging ---", flush=True)
            for l in output_lines[-20:]:
                print(f"  | {l.rstrip()}", flush=True)
            return False
        
        print("\n✅ Command successful.", flush=True)
        return True
    except Exception as e:
        print(f"❌ Failed to execute command: {e}", flush=True)
        return False

def main():
    """
    Main function to orchestrate data generation, profiling, and reporting.
    It generates data for all workloads, then runs them sequentially under a
    single `perf record` session for a combined report.
    """
    WORKDIR.mkdir(exist_ok=True)
    
    commands_to_profile = []
    generators = []

    # --- 1. Generate Data and Commands for ALL Workloads ---
    try:
        for name, params in WORKLOADS.items():
            print_header(f"Generating data for workload: {name}", char="*")
            run_workdir = WORKDIR / name
            run_workdir.mkdir(exist_ok=True)

            generator = RealisticDataGenerator(workload_params=params, workdir=run_workdir)
            generators.append(generator) # Save for cleanup
            data_prefix, score_files = generator.generate_all_files()
            
            # Gnomon can accept a directory of score files
            score_dir = run_workdir / f"scores_{name}"
            score_dir.mkdir(exist_ok=True)
            for sf_path in score_files:
                sf_path.rename(score_dir / sf_path.name)
            
            # Build the full command string
            command_str = f'echo "--- Running workload: {name} ---" && '
            command_str += f'"{GNOMON_BINARY}" --score "{score_dir}" "{data_prefix}"'
            commands_to_profile.append(command_str)

        # --- 2. Create a single script to run all workloads ---
        runner_script_path = WORKDIR / "run_all_workloads.sh"
        with open(runner_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -e\n") # Exit on any error
            for cmd in commands_to_profile:
                f.write(f"{cmd}\n")
        os.chmod(runner_script_path, 0o755)

        # --- 3. Run `perf record` on the combined script ---
        perf_data_file = WORKDIR / "perf.combined.data"
        perf_record_cmd = [
            "perf", "record", "-g", "-o", str(perf_data_file),
            "--", str(runner_script_path)
        ]
        if not run_command(perf_record_cmd, title="Running ALL workloads under `perf record`"):
            sys.exit("Profiling run failed.")

        # --- 4. Generate a single, combined `perf report` ---
        if perf_data_file.exists():
            perf_report_cmd = [
                "perf", "report", "--stdio", "--call-graph=graph",
                "--percent-limit=2", "-i", str(perf_data_file)
            ]
            run_command(perf_report_cmd, title="Combined Granular Profile Report")
        else:
            print(f"Combined perf data file not found at '{perf_data_file}'. The profiling run may have failed.")
            sys.exit(1)

    finally:
        # --- 5. Cleanup ---
        print_header("Cleaning up all generated data", char="~")
        for gen in generators:
            gen.cleanup()
        runner_script = WORKDIR / "run_all_workloads.sh"
        if runner_script.exists():
            runner_script.unlink()
            print(f"Removed temporary script: {runner_script}")

    print("\n✅ All profiling workloads completed successfully.")


if __name__ == "__main__":
    if not GNOMON_BINARY.exists():
        print(f"❌ Error: Compiled profiling binary not found at {GNOMON_BINARY}")
        print("Please ensure you have built the project with: cargo build --profile profiling")
        sys.exit(1)

    main()
