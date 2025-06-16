import argparse
import subprocess
import sys
from pathlib import Path
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
            "score_sparsity": 1.0,
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
    Runs a command, streams its output, and on failure, prints the last 20 lines
    of output for easier debugging.

    Returns:
        bool: True if the command succeeded, False otherwise.
    """
    print_header(title, char="-")
    cmd_str_list = [str(c) for c in cmd]
    print(f"Executing: {' '.join(cmd_str_list)}", flush=True)
    
    output_lines = []
    try:
        proc = subprocess.Popen(
            cmd_str_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace', # Prevents crashes on weird characters
            **kwargs
        )
        
        # Stream output in real-time while also capturing it
        with proc.stdout:
            for line in iter(proc.stdout.readline, ''):
                line_stripped = line.rstrip()
                sys.stdout.write(f"  > {line_stripped}\n")
                sys.stdout.flush()
                output_lines.append(line_stripped)

        retcode = proc.wait()
        
        if retcode != 0:
            print(f"\n❌ Command failed with exit code {retcode}", flush=True)
            print("--- Last 20 lines of output for debugging ---", flush=True)
            for l in output_lines[-20:]:
                print(f"  | {l}", flush=True)
            print("---------------------------------------------", flush=True)
            return False
            
        print(f"\n✅ Command successful.", flush=True)
        return True
        
    except Exception as e:
        print(f"❌ Failed to execute command: {e}", flush=True)
        return False

def main(args):
    """Main function to orchestrate data generation, profiling, and reporting."""
    print_header("Setting up synthetic data for profiling run")
    WORKDIR.mkdir(exist_ok=True)
    
    generator = RealisticDataGenerator(workload_params=PERF_WORKLOAD, workdir=WORKDIR)
    data_prefix, score_files = generator.generate_all_files()

    gnomon_args = ["--score", str(score_files[0]), str(data_prefix)]

    # --- Causal Profiling (coz) ---
    if args.tool in ['coz', 'all']:
        coz_cmd = ["coz", "run", "---", str(GNOMON_BINARY)] + gnomon_args
        coz_succeeded = run_command(coz_cmd, title="Running Causal Profiler (coz)")

        print_header("Causal Profile Report (coz)", char="*")
        print("# Answers: 'If I speed this line up, what's the throughput impact?'")
        profile_file = Path("profile.coz")
        
        if coz_succeeded and profile_file.exists():
            # Use a shell command to sort the text output by the 4th column (speedup %).
            subprocess.run(
                f"(head -n 2 {profile_file} && tail -n +3 {profile_file} | sort -t$'\\t' -k4,4nr) || cat {profile_file}",
                shell=True, check=False, executable='/bin/bash'
            )
        else:
            print("profile.coz not found. The profiling run may have failed.")
            print("\nNOTE: `coz` often fails with complex multi-threaded applications (e.g., those using Tokio and Rayon).")
            print("This is a known limitation of the profiler, not necessarily a bug in your code.")
            print("The `perf` profiler is more robust and should be used as the primary source of truth.\n")

    # --- Traditional Profiling (perf) with Call-Graph and Filtering ---
    if args.tool in ['perf', 'all']:
        perf_data_file = Path("./perf.data")
        perf_record_cmd = ["perf", "record", "-g", "-o", str(perf_data_file), "--", str(GNOMON_BINARY)] + gnomon_args
        run_command(perf_record_cmd, title="Running Traditional Profiler (perf record)")

        print_header("Granular Profile Report (perf)", char="*")
        print("# Answers: 'Where did my program spend its time?'")
        print("# Displaying call-graph, showing only branches consuming > 2% of total time.")
        
        if perf_data_file.exists():
            # This is the key change:
            # --call-graph=graph : Shows an indented call tree.
            # --percent-limit=2  : Hides all entries that account for less than 2% of the total samples.
            perf_report_cmd = [
                "perf", "report", 
                "--stdio",
                "--call-graph=graph", 
                "--percent-limit=2",
                "-i", str(perf_data_file)
            ]
            run_command(perf_report_cmd, title="Generating Granular `perf` Call-Graph Report")
        else:
            print("perf.data not found. The profiling run may have failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance Profiling Harness for Gnomon")
    parser.add_argument('--tool', choices=['coz', 'perf', 'all'], default='all', help="Which profiler(s) to run.")
    args = parser.parse_args()

    if not GNOMON_BINARY.exists():
        print(f"❌ Error: Compiled binary not found at {GNOMON_BINARY}")
        print("Please ensure you have built the project with: cargo build --profile coz")
        sys.exit(1)

    main(args)
