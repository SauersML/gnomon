import subprocess
import sys
from pathlib import Path
from bench import RealisticDataGenerator

# --- Configuration ---
WORKDIR = Path("./perf_workdir")
GNOMON_BINARY = Path("./target/profiling/gnomon").resolve()

# Define a variety of workloads to profile the program under different conditions.
# The script will loop through each of these and generate a separate perf report.
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
    # This simulates a more realistic, demanding use case.
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
    # A workload specifically designed to test performance on "messy" data,
    # with poor overlap, variant mismatches, and missing data.
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
    of output for easier debugging.

    Returns:
        bool: True if the command succeeded, False otherwise.
    """
    print_header(title, char="-")
    cmd_str_list = [str(c) for c in cmd]
    print(f"Executing: {' '.join(cmd_str_list)}", flush=True)
    
    output_lines = []
    process_failed = False
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
                # The traceback is printed via the test harness, no need to duplicate it here
                if not process_failed:
                    sys.stdout.write(f"  > {line_stripped}\n")
                    sys.stdout.flush()
                output_lines.append(line_stripped)

        retcode = proc.wait()
        
        if retcode != 0:
            process_failed = True
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

def main():
    """
    Main function to orchestrate data generation, profiling, and reporting.
    It iterates through a set of predefined workloads and runs perf on each.
    """
    WORKDIR.mkdir(exist_ok=True)
    
    # Track if any workload fails to exit with an error code at the end
    any_failures = False

    for name, params in WORKLOADS.items():
        print_header(f"STARTING WORKLOAD: {name}", char="*")
        
        # Create a dedicated subdirectory for this workload's data
        run_workdir = WORKDIR / name
        run_workdir.mkdir(exist_ok=True)

        generator = None
        try:
            # --- 1. Generate Synthetic Data ---
            print_header(f"[{name}] Generating synthetic data", char="=")
            generator = RealisticDataGenerator(workload_params=params, workdir=run_workdir)
            data_prefix, score_files = generator.generate_all_files()
            
            # Construct the arguments for the gnomon binary
            gnomon_args = []
            score_dir = run_workdir / f"scores_{name}"
            score_dir.mkdir(exist_ok=True)
            
            # Move score files to a dedicated directory and build args
            for sf_path in score_files:
                new_path = score_dir / sf_path.name
                sf_path.rename(new_path)
            
            # Gnomon can take a directory as input
            gnomon_args.extend(["--score", str(score_dir)])
            gnomon_args.append(str(data_prefix))

            # --- 2. Run `perf record` ---
            perf_data_file = WORKDIR / f"perf.{name}.data"
            perf_record_cmd = [
                "perf", "record", "-g", 
                "-o", str(perf_data_file), 
                "--", str(GNOMON_BINARY)
            ] + gnomon_args
            
            if not run_command(perf_record_cmd, title=f"[{name}] Running `perf record`"):
                print(f"❌ Profiling failed for workload '{name}'. Skipping report.")
                any_failures = True
                continue # Proceed to the next workload

            # --- 3. Generate `perf report` ---
            print_header(f"[{name}] Granular Profile Report", char="#")
            print("# Answers: 'Where did my program spend its time?'")
            print("# Displaying call-graph, showing only branches consuming > 2% of total time.")
            
            if perf_data_file.exists():
                perf_report_cmd = [
                    "perf", "report", 
                    "--stdio",
                    "--call-graph=graph", 
                    "--percent-limit=2",
                    "-i", str(perf_data_file)
                ]
                if not run_command(perf_report_cmd, title=f"[{name}] Generating `perf` Call-Graph Report"):
                    any_failures = True
            else:
                print(f"perf.data file not found for workload '{name}'. The profiling run may have failed.")
                any_failures = True

        except Exception as e:
            print(f"An unexpected error occurred during workload '{name}': {e}")
            import traceback
            traceback.print_exc()
            any_failures = True
        
        finally:
            if generator:
                # Cleanup the generated files for the current workload
                generator.cleanup()

    if any_failures:
        print("\n❌ One or more profiling workloads failed.")
        sys.exit(1)
    else:
        print("\n✅ All profiling workloads completed successfully.")


if __name__ == "__main__":
    if not GNOMON_BINARY.exists():
        print(f"❌ Error: Compiled profiling binary not found at {GNOMON_BINARY}")
        print("Please ensure you have built the project with: cargo build --profile profiling")
        sys.exit(1)

    main()
