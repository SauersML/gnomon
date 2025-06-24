import re
import itertools
import io
import csv
from typing import List, Dict, Any, Set, Tuple

def parse_and_validate_benchmark_logs(log_content: str) -> str:
    # === 1. Source of Truth Definition ===
    # These parameter lists are derived from the benchmark logs and must be
    # updated if the benchmark suite changes.
    EXPECTED_N_VALUES = [1, 100, 1000, 5000, 10000, 40000]
    EXPECTED_K_VALUES = [1, 5, 50, 100]
    EXPECTED_SUBSET_VALUES = [1, 5, 50, 100]
    EXPECTED_FREQ_VALUES = [0.00001, 0.001, 0.02, 0.4]
    EXPECTED_PATH_VALUES = ['No-Pivot', 'Pivot']

    # Generate all expected unique benchmark combinations
    expected_combinations: Set[Tuple] = set(itertools.product(
        EXPECTED_N_VALUES,
        EXPECTED_K_VALUES,
        EXPECTED_SUBSET_VALUES,
        EXPECTED_FREQ_VALUES,
        EXPECTED_PATH_VALUES
    ))
    expected_count = len(expected_combinations)

    # === 2. Parsing Logic ===
    # Regex to capture parameters from "Benchmarking..." lines.
    # e.g., "Benchmarking Path Crossover (Multi-Dimensional)/No-Pivot__N=1_K=1_Subset=1%_Freq=0.000/0.00001"
    param_regex = re.compile(
        r"Benchmarking Path Crossover \(Multi-Dimensional\)/"
        r"((?:No-)?Pivot)__N=(\d+)_K=(\d+)_Subset=(\d+)%_Freq=[\d.]+/([\d.]+)"
    )

    # Regex to capture median time and unit from "time: [...]" lines.
    # e.g., "time:   [5.1482 ns 5.1597 ns 5.1757 ns]"
    time_regex = re.compile(r"time:\s+\[.*? ([\d.]+) (ns|µs|ms) .*?\]")

    parsed_records: List[Dict[str, Any]] = []
    current_params: Dict[str, Any] = {}

    def normalize_to_ns(value: float, unit: str) -> float:
        """Converts a time value to nanoseconds for consistency."""
        if unit == 'µs':
            return value * 1_000
        if unit == 'ms':
            return value * 1_000_000
        return value

    for line in log_content.splitlines():
        param_match = param_regex.search(line)
        if param_match:
            groups = param_match.groups()
            current_params = {
                'Path': groups[0],
                'N (Cohort)': int(groups[1]),
                'K (Scores)': int(groups[2]),
                'Subset': int(groups[3]),
                'Freq': float(groups[4]),
            }
            continue

        time_match = time_regex.search(line)
        if time_match and current_params:
            time_value = float(time_match.group(1))
            time_unit = time_match.group(2)
            
            record = current_params.copy()
            record['Time (Median)'] = normalize_to_ns(time_value, time_unit)
            
            parsed_records.append(record)
            current_params = {} # Reset state for the next benchmark

    # === 3. Validation Phase ===
    parsed_count = len(parsed_records)

    # Check 1: Simple row count
    if parsed_count != expected_count:
        raise ValueError(
            f"Validation Error: Mismatch in record counts. "
            f"Expected: {expected_count}, Parsed: {parsed_count}."
        )

    # Check 2: Integrity check for missing or extra combinations
    parsed_combinations: Set[Tuple] = set(
        (
            rec['N (Cohort)'],
            rec['K (Scores)'],
            rec['Subset'],
            rec['Freq'],
            rec['Path']
        )
        for rec in parsed_records
    )

    missing_benchmarks = expected_combinations - parsed_combinations
    extra_benchmarks = parsed_combinations - expected_combinations

    error_messages = []
    if missing_benchmarks:
        examples = [f'N={b[0]}, K={b[1]}, Subset={b[2]}%, Freq={b[3]}, Path={b[4]}' for b in list(missing_benchmarks)[:5]]
        error_messages.append(
            f"{len(missing_benchmarks)} benchmark(s) are missing. First 5 examples:\n"
            f"{examples}"
        )
    
    if extra_benchmarks:
        examples = [f'N={b[0]}, K={b[1]}, Subset={b[2]}%, Freq={b[3]}, Path={b[4]}' for b in list(extra_benchmarks)[:5]]
        error_messages.append(
            f"{len(extra_benchmarks)} unexpected or duplicate benchmark(s) were found. First 5 examples:\n"
            f"{examples}"
        )

    if error_messages:
        raise ValueError("Validation Failed:\n\n" + "\n\n".join(error_messages))
        
    # === 4. Output Generation ===
    if not parsed_records:
        return "" # Return empty string if no records were found

    output = io.StringIO()
    header = ['N (Cohort)', 'K (Scores)', 'Subset', 'Freq', 'Path', 'Time (Median)']
    writer = csv.DictWriter(output, fieldnames=header)
    
    writer.writeheader()
    writer.writerows(parsed_records)
    
    return output.getvalue()
