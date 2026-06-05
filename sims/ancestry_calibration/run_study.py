"""Unified real-P+T ancestry-calibration simulation runner.

This is the only simulation entry point used by sims/main.py. It runs:
1. real P+T data generation for serial1d/grid2d
2. binary risk-model fitting on model_train
3. held-out test metrics for Figure 2

Only binary real-P+T data and metrics are produced here.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
GEN = HERE / "gen_real_pt.py"
FIT = HERE / "fit_binary.py"
PLOT = HERE / "plot_results.py"
OUT = Path("sims/results_hpc/ancestry_calibration")

DEMOGRAPHIES = ("serial1d", "grid2d")
SEEDS = tuple(range(1, 21))  # 20 replicates: seed = inferential unit (power)
CENTERS = 20
CHUNKS = 200          # provenance; live values are gen_real_pt.DEFAULT_CHUNKS/_CHUNK_BP
CHUNK_BP = 500_000
# Live value is gen_real_pt.DEFAULT_THREADS; kept here for the cpu-budget formula.
THREADS_PER_GENERATION_JOB = 8
PLINK_MEMORY_MB_PER_GENERATION_JOB = 24_000  # provenance; live: gen_real_pt.DEFAULT_MEM_MB
N_CAUSAL = 4500  # provenance only; must match gen_real_pt.N_CAUSAL
# Peak RAM of ONE int8-streaming generate task: a chunk's int8 genotype matrix
# (~8 GiB at biobank n) + its dosage view (~4 GiB) + the kept reservoirs (~6 GiB),
# rounded up for slack. The job count is NOT hardcoded -- it is computed at launch
# from whatever RAM and cores are actually free (the node is shared).
GEN_TASK_PEAK_GIB = 22.0
MEM_HEADROOM_FRAC = 0.70  # only spend this share of free RAM; leave room for others
FIT_JOBS = 6  # provenance only; fit_binary runs as one process over all datasets


def auto_generate_jobs(n_tasks: int) -> int:
    """Pick the generate fan-out from currently-free RAM and cores, not a fixed
    number: the node is shared, so the safe parallelism depends on who else is on it."""
    mem_avail_gib = 64.0
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    mem_avail_gib = int(line.split()[1]) / (1024 ** 2)  # kB -> GiB
                    break
    except OSError:
        pass
    try:
        free_cores = max(1, int((os.cpu_count() or 8) - os.getloadavg()[0]))
    except OSError:
        free_cores = os.cpu_count() or 8
    by_mem = int((mem_avail_gib * MEM_HEADROOM_FRAC) // GEN_TASK_PEAK_GIB)
    by_cpu = free_cores // THREADS_PER_GENERATION_JOB
    jobs = max(1, min(n_tasks, by_mem, by_cpu))
    print(f"auto generate jobs={jobs} (free RAM {mem_avail_gib:.0f} GiB -> {by_mem} by mem; "
          f"free cores {free_cores} / {THREADS_PER_GENERATION_JOB} thr -> {by_cpu} by cpu)", flush=True)
    return jobs


def tail(path: Path, n: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def run_logged(cmd: list[object], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text_cmd = " ".join(str(c) for c in cmd)
    print(f"+ {text_cmd}  > {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"+ {text_cmd}\n")
        log.flush()
        result = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}: {text_cmd}\n\nLog tail:\n{tail(log_path)}")


def run_many(tasks: list[tuple[str, list[object], Path]], jobs: int) -> None:
    if not tasks:
        return
    jobs = max(1, min(int(jobs), len(tasks)))
    print(f"running {len(tasks)} tasks with jobs={jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_logged, cmd, log_path): name for name, cmd, log_path in tasks}
        for fut in as_completed(futures):
            name = futures[fut]
            fut.result()
            print(f"done {name}", flush=True)


def main() -> None:
    out = OUT.resolve()
    data_dir = out / "data"
    results_dir = out / "results"
    log_dir = out / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    generate_jobs = auto_generate_jobs(len(DEMOGRAPHIES) * len(SEEDS))

    config = {
        "demographies": list(DEMOGRAPHIES),
        "seeds": list(SEEDS),
        "centers": CENTERS,
        "pgs": "real P+T only",
        "outcome": "binary only",
        "split": {
            "pgs_train": "single training deme, large cohort, P+T only",
            "model_train": "all demes, risk-model training only",
            "test": "all demes, reported metrics only",
        },
        "metrics": [
            "global AUC",
            "global liability-scale pseudo-R2",
            "global Brier",
            "distance-bin AUC",
            "distance-bin liability-scale pseudo-R2",
            "distance-bin Brier",
            "distance/deme prevalence error vs known true prevalence",
            "mean absolute error vs p_true",
            "individual absolute error vs p_true",
        ],
        "parallelism": {
            "generate_jobs": generate_jobs,
            "fit_jobs": FIT_JOBS,
            "threads_per_generation_job": THREADS_PER_GENERATION_JOB,
            "plink_memory_mb_per_generation_job": PLINK_MEMORY_MB_PER_GENERATION_JOB,
            "detected_cpus": int(os.cpu_count() or 1),
        },
        "generation": {
            "chunks": CHUNKS,
            "chunk_bp": CHUNK_BP,
            "n_causal": N_CAUSAL,
        },
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2))

    generate_tasks: list[tuple[str, list[object], Path]] = []
    for dem in DEMOGRAPHIES:
        for seed in SEEDS:
            tag = f"_s{seed}"
            name = f"generate_{dem}_s{seed}"
            generate_tasks.append(
                (
                    name,
                    [
                        sys.executable,
                        GEN,
                        dem,
                        seed,
                    ],
                    log_dir / f"{name}.log",
                )
            )
    run_many(generate_tasks, generate_jobs)

    datasets = []
    for dem in DEMOGRAPHIES:
        for seed in SEEDS:
            for pheno in ("phenoA", "phenoB"):
                path = data_dir / f"{dem}_{pheno}_realpt_s{seed}.parquet"
                if not path.exists():
                    raise FileNotFoundError(path)
                datasets.append(path)

    run_logged(
        [
            sys.executable,
            FIT,
            *datasets,
        ],
        log_dir / "fit_binary.log",
    )
    run_logged(
        [
            sys.executable,
            PLOT,
        ],
        log_dir / "plot_results.log",
    )


if __name__ == "__main__":
    main()
