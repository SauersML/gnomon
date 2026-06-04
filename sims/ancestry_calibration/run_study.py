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
SEEDS = (1, 2, 3, 7)
CENTERS = 20
CHUNKS = 20
CHUNK_BP = 5_000_000
THREADS_PER_GENERATION_JOB = 4
PLINK_MEMORY_MB_PER_GENERATION_JOB = 48_000
N_CAUSAL = 150
GENERATE_JOBS = 6
FIT_JOBS = 6


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
            "distance/deme slope error vs known true slope",
            "distance/deme prevalence error vs known true prevalence",
            "mean absolute error vs p_true",
            "individual absolute error vs p_true",
        ],
        "parallelism": {
            "generate_jobs": GENERATE_JOBS,
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
    run_many(generate_tasks, GENERATE_JOBS)

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
