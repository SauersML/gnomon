"""Real-P+T ancestry-calibration study runner (the only sims entry point).

Pipeline:
  1. generate real-P+T data for serial1d + grid2d x seeds (3 phenotypes each)
  2. fit binary recalibration models + ground-truth metrics (per dataset)
  3. fit survival recalibration models + ground-truth metrics (per dataset)
  4. aggregate the per-dataset CSVs into study-level tables
  5. render the figures

Generation is RAM-bound (each streamed sim holds reservoirs + one chunk), so its
fan-out is sized from currently-free RAM/cores. Fitting is light and fans out per
dataset. The survival fit is run under a wall-clock timeout because gamfit survival
marginal-slope can stall (gam#979); fit_survival flushes the Cox baselines before
attempting gamfit, so a timeout still leaves the baselines on disk.
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
FIT_BINARY = HERE / "fit_binary.py"
FIT_SURVIVAL = HERE / "fit_survival.py"
ANALYZE = HERE / "analyze_results.py"
PLOT = HERE / "plot_results.py"
OUT = Path("sims/results_hpc/ancestry_calibration")

DEMOGRAPHIES = ("serial1d", "grid2d")
PHENOS = ("phenoA", "phenoB", "phenoC")
SEEDS = tuple(range(1, 11))      # seed = inferential unit; averages out P+T threshold noise
CENTERS = 12                     # gamfit marginal-slope surface centers (gam#979: keep modest)
SURVIVAL_TIMEOUT_S = 1500        # cap a gamfit survival-MS stall; baselines already flushed

GEN_TASK_PEAK_GIB = 22.0         # one streamed sim's peak RAM (reservoirs + a chunk)
MEM_HEADROOM_FRAC = 0.70
THREADS_PER_GEN_JOB = 1
FIT_JOBS = 12


def auto_generate_jobs(n_tasks: int) -> int:
    """Size generation fan-out from currently-free RAM/cores (the node is shared)."""
    mem_avail_gib = 64.0
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    mem_avail_gib = int(line.split()[1]) / (1024 ** 2)
                    break
    except OSError:
        pass
    try:
        free_cores = max(1, int((os.cpu_count() or 8) - os.getloadavg()[0]))
    except OSError:
        free_cores = os.cpu_count() or 8
    by_mem = int((mem_avail_gib * MEM_HEADROOM_FRAC) // GEN_TASK_PEAK_GIB)
    jobs = max(1, min(n_tasks, by_mem, free_cores))
    print(f"auto generate jobs={jobs} (free RAM {mem_avail_gib:.0f} GiB, free cores {free_cores})", flush=True)
    return jobs


def run_logged(cmd, log_path: Path, timeout=None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = " ".join(str(c) for c in cmd)
    print(f"+ {text}  > {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"+ {text}\n")
        log.flush()
        try:
            r = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout}s (partial results, if any, were flushed)\n")
            print(f"  TIMEOUT {log_path.name} (non-fatal)", flush=True)
            return
    if r.returncode != 0:
        # a single dataset/method failing must not abort the study
        print(f"  WARN rc={r.returncode} {log_path.name} (non-fatal)", flush=True)


def run_many(tasks, jobs: int) -> None:
    if not tasks:
        return
    jobs = max(1, min(int(jobs), len(tasks)))
    print(f"running {len(tasks)} tasks with jobs={jobs}", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(run_logged, *t): t[1].name for t in tasks}
        for fut in as_completed(futures):
            fut.result()
            print(f"done {futures[fut]}", flush=True)


def main() -> None:
    out = OUT.resolve()
    data_dir = out / "data"
    res_b = out / "results" / "binary"
    res_s = out / "results" / "survival"
    log_dir = out / "logs"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    os.environ.update(env)
    for d in (data_dir, res_b, res_s, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    (out / "run_config.json").write_text(json.dumps({
        "demographies": list(DEMOGRAPHIES), "phenotypes": list(PHENOS), "seeds": list(SEEDS),
        "centers": CENTERS, "pgs": "real P+T only", "outcomes": ["binary", "survival"],
        "methods_binary": ["gamfit", "linpc", "znorm", "calpred", "rawpgs"],
        "methods_survival": ["gamfit", "linpc", "znorm", "rawpgs"],
        "calibration": "ground-truth: probit slope + Brier Skill Score (Murphy reliability/resolution)",
        "discrimination": "GLOBAL only (binary AUC / survival Harrell C); never within-stratum",
    }, indent=2))

    # 1. generation
    gen_tasks = []
    for dem in DEMOGRAPHIES:
        for seed in SEEDS:
            cmd = [sys.executable, GEN, dem, data_dir, seed, "--tag", f"_s{seed}", "--threads", "1"]
            gen_tasks.append((cmd, log_dir / f"gen_{dem}_s{seed}.log"))
    run_many(gen_tasks, auto_generate_jobs(len(gen_tasks)))

    # 2+3. fitting (per dataset, both arms)
    datasets = []
    for dem in DEMOGRAPHIES:
        for seed in SEEDS:
            for pheno in PHENOS:
                p = data_dir / f"{dem}_{pheno}_realpt_s{seed}.parquet"
                if p.exists():
                    datasets.append((dem, pheno, seed, p))
                else:
                    print(f"WARN missing dataset {p}", flush=True)

    fit_tasks = []
    for dem, pheno, seed, p in datasets:
        stem = f"{dem}_{pheno}_s{seed}"
        fit_tasks.append((
            [sys.executable, FIT_BINARY, "--data", p, "--dem", dem, "--pheno", pheno,
             "--centers", CENTERS, "--out-acc", res_b / f"{stem}_acc.csv",
             "--out-cal", res_b / f"{stem}_cal.csv", "--out-pred", res_b / f"{stem}_pred.parquet"],
            log_dir / f"fitb_{stem}.log", None))
        fit_tasks.append((
            [sys.executable, FIT_SURVIVAL, "--data", p, "--dem", dem, "--pheno", pheno,
             "--centers", CENTERS, "--out-acc", res_s / f"{stem}_acc.csv",
             "--out-cal", res_s / f"{stem}_cal.csv"],
            log_dir / f"fits_{stem}.log", SURVIVAL_TIMEOUT_S))
    run_many(fit_tasks, FIT_JOBS)

    # 4. aggregate + 5. plot
    run_logged([sys.executable, ANALYZE], log_dir / "analyze.log")
    run_logged([sys.executable, PLOT], log_dir / "plot.log")


if __name__ == "__main__":
    main()
