"""Real-P+T ancestry-calibration study runner (the only sims entry point).

Pipeline:
  1. generate real-P+T data for serial1d + grid2d x seeds (4 phenotypes each)
  2. fit binary recalibration models + ground-truth metrics (per dataset)
  3. require every prespecified binary method to complete
  4. aggregate the per-dataset CSVs and render figures

Generation is RAM-bound (each streamed sim holds reservoirs + one chunk), so its
fan-out is sized from currently-free RAM/cores. Fitting is light and fans out per
dataset. Partial results are debugging artifacts and never satisfy completion.
This legacy PGS_z study is not yet the external-reference CTN experiment.
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
ANALYZE = HERE / "analyze_results.py"
PLOT = HERE / "plot_results.py"
PLOT_DESIGN = HERE / "plot_figure1.py"
OUT = Path("sims/results_hpc/ancestry_calibration")

DEMOGRAPHIES = ("serial1d", "grid2d")
PHENOS = ("phenoA", "phenoR", "phenoB", "phenoC")
SEEDS = tuple(range(1, 11))      # seed = inferential unit; averages out P+T threshold noise
CENTERS = 12                     # gamfit marginal-slope surface centers (gam#979: keep modest)

GEN_TASK_PEAK_GIB = 22.0         # one streamed sim's peak RAM (reservoirs + a chunk)
MEM_HEADROOM_FRAC = 0.70
THREADS_PER_GEN_JOB = 1
FIT_JOBS = 1


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
    jobs = max(1, min(n_tasks, by_mem, free_cores, 1))
    print(f"auto generate jobs={jobs} (free RAM {mem_avail_gib:.0f} GiB, free cores {free_cores})", flush=True)
    return jobs


def run_logged(cmd, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = " ".join(str(c) for c in cmd)
    print(f"+ {text}  > {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"+ {text}\n")
        log.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, cmd)


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
    log_dir = out / "logs"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    os.environ.update(env)
    for d in (data_dir, res_b, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    (out / "run_config.json").write_text(json.dumps({
        "demographies": list(DEMOGRAPHIES), "phenotypes": list(PHENOS), "seeds": list(SEEDS),
        "centers": CENTERS, "pgs": "real P+T only", "outcomes": ["binary"],
        "score_transform": "discovery-standardized PGS_z; not the external-CTN experiment",
        "methods_binary": ["gamfit", "linpc", "znorm", "calpred", "rawpgs"],
        "risk_metrics": "ground-truth vs p_true: average prediction error, probit risk spread ratio, "
                        "RMSE, MAE, Spearman/Pearson/R2 to true risk, high-risk-tail RMSE, "
                        "top-10%/20% risk-ratio (true+observed), %-at->=2x/3x risk (true+predicted); "
                        "plus Brier Skill Score, OR per SD, and calibration slope+intercept",
        "discrimination": "binary AUC globally and within ancestry strata",
    }, indent=2))

    # 1. generation (skip a (dem, seed) whose full pheno set is already on disk;
    #    set ANC_REGEN=1 to force a clean regeneration)
    force_regen = os.environ.get("ANC_REGEN", "0") != "0"
    gen_tasks = []
    for dem in DEMOGRAPHIES:
        for seed in SEEDS:
            cached = all((data_dir / f"{dem}_{ph}_realpt_s{seed}.parquet").exists() for ph in PHENOS)
            if cached and not force_regen:
                print(f"cache hit: {dem} s{seed} ({len(PHENOS)} phenos) — skipping generation", flush=True)
                continue
            cmd = [sys.executable, GEN, dem, data_dir, seed, "--tag", f"_s{seed}", "--threads", "1"]
            gen_tasks.append((cmd, log_dir / f"gen_{dem}_s{seed}.log"))
    run_many(gen_tasks, auto_generate_jobs(len(gen_tasks)))

    # 2. Fit every prespecified binary dataset.
    datasets = []
    for dem in DEMOGRAPHIES:
        for seed in SEEDS:
            for pheno in PHENOS:
                p = data_dir / f"{dem}_{pheno}_realpt_s{seed}.parquet"
                if p.exists():
                    datasets.append((dem, pheno, seed, p))
                else:
                    raise FileNotFoundError(f"required dataset missing: {p}")

    fit_tasks = []
    for dem, pheno, seed, p in datasets:
        stem = f"{dem}_{pheno}_s{seed}"
        fit_tasks.append((
            [sys.executable, FIT_BINARY, "--data", p, "--dem", dem, "--pheno", pheno,
             "--centers", CENTERS, "--out-acc", res_b / f"{stem}_acc.csv",
             "--out-cal", res_b / f"{stem}_cal.csv", "--out-pred", res_b / f"{stem}_pred.parquet",
             "--out-status", res_b / f"{stem}_status.json"],
            log_dir / f"fitb_{stem}.log"))
    run_many(fit_tasks, FIT_JOBS)

    # Required failures above prevent aggregation and figure generation.
    run_logged([sys.executable, ANALYZE], log_dir / "analyze.log")
    run_logged([sys.executable, PLOT_DESIGN], log_dir / "plot_figure1.log")
    run_logged([sys.executable, PLOT], log_dir / "plot.log")


if __name__ == "__main__":
    main()
