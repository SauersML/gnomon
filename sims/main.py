from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG1_SCRIPT = REPO_ROOT / "sims" / "figure1_msprime_portability.py"
FIG2_SCRIPT = REPO_ROOT / "sims" / "figure2_stdpopsim_confounding.py"


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _assert_mgcv_available(env: dict[str, str]) -> None:
    # Hard requirement: GAM must run with R mgcv via rpy2.
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "os.environ.setdefault('RPY2_CFFI_MODE', 'API'); "
                "import rpy2.robjects as ro; "
                "from rpy2.robjects.packages import importr; "
                "importr('mgcv'); "
                "print('mgcv backend OK')"
            ),
        ],
        check=True,
        env=env,
    )


def _default_work_root() -> Path:
    for p in [Path("/dev/shm"), Path("/tmp")]:
        if p.exists():
            return (p / "gnomon_sims_work").resolve()
    return (Path("/tmp") / "gnomon_sims_work").resolve()


def _default_out_root() -> Path:
    return (REPO_ROOT / "sims" / "results_hpc").resolve()


def _jobs() -> int:
    return max(1, int(os.cpu_count() or 8))


def _apply_thread_env(env: dict[str, str], threads: int) -> None:
    t = str(max(1, int(threads)))
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "GCTB_THREADS",
    ):
        env[k] = t


def _detect_total_mem_mb() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return max(1, kb // 1024)
    except Exception:
        return None
    return None


def _keep_intermediates() -> bool:
    return False


def _clear_ramdisk_after() -> bool:
    return False


def _cohort_sizes(small: bool) -> dict[str, int]:
    if small:
        return {
            "fig1_n_afr": 300,
            "fig1_n_ooa_train": 300,
            "fig1_n_ooa_test": 300,
            "fig2_n_train_eur": 600,
            "fig2_n_test_per_pop": 150,
        }
    return {
        "fig1_n_afr": 10000,
        "fig1_n_ooa_train": 10000,
        "fig1_n_ooa_test": 10000,
        "fig2_n_train_eur": 12000,
        "fig2_n_test_per_pop": 3000,
    }


def setup_env() -> None:
    py = sys.executable
    pip = [py, "-m", "pip"]

    _run(pip + ["install", "--upgrade", "pip", "setuptools", "wheel"])
    _run(
        pip
        + [
            "install",
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "numba",
            "matplotlib",
            "demes",
            "demesdraw",
            "msprime",
            "stdpopsim",
            "rpy2",
            "pgscatalog-calc",
        ]
    )

    mamba = shutil.which("micromamba") or shutil.which("mamba") or shutil.which("conda")

    if not _exists("Rscript"):
        if not mamba:
            raise RuntimeError("Rscript not found and no conda/mamba executable was found on PATH.")
        _run([mamba, "install", "-y", "-c", "conda-forge", "r-base", "r-mgcv"])

    _run(
        [
            "Rscript",
            "-e",
            'if (!requireNamespace("mgcv", quietly=TRUE)) install.packages("mgcv", repos="https://cloud.r-project.org")',
        ]
    )

    if not (_exists("plink2") and _exists("gctb")):
        if mamba:
            _run(
                [
                    mamba,
                    "install",
                    "-y",
                    "-c",
                    "conda-forge",
                    "-c",
                    "bioconda",
                    "plink2",
                    "gctb",
                    "r-base",
                    "r-mgcv",
                ]
            )
        else:
            raise RuntimeError(
                "plink2 and/or gctb missing and no conda/mamba executable was found on PATH. "
                "Install micromamba/mamba/conda or preinstall plink2 and gctb."
            )

    for tool in ["python3", "plink2", "gctb", "Rscript"]:
        print(f"tool {tool}:", shutil.which(tool) or "MISSING")


def _run_fig1(
    env: dict[str, str],
    out_root: Path,
    work_root: Path,
    small: bool,
    total_threads: int,
    total_memory_mb: int | None,
    use_existing: bool,
    use_existing_dir: str | None,
) -> None:
    sizes = _cohort_sizes(small)
    out = out_root / "figure1"
    out.mkdir(parents=True, exist_ok=True)
    work = work_root / "figure1"
    work.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(FIG1_SCRIPT),
        "--out",
        str(out),
        "--work-root",
        str(work),
        "--n-afr",
        str(sizes["fig1_n_afr"]),
        "--n-ooa-train",
        str(sizes["fig1_n_ooa_train"]),
        "--n-ooa-test",
        str(sizes["fig1_n_ooa_test"]),
        "--total-threads",
        str(max(1, int(total_threads))),
    ]
    if total_memory_mb is not None:
        cmd.extend(["--memory-mb-total", str(max(1024, int(total_memory_mb)))])
    if small:
        cmd.extend(["--gens", "5", "20", "100", "500", "2000"])
    if use_existing:
        cmd.append("--use-existing")
        if use_existing_dir:
            cmd.extend(["--use-existing-dir", str(use_existing_dir)])
    if _keep_intermediates():
        cmd.append("--keep-intermediates")
    _run(cmd, env=env)


def _run_fig2(
    env: dict[str, str],
    out_root: Path,
    work_root: Path,
    small: bool,
    total_threads: int,
    total_memory_mb: int | None,
    use_existing: bool,
    use_existing_dir: str | None,
) -> None:
    sizes = _cohort_sizes(small)
    out = out_root / "figure2"
    out.mkdir(parents=True, exist_ok=True)
    work = work_root / "figure2"
    work.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(FIG2_SCRIPT),
        "--out",
        str(out),
        "--work-root",
        str(work),
        "--n-train-eur",
        str(sizes["fig2_n_train_eur"]),
        "--n-test-per-pop",
        str(sizes["fig2_n_test_per_pop"]),
        "--threads",
        str(max(1, int(total_threads))),
    ]
    if total_memory_mb is not None:
        cmd.extend(["--memory-mb", str(max(1024, int(total_memory_mb)))])
    if use_existing:
        cmd.append("--use-existing")
        if use_existing_dir:
            cmd.extend(["--use-existing-dir", str(use_existing_dir)])
    if _keep_intermediates():
        cmd.append("--keep-intermediates")
    _run(cmd, env=env)


def _resource_split(run_fig1: bool, run_fig2: bool) -> dict[str, int | None]:
    total_threads = _jobs()
    total_mem_mb = _detect_total_mem_mb()
    usable_mem_mb = int(0.90 * total_mem_mb) if total_mem_mb is not None else None

    if run_fig1 and run_fig2:
        fig1_threads = max(1, int(round(total_threads * 0.75)))
        fig2_threads = max(1, total_threads - fig1_threads)
        if total_threads >= 8 and fig2_threads < 4:
            fig2_threads = 4
            fig1_threads = max(1, total_threads - fig2_threads)
        if usable_mem_mb is not None:
            fig1_mem = max(4096, int(round(usable_mem_mb * 0.70)))
            fig2_mem = max(2048, usable_mem_mb - fig1_mem)
        else:
            fig1_mem = None
            fig2_mem = None
    elif run_fig1:
        fig1_threads, fig2_threads = total_threads, 0
        fig1_mem, fig2_mem = usable_mem_mb, None
    elif run_fig2:
        fig1_threads, fig2_threads = 0, total_threads
        fig1_mem, fig2_mem = None, usable_mem_mb
    else:
        fig1_threads = fig2_threads = 0
        fig1_mem = fig2_mem = None

    return {
        "total_threads": total_threads,
        "fig1_threads": fig1_threads,
        "fig2_threads": fig2_threads,
        "fig1_mem_mb": fig1_mem,
        "fig2_mem_mb": fig2_mem,
    }


def _zip_png_outputs(out_root: Path) -> Path | None:
    pngs: list[Path] = []
    for d in (out_root / "figure1", out_root / "figure2"):
        if d.exists():
            pngs.extend(sorted(d.glob("*.png")))
    if not pngs:
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = out_root / f"figures_{ts}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in pngs:
            zf.write(p, arcname=p.name)
    return zip_path


def _resolve_existing_dir(base: str | None, figure_subdir: str) -> str | None:
    if base is None:
        return None
    p = Path(base)
    sub = p / figure_subdir
    if sub.exists():
        return str(sub)
    return str(p)


def run_pipeline(figure: str, small: bool, use_existing: bool = False, use_existing_dir: str | None = None) -> None:
    out_root = _default_out_root()
    out_root.mkdir(parents=True, exist_ok=True)

    work_root = _default_work_root()
    work_root.mkdir(parents=True, exist_ok=True)

    jobs = _jobs()
    env = os.environ.copy()
    env.setdefault("RPY2_CFFI_MODE", "API")
    _apply_thread_env(env, jobs)
    _assert_mgcv_available(env)

    run_fig1 = figure in ("1", "both")
    run_fig2 = figure in ("2", "both")
    split = _resource_split(run_fig1=run_fig1, run_fig2=run_fig2)

    print(
        "Resource plan:",
        f"total_threads={split['total_threads']}",
        f"fig1_threads={split['fig1_threads']}",
        f"fig2_threads={split['fig2_threads']}",
        f"fig1_mem_mb={split['fig1_mem_mb']}",
        f"fig2_mem_mb={split['fig2_mem_mb']}",
    )

    # Run both figures concurrently when requested.
    if run_fig1 and run_fig2:
        fig1_env = env.copy()
        _apply_thread_env(fig1_env, int(split["fig1_threads"]))
        fig2_env = env.copy()
        _apply_thread_env(fig2_env, int(split["fig2_threads"]))

        fig1_existing_dir = _resolve_existing_dir(use_existing_dir, "figure1")
        fig2_existing_dir = _resolve_existing_dir(use_existing_dir, "figure2")
        p1 = subprocess.Popen(
            [
                sys.executable,
                str(FIG1_SCRIPT),
                "--out",
                str(out_root / "figure1"),
                "--work-root",
                str(work_root / "figure1"),
                "--n-afr",
                str(_cohort_sizes(small)["fig1_n_afr"]),
                "--n-ooa-train",
                str(_cohort_sizes(small)["fig1_n_ooa_train"]),
                "--n-ooa-test",
                str(_cohort_sizes(small)["fig1_n_ooa_test"]),
                "--total-threads",
                str(split["fig1_threads"]),
                *([] if split["fig1_mem_mb"] is None else ["--memory-mb-total", str(split["fig1_mem_mb"])]),
                *(["--gens", "5", "20", "100", "500", "2000"] if small else []),
                *(["--use-existing"] if use_existing else []),
                *([] if (not use_existing or fig1_existing_dir is None) else ["--use-existing-dir", fig1_existing_dir]),
                *(["--keep-intermediates"] if _keep_intermediates() else []),
            ],
            env=fig1_env,
        )
        p2 = subprocess.Popen(
            [
                sys.executable,
                str(FIG2_SCRIPT),
                "--out",
                str(out_root / "figure2"),
                "--work-root",
                str(work_root / "figure2"),
                "--n-train-eur",
                str(_cohort_sizes(small)["fig2_n_train_eur"]),
                "--n-test-per-pop",
                str(_cohort_sizes(small)["fig2_n_test_per_pop"]),
                "--threads",
                str(split["fig2_threads"]),
                *([] if split["fig2_mem_mb"] is None else ["--memory-mb", str(split["fig2_mem_mb"])]),
                *(["--use-existing"] if use_existing else []),
                *([] if (not use_existing or fig2_existing_dir is None) else ["--use-existing-dir", fig2_existing_dir]),
                *(["--keep-intermediates"] if _keep_intermediates() else []),
            ],
            env=fig2_env,
        )

        rc1 = p1.wait()
        rc2 = p2.wait()
        if rc1 != 0 or rc2 != 0:
            raise subprocess.CalledProcessError(
                rc1 if rc1 != 0 else rc2,
                "parallel figure run",
            )
    else:
        if run_fig1:
            fig1_existing_dir = _resolve_existing_dir(use_existing_dir, "figure1")
            _run_fig1(
                env,
                out_root,
                work_root,
                small,
                total_threads=int(split["fig1_threads"]),
                total_memory_mb=split["fig1_mem_mb"],
                use_existing=use_existing,
                use_existing_dir=fig1_existing_dir,
            )
        if run_fig2:
            fig2_existing_dir = _resolve_existing_dir(use_existing_dir, "figure2")
            _run_fig2(
                env,
                out_root,
                work_root,
                small,
                total_threads=int(split["fig2_threads"]),
                total_memory_mb=split["fig2_mem_mb"],
                use_existing=use_existing,
                use_existing_dir=fig2_existing_dir,
            )

    if _clear_ramdisk_after() and work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)

    zip_path = _zip_png_outputs(out_root)

    print("Done. Outputs:")
    if run_fig1:
        print(" -", out_root / "figure1")
    if run_fig2:
        print(" -", out_root / "figure2")
    if zip_path is not None:
        print(" -", zip_path)


def clean_ramdisk() -> None:
    p = _default_work_root()
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
        print("Removed", p)
    else:
        print("No ramdisk work dir found at", p)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Figure 1/2 HPC runner (defaults to full cohort)")
    sub = p.add_subparsers(dest="cmd", required=False)

    ps = sub.add_parser("setup", help="Install dependencies, then run")
    ps.add_argument("--small", action="store_true", help="Use smaller debug cohorts")
    ps.add_argument("--figure", choices=["1", "2", "both"], default="both")
    ps.add_argument("--use-existing", action="store_true", help="Reuse existing per-seed PLINK/TSV files and skip simulation.")
    ps.add_argument("--use-existing-dir", default=None, help="Directory holding existing reuse files (or parent containing figure1/figure2).")

    pr = sub.add_parser("run", help="Run simulations")
    pr.add_argument("--small", action="store_true", help="Use smaller debug cohorts")
    pr.add_argument("--figure", choices=["1", "2", "both"], default="both")
    pr.add_argument("--use-existing", action="store_true", help="Reuse existing per-seed PLINK/TSV files and skip simulation.")
    pr.add_argument("--use-existing-dir", default=None, help="Directory holding existing reuse files (or parent containing figure1/figure2).")

    sub.add_parser("clean-ramdisk", help="Delete transient work directory")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # zero-arg => setup + run (full cohort)
    if getattr(args, "cmd", None) is None:
        setup_env()
        run_pipeline(figure="both", small=False)
        return

    if args.cmd == "setup":
        setup_env()
        run_pipeline(
            figure=args.figure,
            small=bool(args.small),
            use_existing=bool(args.use_existing),
            use_existing_dir=args.use_existing_dir,
        )
        return

    if args.cmd == "run":
        run_pipeline(
            figure=args.figure,
            small=bool(args.small),
            use_existing=bool(args.use_existing),
            use_existing_dir=args.use_existing_dir,
        )
        return

    if args.cmd == "clean-ramdisk":
        clean_ramdisk()
        return


if __name__ == "__main__":
    main()
