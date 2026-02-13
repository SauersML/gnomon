from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG1_SCRIPT = REPO_ROOT / "sims" / "figure1_msprime_portability.py"
FIG2_SCRIPT = REPO_ROOT / "sims" / "figure2_stdpopsim_confounding.py"


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _default_work_root() -> Path:
    env_val = os.environ.get("SIMS_WORK_ROOT")
    if env_val:
        return Path(env_val).resolve()
    for p in [Path("/dev/shm"), Path("/tmp")]:
        if p.exists():
            return (p / "gnomon_sims_work").resolve()
    return (Path("/tmp") / "gnomon_sims_work").resolve()


def _default_out_root() -> Path:
    env_val = os.environ.get("SIMS_OUT_ROOT")
    if env_val:
        return Path(env_val).resolve()
    return (REPO_ROOT / "sims" / "results_hpc").resolve()


def _jobs() -> int:
    return max(1, int(os.environ.get("SIMS_JOBS", max(1, (os.cpu_count() or 8) // 2))))


def _install_tools_with_conda() -> bool:
    return os.environ.get("SIMS_INSTALL_TOOLS_WITH_CONDA", "0").strip().lower() in {"1", "true", "yes"}


def _keep_intermediates() -> bool:
    return os.environ.get("SIMS_KEEP_INTERMEDIATES", "0").strip().lower() in {"1", "true", "yes"}


def _clear_ramdisk_after() -> bool:
    return os.environ.get("SIMS_CLEAR_RAMDISK_AFTER", "0").strip().lower() in {"1", "true", "yes"}


def _cohort_sizes(small: bool) -> dict[str, int]:
    if small:
        return {
            "fig1_n_afr": 100,
            "fig1_n_ooa_train": 100,
            "fig1_n_ooa_test": 100,
            "fig2_n_train_eur": 120,
            "fig2_n_test_per_pop": 30,
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
            "matplotlib",
            "msprime",
            "stdpopsim",
            "rpy2",
            "pgscatalog-calc",
        ]
    )

    if _exists("Rscript"):
        _run(
            [
                "Rscript",
                "-e",
                'if (!requireNamespace("mgcv", quietly=TRUE)) install.packages("mgcv", repos="https://cloud.r-project.org")',
            ]
        )
    else:
        raise RuntimeError("Rscript not found. mgcv is required and must be installed.")

    if not (_exists("plink2") and _exists("gctb")):
        mamba = shutil.which("micromamba") or shutil.which("mamba") or shutil.which("conda")
        if mamba and _install_tools_with_conda():
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
                "plink2 and/or gctb missing. Install them, or set SIMS_INSTALL_TOOLS_WITH_CONDA=1 "
                "with a conda/mamba executable on PATH."
            )

    for tool in ["python3", "plink2", "gctb", "Rscript"]:
        print(f"tool {tool}:", shutil.which(tool) or "MISSING")


def _run_fig1(env: dict[str, str], out_root: Path, work_root: Path, small: bool) -> None:
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
    ]
    if _keep_intermediates():
        cmd.append("--keep-intermediates")
    _run(cmd, env=env)


def _run_fig2(env: dict[str, str], out_root: Path, work_root: Path, small: bool) -> None:
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
    ]
    if _keep_intermediates():
        cmd.append("--keep-intermediates")
    _run(cmd, env=env)


def run_pipeline(figure: str, small: bool) -> None:
    out_root = _default_out_root()
    out_root.mkdir(parents=True, exist_ok=True)

    work_root = _default_work_root()
    work_root.mkdir(parents=True, exist_ok=True)

    jobs = _jobs()
    env = os.environ.copy()
    env.setdefault("RPY2_CFFI_MODE", "ABI")
    env.setdefault("OMP_NUM_THREADS", str(jobs))
    env.setdefault("OPENBLAS_NUM_THREADS", str(jobs))
    env.setdefault("MKL_NUM_THREADS", str(jobs))
    env.setdefault("NUMEXPR_NUM_THREADS", str(jobs))
    env.setdefault("GCTB_THREADS", str(jobs))

    run_fig1 = figure in ("1", "both")
    run_fig2 = figure in ("2", "both")

    # Deterministic orchestration: fixed execution order.
    if run_fig1:
        _run_fig1(env, out_root, work_root, small)
    if run_fig2:
        _run_fig2(env, out_root, work_root, small)

    if _clear_ramdisk_after() and work_root.exists():
        shutil.rmtree(work_root, ignore_errors=True)

    print("Done. Outputs:")
    if run_fig1:
        print(" -", out_root / "figure1")
    if run_fig2:
        print(" -", out_root / "figure2")


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

    pr = sub.add_parser("run", help="Run simulations")
    pr.add_argument("--small", action="store_true", help="Use smaller debug cohorts")
    pr.add_argument("--figure", choices=["1", "2", "both"], default="both")

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
        run_pipeline(figure=args.figure, small=bool(args.small))
        return

    if args.cmd == "run":
        run_pipeline(figure=args.figure, small=bool(args.small))
        return

    if args.cmd == "clean-ramdisk":
        clean_ramdisk()
        return


if __name__ == "__main__":
    main()
