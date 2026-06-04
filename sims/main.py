from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_STUDY = REPO_ROOT / "sims" / "ancestry_calibration" / "run_study.py"
PLINK1 = Path("/common/software/install/migrated/plink/1.90b6.10/bin/plink")
PLINK2 = Path("/users/0/sauer354/bin/plink2")

REQUIRED_IMPORTS = [
    "gamfit",
    "matplotlib",
    "msprime",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "sklearn",
]


def run(cmd: list[object]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def check() -> None:
    missing = []
    for module in REQUIRED_IMPORTS:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(f"missing Python packages: {', '.join(missing)}")
    for tool in (PLINK1, PLINK2):
        if not tool.exists():
            raise RuntimeError(f"missing required executable: {tool}")


def main() -> None:
    check()
    run([sys.executable, RUN_STUDY])


if __name__ == "__main__":
    main()
