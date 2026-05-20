"""`gnomon calibrate` — train / apply GAM and survival models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

from ._api import (
    CalibrateResult,
    PathLike,
    _run,
    locate_binary,
)

__all__ = ["train", "infer"]


def train(
    training_data: PathLike,
    *,
    num_pcs: int,
    model_family: str = "gam",
    pgs_knots: int = 10,
    pgs_degree: int = 3,
    pc_knots: int = 5,
    pc_degree: int = 2,
    penalty_order: int = 2,
    max_iterations: int = 50,
    convergence_tolerance: float = 1e-7,
    reml_max_iterations: int = 100,
    reml_convergence_tolerance: float = 1e-3,
    output: Optional[PathLike] = None,
    binary: Optional[PathLike] = None,
    extra_args: Optional[Iterable[str]] = None,
    timeout: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[PathLike] = None,
) -> CalibrateResult:
    """Run ``gnomon train``.

    ``training_data`` is the path to a TSV with phenotype/score/PC columns
    (see the gnomon docs for the expected schema).
    """
    bin_path = locate_binary(binary)
    argv = [
        "train",
        str(training_data),
        "--model-family", model_family,
        "--num-pcs", str(num_pcs),
        "--pgs-knots", str(pgs_knots),
        "--pgs-degree", str(pgs_degree),
        "--pc-knots", str(pc_knots),
        "--pc-degree", str(pc_degree),
        "--penalty-order", str(penalty_order),
        "--max-iterations", str(max_iterations),
        "--convergence-tolerance", str(convergence_tolerance),
        "--reml-max-iterations", str(reml_max_iterations),
        "--reml-convergence-tolerance", str(reml_convergence_tolerance),
    ]
    if output is not None:
        argv += ["--output", str(output)]
    if extra_args:
        argv += list(extra_args)
    completed = _run(bin_path, argv, timeout=timeout, env=env, cwd=cwd)
    outs = ()
    if output is not None and Path(output).exists():
        outs = (Path(output),)
    return CalibrateResult(
        output_paths=outs,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def infer(
    test_data: PathLike,
    *,
    model: PathLike,
    output: Optional[PathLike] = None,
    binary: Optional[PathLike] = None,
    extra_args: Optional[Iterable[str]] = None,
    timeout: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[PathLike] = None,
) -> CalibrateResult:
    """Run ``gnomon infer`` (apply a trained calibration model)."""
    bin_path = locate_binary(binary)
    argv = ["infer", str(test_data), "--model", str(model)]
    if output is not None:
        argv += ["--output", str(output)]
    if extra_args:
        argv += list(extra_args)
    completed = _run(bin_path, argv, timeout=timeout, env=env, cwd=cwd)
    outs = ()
    if output is not None and Path(output).exists():
        outs = (Path(output),)
    return CalibrateResult(
        output_paths=outs,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )
