"""`gnomon map` — fit / project HWE PCA models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

from ._api import (
    InvalidConfig,
    MapResult,
    PathLike,
    _run,
    locate_binary,
)

__all__ = ["fit", "project"]


def fit(
    genotype_path: PathLike,
    *,
    components: int,
    variant_list: Optional[PathLike] = None,
    maf: Optional[float] = None,
    ld: bool = False,
    sites_window: Optional[int] = None,
    bp_window: Optional[int] = None,
    binary: Optional[PathLike] = None,
    extra_args: Optional[Iterable[str]] = None,
    timeout: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[PathLike] = None,
) -> MapResult:
    """Run ``gnomon fit`` (HWE PCA model fitting)."""
    if components <= 0:
        raise InvalidConfig("components must be > 0")
    if (sites_window is not None or bp_window is not None) and not ld:
        raise InvalidConfig("sites_window/bp_window require ld=True")
    if sites_window is not None and bp_window is not None:
        raise InvalidConfig("sites_window and bp_window are mutually exclusive")

    bin_path = locate_binary(binary)
    argv = ["fit", str(genotype_path), "--components", str(components)]
    if variant_list is not None:
        argv += ["--list", str(variant_list)]
    if maf is not None:
        argv += ["--maf", str(maf)]
    if ld:
        argv.append("--ld")
    if sites_window is not None:
        argv += ["--sites_window", str(sites_window)]
    if bp_window is not None:
        argv += ["--bp_window", str(bp_window)]
    if extra_args:
        argv += list(extra_args)
    completed = _run(bin_path, argv, timeout=timeout, env=env, cwd=cwd)
    return MapResult(
        output_paths=_glob_artifacts(genotype_path, ("model", "loadings", "scores")),
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def project(
    genotype_path: PathLike,
    *,
    model: Optional[str] = None,
    output_manifest: Optional[PathLike] = None,
    binary: Optional[PathLike] = None,
    extra_args: Optional[Iterable[str]] = None,
    timeout: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[PathLike] = None,
) -> MapResult:
    """Run ``gnomon project`` (projection through a previously fit model)."""
    bin_path = locate_binary(binary)
    argv = ["project", str(genotype_path)]
    if model is not None:
        argv += ["--model", model]
    if output_manifest is not None:
        argv += ["--output-manifest", str(output_manifest)]
    if extra_args:
        argv += list(extra_args)
    completed = _run(bin_path, argv, timeout=timeout, env=env, cwd=cwd)
    outs: list = []
    if output_manifest is not None:
        outs.append(Path(output_manifest))
    outs.extend(_glob_artifacts(genotype_path, ("scores", "projection")))
    return MapResult(
        output_paths=tuple(p for p in outs if p.exists()),
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def _glob_artifacts(prefix: PathLike, tags: Iterable[str]) -> tuple:
    """Best-effort: find files near the genotype prefix that look like outputs."""
    p = Path(prefix)
    parent = p.parent if p.parent != Path("") else Path(".")
    base = p.stem if p.suffix in {".bed", ".vcf", ".bcf", ".gz"} else p.name
    out = []
    if parent.exists():
        for entry in parent.iterdir():
            if not entry.name.startswith(base):
                continue
            if any(t in entry.name.lower() for t in tags):
                out.append(entry)
    return tuple(sorted(out))
