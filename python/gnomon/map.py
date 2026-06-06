"""`gnomon map` — fit / project HWE PCA models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

from ._api import (
    GnomonFailed,
    InvalidConfig,
    MapResult,
    PathLike,
    _run,
    locate_binary,
)

__all__ = ["fit", "project", "VariantKey", "ModelVariantKeys", "model_variant_keys"]


@dataclass(frozen=True)
class VariantKey:
    """One marker in a projection model's variant set.

    Mirrors gnomon's Rust ``map::variant_filter::VariantKey``:
    ``chromosome`` (normalized, no ``chr`` prefix), ``position`` (1-based bp),
    and an optional ``alleles`` ``(ref, alt)`` tuple (``None`` when the model
    stores positions only).
    """

    chromosome: str
    position: int
    alleles: Optional[Tuple[str, str]] = None


@dataclass(frozen=True)
class ModelVariantKeys:
    """A projection model's marker set, resolved from gnomon's own model store.

    Returned by :func:`model_variant_keys`. ``variant_keys`` is the canonical
    site list the model was fit on — use it to build a site subset that is
    guaranteed to match what ``gnomon project --model <name>`` will select,
    instead of reading a hand-maintained JSON sidecar.
    """

    model: str
    genome_build: Optional[str]
    variant_keys: Tuple[VariantKey, ...]

    def __len__(self) -> int:
        return len(self.variant_keys)

    def __iter__(self):
        return iter(self.variant_keys)


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


def model_variant_keys(
    model: str,
    *,
    binary: Optional[PathLike] = None,
    timeout: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[PathLike] = None,
) -> ModelVariantKeys:
    """Return a built-in projection model's variant/marker keys.

    Runs ``gnomon model-keys <model>``, which resolves the model by name from
    gnomon's own model store (downloading it on first use) and prints its
    variant keys as JSON on stdout. This is the first-class replacement for
    callers that used to read ``~/.gnomon/models/<model>.json`` by hand — the
    site list now comes straight from gnomon's real baked model, so a
    site-subset built from it cannot drift from what ``gnomon project`` selects.

    Parameters
    ----------
    model : str
        Built-in model name (e.g. ``"hwe_1kg_hgdp_gsa_v3"``).
    binary, timeout, env, cwd
        Standard subprocess controls (see :func:`gnomon.score`).

    Returns
    -------
    ModelVariantKeys
        ``model``, ``genome_build`` (may be ``None``), and the tuple of
        :class:`VariantKey`.

    Raises
    ------
    GnomonFailed
        If gnomon exits non-zero (e.g. unknown model name) or emits
        unparseable output.
    """
    bin_path = locate_binary(binary)
    completed = _run(
        bin_path,
        ["model-keys", str(model)],
        timeout=timeout,
        env=env,
        cwd=cwd,
    )
    try:
        doc = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as e:
        raise GnomonFailed(
            f"gnomon model-keys produced unparseable JSON for model {model!r}: {e}",
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            returncode=completed.returncode,
        ) from e

    keys = []
    for entry in doc.get("variant_keys", []):
        alleles = entry.get("alleles")
        keys.append(
            VariantKey(
                chromosome=str(entry["chromosome"]),
                position=int(entry["position"]),
                alleles=(tuple(alleles) if alleles is not None else None),
            )
        )
    return ModelVariantKeys(
        model=str(doc.get("model", model)),
        genome_build=doc.get("genome_build"),
        variant_keys=tuple(keys),
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
