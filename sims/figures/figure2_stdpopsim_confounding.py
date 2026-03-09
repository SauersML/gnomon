from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import shutil
import sys
from datetime import datetime
from multiprocessing import shared_memory
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stdpopsim
import tskit
import numba as nb
from scipy.stats import gaussian_kde
from scipy.special import expit as sigmoid
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
SIMS_DIR = THIS_DIR.parent
if str(SIMS_DIR) not in sys.path:
    sys.path.append(str(SIMS_DIR))

from .common import (
    simulate_effect_size_distribution,
    diploid_index_pairs,
    sample_two_site_sets_for_maf,
    compute_pcs_risk_and_diagnostics,
    solve_intercept_for_prevalence,
    pop_names_from_ts,
    plink_safe_individual_id,
)
from methods.raw_pgs import RawPGSMethod
from methods.linear_interaction import LinearInteractionMethod
from methods.normalization import NormalizationMethod
from methods.gam_mgcv import GAMMethod
from methods.thinplate_mgcv import ThinPlateMethod
from plink_utils import run_plink_conversion
from prs_tools import BayesR, PPlusT


# Requested scale-down by 10.
N_TRAIN_EUR = 1200
N_TEST_PER_POP = 300
N_PCS = 5

CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "black": "#111111",
}


_LINK_SHARED: dict[str, np.ndarray] = {}
_LINK_SHARED_HANDLES: list[shared_memory.SharedMemory] = []
_LINK_N_DIP = 0


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _rss_gb() -> float:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux ru_maxrss is KiB; macOS/BSD is bytes.
        if rss > 10_000_000:
            return rss / (1024.0 ** 3)
        return rss / (1024.0 ** 2)
    except Exception:
        return float("nan")


def _stage_done(prefix: str, stage: str, started_at: float, extra: str = "") -> None:
    elapsed = time.perf_counter() - started_at
    tail = f" {extra}" if extra else ""
    _log(f"[{prefix}] {stage} done in {elapsed:.1f}s (rss_gb={_rss_gb():.2f}){tail}")


def _stage_mark(prefix: str, step: int, total: int, label: str) -> None:
    pct = 100.0 * float(step) / float(max(1, total))
    _log(f"[{prefix}] stage {step}/{total} ({pct:.1f}%): {label}")


def _ancestry_mark(prefix: str | None, pct: float, label: str) -> None:
    if prefix is None:
        return
    p = min(100.0, max(0.0, float(pct)))
    _log(f"[{prefix}] ancestry step progress: {p:.1f}% - {label}")


def _create_shared_array(arr: np.ndarray) -> tuple[shared_memory.SharedMemory, dict[str, object]]:
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[...] = arr
    meta = {"name": shm.name, "shape": arr.shape, "dtype": arr.dtype.str}
    return shm, meta


def _init_link_worker(
    child_meta: dict[str, object],
    parent_meta: dict[str, object],
    span_meta: dict[str, object],
    hap_meta: dict[str, object],
    n_dip: int,
) -> None:
    global _LINK_SHARED
    global _LINK_SHARED_HANDLES
    global _LINK_N_DIP

    _LINK_SHARED_HANDLES = []
    _LINK_SHARED = {}
    for key, meta in (
        ("child_idx", child_meta),
        ("parent_src", parent_meta),
        ("span", span_meta),
        ("hap_to_dip", hap_meta),
    ):
        shm = shared_memory.SharedMemory(name=str(meta["name"]))
        arr = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(str(meta["dtype"])), buffer=shm.buf)
        _LINK_SHARED_HANDLES.append(shm)
        _LINK_SHARED[key] = arr
    _LINK_N_DIP = int(n_dip)


def _link_chunk_worker(start: int, stop: int) -> tuple[np.ndarray, int, int]:
    child = _LINK_SHARED["child_idx"][start:stop]
    parent = _LINK_SHARED["parent_src"][start:stop]
    span = _LINK_SHARED["span"][start:stop]
    hap_to_dip = _LINK_SHARED["hap_to_dip"]

    valid = (child >= 0) & (parent >= 0)
    out = np.zeros((_LINK_N_DIP, 3), dtype=np.float64)
    if np.any(valid):
        dip_idx = hap_to_dip[child[valid]]
        src_idx = parent[valid].astype(np.int64, copy=False)
        weights = 0.5 * span[valid]
        np.add.at(out, (dip_idx, src_idx), weights)
    return out, int(stop - start), int(np.count_nonzero(valid))


@nb.njit(cache=True)
def _accumulate_link_edges_numba(
    child_idx: np.ndarray,
    parent_src: np.ndarray,
    span: np.ndarray,
    hap_to_dip: np.ndarray,
    out_dip: np.ndarray,
) -> None:
    for i in range(child_idx.shape[0]):
        c = int(child_idx[i])
        s = int(parent_src[i])
        if c >= 0 and s >= 0:
            d = int(hap_to_dip[c])
            if d >= 0:
                out_dip[d, s] += 0.5 * span[i]


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#2a2a2a",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#d4d9de",
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def _style_axes(ax, y_grid: bool = True) -> None:
    if y_grid:
        ax.grid(True, axis="y", alpha=0.55)
    else:
        ax.grid(True, axis="both", alpha=0.45)
    ax.set_axisbelow(True)


def _set_runtime_thread_env(threads: int) -> None:
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
        os.environ[k] = t


def _default_total_threads() -> int:
    candidates: list[int] = []
    if hasattr(os, "sched_getaffinity"):
        try:
            candidates.append(max(1, len(os.sched_getaffinity(0))))  # type: ignore[attr-defined]
        except Exception:
            pass
    candidates.append(max(1, int(os.cpu_count() or 1)))
    return max(1, min(candidates))


def _detect_physical_mem_mb() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return max(1, kb // 1024)
    except Exception:
        return None
    return None


def _detect_cgroup_mem_limit_mb() -> int | None:
    limit_bytes: int | None = None
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            txt = Path(p).read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not txt or txt == "max":
            continue
        try:
            b = int(txt)
        except Exception:
            continue
        if b <= 0 or b >= (1 << 60):
            continue
        limit_bytes = b if limit_bytes is None else min(limit_bytes, b)
    if limit_bytes is None:
        return None
    return max(1, limit_bytes // (1024 * 1024))


def _detect_total_mem_mb() -> int | None:
    caps = [
        v
        for v in (
            _detect_physical_mem_mb(),
            _detect_cgroup_mem_limit_mb(),
        )
        if v is not None and v > 0
    ]
    if not caps:
        return None
    return max(1, min(caps))


def _default_work_root(out_dir: Path) -> Path:
    for base in (Path("/dev/shm"), Path("/tmp")):
        if base.exists():
            return (base / "gnomon_sims_work" / "figure2").resolve()
    return out_dir / "work"


def _stage8_checkpoint_paths(out_dir: Path, seed: int) -> dict[str, Path]:
    stem = out_dir / f"fig2_s{int(seed)}.stage8"
    return {
        "meta": Path(f"{stem}.meta.json"),
        "npz": Path(f"{stem}.npz"),
        "trees": Path(f"{stem}.trees"),
    }


def _write_stage8_checkpoint(
    out_dir: Path,
    seed: int,
    run_id: str,
    expected_meta: dict[str, object],
    ts,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    pop_label: np.ndarray,
    ts_ind_id: np.ndarray,
    pcs: np.ndarray,
    G_true: np.ndarray,
    causal_pos_1based: set[int],
    het_by_pop: dict[str, float],
    ts_sites: int,
    causal_overlap: int,
) -> None:
    paths = _stage8_checkpoint_paths(out_dir, seed)
    meta = dict(expected_meta)
    meta.update(
        {
            "schema_version": 1,
            "ts_sites": int(ts_sites),
            "causal_overlap": int(causal_overlap),
            "het_by_pop": {str(k): float(v) for k, v in het_by_pop.items()},
        }
    )

    tmp_trees = Path(f"{paths['trees']}.tmp")
    tmp_npz = Path(f"{paths['npz']}.tmp")
    tmp_meta = Path(f"{paths['meta']}.tmp")
    _log(f"[{run_id}] Writing stage-8 checkpoint")
    ts.dump(str(tmp_trees))
    with open(tmp_npz, "wb") as f_npz:
        np.savez(
            f_npz,
            a_idx=np.asarray(a_idx, dtype=np.int64),
            b_idx=np.asarray(b_idx, dtype=np.int64),
            pop_label=np.asarray(pop_label, dtype="U16"),
            ts_ind_id=np.asarray(ts_ind_id, dtype=np.int64),
            pcs=np.asarray(pcs, dtype=np.float64),
            G_true=np.asarray(G_true, dtype=np.float64),
            causal_pos_1based=np.asarray(sorted(int(x) for x in causal_pos_1based), dtype=np.int64),
        )
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, sort_keys=True)

    tmp_trees.replace(paths["trees"])
    tmp_npz.replace(paths["npz"])
    tmp_meta.replace(paths["meta"])
    _log(f"[{run_id}] Stage-8 checkpoint ready: {paths['meta']}, {paths['npz']}, {paths['trees']}")


def _load_stage8_checkpoint(
    out_dir: Path,
    seed: int,
    run_id: str,
    expected_meta: dict[str, object],
):
    paths = _stage8_checkpoint_paths(out_dir, seed)
    if not (paths["meta"].exists() and paths["npz"].exists() and paths["trees"].exists()):
        return None

    try:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        _log(f"[{run_id}] Stage-8 checkpoint metadata unreadable; ignoring ({type(e).__name__}: {e})")
        return None

    for k, v in expected_meta.items():
        if k not in meta:
            _log(f"[{run_id}] Stage-8 checkpoint missing key={k}; ignoring")
            return None
        if isinstance(v, float):
            if abs(float(meta[k]) - float(v)) > 1e-9:
                _log(f"[{run_id}] Stage-8 checkpoint key mismatch for {k}; ignoring")
                return None
        else:
            if meta[k] != v:
                _log(f"[{run_id}] Stage-8 checkpoint key mismatch for {k}; ignoring")
                return None

    try:
        ts = tskit.load(str(paths["trees"]))
        z = np.load(str(paths["npz"]), allow_pickle=False)
        a_idx = z["a_idx"].astype(np.int64, copy=False)
        b_idx = z["b_idx"].astype(np.int64, copy=False)
        pop_label = z["pop_label"].astype(object, copy=False)
        ts_ind_id = z["ts_ind_id"].astype(np.int64, copy=False)
        pcs = z["pcs"].astype(np.float64, copy=False)
        G_true = z["G_true"].astype(np.float64, copy=False)
        causal_pos_1based = set(z["causal_pos_1based"].astype(np.int64, copy=False).tolist())
    except Exception as e:
        _log(f"[{run_id}] Stage-8 checkpoint payload unreadable; ignoring ({type(e).__name__}: {e})")
        return None

    het_by_pop = meta.get("het_by_pop", {})
    ts_sites = int(meta.get("ts_sites", int(ts.num_sites)))
    causal_overlap = int(meta.get("causal_overlap", len(causal_pos_1based)))
    _log(f"[{run_id}] Reusing stage-8 checkpoint from disk")
    return (
        ts,
        a_idx,
        b_idx,
        pop_label,
        ts_ind_id,
        pcs,
        G_true,
        ts_sites,
        causal_overlap,
        {str(k): float(v) for k, v in het_by_pop.items()},
        causal_pos_1based,
    )


def _true_ancestry_proportions(
    ts,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    pop_lookup: dict[int, str],
    census_time: float,
    ancestry_threads: int,
    log_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _link_ancestors_edges(ts_obj, samples_arr: np.ndarray, ancestors_arr: np.ndarray):
        if hasattr(ts_obj, "link_ancestors"):
            return ts_obj.link_ancestors(samples_arr, ancestors_arr)
        tbl = getattr(ts_obj, "tables", None)
        if tbl is not None and hasattr(tbl, "link_ancestors"):
            return tbl.link_ancestors(samples_arr, ancestors_arr)
        raise RuntimeError(
            "Neither TreeSequence.link_ancestors nor tables.link_ancestors is available "
            "in this tskit/msprime build."
        )

    def _source_index_from_pop_name(name: str) -> int:
        u = str(name).upper()
        if "AFR" in u:
            return 0
        if "EUR" in u:
            return 1
        if "ASIA" in u or "EAS" in u:
            return 2
        return -1

    sample_nodes = ts.samples()
    n_hap = len(sample_nodes)
    n_dip = int(a_idx.shape[0])
    src_names = ["AFR", "EUR", "ASIA"]
    src_to_col = {name: i for i, name in enumerate(src_names)}
    dip_acc = np.zeros((n_dip, len(src_names)), dtype=np.float64)
    total_len = float(ts.sequence_length)
    sample_nodes_arr = np.asarray(sample_nodes, dtype=np.int32)
    n_nodes = int(ts.num_nodes)
    sample_lookup = np.full(n_nodes, -1, dtype=np.int64)
    sample_lookup[sample_nodes_arr.astype(np.int64)] = np.arange(n_hap, dtype=np.int64)
    hap_to_dip = np.full(n_hap, -1, dtype=np.int64)
    dip_ids = np.arange(n_dip, dtype=np.int64)
    hap_to_dip[np.asarray(a_idx, dtype=np.int64)] = dip_ids
    hap_to_dip[np.asarray(b_idx, dtype=np.int64)] = dip_ids

    node_pop = np.asarray(ts.tables.nodes.population, dtype=np.int32)
    node_time = np.asarray(ts.tables.nodes.time, dtype=np.float64)
    max_pop = max(pop_lookup.keys()) if len(pop_lookup) > 0 else -1
    pop_to_src = np.full(max_pop + 1, -1, dtype=np.int8) if max_pop >= 0 else np.full(0, -1, dtype=np.int8)
    for pop_id, pop_name in pop_lookup.items():
        src_idx = _source_index_from_pop_name(str(pop_name))
        if src_idx >= 0 and int(pop_id) >= 0:
            pop_to_src[int(pop_id)] = int(src_idx)
    node_src_from_pop = np.full(n_nodes, -1, dtype=np.int8)
    if pop_to_src.size > 0:
        ok = (node_pop >= 0) & (node_pop < pop_to_src.shape[0])
        node_src_from_pop[ok] = pop_to_src[node_pop[ok]]

    candidate_mask = (node_time > 0.0) & (node_src_from_pop >= 0)
    candidate_times = node_time[candidate_mask]
    if candidate_times.size == 0:
        raise RuntimeError("No source-labeled ancestor nodes found; cannot compute ancestry proportions.")

    uniq_times = np.unique(candidate_times)
    requested = float(census_time)
    exact_match = bool(np.any(np.isclose(candidate_times, requested, rtol=0.0, atol=1e-9)))
    if not exact_match:
        raise RuntimeError(
            f"Required internal census_time={requested:.6f} is not present. "
            f"Available times include: {uniq_times[:20].tolist()} (showing up to 20)."
        )
    selected_time = requested

    anc_mask = np.isclose(node_time, selected_time, rtol=0.0, atol=1e-9) & (node_src_from_pop >= 0)
    anc_nodes = np.nonzero(anc_mask)[0].astype(np.int32, copy=False)
    if anc_nodes.size == 0:
        raise RuntimeError(
            f"No ancestors found at required internal census_time={selected_time:.6f}."
        )

    # Only census-time ancestors are valid ancestry sources for attribution.
    node_src = np.full(n_nodes, -1, dtype=np.int8)
    node_src[anc_nodes.astype(np.int64)] = node_src_from_pop[anc_nodes.astype(np.int64)]

    if log_prefix:
        _log(
            f"[{log_prefix}] ancestry proportions via link_ancestors "
            f"(haplotypes={n_hap}, diploids={n_dip}, trees={int(getattr(ts, 'num_trees', 0))}, "
            f"census_time={selected_time:.6f}, ancestors={anc_nodes.size})"
        )
        _log(f"[{log_prefix}] numba edge-accumulation enabled")
    _ancestry_mark(log_prefix, 0.0, "starting ancestry computation")

    t0 = time.perf_counter()
    _ancestry_mark(log_prefix, 5.0, "running link_ancestors")
    edges = _link_ancestors_edges(ts, sample_nodes_arr, anc_nodes)
    _ancestry_mark(log_prefix, 20.0, f"link_ancestors complete (edges={len(edges)})")
    parents = np.asarray(edges.parent, dtype=np.int64)
    children = np.asarray(edges.child, dtype=np.int64)
    left = np.asarray(edges.left, dtype=np.float64)
    right = np.asarray(edges.right, dtype=np.float64)
    span = right - left
    child_idx = sample_lookup[children]
    parent_src = node_src[parents]
    n_edges = int(len(edges))
    worker_target = max(1, int(0.8 * max(1, int(ancestry_threads))))
    workers = min(worker_target, n_edges) if n_edges > 0 else 1
    if log_prefix is not None:
        _log(
            f"[{log_prefix}] link_ancestors accumulation mode="
            f"{'parallel' if workers > 1 else 'serial'} workers={workers} "
            f"(threads={ancestry_threads}, edges={n_edges})"
        )

    matched_edges = 0
    if workers <= 1 or n_edges < 1_000_000:
        chunk_size = 2_000_000
        last_log_t = time.perf_counter()
        for start in range(0, n_edges, chunk_size):
            stop = min(n_edges, start + chunk_size)
            _accumulate_link_edges_numba(
                child_idx[start:stop],
                parent_src[start:stop],
                span[start:stop],
                hap_to_dip,
                dip_acc,
            )
            if log_prefix is not None:
                now_t = time.perf_counter()
                if (stop == n_edges) or (now_t - last_log_t >= 20.0):
                    pct = 100.0 * float(stop) / float(max(1, n_edges))
                    _log(
                        f"[{log_prefix}] link_ancestors accumulation progress: "
                        f"edges={stop}/{n_edges} ({pct:.1f}%)"
                    )
                    _ancestry_mark(log_prefix, 20.0 + 75.0 * (float(stop) / float(max(1, n_edges))), "accumulating edge chunks")
                    last_log_t = now_t
        valid = (child_idx >= 0) & (parent_src >= 0)
        matched_edges = int(np.count_nonzero(valid))
    else:
        split_points = np.linspace(0, n_edges, num=workers + 1, dtype=np.int64)
        tasks = [(int(split_points[i]), int(split_points[i + 1])) for i in range(workers) if int(split_points[i + 1]) > int(split_points[i])]
        shms: list[shared_memory.SharedMemory] = []
        try:
            child_shm, child_meta = _create_shared_array(np.asarray(child_idx, dtype=np.int64))
            parent_shm, parent_meta = _create_shared_array(np.asarray(parent_src, dtype=np.int8))
            span_shm, span_meta = _create_shared_array(np.asarray(span, dtype=np.float64))
            hap_shm, hap_meta = _create_shared_array(np.asarray(hap_to_dip, dtype=np.int64))
            shms.extend([child_shm, parent_shm, span_shm, hap_shm])

            ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
            done_edges = 0
            last_log_t = time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(tasks),
                mp_context=ctx,
                initializer=_init_link_worker,
                initargs=(child_meta, parent_meta, span_meta, hap_meta, int(n_dip)),
            ) as ex:
                futs = [ex.submit(_link_chunk_worker, start, stop) for (start, stop) in tasks]
                for fut in concurrent.futures.as_completed(futs):
                    part, processed, matched = fut.result()
                    dip_acc += part
                    done_edges += int(processed)
                    matched_edges += int(matched)
                    if log_prefix is not None:
                        now_t = time.perf_counter()
                        if (done_edges == n_edges) or (now_t - last_log_t >= 20.0):
                            pct = 100.0 * float(done_edges) / float(max(1, n_edges))
                            _log(
                                f"[{log_prefix}] link_ancestors accumulation progress: "
                                f"edges={done_edges}/{n_edges} ({pct:.1f}%)"
                            )
                            _ancestry_mark(log_prefix, 20.0 + 75.0 * (float(done_edges) / float(max(1, n_edges))), "accumulating edge chunks (parallel)")
                            last_log_t = now_t
        finally:
            for shm in shms:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass

    if log_prefix:
        _stage_done(
            log_prefix,
            "link_ancestors combined",
            t0,
            extra=f"(edges={len(edges)}, matched_edges={matched_edges})",
        )
    _ancestry_mark(log_prefix, 95.0, "normalizing ancestry proportions")

    denom = np.where(total_len > 0, total_len, 1.0)
    props = dip_acc / denom
    afr = props[:, src_to_col["AFR"]]
    eur = props[:, src_to_col["EUR"]]
    asia = props[:, src_to_col["ASIA"]]
    s = afr + eur + asia
    bad = s <= 0
    if np.any(bad):
        n_bad = int(np.count_nonzero(bad))
        prefix = log_prefix if log_prefix is not None else "fig2"
        _log(
            f"[{prefix}] WARNING: unresolved ancestry for {n_bad} diploids "
            "(no AFR/EUR/ASIA source found on traced paths); leaving proportions as 0.0"
        )
        afr[bad] = 0.0
        eur[bad] = 0.0
        asia[bad] = 0.0
    good = ~bad
    if np.any(good):
        afr[good] /= s[good]
        eur[good] /= s[good]
        asia[good] /= s[good]
    _ancestry_mark(log_prefix, 100.0, "ancestry proportions ready")
    return afr, eur, asia


def _model_defined_census_time(model) -> float:
    """
    Choose the three-pop ancestry census time from the model itself.
    For AmericanAdmixture_4B18, use just after the ADMIX->(AFR,EUR,ASIA)
    mass-migration event when tracing backward in time.
    """
    pop_name_to_id: dict[str, int] = {}
    for i, p in enumerate(getattr(model, "populations", [])):
        name = str(getattr(p, "name", f"pop_{i}")).upper()
        pop_id = int(getattr(p, "id", i))
        pop_name_to_id[name] = pop_id

    if "ADMIX" not in pop_name_to_id:
        raise RuntimeError(
            "Model is missing ADMIX population; cannot derive three-pop ancestry census time."
        )
    admix_id = int(pop_name_to_id["ADMIX"])

    admix_source_times: list[float] = []
    for ev in getattr(model, "events", []):
        ev_name = type(ev).__name__
        if ev_name != "MassMigration":
            continue
        if int(getattr(ev, "source", -1)) == admix_id:
            admix_source_times.append(float(getattr(ev, "time")))

    if not admix_source_times:
        raise RuntimeError(
            "Could not find ADMIX source mass-migration event in model; "
            "cannot derive three-pop ancestry census time."
        )

    tadmix = min(admix_source_times)
    eps = max(1e-6, 1e-6 * max(1.0, abs(tadmix)))
    return float(tadmix + eps)


def _simulate(
    seed: int,
    pgs_effects: np.ndarray,
    out_dir: Path,
    n_train_eur: int,
    n_test_per_pop: int,
    plink_threads: int,
    plink_memory_mb: int | None,
    msprime_model: str,
    pca_n_sites: int,
    causal_max_sites: int,
    bp_cap: int | None,
    ancestry_threads: int,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"fig2_s{seed}"
    total_steps = 14
    run_t0 = time.perf_counter()
    _log(
        f"[{run_id}] Simulation start "
        f"(model={msprime_model}, n_train_eur={n_train_eur}, n_test_per_pop={n_test_per_pop}, "
        f"bp_cap={bp_cap})"
    )
    _stage_mark(run_id, 1, total_steps, "stdpopsim setup")
    _log(f"[{run_id}] Loading stdpopsim species/model")
    t0 = time.perf_counter()
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("AmericanAdmixture_4B18")
    census_time = _model_defined_census_time(model.model)
    model.model.add_census(census_time)
    model.model.sort_events()
    _log(f"[{run_id}] Using model-defined ancestry census time: {census_time:.6f} generations ago")
    if bp_cap is not None:
        # For short debug runs, switch to a generic capped contig.
        contig = species.get_contig(
            chromosome=None,
            length=int(bp_cap),
            mutation_rate=model.mutation_rate,
            recombination_rate=1e-8,
        )
    else:
        contig = species.get_contig("chr22", genetic_map="HapMapII_GRCh38", mutation_rate=model.mutation_rate)
    engine = stdpopsim.get_engine("msprime")
    _stage_done(run_id, "stdpopsim setup", t0, extra=f"(sequence_length={int(contig.length)})")

    samples = {
        "AFR": n_test_per_pop,
        "EUR": n_train_eur + n_test_per_pop,
        "ASIA": n_test_per_pop,
        "ADMIX": n_test_per_pop,
    }
    expected_ckpt_meta = {
        "seed": int(seed),
        "msprime_model": str(msprime_model),
        "n_train_eur": int(n_train_eur),
        "n_test_per_pop": int(n_test_per_pop),
        "pca_n_sites": int(pca_n_sites),
        "causal_max_sites": int(causal_max_sites),
        "bp_cap": None if bp_cap is None else int(bp_cap),
        "sequence_length": int(contig.length),
        "n_pcs": int(N_PCS),
        "census_time": float(census_time),
    }
    ancestry_future: concurrent.futures.Future | None = None
    ancestry_pool: concurrent.futures.ThreadPoolExecutor | None = None
    overlap_ancestry_threads = max(1, int(round(float(max(1, int(ancestry_threads))) * 0.35)))

    ckpt = _load_stage8_checkpoint(out_dir, int(seed), run_id, expected_ckpt_meta)
    if ckpt is not None:
        (
            ts,
            a_idx,
            b_idx,
            pop_label,
            ts_ind_id,
            pcs,
            G_true,
            ts_sites,
            causal_overlap,
            het_by_pop,
            causal_pos_1based,
        ) = ckpt
        pop_lookup = pop_names_from_ts(ts)
        _stage_mark(run_id, 6, total_steps, "feature extraction complete (checkpoint)")
        _stage_mark(run_id, 7, total_steps, "true-effect diagnostics (checkpoint)")
        _stage_mark(run_id, 8, total_steps, "ready for ancestry (checkpoint)")
    else:
        simulate_kwargs = {
            "seed": seed,
            "msprime_model": msprime_model,
            "record_migrations": True,
        }
        _stage_mark(run_id, 2, total_steps, "stdpopsim/msprime simulate")
        _log(f"[{run_id}] Running stdpopsim/msprime simulate")
        t0 = time.perf_counter()
        ts = engine.simulate(model, contig, samples, **simulate_kwargs)
        _stage_done(
            run_id,
            "stdpopsim simulate",
            t0,
            extra=f"(nodes={ts.num_nodes}, edges={ts.num_edges}, sites={ts.num_sites}, trees={ts.num_trees})",
        )

        _stage_mark(run_id, 3, total_steps, "diploid sample mapping")
        _log(f"[{run_id}] Extracting diploid sample mappings")
        t0 = time.perf_counter()
        a_idx, b_idx, pop_idx, ts_ind_id = diploid_index_pairs(ts)
        pop_lookup = pop_names_from_ts(ts)
        pop_label = np.array([pop_lookup.get(int(p), f"pop_{int(p)}") for p in pop_idx], dtype=object)
        _stage_done(run_id, "sample mapping", t0, extra=f"(diploids={len(pop_label)}, haploids={len(ts.samples())})")
        _stage_mark(run_id, 9, total_steps, "ancestry proportions (async start)")
        _log(
            f"[{run_id}] Starting ancestry proportions in parallel "
            f"(ancestry_threads={overlap_ancestry_threads})"
        )
        ancestry_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        ancestry_future = ancestry_pool.submit(
            _true_ancestry_proportions,
            ts,
            a_idx,
            b_idx,
            pop_lookup,
            census_time,
            overlap_ancestry_threads,
            run_id,
        )

        _stage_mark(run_id, 4, total_steps, "sample PCA + causal sites")
        _log(f"[{run_id}] Sampling PCA+causal sites in one pass")
        t0 = time.perf_counter()
        pca_sites, causal_sites = sample_two_site_sets_for_maf(
            ts,
            a_idx,
            b_idx,
            pca_n_sites=pca_n_sites,
            pca_maf_min=0.05,
            pca_seed=seed + 11,
            causal_n_sites=min(causal_max_sites, int(ts.num_sites)),
            causal_maf_min=0.01,
            causal_seed=seed + 17,
            log_fn=_log,
            progress_label=f"{run_id} site_scan",
        )
        _stage_done(
            run_id,
            "sample PCA+causal sites",
            t0,
            extra=f"(pca_selected={len(pca_sites)}, causal_selected={len(causal_sites)})",
        )
        _stage_mark(run_id, 5, total_steps, "compute PCs")
        _log(f"[{run_id}] Computing PCs + risk + diagnostics in one pass")
        t0 = time.perf_counter()
        pcs, G_true, ts_sites, causal_overlap, het_by_pop, causal_pos_1based = compute_pcs_risk_and_diagnostics(
            ts,
            a_idx,
            b_idx,
            pop_label,
            pca_site_ids=pca_sites,
            n_pcs=N_PCS,
            pca_seed=seed + 13,
            causal_site_ids=causal_sites,
            real_effects=pgs_effects,
            causal_seed=seed + 19,
            log_fn=_log,
            progress_label=f"{run_id} feature_build",
        )
        _stage_done(run_id, "compute PCs+risk+diagnostics", t0)
        _stage_mark(run_id, 6, total_steps, "feature extraction complete")
        _stage_mark(run_id, 7, total_steps, "true-effect diagnostics")
        _stage_mark(run_id, 8, total_steps, "ready for ancestry")
        _write_stage8_checkpoint(
            out_dir=out_dir,
            seed=int(seed),
            run_id=run_id,
            expected_meta=expected_ckpt_meta,
            ts=ts,
            a_idx=a_idx,
            b_idx=b_idx,
            pop_label=pop_label,
            ts_ind_id=ts_ind_id,
            pcs=pcs,
            G_true=G_true,
            causal_pos_1based=causal_pos_1based,
            het_by_pop=het_by_pop,
            ts_sites=int(ts_sites),
            causal_overlap=int(causal_overlap),
        )

    true_effect_sites = int(len(causal_pos_1based))
    het_bits = ", ".join(f"{k}={v:.4f}" for k, v in sorted(het_by_pop.items()))
    _log(
        f"[{run_id}] Variant diagnostics: ts_sites={ts_sites}, "
        f"true_effect_sites={true_effect_sites}, overlap={causal_overlap}"
    )
    _log(f"[{run_id}] Mean heterozygosity at true-effect sites by pop: {het_bits}")

    rng = np.random.default_rng(seed + 23)
    if ancestry_future is not None:
        _log(f"[{run_id}] Waiting for asynchronous ancestry proportions")
        t0 = time.perf_counter()
        afr_prop, eur_prop, asia_prop = ancestry_future.result()
        _stage_done(run_id, "ancestry proportions (async)", t0)
        ancestry_pool.shutdown(wait=True)
        ancestry_pool = None
    else:
        _stage_mark(run_id, 9, total_steps, "ancestry proportions")
        _log(f"[{run_id}] Computing true ancestry proportions")
        t0 = time.perf_counter()
        afr_prop, eur_prop, asia_prop = _true_ancestry_proportions(
            ts,
            a_idx,
            b_idx,
            pop_lookup,
            census_time=census_time,
            ancestry_threads=ancestry_threads,
            log_prefix=run_id,
        )
        _stage_done(run_id, "ancestry proportions", t0)

    _stage_mark(run_id, 10, total_steps, "sample phenotypes")
    _log(f"[{run_id}] Sampling phenotypes from liability model")
    t0 = time.perf_counter()
    env = 0.8 * afr_prop + 0.3 * asia_prop + 0.1 * eur_prop
    eta = 0.75 * G_true + env
    b0 = solve_intercept_for_prevalence(0.10, eta)
    y = rng.binomial(1, sigmoid(b0 + eta)).astype(np.int32)
    _stage_done(run_id, "sample phenotypes", t0)

    rows = []
    eur_idx = np.where(pop_label == "EUR")[0]
    eur_perm = rng.permutation(eur_idx)
    eur_train = eur_perm[:n_train_eur]
    eur_holdout = eur_perm[n_train_eur:n_train_eur + n_test_per_pop]
    eur_n_cal = int(len(eur_holdout) // 2)
    eur_cal = eur_holdout[:eur_n_cal]
    eur_test = eur_holdout[eur_n_cal:]

    holdout_idx_by_pop = {pop: rng.permutation(np.where(pop_label == pop)[0]) for pop in ("AFR", "ASIA", "ADMIX")}
    cal_set_by_pop: dict[str, set[int]] = {}
    test_set_by_pop: dict[str, set[int]] = {}
    for pop, idx in holdout_idx_by_pop.items():
        n_cal = int(len(idx) // 2)
        cal_set_by_pop[pop] = set(idx[:n_cal].tolist())
        test_set_by_pop[pop] = set(idx[n_cal:].tolist())

    eur_train_set = set(eur_train.tolist())
    eur_cal_set = set(eur_cal.tolist())
    eur_test_set = set(eur_test.tolist())
    group_arr = np.empty(len(pop_label), dtype=object)
    for i in range(len(pop_label)):
        grp = "unused"
        if i in eur_train_set:
            grp = "EUR_train"
        elif i in eur_cal_set:
            grp = "EUR_cal"
        elif i in eur_test_set:
            grp = "EUR_test"
        elif pop_label[i] in cal_set_by_pop and i in cal_set_by_pop[pop_label[i]]:
            grp = f"{pop_label[i]}_cal"
        elif pop_label[i] in test_set_by_pop and i in test_set_by_pop[pop_label[i]]:
            grp = f"{pop_label[i]}_test"
        group_arr[i] = grp

    for i in range(len(pop_label)):
        grp = str(group_arr[i])
        iid = plink_safe_individual_id(i)
        row = {
            "IID": iid,
            "individual_id": iid,
            "tskit_individual_id": int(ts_ind_id[i]),
            "pop_label": str(pop_label[i]),
            "group": grp,
            "y": int(y[i]),
            "G_true": float(G_true[i]),
            "afr_prop": float(afr_prop[i]),
            "eur_prop": float(eur_prop[i]),
            "asia_prop": float(asia_prop[i]),
            "seed": int(seed),
        }
        for k in range(N_PCS):
            row[f"pc{k+1}"] = float(pcs[i, k])
        rows.append(row)

    _stage_mark(run_id, 11, total_steps, "build dataframe")
    _log(f"[{run_id}] Building dataframe (n_rows={len(rows)})")
    t0 = time.perf_counter()
    df = pd.DataFrame(rows)
    _stage_done(run_id, "dataframe build", t0)

    prefix = out_dir / f"fig2_s{seed}"
    _stage_mark(run_id, 12, total_steps, "stream VCF")
    _log(f"[{run_id}] Streaming VCF to PLINK")
    t0 = time.perf_counter()

    def _write_vcf_stream(handle) -> None:
        names = [plink_safe_individual_id(i) for i in range(ts.num_individuals)]
        ts.write_vcf(handle, individual_names=names, position_transform=lambda x: np.asarray(x) + 1)

    _stage_done(run_id, "stream VCF", t0)
    _stage_mark(run_id, 13, total_steps, "VCF->PLINK conversion")
    _log(f"[{run_id}] Converting VCF to PLINK")
    t0 = time.perf_counter()
    run_plink_conversion(
        _write_vcf_stream,
        str(prefix),
        cm_map_path=None,
        threads=plink_threads,
        memory_mb=plink_memory_mb,
    )
    _stage_done(run_id, "PLINK conversion", t0)
    bim_path = prefix.with_suffix(".bim")
    bim_n_variants = sum(1 for _ in open(bim_path, "r", encoding="utf-8", errors="replace"))
    bim_pos = set()
    with open(bim_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                try:
                    bim_pos.add(int(parts[3]))
                except ValueError:
                    pass
    overlap_bim_true_effect = len(bim_pos & causal_pos_1based)
    _log(
        f"[{run_id}] Sample-variant diagnostics: "
        f"bim_variants={bim_n_variants}, overlap_true_effect_positions={overlap_bim_true_effect}"
    )
    _stage_mark(run_id, 14, total_steps, "write simulation table")
    _log(f"[{run_id}] Writing simulation table to {prefix.with_suffix('.tsv')}")
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(prefix.with_suffix(".tsv"), sep="\t", index=False)
    _stage_done(run_id, "write simulation table", t0)
    _stage_done(run_id, "simulation stage total", run_t0)
    return df


def _run_prs_and_predict(
    df: pd.DataFrame,
    prefix: str,
    out_dir: Path,
    plink_threads: int,
    plink_memory_mb: int | None,
    bayesr_threads: int,
    use_bayesr: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object] | None]:
    import shutil
    import subprocess

    run_id = Path(prefix).name
    total_steps = 8 if use_bayesr else 6
    run_t0 = time.perf_counter()
    _stage_mark(run_id, 1, total_steps, "prepare PRS files")
    _log(f"[{run_id}] Preparing PRS files in {out_dir}")
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    fam = pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, names=["FID", "IID", "PID", "MID", "SEX", "PHENO"], dtype=str)
    iid_to_fid = dict(zip(fam["IID"], fam["FID"]))

    train_ids = df.loc[df["group"] == "EUR_train", "IID"].astype(str).tolist()
    pred_ids = df.loc[df["group"].str.endswith("_test") | df["group"].str.endswith("_cal"), "IID"].astype(str).tolist()

    pd.DataFrame({"FID": [iid_to_fid[i] for i in train_ids], "IID": train_ids}).to_csv(work / "train.keep", sep="\t", header=False, index=False)
    pd.DataFrame({"FID": [iid_to_fid[i] for i in pred_ids], "IID": pred_ids}).to_csv(work / "test.keep", sep="\t", header=False, index=False)
    _stage_done(run_id, "PRS file prep", t0, extra=f"(n_train={len(train_ids)}, n_pred={len(pred_ids)})")

    plink = shutil.which("plink2") or "plink2"
    common = ["--threads", str(plink_threads)]
    if plink_memory_mb is not None and plink_memory_mb > 0:
        common.extend(["--memory", str(plink_memory_mb)])
    _stage_mark(run_id, 2, total_steps, "plink2 --freq")
    _log(f"[{run_id}] Running plink2 --freq")
    t0 = time.perf_counter()
    subprocess.run([plink, "--bfile", prefix, "--freq", *common, "--out", str(work / "ref")], check=True)
    _stage_done(run_id, "plink2 --freq", t0)
    _stage_mark(run_id, 3, total_steps, "plink2 train split")
    _log(f"[{run_id}] Running plink2 train split")
    t0 = time.perf_counter()
    subprocess.run([plink, "--bfile", prefix, "--keep", str(work / "train.keep"), "--make-bed", *common, "--out", str(work / "train")], check=True)
    _stage_done(run_id, "plink2 train split", t0)
    _stage_mark(run_id, 4, total_steps, "plink2 test split")
    _log(f"[{run_id}] Running plink2 test split")
    t0 = time.perf_counter()
    subprocess.run([plink, "--bfile", prefix, "--keep", str(work / "test.keep"), "--make-bed", *common, "--out", str(work / "test")], check=True)
    _stage_done(run_id, "plink2 test split", t0)

    train_df = df[df["IID"].isin(train_ids)].copy()
    t0 = time.perf_counter()
    pd.DataFrame({
        "FID": [iid_to_fid[i] for i in train_df["IID"].astype(str)],
        "IID": train_df["IID"].astype(str),
        "y": train_df["y"].astype(int),
    }).to_csv(work / "train.phen", sep=" ", header=False, index=False)

    pd.DataFrame({
        "FID": [iid_to_fid[i] for i in train_df["IID"].astype(str)],
        "IID": train_df["IID"].astype(str),
        **{f"pc{i+1}": train_df[f"pc{i+1}"].to_numpy() for i in range(N_PCS)},
    }).to_csv(work / "train.covar", sep=" ", header=False, index=False)
    _stage_done(run_id, "write PRS phenotype/covariate files", t0)

    if use_bayesr:
        br = BayesR(threads=bayesr_threads, plink_memory_mb=plink_memory_mb)
        _stage_mark(run_id, 5, total_steps, "BayesR fit")
        _log(f"[{run_id}] BayesR fit starting")
        t0 = time.perf_counter()
        eff = br.fit(str(work / "train"), str(work / "train.phen"), str(work / "bayesr"), covar_file=str(work / "train.covar"))
        _stage_done(run_id, "BayesR fit", t0)
        _stage_mark(run_id, 6, total_steps, "BayesR score train")
        _log(f"[{run_id}] BayesR scoring train")
        t0 = time.perf_counter()
        train_scores = br.predict(str(work / "train"), eff, str(work / "bayesr_train"), freq_file=str(work / "ref.afreq"))
        _stage_done(run_id, "BayesR score train", t0)
        _stage_mark(run_id, 7, total_steps, "BayesR score test")
        _log(f"[{run_id}] BayesR scoring test")
        t0 = time.perf_counter()
        test_scores = br.predict(str(work / "test"), eff, str(work / "bayesr_test"), freq_file=str(work / "ref.afreq"))
        _stage_done(run_id, "BayesR score test", t0)
        _stage_mark(run_id, 8, total_steps, "BayesR stage complete")
        _stage_done(run_id, "BayesR/predict stage total", run_t0)
        return train_scores, test_scores, None
    else:
        _stage_mark(run_id, 5, total_steps, "P+T fit/score")
        _log(f"[{run_id}] P+T fit/scoring starting")
        t0 = time.perf_counter()
        pt = PPlusT(threads=plink_threads, plink_memory_mb=plink_memory_mb)
        train_scores, test_scores, pt_meta = pt.fit_and_predict(
            bfile_train=str(work / "train"),
            bfile_test=str(work / "test"),
            pheno_file=str(work / "train.phen"),
            covar_file=str(work / "train.covar"),
            freq_file=str(work / "ref.afreq"),
            out_prefix=str(work / "pt"),
        )
        _stage_done(run_id, "P+T fit/score", t0)
        _stage_mark(run_id, 6, total_steps, "P+T stage complete")
        _stage_done(run_id, "P+T/predict stage total", run_t0)
        return train_scores, test_scores, pt_meta


def _cleanup_seed_artifacts(prefix: Path, seed_work_dir: Path) -> None:
    _log(f"[{prefix.name}] Cleaning seed artifacts")
    # Keep .bed/.bim/.fam/.tsv for later --use-existing reuse.
    for ext in (".log", ".vcf"):
        p = Path(f"{prefix}{ext}")
        if p.exists():
            p.unlink(missing_ok=True)
    if seed_work_dir.exists():
        shutil.rmtree(seed_work_dir, ignore_errors=True)


def _fit_predict_single_method(
    method_name: str,
    P_train: np.ndarray,
    PC_train: np.ndarray,
    y_train: np.ndarray,
    P_test: np.ndarray,
    PC_test: np.ndarray,
    train_pop_labels: np.ndarray,
    test_pop_labels: np.ndarray,
) -> tuple[str, np.ndarray]:
    if method_name == "raw":
        method = RawPGSMethod(max_iter=1000)
        method.fit(P_train, PC_train, y_train)
        return method_name, method.predict_proba(P_test, PC_test)

    if method_name == "linear":
        method = LinearInteractionMethod(max_iter=1000)
        method.fit(P_train, PC_train, y_train)
        return method_name, method.predict_proba(P_test, PC_test)

    if method_name == "normalized":
        method = NormalizationMethod(n_pcs=N_PCS, max_iter=1000)
        method.set_pop_labels(train_pop_labels)
        method.fit(P_train, PC_train, y_train)
        method.set_pop_labels(test_pop_labels)
        return method_name, method.predict_proba(P_test, PC_test)

    raise RuntimeError(f"Unknown method_name={method_name}")


def _method_preds(cal_df: pd.DataFrame, test_df: pd.DataFrame, cal_prs: np.ndarray, test_prs: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    prs_scaler = StandardScaler()
    P_train = prs_scaler.fit_transform(cal_prs.reshape(-1, 1)).ravel()
    P_test = prs_scaler.transform(test_prs.reshape(-1, 1)).ravel()
    PC_train = cal_df[[f"pc{i+1}" for i in range(N_PCS)]].to_numpy()
    PC_test = test_df[[f"pc{i+1}" for i in range(N_PCS)]].to_numpy()
    y_train = cal_df["y"].to_numpy(dtype=np.int32)
    train_pop = cal_df["pop_label"].to_numpy(dtype=object)
    test_pop = test_df["pop_label"].to_numpy(dtype=object)

    method_names = ["raw", "linear", "normalized"]
    worker_count = len(method_names)
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as ex:
            futs = [
                ex.submit(
                    _fit_predict_single_method,
                    name,
                    P_train,
                    PC_train,
                    y_train,
                    P_test,
                    PC_test,
                    train_pop,
                    test_pop,
                )
                for name in method_names
            ]
            for fut in concurrent.futures.as_completed(futs):
                name, pred = fut.result()
                out[name] = np.asarray(pred, dtype=float)
    except Exception as e:
        _log(f"[fig2] parallel method fitting failed; falling back to sequential ({type(e).__name__}: {e})")
        for name in method_names:
            key, pred = _fit_predict_single_method(
                name,
                P_train,
                PC_train,
                y_train,
                P_test,
                PC_test,
                train_pop,
                test_pop,
            )
            out[key] = np.asarray(pred, dtype=float)

    gm = GAMMethod(k_pgs=4, k_pc=4, k_interaction=3, use_ti=True)
    gm.fit(P_train, PC_train, y_train)
    out["pspline"] = gm.predict_proba(P_test, PC_test)
    tp = ThinPlateMethod(k_pgs=4, k_pc=4, k_interaction=3, use_ti=True)
    tp.fit(P_train, PC_train, y_train)
    out["thinplate"] = tp.predict_proba(P_test, PC_test)

    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss

    if y.size == 0:
        return np.nan
    return float(brier_score_loss(y.astype(float), np.clip(p.astype(float), 0.0, 1.0)))


def _assert_noncollapsed_prs(labels: np.ndarray, prs: np.ndarray, context: str) -> None:
    lab = np.asarray(labels, dtype=object)
    x = np.asarray(prs, dtype=float)
    if lab.shape[0] != x.shape[0]:
        raise RuntimeError(f"{context}: label/score length mismatch ({lab.shape[0]} vs {x.shape[0]})")
    if x.size == 0:
        raise RuntimeError(f"{context}: empty PRS vector")

    finite = np.isfinite(x)
    if not np.all(finite):
        raise RuntimeError(f"{context}: non-finite PRS values detected")

    rows: list[dict[str, object]] = []
    collapsed: list[str] = []
    for g in sorted(pd.unique(lab)):
        mask = lab == g
        s = x[mask]
        n = int(s.size)
        n_unique = int(np.unique(s).size) if n > 0 else 0
        sd = float(np.std(s, ddof=1)) if n > 1 else float("nan")
        rows.append({"group": str(g), "n": n, "n_unique_prs": n_unique, "sd_prs": sd})
        if n > 1 and n_unique <= 1:
            collapsed.append(str(g))

    overall_unique = int(np.unique(x).size)
    _log_results_table(
        f"[{context}] PRS variation by group",
        pd.DataFrame(rows).sort_values("group").reset_index(drop=True),
    )
    _log(f"[{context}] Overall PRS unique values: {overall_unique}")

    if overall_unique <= 1 or collapsed:
        raise RuntimeError(
            f"{context}: collapsed PRS detected "
            f"(overall_unique={overall_unique}, collapsed_groups={collapsed})"
        )


def _plot_main(df: pd.DataFrame, out_dir: Path) -> None:
    _apply_plot_style()
    prs_sources = sorted(df["prs_source"].unique()) if "prs_source" in df.columns else ["estimated"]
    pops = ["EUR", "AFR", "ASIA", "ADMIX"]
    pop_colors = {"EUR": CB["blue"], "AFR": CB["orange"], "ASIA": CB["green"], "ADMIX": CB["purple"]}

    for prs_source in prs_sources:
        sub_df = df[df["prs_source"] == prs_source] if "prs_source" in df.columns else df
        methods = [m for m in ["raw", "normalized", "linear", "pspline", "thinplate"] if m in set(sub_df["method"])]
        x = np.arange(len(methods))
        width = 0.18
        source_suffix = "" if prs_source == "estimated" else f"_{prs_source}"

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, pop in enumerate(pops):
            vals = []
            for m in methods:
                sub = sub_df[(sub_df["method"] == m) & (sub_df["population"] == pop)]
                vals.append(float(sub["auc"].iloc[0]) if len(sub) else np.nan)
            ax.bar(
                x + (i - 1.5) * width,
                vals,
                width=width,
                label=pop,
                color=pop_colors[pop],
                edgecolor="#222222",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel(f"AUC ({prs_source} PRS)")
        ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False)
        _style_axes(ax, y_grid=True)
        fig.tight_layout()
        fig.savefig(out_dir / f"figure2_auc_by_method_population{source_suffix}.png", dpi=240)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, pop in enumerate(pops):
            vals = []
            for m in methods:
                sub = sub_df[(sub_df["method"] == m) & (sub_df["population"] == pop)]
                vals.append(float(sub["brier"].iloc[0]) if len(sub) else np.nan)
            ax.bar(
                x + (i - 1.5) * width,
                vals,
                width=width,
                label=pop,
                color=pop_colors[pop],
                edgecolor="#222222",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel(f"Brier score ({prs_source} PRS)")
        ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False)
        _style_axes(ax, y_grid=True)
        fig.tight_layout()
        fig.savefig(out_dir / f"figure2_brier_by_method_population{source_suffix}.png", dpi=240)
        plt.close(fig)


def _plot_prs_distribution(test_df: pd.DataFrame, prs: np.ndarray, out_dir: Path, prs_source: str = "estimated") -> None:
    _apply_plot_style()
    def _draw_kde(ax, x: np.ndarray, color: str, label: str) -> None:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        if x.size < 2 or float(np.std(x)) < 1e-8:
            xv = float(np.mean(x))
            ax.axvline(xv, color=color, alpha=0.9, linewidth=2, label=label)
            return
        lo, hi = float(np.min(x)), float(np.max(x))
        pad = max(1e-6, 0.25 * (hi - lo))
        grid = np.linspace(lo - pad, hi + pad, 200)
        kde = gaussian_kde(x)
        dens = kde(grid)
        ax.plot(grid, dens, color=color, linewidth=2, label=label)
        ax.fill_between(grid, 0.0, dens, color=color, alpha=0.22)

    fig, ax = plt.subplots(figsize=(8, 5))
    for pop, color in [("EUR", CB["blue"]), ("AFR", CB["orange"]), ("ASIA", CB["green"]), ("ADMIX", CB["purple"])]:
        mask = test_df["pop_label"] == pop
        x = prs[mask.to_numpy()]
        if x.size:
            _draw_kde(ax, x, color=color, label=pop)
    ax.set_xlabel(f"PRS ({prs_source})")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    _style_axes(ax, y_grid=False)
    fig.tight_layout()
    suffix = "" if prs_source == "estimated" else f"_{prs_source}"
    fig.savefig(out_dir / f"figure2_prs_distributions{suffix}.png", dpi=240)
    plt.close(fig)


def _plot_pcs(df: pd.DataFrame, out_dir: Path) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    for pop, color in [("EUR", CB["blue"]), ("AFR", CB["orange"]), ("ASIA", CB["green"]), ("ADMIX", CB["purple"])]:
        s = df[df["pop_label"] == pop]
        if len(s):
            ax.scatter(s["pc1"], s["pc2"], s=24, alpha=0.75, label=pop, color=color, edgecolors="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    _style_axes(ax, y_grid=False)
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_pc12.png", dpi=240)
    plt.close(fig)


def _ensure_calibration_groups(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Backfill *_cal groups for legacy cached simulations that only have *_test."""
    if "group" not in df.columns or "pop_label" not in df.columns:
        raise RuntimeError("Simulation table must include group and pop_label columns.")
    has_cal = bool(df["group"].astype(str).str.endswith("_cal").any())
    if has_cal:
        return df

    out = df.copy()
    rng = np.random.default_rng(int(seed) + 31)
    pops = ["EUR", "AFR", "ASIA", "ADMIX"]
    cal_counts: dict[str, int] = {}
    test_counts: dict[str, int] = {}
    for pop in pops:
        mask = (out["pop_label"].astype(str) == pop) & (
            out["group"].astype(str).str.endswith("_test") | (out["group"].astype(str) == "test")
        )
        idx = out.index[mask].to_numpy()
        if idx.size == 0:
            continue
        perm = rng.permutation(idx)
        n_cal = int(perm.size // 2)
        cal_idx = perm[:n_cal]
        test_idx = perm[n_cal:]
        out.loc[cal_idx, "group"] = f"{pop}_cal"
        out.loc[test_idx, "group"] = f"{pop}_test"
        cal_counts[pop] = int(cal_idx.size)
        test_counts[pop] = int(test_idx.size)

    if not out["group"].astype(str).str.endswith("_cal").any():
        raise RuntimeError(
            "Could not derive calibration groups from cached simulation table. "
            "Expected *_test (or test) rows by population."
        )
    _log(
        "[fig2] Upgraded cached split to include calibration groups: "
        + ", ".join(
            f"{pop}(cal={cal_counts.get(pop, 0)},test={test_counts.get(pop, 0)})"
            for pop in pops
        )
    )
    return out


def _log_results_table(title: str, df: pd.DataFrame) -> None:
    if df.empty:
        _log(f"{title}: <empty>")
        return
    txt = df.to_string(index=False, justify="left")
    _log(f"{title}\n{txt}")


def _plot_pt_train_accuracy(metrics_df: pd.DataFrame, out_path: Path) -> None:
    if metrics_df.empty:
        return
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = metrics_df["p_threshold"].to_numpy(dtype=float)
    ax.plot(x, metrics_df["train_accuracy"].to_numpy(dtype=float), marker="o", linewidth=2, color=CB["blue"], label="Accuracy")
    if "train_balanced_accuracy" in metrics_df.columns:
        ax.plot(
            x,
            metrics_df["train_balanced_accuracy"].to_numpy(dtype=float),
            marker="s",
            linewidth=2,
            color=CB["orange"],
            label="Balanced Accuracy",
        )
    if "train_auc" in metrics_df.columns:
        ax.plot(
            x,
            metrics_df["train_auc"].to_numpy(dtype=float),
            marker="^",
            linewidth=2,
            color=CB["green"],
            label="AUC",
        )
    ax.set_xscale("log")
    ax.set_xlabel("P+T p-value threshold")
    ax.set_ylabel("Train metric")
    ax.set_title("Figure2 P+T train metrics across thresholds")
    ax.legend(frameon=False, loc="best")
    _style_axes(ax, y_grid=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _force_plots_from_existing(out_dir: Path) -> None:
    res_path = out_dir / "figure2_auc_by_method_population.tsv"
    if not res_path.exists():
        alt = out_dir / "figure2_auc_brier_by_method_population.tsv"
        if alt.exists():
            res_path = alt
    if not res_path.exists():
        raise RuntimeError(
            f"No Figure2 aggregate table found in {out_dir}. "
            "Expected figure2_auc_by_method_population.tsv or figure2_auc_brier_by_method_population.tsv."
        )
    res = pd.read_csv(res_path, sep="\t")
    if res.empty:
        raise RuntimeError(f"Figure2 aggregate table is empty: {res_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "figure2_auc_by_method_population.tsv", sep="\t", index=False)
    res.to_csv(out_dir / "figure2_auc_brier_by_method_population.tsv", sep="\t", index=False)
    _log("[fig2] Wrote figure2_auc_by_method_population.tsv")
    _log("[fig2] Wrote figure2_auc_brier_by_method_population.tsv")
    _log_results_table(
        "[fig2] AUC/Brier/statistics by method and population",
        res.sort_values(["prs_source", "method", "population"]).reset_index(drop=True),
    )
    _plot_main(res, out_dir)

    pred_path = out_dir / "figure2_test_predictions.tsv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path, sep="\t")
        if not pred_df.empty:
            for prs_source in sorted(pred_df["prs_source"].dropna().astype(str).unique()):
                sub = pred_df[pred_df["prs_source"].astype(str) == prs_source].copy()
                if sub.empty:
                    continue
                method_order = sorted(sub["method"].dropna().astype(str).unique())
                if method_order:
                    sub = sub[sub["method"].astype(str) == method_order[0]].copy()
                _plot_prs_distribution(sub, sub["prs"].to_numpy(dtype=float), out_dir, prs_source=prs_source)
            pc_df = (
                pred_df[["IID", "pop_label", "pc1", "pc2"]]
                .dropna(subset=["IID"])
                .drop_duplicates(subset=["IID"])
                .rename(columns={"pop_label": "pop_label"})
            )
            if not pc_df.empty:
                _plot_pcs(pc_df, out_dir)
    pt_path = out_dir / "figure2_pt_thresholds.tsv"
    if pt_path.exists():
        pt_metrics = pd.read_csv(pt_path, sep="\t")
        if not pt_metrics.empty:
            _plot_pt_train_accuracy(pt_metrics.sort_values("p_threshold"), out_dir / "figure2_pt_train_accuracy.png")
    _log("[fig2] Force-figs regeneration complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sims/results_figure2_local")
    parser.add_argument("--seed", type=int, default=99173)
    parser.add_argument("--cache", default="sims/.cache", help="Deprecated: external PGS cache is no longer used.")
    parser.add_argument("--n-train-eur", type=int, default=N_TRAIN_EUR)
    parser.add_argument("--n-test-per-pop", type=int, default=N_TEST_PER_POP)
    parser.add_argument(
        "--msprime-model",
        choices=["dtwf", "hudson"],
        default="hudson",
        help="msprime ancestry model (hudson is usually faster).",
    )
    parser.add_argument("--pca-sites", type=int, default=2000)
    parser.add_argument("--causal-sites", type=int, default=5000)
    parser.add_argument("--bp-cap", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None, help="Thread count for PLINK/PRS. Default uses node allocation.")
    parser.add_argument("--memory-mb", type=int, default=None, help="PLINK memory limit (MB). Default auto-sizes from node RAM.")
    parser.add_argument("--work-root", default=None, help="Directory for heavy transient work files (e.g., RAM disk).")
    parser.add_argument("--keep-intermediates", action="store_true", help="Keep PLINK/GCTB intermediate files.")
    parser.add_argument("--bayesr", action="store_true", help="Use BayesR backend (default is fast P+T).")
    parser.add_argument("--use-existing", action="store_true", help="Reuse existing fig2_s* PLINK/TSV files; skip simulation.")
    parser.add_argument("--use-existing-dir", default=None, help="Directory containing existing fig2_s* PLINK/TSV files.")
    parser.add_argument(
        "--force-figs",
        action="store_true",
        help="Force regeneration of Figure2 aggregate tables/plots from existing available outputs.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root) if args.work_root else _default_work_root(out_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    if bool(args.force_figs):
        _log("Figure2 force-figs mode: regenerating from existing outputs only")
        try:
            _force_plots_from_existing(out_dir)
            return
        except RuntimeError as e:
            source_base = Path(args.use_existing_dir) if args.use_existing_dir else out_dir
            seed = int(args.seed)
            source_prefix = str(source_base / f"fig2_s{seed}")
            sim_tsv = source_base / f"fig2_s{seed}.tsv"
            needed = [Path(f"{source_prefix}.bed"), Path(f"{source_prefix}.bim"), Path(f"{source_prefix}.fam"), sim_tsv]
            miss = [str(p) for p in needed if not p.exists()]
            if miss:
                raise RuntimeError(
                    f"{e} Recovery without rerunning simulation requires existing "
                    f"fig2_s{seed}.bed/.bim/.fam and fig2_s{seed}.tsv. Missing: {miss}"
                ) from e
            _log(
                f"[fig2] Force-figs mode: aggregate table missing; recovering real outputs from existing artifacts in {source_base}"
            )
            args.use_existing = True
    _log("Figure2 spline backends: pspline + thinplate (mgcv)")
    _log(f"Figure2 PRS backend: {'BayesR' if args.bayesr else 'P+T'}")
    threads = max(1, int(args.threads)) if args.threads is not None else _default_total_threads()
    if args.memory_mb is not None:
        memory_mb = max(512, int(args.memory_mb))
    else:
        total_mb = _detect_total_mem_mb()
        memory_mb = max(2048, int(0.80 * total_mb)) if total_mb is not None else None
    _log(f"Figure2 resources: threads={threads} memory_mb={memory_mb}")
    _set_runtime_thread_env(threads)

    pgs_effects = simulate_effect_size_distribution(
        n_effects=200000,
        seed=int(args.seed) + 4049,
    )
    _log(f"Figure2 using runtime-simulated effect-size distribution (n={len(pgs_effects)})")

    df = None
    prefix = None
    train_scores = None
    test_scores = None
    seed = int(args.seed)
    _log(f"[fig2_s{seed}] Starting full Figure2 pipeline")
    prefix = str(out_dir / f"fig2_s{seed}")
    source_base = Path(args.use_existing_dir) if args.use_existing_dir else out_dir
    source_prefix = str(source_base / f"fig2_s{seed}")
    sim_tsv = source_base / f"fig2_s{seed}.tsv"

    def _simulate_current() -> pd.DataFrame:
        return _simulate(
            seed,
            pgs_effects,
            out_dir,
            n_train_eur=int(args.n_train_eur),
            n_test_per_pop=int(args.n_test_per_pop),
            plink_threads=threads,
            plink_memory_mb=memory_mb,
            msprime_model=str(args.msprime_model),
            pca_n_sites=int(args.pca_sites),
            causal_max_sites=int(args.causal_sites),
            bp_cap=args.bp_cap,
            ancestry_threads=threads,
        )

    reused_existing = False
    if args.use_existing:
        try:
            needed = [Path(f"{source_prefix}.bed"), Path(f"{source_prefix}.bim"), Path(f"{source_prefix}.fam"), sim_tsv]
            miss = [str(p) for p in needed if not p.exists()]
            if miss:
                raise FileNotFoundError(f"[fig2_s{seed}] --use-existing requested, but missing files: {miss}")
            _log(f"[fig2_s{seed}] Reusing existing simulation/PLINK files from {source_base}")
            df = pd.read_csv(sim_tsv, sep="\t")
            reused_existing = True
        except Exception as e:
            _log(f"[fig2_s{seed}] Existing artifacts unavailable/unreadable; regenerating ({type(e).__name__}: {e})")
            source_prefix = prefix
            df = _simulate_current()
    else:
        source_prefix = prefix
        df = _simulate_current()
    try:
        df = _ensure_calibration_groups(df, seed=seed)
    except Exception as e:
        if args.use_existing and reused_existing:
            _log(f"[fig2_s{seed}] Existing split invalid; regenerating ({type(e).__name__}: {e})")
            source_prefix = prefix
            df = _ensure_calibration_groups(_simulate_current(), seed=seed)
            reused_existing = False
        else:
            raise

    seed_work = work_root / f"work_s{seed}"
    _log(f"[fig2_s{seed}] Starting PRS/prediction stage")
    try:
        train_scores, test_scores, pt_meta = _run_prs_and_predict(
            df,
            source_prefix,
            seed_work,
            plink_threads=threads,
            plink_memory_mb=memory_mb,
            bayesr_threads=threads,
            use_bayesr=bool(args.bayesr),
        )
    except Exception as e:
        if args.use_existing and reused_existing:
            _log(
                f"[fig2_s{seed}] Existing PLINK/TSV artifacts failed during PRS prep; "
                f"regenerating and retrying ({type(e).__name__}: {e})"
            )
            source_prefix = prefix
            df = _ensure_calibration_groups(_simulate_current(), seed=seed)
            train_scores, test_scores, pt_meta = _run_prs_and_predict(
                df,
                source_prefix,
                seed_work,
                plink_threads=threads,
                plink_memory_mb=memory_mb,
                bayesr_threads=threads,
                use_bayesr=bool(args.bayesr),
            )
        else:
            raise
    if pt_meta is not None:
        pt_metrics = pt_meta["threshold_metrics"].copy()
        pt_metrics["seed"] = int(seed)
        pt_metrics["is_selected"] = pt_metrics["p_threshold"] == float(pt_meta["best_p_threshold"])
        pt_metrics.to_csv(out_dir / "figure2_pt_thresholds.tsv", sep="\t", index=False)
        _plot_pt_train_accuracy(pt_metrics.sort_values("p_threshold"), out_dir / "figure2_pt_train_accuracy.png")
        _log(
            f"[fig2_s{seed}] P+T selected p={float(pt_meta['best_p_threshold']):g} "
            f"(train_accuracy={float(pt_meta['best_train_accuracy']):.4f}, "
            f"train_balanced_accuracy={float(pt_meta.get('best_train_balanced_accuracy', float('nan'))):.4f}, "
            f"train_auc={float(pt_meta.get('best_train_auc', float('nan'))):.4f}, "
            f"n_snps={int(pt_meta['best_n_snps'])})"
        )

    train_df = df[df["group"] == "EUR_train"].copy()
    pred_df = df[df["group"].str.endswith("_test") | df["group"].str.endswith("_cal")].copy()
    train_df = train_df.set_index("IID").loc[train_scores["IID"].astype(str)].reset_index()
    pred_df = pred_df.set_index("IID").loc[test_scores["IID"].astype(str)].reset_index()
    cal_df = pred_df[pred_df["group"].str.endswith("_cal")].copy()
    test_df = pred_df[pred_df["group"].str.endswith("_test")].copy()
    cal_scores = test_scores[test_scores["IID"].astype(str).isin(cal_df["IID"].astype(str))].copy()
    test_scores = test_scores[test_scores["IID"].astype(str).isin(test_df["IID"].astype(str))].copy()
    cal_df = cal_df.set_index("IID").loc[cal_scores["IID"].astype(str)].reset_index()
    test_df = test_df.set_index("IID").loc[test_scores["IID"].astype(str)].reset_index()
    _assert_noncollapsed_prs(
        train_df["pop_label"].to_numpy(),
        train_scores["PRS"].to_numpy(dtype=float),
        context=f"fig2_s{seed} train",
    )
    _assert_noncollapsed_prs(
        test_df["pop_label"].to_numpy(),
        test_scores["PRS"].to_numpy(dtype=float),
        context=f"fig2_s{seed} pred",
    )
    _assert_noncollapsed_prs(
        cal_df["pop_label"].to_numpy(),
        cal_scores["PRS"].to_numpy(dtype=float),
        context=f"fig2_s{seed} cal",
    )
    _assert_noncollapsed_prs(
        test_df["pop_label"].to_numpy(),
        test_scores["PRS"].to_numpy(dtype=float),
        context=f"fig2_s{seed} test",
    )

    rows = []
    prs_sources = {
        "estimated": {
            "train": train_scores["PRS"].to_numpy(dtype=float),
            "test": test_scores["PRS"].to_numpy(dtype=float),
            "cal": cal_scores["PRS"].to_numpy(dtype=float),
        },
        "oracle": {
            "train": train_df["G_true"].to_numpy(dtype=float),
            "test": test_df["G_true"].to_numpy(dtype=float),
            "cal": cal_df["G_true"].to_numpy(dtype=float),
        },
    }

    preds_by_source: dict[str, dict[str, np.ndarray]] = {}
    for prs_source, prs_vals in prs_sources.items():
        _assert_noncollapsed_prs(
            train_df["pop_label"].to_numpy(),
            prs_vals["train"],
            context=f"fig2_s{seed} train ({prs_source})",
        )
        _assert_noncollapsed_prs(
            test_df["pop_label"].to_numpy(),
            prs_vals["test"],
            context=f"fig2_s{seed} pred ({prs_source})",
        )
        _assert_noncollapsed_prs(
            cal_df["pop_label"].to_numpy(),
            prs_vals["cal"],
            context=f"fig2_s{seed} cal ({prs_source})",
        )
        preds = _method_preds(cal_df, test_df, prs_vals["cal"], prs_vals["test"])
        preds_by_source[prs_source] = preds
        _log(f"[fig2_s{seed}] Method fitting/prediction complete (prs_source={prs_source})")

        test_prs = prs_vals["test"]
        for method, y_prob in preds.items():
            for pop in ["EUR", "AFR", "ASIA", "ADMIX"]:
                mask = test_df["pop_label"] == pop
                y_pop = test_df.loc[mask, "y"].to_numpy()
                p_pop = np.asarray(y_prob, dtype=float)[mask.to_numpy()]
                prs_pop = test_prs[mask.to_numpy()]
                g_pop = test_df.loc[mask, "G_true"].to_numpy(dtype=float)
                auc_pop = _auc(y_pop, p_pop)
                brier_pop = _brier(y_pop, p_pop)
                _log(
                    f"[fig2_s{seed}] prs_source={prs_source} method={method} pop={pop} n={int(mask.sum())} "
                    f"auc={auc_pop:.4f} brier={brier_pop:.4f}"
                )
                rows.append(
                    {
                        "prs_source": prs_source,
                        "method": method,
                        "population": pop,
                        "n_train_prs": int(len(train_df)),
                        "n_calibration": int(len(cal_df)),
                        "n_test_total": int(len(test_df)),
                        "n": int(mask.sum()),
                        "prevalence": float(np.mean(y_pop)) if y_pop.size > 0 else np.nan,
                        "auc": auc_pop,
                        "brier": brier_pop,
                        "mean_prs": float(np.mean(prs_pop)) if prs_pop.size > 0 else np.nan,
                        "sd_prs": float(np.std(prs_pop, ddof=1)) if prs_pop.size > 1 else np.nan,
                        "mean_g_true": float(np.mean(g_pop)) if g_pop.size > 0 else np.nan,
                        "mean_y_prob": float(np.mean(p_pop)) if p_pop.size > 0 else np.nan,
                    }
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "figure2_auc_by_method_population.tsv", sep="\t", index=False)
    res.to_csv(out_dir / "figure2_auc_brier_by_method_population.tsv", sep="\t", index=False)
    _log("[fig2] Wrote figure2_auc_by_method_population.tsv")
    _log("[fig2] Wrote figure2_auc_brier_by_method_population.tsv")
    _log_results_table(
        "[fig2] AUC/Brier/statistics by method and population",
        res.sort_values(["prs_source", "method", "population"]).reset_index(drop=True),
    )
    pred_rows = []
    for prs_source, preds in preds_by_source.items():
        test_prs = prs_sources[prs_source]["test"]
        for method, y_prob in preds.items():
            part = test_df[
                [
                    "IID",
                    "group",
                    "pop_label",
                    "y",
                    "G_true",
                    "afr_prop",
                    "eur_prop",
                    "asia_prop",
                    "pc1",
                    "pc2",
                    "pc3",
                    "pc4",
                    "pc5",
                    "seed",
                ]
            ].copy()
            part["prs_source"] = prs_source
            part["method"] = method
            part["prs"] = test_prs
            part["y_prob"] = np.asarray(y_prob, dtype=float)
            pred_rows.append(part)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "figure2_test_predictions.tsv", sep="\t", index=False)
    _log("[fig2] Wrote figure2_test_predictions.tsv")

    _log("[fig2] Plotting outputs")
    _plot_main(res, out_dir)
    for prs_source, prs_vals in prs_sources.items():
        _plot_prs_distribution(test_df, prs_vals["test"], out_dir, prs_source=prs_source)
    _plot_pcs(df, out_dir)

    if not args.keep_intermediates and not args.use_existing:
        _cleanup_seed_artifacts(prefix=Path(prefix), seed_work_dir=seed_work)
    elif not args.keep_intermediates and args.use_existing:
        shutil.rmtree(seed_work, ignore_errors=True)
    _log("[fig2] Figure2 run complete")


if __name__ == "__main__":
    main()
