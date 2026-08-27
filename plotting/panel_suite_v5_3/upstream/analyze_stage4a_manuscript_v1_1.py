#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage 4A manuscript-oriented analysis v1.1
=====================================

Purpose
-------
Analyze the Stage 4A discovery cohort as the model-side bridge from
static structural allocation to dynamical leverage.

This script deliberately does NOT frame the study around named neuron
models. In all outputs:
    high-complexity unit = model abstraction of richer intrinsic dynamics
    dynamics-aware allocation = optimizer-discovered placement
    relative dynamical leverage = validated score gain over a matched baseline

The script:
1. finds Stage4/Stage4A method tables;
2. isolates the Stage4A baseline cohort (default graph seeds 41001, 41002);
3. deduplicates byte-identical and scientifically identical task tables;
4. reconstructs matched task-level comparisons;
5. analyzes budget-, state-, and structure-dependent allocation effects;
6. outputs a model-side bridge table for later physiological validation;
7. keeps search-stage maxima separate from independent validation.

No simulation is rerun.

Default project root:
    D:\\Research\\Neural Science

Default output:
    D:\\Research\\Neural Science\\plot\\Stage4A\\analysis
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon, spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


DEFAULT_ROOT = Path.cwd()

ALIASES = {
    "method": [
        "method", "placement", "allocation_rule", "selector",
    ],
    "task_id": [
        "task_id", "complete_task_id", "runner_task_id", "task",
    ],
    "graph_seed": [
        "graph_seed", "seed", "network_seed",
    ],
    "regime": [
        "regime", "state", "operating_regime",
    ],
    "topology": [
        "topology", "graph_type", "network_topology",
    ],
    "k": [
        "k", "n_hh", "hh_count", "n_complex", "complexity_budget",
        "high_complexity_count",
    ],
    "N": [
        "N", "n_neurons", "network_size", "n_total",
    ],
    "score": [
        "final_score_mean", "validation_score_mean", "score_mean",
        "final_score", "rhythm_score_mean", "rhythm_score",
    ],
    "score_std": [
        "final_score_std", "validation_score_std", "score_std",
    ],
    "search_best": [
        "optimizer_search_best", "search_best", "best_search_score",
    ],
    "selected_nodes": [
        "selected_nodes", "selected_indices", "selected_ids",
        "mask_nodes", "mask_indices",
    ],
    "mask_hash": [
        "mask_hash", "mask_sha256", "selected_nodes_sha256",
    ],
    # structural phenotype
    "degree": [
        "sel_total_degree", "selected_total_degree", "selected_degree",
        "mean_selected_degree", "selection_degree",
    ],
    "coverage1": [
        "coverage1", "coverage_1", "one_hop_coverage", "coverage_one_hop",
    ],
    "coverage2": [
        "coverage2", "coverage_2", "two_hop_coverage", "coverage_two_hop",
    ],
    "redundancy": [
        "redundancy", "selection_redundancy", "neighbor_redundancy",
    ],
    "dispersion": [
        "dispersion", "selection_dispersion", "pairwise_dispersion",
    ],
    "spectral_descriptor": [
        "sel_spectral", "selected_spectral", "selection_spectral",
        "spectral_centrality",
    ],
    "feedback_descriptor": [
        "sel_feedback", "selected_feedback", "selection_feedback",
        "feedback_score",
    ],
    "cycle_descriptor": [
        "sel_cycle3", "selected_cycle3", "selection_cycle3", "cycle3",
    ],
    "bridge_descriptor": [
        "sel_bridge", "selected_bridge", "selection_bridge",
        "bridge_score",
    ],
}

DESCRIPTOR_OUTPUTS = {
    "degree": "sel_total_degree",
    "coverage1": "coverage1",
    "coverage2": "coverage2",
    "redundancy": "redundancy",
    "dispersion": "dispersion",
    "spectral_descriptor": "sel_spectral",
    "feedback_descriptor": "sel_feedback",
    "cycle_descriptor": "sel_cycle3",
    "bridge_descriptor": "sel_bridge",
}

METHOD_ALIASES = {
    "gpu_optimized": "dynamics_aware",
    "gpu optimized": "dynamics_aware",
    "gpu_optimizer": "dynamics_aware",
    "gpu surrogate": "dynamics_aware",
    "gpu_surrogate": "dynamics_aware",
    "optimized": "dynamics_aware",
    "optimizer": "dynamics_aware",
    "dynamics_aware": "dynamics_aware",
    "dynamics-aware": "dynamics_aware",
    "spectral": "spectral",
    "high_degree": "high_degree",
    "degree": "high_degree",
    "feedback_hub": "feedback_hub",
    "module_bridge": "module_bridge",
    "cycle_proxy": "cycle_proxy",
    "random": "random",
}

METHOD_LABELS = {
    "dynamics_aware": "Dynamics-aware allocation",
    "spectral": "Spectral",
    "feedback_hub": "Feedback hub",
    "high_degree": "High degree",
    "module_bridge": "Module bridge",
    "cycle_proxy": "Cycle proxy",
    "random": "Random",
}


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_col(df: pd.DataFrame, semantic: str) -> str | None:
    cmap = {str(c).lower(): str(c) for c in df.columns}
    for candidate in ALIASES.get(semantic, []):
        if candidate.lower() in cmap:
            return cmap[candidate.lower()]
    # conservative fuzzy fallback
    for c in df.columns:
        cl = str(c).lower()
        for candidate in ALIASES.get(semantic, []):
            cc = candidate.lower()
            if len(cc) >= 8 and cc in cl:
                return str(c)
    return None


def normalize_method(value) -> str:
    s = str(value).strip().lower()
    return METHOD_ALIASES.get(s, s.replace(" ", "_"))


def method_label(value) -> str:
    return METHOD_LABELS.get(str(value), str(value).replace("_", " ").title())


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    vals = p[finite]
    if vals.size == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    inv = np.empty_like(order)
    inv[order] = np.arange(m)
    corrected = q[inv]
    out[np.where(finite)[0]] = corrected
    return out


def bootstrap_mean_ci(values, n_boot=10000, seed=20260818):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    if arr.size == 1:
        x = float(arr[0])
        return x, x, x
    rng = np.random.default_rng(seed)
    # Chunk to avoid unnecessarily large memory.
    means = np.empty(n_boot, dtype=float)
    chunk = 1000
    cursor = 0
    while cursor < n_boot:
        b = min(chunk, n_boot - cursor)
        idx = rng.integers(0, arr.size, size=(b, arr.size))
        means[cursor:cursor+b] = arr[idx].mean(axis=1)
        cursor += b
    return (
        float(arr.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def paired_wilcoxon(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if not HAVE_SCIPY:
        return np.nan
    if np.allclose(arr, 0):
        return 1.0
    try:
        return float(wilcoxon(arr, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        try:
            return float(wilcoxon(arr).pvalue)
        except Exception:
            return np.nan


def spearman_xy(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())
    if HAVE_SCIPY:
        r = spearmanr(x[m], y[m])
        return float(r.statistic), float(r.pvalue), int(m.sum())
    # fallback rank correlation without p
    xr = pd.Series(x[m]).rank().to_numpy()
    yr = pd.Series(y[m]).rank().to_numpy()
    rho = np.corrcoef(xr, yr)[0, 1]
    return float(rho), np.nan, int(m.sum())


def read_table(path: Path):
    try:
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t")
        return pd.read_csv(path)
    except Exception:
        return None


def discover_candidate_tables(root: Path):
    """
    Search Stage4-like areas only.
    We intentionally do not scan virtual environments or plotting outputs.
    """
    exclude_dirs = {
        ".git", ".venv", "venv", "__pycache__", "node_modules",
        "plot", "plots", "figure", "figures",
    }
    candidates = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in exclude_dirs
            and not d.lower().startswith(".")
        ]
        pdir = Path(dirpath)
        low_dir = str(pdir).lower()

        # Stage4A may historically be stored under a generic results_stage4 tree.
        stageish = (
            "stage4a" in low_dir
            or "stage_4a" in low_dir
            or "stage4" in low_dir
            or "stage_4" in low_dir
        )

        for fn in filenames:
            suffix = Path(fn).suffix.lower()
            if suffix not in {".csv", ".tsv"}:
                continue
            low_fn = fn.lower()
            if not stageish and not any(
                token in low_fn for token in [
                    "stage4a", "stage4_a", "stage4_discovery",
                    "discovery_methods",
                ]
            ):
                continue
            candidates.append(pdir / fn)

    return sorted(set(candidates), key=lambda x: str(x).lower())


def normalize_table(df: pd.DataFrame, path: Path):
    sem = {k: find_col(df, k) for k in ALIASES}
    if not sem["method"] or not sem["score"]:
        return None, sem

    d = df.copy()
    rename = {}

    direct = {
        "method": "method_raw",
        "task_id": "task_id",
        "graph_seed": "graph_seed",
        "regime": "regime",
        "topology": "topology",
        "k": "k",
        "N": "N",
        "score": "final_score_mean",
        "score_std": "final_score_std",
        "search_best": "optimizer_search_best",
        "selected_nodes": "selected_nodes",
        "mask_hash": "mask_hash",
    }
    for semantic, outname in direct.items():
        c = sem.get(semantic)
        if c and c != outname:
            rename[c] = outname

    for semantic, outname in DESCRIPTOR_OUTPUTS.items():
        c = sem.get(semantic)
        if c and c != outname:
            rename[c] = outname

    d = d.rename(columns=rename)
    d["method"] = d["method_raw"].map(normalize_method)
    d["source_file"] = str(path)

    for col in [
        "graph_seed", "k", "N", "final_score_mean", "final_score_std",
        "optimizer_search_best", *DESCRIPTOR_OUTPUTS.values()
    ]:
        if col in d.columns:
            d[col] = safe_numeric(d[col])

    if "k" in d.columns and "N" in d.columns:
        d["complexity_fraction"] = d["k"] / d["N"].replace(0, np.nan)
    else:
        d["complexity_fraction"] = np.nan

    if "task_id" not in d.columns:
        d["task_id"] = ""
    d["task_id"] = d["task_id"].fillna("").astype(str)

    return d, sem


def build_canonical(root: Path, graph_seeds):
    candidate_paths = discover_candidate_tables(root)
    seen_hashes = {}
    frames = []
    audit = []

    for p in candidate_paths:
        df = read_table(p)
        if df is None:
            audit.append({
                "path": str(p),
                "status": "read_failed",
                "rows": np.nan,
                "file_sha256": "",
                "used": False,
                "reason": "read_failed",
            })
            continue

        sem = {k: find_col(df, k) for k in ALIASES}
        useful = bool(sem["method"] and sem["score"])
        if not useful:
            audit.append({
                "path": str(p),
                "status": "not_method_score_table",
                "rows": len(df),
                "file_sha256": "",
                "used": False,
                "reason": "missing_method_or_score",
            })
            continue

        h = sha256_file(p)
        if h in seen_hashes:
            audit.append({
                "path": str(p),
                "status": "duplicate_file",
                "rows": len(df),
                "file_sha256": h,
                "used": False,
                "reason": f"byte_identical_to:{seen_hashes[h]}",
            })
            continue
        seen_hashes[h] = str(p)

        d, sem = normalize_table(df, p)
        if d is None:
            continue

        # Preferred Stage4A isolation:
        # if graph_seed exists, use the known discovery cohort seeds.
        if graph_seeds and "graph_seed" in d.columns:
            before = len(d)
            d = d[d["graph_seed"].isin(graph_seeds)].copy()
            reason = f"filtered_to_graph_seeds:{','.join(map(str, graph_seeds))}"
        else:
            before = len(d)
            reason = "no_graph_seed_filter"

        if d.empty:
            audit.append({
                "path": str(p),
                "status": "no_stage4a_rows_after_filter",
                "rows": before,
                "file_sha256": h,
                "used": False,
                "reason": reason,
            })
            continue

        d["source_sha256"] = h
        frames.append(d)
        audit.append({
            "path": str(p),
            "status": "used",
            "rows": len(d),
            "file_sha256": h,
            "used": True,
            "reason": reason,
        })

    if not frames:
        raise RuntimeError(
            "No Stage4A method-score rows were found. "
            "Check candidate paths or pass --graph-seeds with the correct Stage4A seeds."
        )

    canon = pd.concat(frames, ignore_index=True, sort=False)

    # Construct missing task ids from matched scientific context.
    task_key_cols = [
        c for c in ["graph_seed", "topology", "regime", "k", "N"]
        if c in canon.columns
    ]
    missing = canon["task_id"].astype(str).str.strip().eq("")
    if task_key_cols:
        generated = canon[task_key_cols].astype(str).agg("|".join, axis=1)
        canon.loc[missing, "task_id"] = generated.loc[missing]

    # ------------------------------------------------------------------
    # STRICT STAGE4A TASK CANONICALIZATION
    #
    # Intended experimental design per task:
    #   random = 32 rows
    #   each non-random allocation method = 1 row
    #   total = 38 rows
    #
    # Historical result trees can also contain aggregate/partial tables
    # carrying extra random summaries.  We therefore select exactly one
    # complete per-task source table, rather than truncating random rows or
    # deduplicating by score.
    # ------------------------------------------------------------------
    expected_nonrandom = [
        "dynamics_aware",
        "spectral",
        "feedback_hub",
        "high_degree",
        "module_bridge",
        "cycle_proxy",
    ]

    def is_complete_stage4a_task(g):
        counts = g["method"].value_counts().to_dict()
        if len(g) != 38:
            return False
        if counts.get("random", 0) != 32:
            return False
        return all(counts.get(m, 0) == 1 for m in expected_nonrandom)

    selected_task_frames = []
    source_selection_rows = []
    pre_selection_rows = len(canon)

    for task_id, task_df in canon.groupby("task_id", sort=False, dropna=False):
        valid_sources = []

        if "source_file" in task_df.columns:
            for source_file, sg in task_df.groupby(
                "source_file", sort=False, dropna=False
            ):
                if is_complete_stage4a_task(sg):
                    src_text = str(source_file)
                    basename = Path(src_text).name.lower()

                    # Prefer the explicitly canonical runner output.
                    if basename == "stage4_discovery_methods.csv":
                        priority = 0
                    elif "runner_output" in src_text.lower():
                        priority = 1
                    else:
                        priority = 2

                    valid_sources.append(
                        (priority, len(src_text), src_text, sg.copy())
                    )

        if valid_sources:
            valid_sources.sort(key=lambda x: (x[0], x[1], x[2]))
            _, _, chosen_source, chosen = valid_sources[0]
            selected_task_frames.append(chosen)
            source_selection_rows.append({
                "task_id": task_id,
                "status": "selected_complete_source",
                "chosen_source": chosen_source,
                "rows_before": len(task_df),
                "rows_after": len(chosen),
                "random_before": int((task_df["method"] == "random").sum()),
                "random_after": int((chosen["method"] == "random").sum()),
                "candidate_complete_sources": len(valid_sources),
            })
            continue

        # Recovery path: if the task is assembled from several files, first
        # remove exact scientific duplicates.  Accept it ONLY if the resulting
        # task matches the intended 38-row design exactly.
        identity = [
            c for c in [
                "task_id", "graph_seed", "topology", "regime", "k", "N",
                "method", "final_score_mean", "final_score_std",
                "optimizer_search_best", "sel_total_degree", "coverage1",
                "coverage2", "redundancy", "dispersion", "sel_spectral",
                "sel_feedback", "sel_cycle3", "sel_bridge",
                "selected_nodes", "mask_hash",
            ] if c in task_df.columns
        ]
        recovered = task_df.drop_duplicates(
            subset=identity, keep="first"
        ).copy()

        if is_complete_stage4a_task(recovered):
            selected_task_frames.append(recovered)
            source_selection_rows.append({
                "task_id": task_id,
                "status": "recovered_by_scientific_dedup",
                "chosen_source": "MULTI_SOURCE",
                "rows_before": len(task_df),
                "rows_after": len(recovered),
                "random_before": int((task_df["method"] == "random").sum()),
                "random_after": int((recovered["method"] == "random").sum()),
                "candidate_complete_sources": 0,
            })
            continue

        counts = task_df["method"].value_counts().to_dict()
        raise RuntimeError(
            "Stage4A task cannot be canonicalized safely. "
            f"task_id={task_id}; rows={len(task_df)}; "
            f"method_counts={counts}. "
            "No complete 38-row source table was found. "
            "Do NOT truncate random rows manually; inspect source_file provenance."
        )

    canon = pd.concat(
        selected_task_frames, ignore_index=True, sort=False
    )

    # Final hard validation: every task must be exactly 38 rows.
    task_counts = canon.groupby("task_id").size()
    random_counts = (
        canon[canon["method"] == "random"]
        .groupby("task_id")
        .size()
        .reindex(task_counts.index, fill_value=0)
    )

    bad_tasks = task_counts[task_counts != 38]
    bad_random = random_counts[random_counts != 32]
    if len(bad_tasks) or len(bad_random):
        raise RuntimeError(
            "Final Stage4A cardinality check failed. "
            f"bad_total={bad_tasks.to_dict()}, "
            f"bad_random={bad_random.to_dict()}"
        )

    expected_rows = int(len(task_counts) * 38)
    if len(canon) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} canonical rows "
            f"for {len(task_counts)} tasks, found {len(canon)}."
        )

    source_selection = pd.DataFrame(source_selection_rows)
    source_selection.to_csv(
        root / "plot" / "Stage4A" / "stage4a_task_source_selection_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    scientific_duplicates_removed = pre_selection_rows - len(canon)

    return canon, pd.DataFrame(audit), scientific_duplicates_removed


def task_id_cols(canon):
    cols = [
        c for c in ["task_id", "graph_seed", "topology", "regime", "k", "N"]
        if c in canon.columns
    ]
    return cols if cols else ["task_id"]


def task_method_means(canon):
    ids = task_id_cols(canon)
    out = (
        canon.groupby(ids + ["method"], dropna=False)["final_score_mean"]
        .mean()
        .reset_index()
    )
    return out, ids


def task_wide(canon):
    tm, ids = task_method_means(canon)
    wide = tm.pivot_table(
        index=ids,
        columns="method",
        values="final_score_mean",
        aggfunc="mean",
    ).reset_index()
    return wide, ids


def overall_method_summary(canon):
    tm, _ = task_method_means(canon)
    rows = []
    methods = list(tm["method"].dropna().astype(str).unique())
    for i, m in enumerate(methods):
        vals = tm.loc[tm["method"].eq(m), "final_score_mean"].to_numpy(float)
        mean, lo, hi = bootstrap_mean_ci(vals, seed=20260818+i)
        rows.append({
            "method": m,
            "method_label": method_label(m),
            "n_tasks": len(vals),
            "mean_validated_score": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "median_validated_score": float(np.median(vals)) if len(vals) else np.nan,
            "std_across_tasks": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        "mean_validated_score", ascending=False
    ).reset_index(drop=True)


def paired_baseline_comparisons(canon):
    wide, ids = task_wide(canon)
    if "dynamics_aware" not in wide.columns:
        raise RuntimeError(
            "No dynamics-aware method found after method normalization."
        )

    baselines = [
        "spectral", "feedback_hub", "high_degree",
        "module_bridge", "cycle_proxy", "random"
    ]
    rows = []
    for i, b in enumerate(baselines):
        if b not in wide.columns:
            continue
        d = (
            safe_numeric(wide["dynamics_aware"])
            - safe_numeric(wide[b])
        ).dropna().to_numpy(float)
        if not len(d):
            continue
        mean, lo, hi = bootstrap_mean_ci(d, seed=20260818+i)
        npos = int(np.sum(d > 0))
        nneg = int(np.sum(d < 0))
        denom = npos + nneg
        rank_biserial_sign = (
            (npos - nneg) / denom if denom else 0.0
        )
        rows.append({
            "baseline": b,
            "baseline_label": method_label(b),
            "n_paired_tasks": len(d),
            "mean_gain": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "median_gain": float(np.median(d)),
            "win_fraction": float(np.mean(d > 0)),
            "loss_fraction": float(np.mean(d < 0)),
            "signed_win_balance": rank_biserial_sign,
            "wilcoxon_p": paired_wilcoxon(d),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_fdr_bh"] = bh_fdr(out["wilcoxon_p"].to_numpy())
    return out, wide, ids


def paired_gain_by(canon, baseline, factors):
    wide, ids = task_wide(canon)
    if "dynamics_aware" not in wide.columns or baseline not in wide.columns:
        return pd.DataFrame()
    wide["gain"] = safe_numeric(wide["dynamics_aware"]) - safe_numeric(wide[baseline])
    factors = [f for f in factors if f in wide.columns]
    if not factors:
        return pd.DataFrame()

    rows = []
    grouped = wide.dropna(subset=["gain"]).groupby(factors, dropna=False, sort=True)
    for j, (key, g) in enumerate(grouped):
        if not isinstance(key, tuple):
            key = (key,)
        vals = g["gain"].dropna().to_numpy(float)
        mean, lo, hi = bootstrap_mean_ci(vals, seed=20260818+j)
        row = {f: v for f, v in zip(factors, key)}
        row.update({
            "baseline": baseline,
            "n_paired_tasks": len(vals),
            "mean_gain": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "median_gain": float(np.median(vals)) if len(vals) else np.nan,
            "win_fraction": float(np.mean(vals > 0)) if len(vals) else np.nan,
            "wilcoxon_p": paired_wilcoxon(vals),
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_fdr_bh"] = bh_fdr(out["wilcoxon_p"].to_numpy())
    return out


def build_relative_leverage_table(canon):
    """
    Relative dynamical leverage is defined here purely operationally:
        validated score(method) - validated score(random)
    within the same graph/state/budget task.

    This is a model-side quantity, not a physiological measurement.
    """
    wide, ids = task_wide(canon)
    tm, _ = task_method_means(canon)
    if "random" not in wide.columns:
        return pd.DataFrame()

    rand = wide[ids + ["random"]].rename(
        columns={"random": "random_validated_score"}
    )
    out = tm.merge(rand, on=ids, how="left")
    out["relative_dynamical_leverage"] = (
        safe_numeric(out["final_score_mean"])
        - safe_numeric(out["random_validated_score"])
    )
    return out


def descriptor_columns(canon):
    return [
        c for c in [
            "sel_total_degree", "coverage1", "coverage2", "redundancy",
            "dispersion", "sel_spectral", "sel_feedback",
            "sel_cycle3", "sel_bridge",
        ]
        if c in canon.columns
    ]


def task_descriptor_wide(canon, descriptor):
    ids = task_id_cols(canon)
    d = canon[ids + ["method", descriptor]].copy()
    d[descriptor] = safe_numeric(d[descriptor])
    d = d.dropna(subset=[descriptor])
    if d.empty:
        return pd.DataFrame(), ids
    t = (
        d.groupby(ids + ["method"], dropna=False)[descriptor]
        .mean().reset_index()
    )
    w = t.pivot_table(
        index=ids, columns="method", values=descriptor, aggfunc="mean"
    ).reset_index()
    return w, ids


def descriptor_shifts_vs_baseline(canon, baseline="spectral"):
    rows = []
    for di, desc in enumerate(descriptor_columns(canon)):
        w, ids = task_descriptor_wide(canon, desc)
        if w.empty or "dynamics_aware" not in w.columns or baseline not in w.columns:
            continue
        w["shift"] = safe_numeric(w["dynamics_aware"]) - safe_numeric(w[baseline])

        grouping_options = [
            [],
            ["regime"] if "regime" in w.columns else [],
            ["k"] if "k" in w.columns else [],
            [c for c in ["regime", "k"] if c in w.columns],
        ]
        seen = set()
        for factors in grouping_options:
            key_tuple = tuple(factors)
            if key_tuple in seen:
                continue
            seen.add(key_tuple)

            if not factors:
                groups = [((), w)]
            else:
                groups = w.groupby(factors, dropna=False, sort=True)

            for gi, (key, g) in enumerate(groups):
                if factors and not isinstance(key, tuple):
                    key = (key,)
                vals = g["shift"].dropna().to_numpy(float)
                if not len(vals):
                    continue
                mean, lo, hi = bootstrap_mean_ci(
                    vals, seed=20260818 + di * 100 + gi
                )
                row = {
                    "descriptor": desc,
                    "baseline": baseline,
                    "grouping": "+".join(factors) if factors else "overall",
                    "n_tasks": len(vals),
                    "mean_shift": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "median_shift": float(np.median(vals)),
                    "positive_fraction": float(np.mean(vals > 0)),
                    "wilcoxon_p": paired_wilcoxon(vals),
                }
                if factors:
                    for f, v in zip(factors, key):
                        row[f] = v
                rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        mask = out["wilcoxon_p"].notna()
        out.loc[mask, "wilcoxon_fdr_bh"] = bh_fdr(
            out.loc[mask, "wilcoxon_p"].to_numpy()
        )
    return out


def state_reconfiguration(canon):
    """
    Within the dynamics-aware method, compare structural phenotype across
    matched graph seed + budget conditions.

    This is an early state-reconfiguration analysis.
    It is NOT optimizer-stability proof; Stage4F is intended to test that.
    """
    if "regime" not in canon.columns:
        return pd.DataFrame()

    d = canon[canon["method"].eq("dynamics_aware")].copy()
    descs = descriptor_columns(d)
    if not descs:
        return pd.DataFrame()

    base_ids = [c for c in ["graph_seed", "topology", "k", "N"] if c in d.columns]
    if not base_ids:
        return pd.DataFrame()

    # First collapse any accidental duplicate rows within state.
    agg = (
        d.groupby(base_ids + ["regime"], dropna=False)[descs + ["final_score_mean"]]
        .mean().reset_index()
    )

    regimes = list(agg["regime"].dropna().astype(str).unique())
    rows = []
    for a in regimes:
        for b in regimes:
            if a >= b:
                continue
            da = agg[agg["regime"].astype(str).eq(a)].copy()
            db = agg[agg["regime"].astype(str).eq(b)].copy()
            m = da.merge(
                db, on=base_ids, suffixes=(f"__{a}", f"__{b}")
            )
            if m.empty:
                continue
            for metric in descs + ["final_score_mean"]:
                va = safe_numeric(m[f"{metric}__{a}"])
                vb = safe_numeric(m[f"{metric}__{b}"])
                diff = (vb - va).dropna().to_numpy(float)
                if not len(diff):
                    continue
                mean, lo, hi = bootstrap_mean_ci(diff)
                rows.append({
                    "state_a": a,
                    "state_b": b,
                    "contrast": f"{b} - {a}",
                    "metric": metric,
                    "n_matched_graph_budget_conditions": len(diff),
                    "mean_shift": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "median_shift": float(np.median(diff)),
                    "positive_fraction": float(np.mean(diff > 0)),
                    "wilcoxon_p": paired_wilcoxon(diff),
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_fdr_bh"] = bh_fdr(out["wilcoxon_p"].to_numpy())
    return out


def descriptor_gain_correlations(canon, baseline="spectral"):
    """
    Exploratory: correlate dynamics-aware structural departure from a baseline
    with dynamics-aware performance gain over the same baseline.

    This supports mechanism generation, not a universal allocation law.
    """
    score_wide, ids = task_wide(canon)
    if "dynamics_aware" not in score_wide.columns or baseline not in score_wide.columns:
        return pd.DataFrame()

    base = score_wide[ids + ["dynamics_aware", baseline]].copy()
    base["score_gain"] = (
        safe_numeric(base["dynamics_aware"]) - safe_numeric(base[baseline])
    )

    rows = []
    for desc in descriptor_columns(canon):
        dw, _ = task_descriptor_wide(canon, desc)
        if dw.empty or "dynamics_aware" not in dw.columns or baseline not in dw.columns:
            continue
        dd = dw[ids + ["dynamics_aware", baseline]].copy()
        dd["descriptor_shift"] = (
            safe_numeric(dd["dynamics_aware"]) - safe_numeric(dd[baseline])
        )
        m = base[ids + ["score_gain"]].merge(
            dd[ids + ["descriptor_shift"]], on=ids, how="inner"
        )
        rho, p, n = spearman_xy(
            m["descriptor_shift"].to_numpy(),
            m["score_gain"].to_numpy(),
        )
        rows.append({
            "descriptor": desc,
            "baseline": baseline,
            "n_tasks": n,
            "spearman_rho_descriptor_shift_vs_score_gain": rho,
            "spearman_p": p,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["spearman_fdr_bh"] = bh_fdr(out["spearman_p"].to_numpy())
    return out


def search_validation_audit(canon):
    if "optimizer_search_best" not in canon.columns:
        return pd.DataFrame(), pd.DataFrame()

    d = canon[canon["method"].eq("dynamics_aware")].copy()
    d["optimizer_search_best"] = safe_numeric(d["optimizer_search_best"])
    d["final_score_mean"] = safe_numeric(d["final_score_mean"])
    d = d.dropna(subset=["optimizer_search_best", "final_score_mean"])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    d["search_minus_validation"] = (
        d["optimizer_search_best"] - d["final_score_mean"]
    )

    rho, p, n = spearman_xy(
        d["optimizer_search_best"].to_numpy(),
        d["final_score_mean"].to_numpy(),
    )
    summary = pd.DataFrame([{
        "n_tasks": len(d),
        "mean_search_minus_validation": float(d["search_minus_validation"].mean()),
        "median_search_minus_validation": float(d["search_minus_validation"].median()),
        "std_search_minus_validation": float(d["search_minus_validation"].std(ddof=1))
            if len(d) > 1 else np.nan,
        "spearman_search_vs_validation": rho,
        "spearman_p": p,
    }])
    return d, summary


def parse_selected_nodes(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple, set, np.ndarray)):
        try:
            return set(int(x) for x in value)
        except Exception:
            return set(str(x) for x in value)
    s = str(value).strip()
    if not s:
        return None
    try:
        x = ast.literal_eval(s)
        if isinstance(x, (list, tuple, set, np.ndarray)):
            try:
                return set(int(v) for v in x)
            except Exception:
                return set(str(v) for v in x)
    except Exception:
        pass
    nums = re.findall(r"-?\d+", s)
    if nums:
        return set(int(x) for x in nums)
    return None


def mask_overlap_across_states(canon):
    """
    Exploratory only.
    Between-state overlap cannot establish state-specific optimization until
    Stage4F quantifies within-state optimizer variability.
    """
    if "selected_nodes" not in canon.columns or "regime" not in canon.columns:
        return pd.DataFrame()

    d = canon[canon["method"].eq("dynamics_aware")].copy()
    d["_nodes"] = d["selected_nodes"].map(parse_selected_nodes)
    d = d[d["_nodes"].notna()].copy()
    if d.empty:
        return pd.DataFrame()

    base_ids = [c for c in ["graph_seed", "topology", "k", "N"] if c in d.columns]
    rows = []

    for key, g in d.groupby(base_ids, dropna=False):
        regimes = list(g["regime"].astype(str).unique())
        for i, a in enumerate(regimes):
            for b in regimes[i+1:]:
                ga = g[g["regime"].astype(str).eq(a)]
                gb = g[g["regime"].astype(str).eq(b)]
                if ga.empty or gb.empty:
                    continue
                A = ga.iloc[0]["_nodes"]
                B = gb.iloc[0]["_nodes"]
                if A is None or B is None:
                    continue
                union = len(A | B)
                jac = len(A & B) / union if union else np.nan
                row = {
                    "state_a": a,
                    "state_b": b,
                    "jaccard": jac,
                    "intersection": len(A & B),
                    "union": union,
                    "size_a": len(A),
                    "size_b": len(B),
                }
                if not isinstance(key, tuple):
                    key = (key,)
                row.update({c: v for c, v in zip(base_ids, key)})
                rows.append(row)

    return pd.DataFrame(rows)


def selector_collision_audit(canon):
    """
    Exact selector collision audit.
    Uses mask_hash when available, otherwise selected_nodes.
    Similar scores are NOT treated as identical masks.
    """
    key_col = None
    if "mask_hash" in canon.columns and canon["mask_hash"].notna().any():
        key_col = "mask_hash"
    elif "selected_nodes" in canon.columns and canon["selected_nodes"].notna().any():
        key_col = "selected_nodes"
    if key_col is None:
        return pd.DataFrame()

    ids = task_id_cols(canon)
    d = canon[ids + ["method", key_col]].dropna().copy()
    structural = [
        m for m in [
            "spectral", "feedback_hub", "high_degree",
            "module_bridge", "cycle_proxy", "dynamics_aware"
        ] if m in set(d["method"])
    ]

    rows = []
    for i, a in enumerate(structural):
        for b in structural[i+1:]:
            w = d[d["method"].isin([a, b])].pivot_table(
                index=ids, columns="method", values=key_col, aggfunc="first"
            ).dropna()
            if a not in w.columns or b not in w.columns or w.empty:
                continue
            eq = w[a].astype(str).eq(w[b].astype(str))
            rows.append({
                "method_a": a,
                "method_b": b,
                "n_paired_tasks": len(eq),
                "identical_count": int(eq.sum()),
                "identical_fraction": float(eq.mean()),
            })
    return pd.DataFrame(rows)


def build_physiology_bridge_table(canon, leverage_table):
    """
    Model-side table designed for later physiological validation.

    It contains no biological measurements. It only makes the simulation-side
    constructs explicit:
      - intrinsic complexity budget
      - collective operating state
      - validated relative dynamical leverage
      - structural phenotype

    Later biological data should be joined through independently defined
    cell-complexity and network-leverage observables, not by relabeling
    model identities as biological cell types.
    """
    ids = task_id_cols(canon)
    descs = descriptor_columns(canon)

    cols = ids + [
        c for c in [
            "complexity_fraction", "method", "final_score_mean",
            "final_score_std"
        ] if c in canon.columns
    ] + descs

    base = (
        canon[cols]
        .groupby(ids + [c for c in ["complexity_fraction", "method"] if c in cols],
                 dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )

    if leverage_table.empty:
        base["relative_dynamical_leverage"] = np.nan
        return base

    lev_cols = ids + ["method", "relative_dynamical_leverage"]
    out = base.merge(
        leverage_table[lev_cols],
        on=ids + ["method"],
        how="left"
    )
    return out


def write_markdown_summary(
    outdir: Path,
    canon: pd.DataFrame,
    audit: pd.DataFrame,
    duplicates_removed: int,
    overall: pd.DataFrame,
    contrasts: pd.DataFrame,
    budget_state: pd.DataFrame,
    search_summary: pd.DataFrame,
):
    lines = []
    lines += [
        "# Stage 4A manuscript-oriented analysis summary",
        "",
        "## Scope",
        "",
        "This analysis treats Stage 4A as the discovery phase for a general problem:",
        "where should a finite budget of richer intrinsic cellular dynamics be allocated",
        "to exert the greatest network-level dynamical leverage?",
        "",
        "Named neuron models and GPU implementation details are intentionally not used",
        "as the manuscript-level scientific framing.",
        "",
        "## Data hygiene",
        "",
        f"- Canonical rows after deduplication: {len(canon):,}",
        f"- Unique matched task IDs: {canon['task_id'].nunique() if 'task_id' in canon.columns else 'NA'}",
        f"- Scientifically duplicate rows removed: {duplicates_removed:,}",
        f"- Candidate files used: {int(audit['used'].sum()) if not audit.empty and 'used' in audit.columns else 'NA'}",
        "",
        "Search-stage maxima and independent validation are analyzed separately.",
        "",
        "## Scientific interpretation",
        "",
        "Stage 4A should support three manuscript transitions:",
        "",
        "1. Static structural centrality is an informative but incomplete proxy for dynamical value.",
        "2. The benefit of dynamics-aware allocation depends on finite complexity budget and collective state.",
        "3. Optimal allocations should therefore be interpreted through a state-conditioned dynamical-leverage landscape,",
        "   which Stage 4F will test for optimizer stability and fresh validation.",
        "",
        "The later physiological-data section should test whether independently defined",
        "intrinsic cellular complexity covaries with network dynamical leverage. It should",
        "not treat the two model endpoints as literal biological cell classes.",
        "",
    ]

    if not overall.empty:
        lines += ["## Overall validated method ranking", ""]
        for _, r in overall.iterrows():
            lines.append(
                f"- {r['method_label']}: mean={r['mean_validated_score']:.4f}, "
                f"95% bootstrap CI [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}], "
                f"n={int(r['n_tasks'])}."
            )
        lines.append("")

    if not contrasts.empty:
        lines += ["## Paired dynamics-aware contrasts", ""]
        for _, r in contrasts.iterrows():
            q = r.get("wilcoxon_fdr_bh", np.nan)
            qtxt = f"{q:.3g}" if np.isfinite(q) else "NA"
            lines.append(
                f"- vs {r['baseline_label']}: Δ={r['mean_gain']:.4f}, "
                f"win={100*r['win_fraction']:.1f}%, "
                f"FDR-adjusted p={qtxt}, n={int(r['n_paired_tasks'])}."
            )
        lines.append("")

    if not budget_state.empty:
        lines += [
            "## State × budget",
            "",
            "See `gain_by_regime_and_budget_vs_spectral.csv` for the discovery-phase",
            "state-conditioned allocation landscape. These results are hypothesis-forming;",
            "Stage 4F is the confirmatory stability layer.",
            "",
        ]

    if not search_summary.empty:
        r = search_summary.iloc[0]
        lines += [
            "## Search / validation separation",
            "",
            f"- Mean search-minus-validation gap: {r['mean_search_minus_validation']:.4f}",
            f"- Median gap: {r['median_search_minus_validation']:.4f}",
            f"- Spearman(search, validation): {r['spearman_search_vs_validation']:.4f}",
            "",
        ]

    lines += [
        "## Output interpretation labels",
        "",
        "- `relative_dynamical_leverage` = validated score relative to matched random allocation.",
        "- Descriptor shifts vs spectral quantify how the dynamics-aware mask departs from a strong structural baseline.",
        "- Between-state mask overlap is exploratory only until Stage 4F establishes within-state optimizer variability.",
        "- `physiology_bridge_model_table.csv` contains model-side quantities only; no biological inference is made by this script.",
        "",
    ]

    (outdir / "stage4a_analysis_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Stage4A manuscript-oriented analysis for NeuralScience"
    )
    ap.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help="Project root"
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Output directory; default ROOT/plot/Stage4A/analysis"
    )
    ap.add_argument(
        "--graph-seeds", nargs="*", type=int, default=[41001, 41002],
        help=(
            "Stage4A discovery graph seeds. Default: 41001 41002. "
            "Pass an empty list only if Stage4A files are isolated by directory."
        )
    )
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Project root not found: {root}")

    outdir = (
        args.output.resolve()
        if args.output is not None
        else root / "plot" / "Stage4A" / "analysis"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    print("[Stage4A] Building canonical discovery cohort ...")
    canon, file_audit, dup_removed = build_canonical(
        root, args.graph_seeds
    )

    # Save canonical data first.
    canon.to_csv(
        outdir / "stage4a_canonical_methods.csv",
        index=False, encoding="utf-8-sig"
    )
    file_audit.to_csv(
        outdir / "stage4a_file_dedup_audit.csv",
        index=False, encoding="utf-8-sig"
    )

    # 1. Overall validated method summary.
    overall = overall_method_summary(canon)
    overall.to_csv(
        outdir / "overall_validated_method_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    # 2. Matched paired comparisons.
    contrasts, wide, ids = paired_baseline_comparisons(canon)
    contrasts.to_csv(
        outdir / "paired_dynamics_aware_vs_baselines.csv",
        index=False, encoding="utf-8-sig"
    )
    wide.to_csv(
        outdir / "task_level_paired_scores.csv",
        index=False, encoding="utf-8-sig"
    )

    # 3. Budget, state, and state×budget.
    gain_budget_spec = paired_gain_by(canon, "spectral", ["k"])
    gain_budget_rand = paired_gain_by(canon, "random", ["k"])
    gain_state_spec = paired_gain_by(canon, "spectral", ["regime"])
    gain_state_rand = paired_gain_by(canon, "random", ["regime"])
    gain_state_budget_spec = paired_gain_by(
        canon, "spectral", ["regime", "k"]
    )
    gain_state_budget_rand = paired_gain_by(
        canon, "random", ["regime", "k"]
    )

    gain_budget_spec.to_csv(
        outdir / "gain_by_budget_vs_spectral.csv",
        index=False, encoding="utf-8-sig"
    )
    gain_budget_rand.to_csv(
        outdir / "gain_by_budget_vs_random.csv",
        index=False, encoding="utf-8-sig"
    )
    gain_state_spec.to_csv(
        outdir / "gain_by_regime_vs_spectral.csv",
        index=False, encoding="utf-8-sig"
    )
    gain_state_rand.to_csv(
        outdir / "gain_by_regime_vs_random.csv",
        index=False, encoding="utf-8-sig"
    )
    gain_state_budget_spec.to_csv(
        outdir / "gain_by_regime_and_budget_vs_spectral.csv",
        index=False, encoding="utf-8-sig"
    )
    gain_state_budget_rand.to_csv(
        outdir / "gain_by_regime_and_budget_vs_random.csv",
        index=False, encoding="utf-8-sig"
    )

    # 4. Relative dynamical leverage.
    leverage = build_relative_leverage_table(canon)
    leverage.to_csv(
        outdir / "relative_dynamical_leverage_table.csv",
        index=False, encoding="utf-8-sig"
    )

    # 5. Structural phenotype.
    shifts_spec = descriptor_shifts_vs_baseline(canon, "spectral")
    shifts_rand = descriptor_shifts_vs_baseline(canon, "random")
    shifts_spec.to_csv(
        outdir / "structural_phenotype_shifts_vs_spectral.csv",
        index=False, encoding="utf-8-sig"
    )
    shifts_rand.to_csv(
        outdir / "structural_phenotype_shifts_vs_random.csv",
        index=False, encoding="utf-8-sig"
    )

    # 6. State reconfiguration of optimized masks.
    reconfig = state_reconfiguration(canon)
    reconfig.to_csv(
        outdir / "state_reconfiguration_of_dynamics_aware_allocation.csv",
        index=False, encoding="utf-8-sig"
    )

    # 7. Descriptor shift -> performance-gain relations.
    corr_spec = descriptor_gain_correlations(canon, "spectral")
    corr_rand = descriptor_gain_correlations(canon, "random")
    corr_spec.to_csv(
        outdir / "descriptor_shift_vs_gain_correlations_spectral.csv",
        index=False, encoding="utf-8-sig"
    )
    corr_rand.to_csv(
        outdir / "descriptor_shift_vs_gain_correlations_random.csv",
        index=False, encoding="utf-8-sig"
    )

    # 8. Search vs fresh validation hygiene.
    search_rows, search_summary = search_validation_audit(canon)
    search_rows.to_csv(
        outdir / "search_vs_validation_task_audit.csv",
        index=False, encoding="utf-8-sig"
    )
    search_summary.to_csv(
        outdir / "search_vs_validation_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    # 9. Mask-level exploratory audits if representations are available.
    overlap = mask_overlap_across_states(canon)
    overlap.to_csv(
        outdir / "between_state_mask_overlap_exploratory.csv",
        index=False, encoding="utf-8-sig"
    )

    collisions = selector_collision_audit(canon)
    collisions.to_csv(
        outdir / "selector_collision_audit.csv",
        index=False, encoding="utf-8-sig"
    )

    # 10. Model-side biological bridge table.
    bridge = build_physiology_bridge_table(canon, leverage)
    bridge.to_csv(
        outdir / "physiology_bridge_model_table.csv",
        index=False, encoding="utf-8-sig"
    )

    # 11. Manifest.
    manifest = {
        "analysis_name": "Stage4A manuscript-oriented dynamics-aware allocation analysis",
        "project_root": str(root),
        "graph_seeds": args.graph_seeds,
        "canonical_rows": int(len(canon)),
        "unique_task_ids": int(canon["task_id"].nunique()) if "task_id" in canon.columns else None,
        "methods": sorted(canon["method"].dropna().astype(str).unique().tolist()),
        "regimes": sorted(canon["regime"].dropna().astype(str).unique().tolist())
            if "regime" in canon.columns else [],
        "budgets_k": sorted(
            [float(x) for x in canon["k"].dropna().unique().tolist()]
        ) if "k" in canon.columns else [],
        "scientific_duplicate_rows_removed": int(dup_removed),
        "scipy_available": HAVE_SCIPY,
        "interpretation": {
            "primary_question": (
                "Does dynamical leverage, rather than static structural centrality alone, "
                "determine where finite intrinsic cellular complexity is most effective?"
            ),
            "stage4a_role": "discovery",
            "stage4f_role": "optimizer-stability and fresh-validation closure",
            "physiology_role": (
                "independent biological test of the relation between intrinsic cellular "
                "complexity and network dynamical leverage"
            ),
        },
    }
    (outdir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    write_markdown_summary(
        outdir=outdir,
        canon=canon,
        audit=file_audit,
        duplicates_removed=dup_removed,
        overall=overall,
        contrasts=contrasts,
        budget_state=gain_state_budget_spec,
        search_summary=search_summary,
    )

    print()
    print("[Stage4A] DONE")
    print(f"  canonical rows : {len(canon):,}")
    print(f"  unique tasks   : {canon['task_id'].nunique() if 'task_id' in canon.columns else 'NA'}")
    print(f"  methods        : {canon['method'].nunique()}")
    print(f"  duplicates rm  : {dup_removed:,}")
    print(f"  output         : {outdir}")
    print()
    print("Key manuscript-oriented outputs:")
    print("  overall_validated_method_summary.csv")
    print("  paired_dynamics_aware_vs_baselines.csv")
    print("  gain_by_regime_and_budget_vs_spectral.csv")
    print("  structural_phenotype_shifts_vs_spectral.csv")
    print("  state_reconfiguration_of_dynamics_aware_allocation.csv")
    print("  physiology_bridge_model_table.csv")
    print("  stage4a_analysis_summary.md")


if __name__ == "__main__":
    main()
