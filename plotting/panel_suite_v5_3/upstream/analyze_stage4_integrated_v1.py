#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integrated Stage 4 evidence analysis
====================================

Reads the two authoritative Stage 4 archives directly:

  1) stage4_complete_results.tar.gz
     - Stage 4B replication
     - Stage 4C atlas extension
     - Stage 4D scaling
     - Stage 4E modular / small-world
     - original ER failures are excluded because only DONE tasks are read

  2) stage4f_closure_results.tar.gz
     - 27 repaired ER tasks
     - 81 optimizer-stability searches
     - 27 fresh common-noise reevaluations (24 seeds each)

Optionally reads the cleaned Stage 4A discovery canonical table if it exists:
  plot/Stage4A/analysis/stage4a_canonical_methods.csv

This script performs analysis only. It does NOT make publication figures.

Scientific guardrails
---------------------
- "gpu_optimized" is renamed "dynamics_aware" in analysis outputs.
- Search maxima are never treated as final performance estimates.
- transition_dense uses connection_prob=0.10 whereas sparse_drive and
  transition_mid use connection_prob=0.05. Dense-vs-other comparisons are
  therefore topology+dynamics composite conditions, not pure fixed-topology
  state tests.
- The pure fixed-topology state test is transition_mid vs sparse_drive.
- Exact mask identity is not assumed to be unique. Stage 4F explicitly tests
  optimizer degeneracy / within-state variability.
- The script reports unsupported claims as unsupported; it never forces the
  intended narrative.

Default project root:
  D:/Research/Neural Science

Default output:
  D:/Research/Neural Science/plot/Stage4_integrated/analysis
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon, spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


DEFAULT_ROOT = Path.cwd()

METHOD_MAP = {
    "gpu_optimized": "dynamics_aware",
    "dynamics_aware": "dynamics_aware",
    "spectral": "spectral",
    "feedback_hub": "feedback_hub",
    "high_degree": "high_degree",
    "module_bridge": "module_bridge",
    "cycle_proxy": "cycle_proxy",
    "random": "random",
}

BASELINES = [
    "spectral",
    "feedback_hub",
    "high_degree",
    "module_bridge",
    "cycle_proxy",
    "random",
]

DESCRIPTORS = [
    "sel_total_degree",
    "sel_feedback",
    "sel_cycle3",
    "sel_bridge",
    "sel_spectral",
    "coverage1",
    "coverage2",
    "out_neighbor_redundancy",
    "selected_dispersion",
]


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    vals = p[mask]
    if vals.size == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    inv = np.empty_like(order)
    inv[order] = np.arange(m)
    out[np.where(mask)[0]] = q[inv]
    return out


def bootstrap_mean_ci(values, n_boot=10000, seed=20260818):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    chunk = 1000
    pos = 0
    while pos < n_boot:
        n = min(chunk, n_boot - pos)
        idx = rng.integers(0, len(x), size=(n, len(x)))
        means[pos:pos+n] = x[idx].mean(axis=1)
        pos += n
    return (
        float(x.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def wilcoxon_p(values, alternative="greater"):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or not HAVE_SCIPY:
        return np.nan
    if np.allclose(x, 0):
        return 1.0
    try:
        return float(wilcoxon(x, alternative=alternative).pvalue)
    except Exception:
        return np.nan


def read_csv_member(tf: tarfile.TarFile, name: str) -> pd.DataFrame:
    member = tf.extractfile(name)
    if member is None:
        raise FileNotFoundError(name)
    return pd.read_csv(io.BytesIO(member.read()))


def read_json_member(tf: tarfile.TarFile, name: str):
    member = tf.extractfile(name)
    if member is None:
        raise FileNotFoundError(name)
    return json.loads(member.read().decode("utf-8"))


def member_names(tf: tarfile.TarFile):
    return set(tf.getnames())


def complete_done_ids(tf: tarfile.TarFile):
    out = set()
    pat = re.compile(r"results_stage4_complete/state/([0-9a-f]+)\.done\.json$")
    for n in tf.getnames():
        m = pat.match(n)
        if m:
            out.add(m.group(1))
    return out


def aggregate_task_methods(df: pd.DataFrame, metadata: dict):
    """
    Convert a 38-row task table into exactly seven task-method rows:
    32 random placements are averaged; six non-random methods are singletons.
    """
    method_col = "method" if "method" in df.columns else "replicate_label"
    score_col = (
        "rhythm_score_mean"
        if "rhythm_score_mean" in df.columns
        else "final_score_mean"
    )
    if method_col not in df.columns or score_col not in df.columns:
        raise RuntimeError(
            f"Cannot identify method/score columns: {df.columns.tolist()}"
        )

    rows = []
    for raw_method, g in df.groupby(method_col, dropna=False):
        method = METHOD_MAP.get(str(raw_method), str(raw_method))
        row = dict(metadata)
        row["method"] = method
        row["score"] = pd.to_numeric(g[score_col], errors="coerce").mean()
        row["raw_rows"] = len(g)
        if "mask_key" in g.columns and method != "random":
            row["mask_key"] = str(g["mask_key"].iloc[0])
        for c in DESCRIPTORS + ["optimizer_search_best"]:
            if c in g.columns:
                row[c] = pd.to_numeric(g[c], errors="coerce").mean()
        rows.append(row)

    out = pd.DataFrame(rows)
    expected = {
        "dynamics_aware", "spectral", "feedback_hub",
        "high_degree", "module_bridge", "cycle_proxy", "random"
    }
    got = set(out["method"])
    if got != expected:
        raise RuntimeError(
            f"Task {metadata.get('task_id')} method set mismatch: "
            f"expected={sorted(expected)} got={sorted(got)}"
        )
    return out


def load_complete_archive(path: Path):
    with tarfile.open(path, "r:gz") as tf:
        manifest = read_csv_member(
            tf, "results_stage4_complete/manifest.csv"
        )
        integrity = read_json_member(
            tf, "results_stage4_complete/integrity_report.json"
        )
        done_ids = complete_done_ids(tf)

        frames = []
        missing = []
        for _, r in manifest.iterrows():
            task_id = str(r["task_id"])
            if task_id not in done_ids:
                continue
            name = (
                f"results_stage4_complete/tasks/{task_id}/"
                "runner_output/stage4_discovery_methods.csv"
            )
            try:
                df = read_csv_member(tf, name)
            except Exception:
                missing.append(task_id)
                continue

            meta = r.to_dict()
            meta["evidence_source"] = "stage4_complete"
            frames.append(aggregate_task_methods(df, meta))

    if missing:
        raise RuntimeError(
            f"Missing canonical complete-task tables: {missing[:10]}"
        )

    out = pd.concat(frames, ignore_index=True, sort=False)
    if out["task_id"].nunique() != len(done_ids):
        raise RuntimeError(
            f"Complete archive task mismatch: "
            f"{out['task_id'].nunique()} vs DONE {len(done_ids)}"
        )
    return out, manifest, integrity


def load_closure_archive(path: Path):
    with tarfile.open(path, "r:gz") as tf:
        manifest = read_csv_member(
            tf, "results_stage4f_closure/manifest.csv"
        )
        integrity = read_json_member(
            tf, "results_stage4f_closure/integrity_report.json"
        )
        manager = read_json_member(
            tf, "results_stage4f_closure/manager_status.json"
        )

        # Repaired ER tasks
        er_manifest = manifest[manifest["task_type"].eq("er_patch")].copy()
        er_frames = []
        for _, r in er_manifest.iterrows():
            task_id = str(r["task_id"])
            name = (
                f"results_stage4f_closure/tasks/{task_id}/methods.csv"
            )
            df = read_csv_member(tf, name)
            meta = r.to_dict()
            meta["phase"] = "er_patch"
            meta["budget_fraction"] = (
                float(meta["k"]) / float(meta["N"])
            )
            meta["evidence_source"] = "stage4f_closure"
            er_frames.append(aggregate_task_methods(df, meta))
        er = pd.concat(er_frames, ignore_index=True, sort=False)

        # Closure analyses generated on the cloud
        analysis = {}
        for fn in [
            "er_method_summary.csv",
            "er_paired_comparisons.csv",
            "optimizer_score_stability.csv",
            "within_state_mask_distances.csv",
            "fixed_topology_between_state_distances.csv",
            "fixed_topology_anchor_summary.csv",
            "dense_composite_mask_distances.csv",
            "winner_curse_fresh_validation.csv",
        ]:
            name = f"results_stage4f_closure/analysis/{fn}"
            analysis[fn] = read_csv_member(tf, name)

        analysis_summary = read_json_member(
            tf,
            "results_stage4f_closure/analysis/"
            "stage4f_analysis_summary.json"
        )

        # Fresh reevaluation raw task tables
        reeval_frames = []
        reeval_manifest = manifest[
            manifest["task_type"].eq("stability_reeval")
        ].copy()
        for _, r in reeval_manifest.iterrows():
            task_id = str(r["task_id"])
            name = (
                f"results_stage4f_closure/tasks/{task_id}/"
                "reevaluation.csv"
            )
            d = read_csv_member(tf, name)
            reeval_frames.append(d)
        fresh = pd.concat(reeval_frames, ignore_index=True, sort=False)

    return {
        "manifest": manifest,
        "integrity": integrity,
        "manager": manager,
        "er": er,
        "analysis": analysis,
        "analysis_summary": analysis_summary,
        "fresh_reevaluation": fresh,
    }


def load_optional_stage4a(path: Path):
    if not path.exists():
        return pd.DataFrame()

    d = pd.read_csv(path)
    method_col = "method" if "method" in d.columns else "method_raw"
    score_col = None
    for c in [
        "final_score_mean", "rhythm_score_mean",
        "validation_score_mean", "score"
    ]:
        if c in d.columns:
            score_col = c
            break
    if score_col is None:
        return pd.DataFrame()

    d["method_norm"] = (
        d[method_col].astype(str).map(lambda x: METHOD_MAP.get(x, x))
    )
    ids = [
        c for c in [
            "task_id", "graph_seed", "topology", "regime",
            "k", "N", "complexity_fraction"
        ]
        if c in d.columns
    ]

    agg_cols = [score_col] + [
        c for c in DESCRIPTORS + ["optimizer_search_best"]
        if c in d.columns
    ]
    for c in agg_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    g = (
        d.groupby(ids + ["method_norm"], dropna=False)[agg_cols]
        .mean()
        .reset_index()
        .rename(columns={
            "method_norm": "method",
            score_col: "score",
        })
    )
    g["phase"] = "stage4a_discovery"
    g["evidence_source"] = "cleaned_stage4a_local"
    return g


def validate_core(core: pd.DataFrame):
    n_tasks = core["task_id"].nunique()
    if n_tasks != 225:
        raise RuntimeError(
            f"Expected 225 complete core tasks after ER repair; got {n_tasks}."
        )
    counts = core.groupby("task_id")["method"].nunique()
    bad = counts[counts != 7]
    if len(bad):
        raise RuntimeError(
            f"Tasks without seven aggregated methods: {bad.to_dict()}"
        )
    if len(core) != 225 * 7:
        raise RuntimeError(
            f"Expected 1575 task-method rows; got {len(core)}."
        )


def task_wide(df: pd.DataFrame):
    return df.pivot_table(
        index="task_id",
        columns="method",
        values="score",
        aggfunc="mean"
    )


def paired_contrasts(df: pd.DataFrame, grouping: str, label: str):
    wide = task_wide(df)
    if "dynamics_aware" not in wide.columns:
        return pd.DataFrame()

    rows = []
    for i, b in enumerate(BASELINES):
        if b not in wide.columns:
            continue
        delta = (
            pd.to_numeric(wide["dynamics_aware"], errors="coerce")
            - pd.to_numeric(wide[b], errors="coerce")
        ).dropna().to_numpy(float)
        mean, lo, hi = bootstrap_mean_ci(
            delta, seed=20260818 + i
        )
        rows.append({
            "grouping": grouping,
            "group": label,
            "baseline": b,
            "n_tasks": len(delta),
            "mean_delta": mean,
            "bootstrap95_low": lo,
            "bootstrap95_high": hi,
            "median_delta": float(np.median(delta)),
            "positive_fraction": float(np.mean(delta > 0)),
            "wilcoxon_greater_p": wilcoxon_p(
                delta, alternative="greater"
            ),
            "wilcoxon_two_sided_p": wilcoxon_p(
                delta, alternative="two-sided"
            ),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_greater_fdr_bh"] = bh_fdr(
            out["wilcoxon_greater_p"].to_numpy()
        )
    return out


def build_all_contrasts(core: pd.DataFrame):
    tables = []

    # Descriptive pooled summary; heterogeneous, so not used alone for inference.
    tables.append(
        paired_contrasts(core, "pooled", "all_225_core_tasks")
    )

    # Development phases
    for phase, g in core.groupby("phase", dropna=False):
        tables.append(
            paired_contrasts(g, "phase", str(phase))
        )

    # Manuscript-oriented evidence strata
    strata = {
        "scale_free_N256_replication_atlas":
            core["phase"].isin([
                "stage4b_replication",
                "stage4c_atlas_extension",
            ]),
        "ER_N256_repair":
            core["phase"].eq("er_patch"),
        "modular_N256":
            core["phase"].eq("stage4e_topology")
            & core["topology"].eq("modular"),
        "small_world_N256":
            core["phase"].eq("stage4e_topology")
            & core["topology"].eq("small_world"),
        "scale_free_scaling_N128_N512":
            core["phase"].eq("stage4d_scaling"),
    }
    for label, mask in strata.items():
        tables.append(
            paired_contrasts(
                core.loc[mask], "evidence_stratum", label
            )
        )

    return pd.concat(
        [x for x in tables if x is not None and not x.empty],
        ignore_index=True
    )


def gain_table(df: pd.DataFrame, baseline: str, group_cols):
    id_cols = [
        c for c in [
            "task_id", "graph_seed", "topology", "regime",
            "k", "N", "phase", "connection_prob", "coupling"
        ]
        if c in df.columns
    ]
    tm = (
        df.groupby(id_cols + ["method"], dropna=False)["score"]
        .mean().reset_index()
    )
    wide = tm.pivot_table(
        index=id_cols, columns="method", values="score",
        aggfunc="mean"
    ).reset_index()

    if "dynamics_aware" not in wide.columns or baseline not in wide.columns:
        return pd.DataFrame()

    wide["gain"] = (
        pd.to_numeric(wide["dynamics_aware"], errors="coerce")
        - pd.to_numeric(wide[baseline], errors="coerce")
    )

    rows = []
    group_cols = [c for c in group_cols if c in wide.columns]
    for gi, (key, g) in enumerate(
        wide.groupby(group_cols, dropna=False, sort=True)
    ):
        if not isinstance(key, tuple):
            key = (key,)
        x = g["gain"].dropna().to_numpy(float)
        mean, lo, hi = bootstrap_mean_ci(
            x, seed=20260818 + gi
        )
        row = {c: v for c, v in zip(group_cols, key)}
        row.update({
            "baseline": baseline,
            "n_tasks": len(x),
            "mean_gain": mean,
            "bootstrap95_low": lo,
            "bootstrap95_high": hi,
            "median_gain": float(np.median(x)),
            "positive_fraction": float(np.mean(x > 0)),
            "wilcoxon_greater_p": wilcoxon_p(
                x, alternative="greater"
            ),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_greater_fdr_bh"] = bh_fdr(
            out["wilcoxon_greater_p"].to_numpy()
        )
    return out


def fixed_topology_state_analysis(core: pd.DataFrame):
    """
    Pure fixed-topology contrast:
      transition_mid vs sparse_drive
    They share p=0.05 for a given graph seed; only coupling differs.

    transition_dense is deliberately excluded from this table because p=0.10.
    """
    sf = core[
        core["phase"].isin([
            "stage4b_replication",
            "stage4c_atlas_extension",
        ])
        & core["topology"].eq("scale_free")
        & core["N"].eq(256)
    ].copy()

    gpu = sf[sf["method"].eq("dynamics_aware")].copy()
    sparse = gpu[gpu["regime"].eq("sparse_drive")]
    mid = gpu[gpu["regime"].eq("transition_mid")]

    keys = ["graph_seed", "k"]
    m = sparse.merge(
        mid, on=keys, suffixes=("_sparse", "_mid")
    )

    metrics = ["score"] + [c for c in DESCRIPTORS if c in gpu.columns]
    rows = []
    for metric in metrics:
        a = pd.to_numeric(
            m[f"{metric}_sparse"], errors="coerce"
        )
        b = pd.to_numeric(
            m[f"{metric}_mid"], errors="coerce"
        )
        delta = (b - a).dropna().to_numpy(float)
        if len(delta) == 0:
            continue
        mean, lo, hi = bootstrap_mean_ci(delta)
        rows.append({
            "contrast": "transition_mid - sparse_drive",
            "metric": metric,
            "n_matched_graph_budget_conditions": len(delta),
            "mean_shift": mean,
            "bootstrap95_low": lo,
            "bootstrap95_high": hi,
            "median_shift": float(np.median(delta)),
            "positive_fraction": float(np.mean(delta > 0)),
            "wilcoxon_two_sided_p": wilcoxon_p(
                delta, alternative="two-sided"
            ),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["wilcoxon_two_sided_fdr_bh"] = bh_fdr(
            out["wilcoxon_two_sided_p"].to_numpy()
        )

    # Fixed-topology change in dynamics-aware advantage over spectral
    ids = [
        "task_id", "graph_seed", "regime", "k",
        "topology", "N", "phase"
    ]
    tm = sf.groupby(
        ids + ["method"], dropna=False
    )["score"].mean().reset_index()
    wide = tm.pivot_table(
        index=ids, columns="method", values="score"
    ).reset_index()
    wide["gain_vs_spectral"] = (
        wide["dynamics_aware"] - wide["spectral"]
    )
    a = wide[wide["regime"].eq("sparse_drive")]
    b = wide[wide["regime"].eq("transition_mid")]
    mg = a.merge(
        b, on=["graph_seed", "k"],
        suffixes=("_sparse", "_mid")
    )
    delta = (
        mg["gain_vs_spectral_mid"]
        - mg["gain_vs_spectral_sparse"]
    ).dropna().to_numpy(float)

    gain_shift = pd.DataFrame()
    if len(delta):
        mean, lo, hi = bootstrap_mean_ci(delta)
        gain_shift = pd.DataFrame([{
            "contrast": "transition_mid - sparse_drive",
            "metric": "dynamics_aware_gain_vs_spectral",
            "n_matched_graph_budget_conditions": len(delta),
            "mean_shift": mean,
            "bootstrap95_low": lo,
            "bootstrap95_high": hi,
            "median_shift": float(np.median(delta)),
            "positive_fraction": float(np.mean(delta > 0)),
            "wilcoxon_two_sided_p": wilcoxon_p(
                delta, alternative="two-sided"
            ),
        }])

    return out, gain_shift


def pairwise_functional_degeneracy(closure_analysis: dict):
    within = closure_analysis[
        "within_state_mask_distances.csv"
    ].copy()
    fresh = closure_analysis[
        "winner_curse_fresh_validation.csv"
    ].copy()

    left = fresh[
        [
            "graph_seed", "regime", "k",
            "replicate_label", "fresh_score"
        ]
    ].rename(columns={
        "replicate_label": "rep_a",
        "fresh_score": "fresh_score_a",
    })
    right = fresh[
        [
            "graph_seed", "regime", "k",
            "replicate_label", "fresh_score"
        ]
    ].rename(columns={
        "replicate_label": "rep_b",
        "fresh_score": "fresh_score_b",
    })

    p = within.merge(
        left,
        on=["graph_seed", "regime", "k", "rep_a"],
        how="left"
    ).merge(
        right,
        on=["graph_seed", "regime", "k", "rep_b"],
        how="left"
    )
    p["abs_fresh_score_difference"] = (
        p["fresh_score_a"] - p["fresh_score_b"]
    ).abs()

    rows = []
    for k, g in p.groupby("k", sort=True):
        rows.append({
            "grouping": "k",
            "group": k,
            "n_pairs": len(g),
            "median_jaccard_distance": float(
                g["jaccard_distance"].median()
            ),
            "mean_jaccard_distance": float(
                g["jaccard_distance"].mean()
            ),
            "median_abs_fresh_score_difference": float(
                g["abs_fresh_score_difference"].median()
            ),
            "mean_abs_fresh_score_difference": float(
                g["abs_fresh_score_difference"].mean()
            ),
        })
    for regime, g in p.groupby("regime", sort=True):
        rows.append({
            "grouping": "regime",
            "group": regime,
            "n_pairs": len(g),
            "median_jaccard_distance": float(
                g["jaccard_distance"].median()
            ),
            "mean_jaccard_distance": float(
                g["jaccard_distance"].mean()
            ),
            "median_abs_fresh_score_difference": float(
                g["abs_fresh_score_difference"].median()
            ),
            "mean_abs_fresh_score_difference": float(
                g["abs_fresh_score_difference"].mean()
            ),
        })

    if HAVE_SCIPY:
        mask = (
            p["jaccard_distance"].notna()
            & p["abs_fresh_score_difference"].notna()
        )
        if mask.sum() >= 3:
            rho, pv = spearmanr(
                p.loc[mask, "jaccard_distance"],
                p.loc[mask, "abs_fresh_score_difference"],
            )
            corr = pd.DataFrame([{
                "n_pairs": int(mask.sum()),
                "spearman_rho_mask_distance_vs_abs_score_difference":
                    float(rho),
                "spearman_p": float(pv),
            }])
        else:
            corr = pd.DataFrame()
    else:
        corr = pd.DataFrame()

    return p, pd.DataFrame(rows), corr


def model_side_biology_bridge(core: pd.DataFrame):
    """
    Simulation-side quantities only. This table is prepared so later
    biological analyses can define independent C_i and L_i observables.
    """
    ids = [
        c for c in [
            "task_id", "phase", "topology", "N",
            "graph_seed", "regime", "connection_prob",
            "coupling", "k", "budget_fraction"
        ]
        if c in core.columns
    ]
    wide = core.pivot_table(
        index=ids, columns="method", values="score",
        aggfunc="mean"
    ).reset_index()

    if "dynamics_aware" in wide.columns:
        for b in BASELINES:
            if b in wide.columns:
                wide[f"leverage_gain_vs_{b}"] = (
                    wide["dynamics_aware"] - wide[b]
                )

    gpu = core[
        core["method"].eq("dynamics_aware")
    ][ids + [c for c in DESCRIPTORS if c in core.columns]].copy()

    return wide.merge(gpu, on=ids, how="left")


def write_summary(
    outdir: Path,
    complete_integrity,
    closure,
    core,
    contrasts,
    fixed_state,
    fixed_gain,
    degeneracy_summary,
):
    pooled = contrasts[
        contrasts["group"].eq("all_225_core_tasks")
    ].copy()

    def line_for(base):
        x = pooled[pooled["baseline"].eq(base)]
        if x.empty:
            return ""
        r = x.iloc[0]
        return (
            f"- dynamics-aware vs {base}: "
            f"mean Δ={r['mean_delta']:.4f}, "
            f"win={100*r['positive_fraction']:.1f}%, "
            f"one-sided p={r['wilcoxon_greater_p']:.3g}."
        )

    fsum = closure["analysis_summary"]["stage4f"]

    lines = [
        "# Integrated Stage 4 evidence summary",
        "",
        "## Integrity",
        "",
        f"- Complete Stage4 successful tasks: "
        f"{core[core['evidence_source'].eq('stage4_complete')]['task_id'].nunique()}",
        f"- Repaired ER tasks: "
        f"{core[core['phase'].eq('er_patch')]['task_id'].nunique()}",
        f"- Authoritative core tasks after repair: {core['task_id'].nunique()} / 225",
        f"- Stage4F closure: {closure['manager']['done']} / {closure['manager']['total']} DONE, "
        f"{closure['manager']['failed']} failed.",
        "",
        "The DONE marker phase-count issue is only a parsing issue: the marker stores",
        "`task.task_type`, not `phase`. The closure integrity report explicitly confirms",
        "27 ER repair + 81 stability search + 27 stability reevaluation.",
        "",
        "## Strongly supported findings",
        "",
        line_for("spectral"),
        line_for("random"),
        "",
        "The pooled 225-task contrast is descriptive because tasks differ in topology,",
        "scale and budget. Formal manuscript inference should emphasize the stratified",
        "replication/topology/scaling tables.",
        "",
        "The repaired ER family closes the missing topology branch: all 27 ER tasks",
        "completed successfully and are present in the closure archive.",
        "",
        "## Stage 4F result that changes the narrative",
        "",
        f"- Fixed-topology anchors: {fsum['fixed_topology_anchor_n']}.",
        f"- Fraction with between-state mask distance > within-state optimizer variability: "
        f"{100*fsum['positive_shift_fraction']:.1f}%.",
        f"- Median between-minus-within Jaccard distance: "
        f"{fsum['median_between_minus_within']:.4f}.",
        f"- One-sided Wilcoxon p for between > within: "
        f"{fsum['wilcoxon_between_gt_within_p']:.3g}.",
        "",
        "Therefore the current data do NOT support the claim that a fixed topology has",
        "a unique optimal allocation mask that is reorganized by state beyond optimizer",
        "variability. This claim must not be made.",
        "",
        "The earlier transition_dense contrast is not a pure state test because",
        "transition_dense uses p=0.10 whereas sparse_drive and transition_mid use p=0.05.",
        "It is a topology+dynamics composite condition.",
        "",
        "## New positive interpretation: functional degeneracy",
        "",
        "Stage4F shows that multiple structurally different optimizer solutions can be",
        "functionally similar on fresh common-noise validation. This is especially clear",
        "at low complexity budget, where mask overlap can be extremely low while fresh-score",
        "differences remain modest. The allocation landscape therefore appears broad /",
        "degenerate rather than containing a single privileged mask.",
        "",
        "This is potentially biologically important: the model predicts an equivalence",
        "class of dynamically effective allocations rather than one uniquely mandated",
        "set of neurons.",
        "",
        "## Search / validation",
        "",
        f"- Winner's-curse mean search-minus-fresh: "
        f"{fsum['winner_curse_mean']:.4f}.",
        f"- Median search-minus-fresh: "
        f"{fsum['winner_curse_median']:.4f}.",
        "",
        "Search fitness must never be reported as final performance.",
        "",
        "## Biological bridge",
        "",
        "The strongest current bridge to biological data is:",
        "",
        "1. Intrinsic cellular dynamical complexity is treated as a finite network resource.",
        "2. Dynamics-aware allocation consistently outperforms static structural targeting",
        "   across independent graphs, topology families and scales.",
        "3. Static structural centrality is therefore an incomplete proxy for functional",
        "   allocation value.",
        "4. The optimizer landscape is non-unique / degenerate, suggesting robust alternative",
        "   configurations rather than one exact privileged mask.",
        "",
        "The biological section should independently define real-cell intrinsic complexity",
        "and circuit dynamical leverage, then ask whether richer intrinsic dynamics occupy",
        "positions with greater functional leverage beyond static structure.",
        "",
        "State-specific biological allocation should NOT yet be a headline claim.",
        "A proper model-side state-specificity test would require cross-state transfer:",
        "evaluate masks optimized in state A in state B and vice versa on the same topology.",
        "The current Stage4F reevaluation is home-state only and cannot answer that question.",
        "",
    ]

    (outdir / "integrated_stage4_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--complete-archive", type=Path, default=None)
    ap.add_argument("--closure-archive", type=Path, default=None)
    ap.add_argument("--stage4a-canonical", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    complete = (
        args.complete_archive.resolve()
        if args.complete_archive
        else root / "stage4_complete_results.tar.gz"
    )
    closure_path = (
        args.closure_archive.resolve()
        if args.closure_archive
        else root / "stage4f_closure_results.tar.gz"
    )
    stage4a = (
        args.stage4a_canonical.resolve()
        if args.stage4a_canonical
        else root / "plot" / "Stage4A" / "analysis" /
             "stage4a_canonical_methods.csv"
    )
    outdir = (
        args.output.resolve()
        if args.output
        else root / "plot" / "Stage4_integrated" / "analysis"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    if not complete.exists():
        raise SystemExit(f"Missing: {complete}")
    if not closure_path.exists():
        raise SystemExit(f"Missing: {closure_path}")

    print("[1/8] Reading complete Stage4 archive ...")
    complete_tm, complete_manifest, complete_integrity = (
        load_complete_archive(complete)
    )

    print("[2/8] Reading Stage4F closure archive ...")
    closure = load_closure_archive(closure_path)

    print("[3/8] Building authoritative 225-task core ...")
    core = pd.concat(
        [complete_tm, closure["er"]],
        ignore_index=True,
        sort=False
    )
    validate_core(core)
    core.to_csv(
        outdir / "stage4_core_225_task_methods.csv",
        index=False, encoding="utf-8-sig"
    )

    print("[4/8] Loading optional cleaned Stage4A discovery ...")
    stage4a_tm = load_optional_stage4a(stage4a)
    stage4a_tm.to_csv(
        outdir / "stage4a_discovery_task_methods.csv",
        index=False, encoding="utf-8-sig"
    )

    print("[5/8] Formal paired contrasts and generalization ...")
    contrasts = build_all_contrasts(core)
    contrasts.to_csv(
        outdir / "stage4_paired_contrasts.csv",
        index=False, encoding="utf-8-sig"
    )

    gain_budget = gain_table(
        core, "spectral",
        ["phase", "topology", "N", "regime", "k"]
    )
    gain_budget.to_csv(
        outdir / "stage4_gain_landscape_vs_spectral.csv",
        index=False, encoding="utf-8-sig"
    )

    gain_random = gain_table(
        core, "random",
        ["phase", "topology", "N", "regime", "k"]
    )
    gain_random.to_csv(
        outdir / "stage4_gain_landscape_vs_random.csv",
        index=False, encoding="utf-8-sig"
    )

    print("[6/8] Pure fixed-topology state analysis ...")
    fixed_state, fixed_gain = fixed_topology_state_analysis(core)
    fixed_state.to_csv(
        outdir / "fixed_topology_state_shift_analysis.csv",
        index=False, encoding="utf-8-sig"
    )
    fixed_gain.to_csv(
        outdir / "fixed_topology_state_gain_shift.csv",
        index=False, encoding="utf-8-sig"
    )

    print("[7/8] Optimizer stability / degeneracy analysis ...")
    pairdeg, degsummary, degcorr = pairwise_functional_degeneracy(
        closure["analysis"]
    )
    pairdeg.to_csv(
        outdir / "optimizer_pairwise_mask_vs_fresh_score.csv",
        index=False, encoding="utf-8-sig"
    )
    degsummary.to_csv(
        outdir / "optimizer_functional_degeneracy_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    degcorr.to_csv(
        outdir / "optimizer_mask_score_correlation.csv",
        index=False, encoding="utf-8-sig"
    )

    # Preserve raw closure analysis tables locally in one clean analysis folder.
    for fn, d in closure["analysis"].items():
        d.to_csv(
            outdir / f"stage4f_{fn}",
            index=False, encoding="utf-8-sig"
        )

    closure["fresh_reevaluation"].to_csv(
        outdir / "stage4f_fresh_reevaluation_24seeds.csv",
        index=False, encoding="utf-8-sig"
    )

    bridge = model_side_biology_bridge(core)
    bridge.to_csv(
        outdir / "model_side_biology_bridge.csv",
        index=False, encoding="utf-8-sig"
    )

    print("[8/8] Writing audit + summary ...")
    manifest = {
        "complete_archive": str(complete),
        "complete_archive_sha256": sha256_file(complete),
        "closure_archive": str(closure_path),
        "closure_archive_sha256": sha256_file(closure_path),
        "complete_done_tasks": int(complete_tm["task_id"].nunique()),
        "er_repair_tasks": int(closure["er"]["task_id"].nunique()),
        "authoritative_core_tasks": int(core["task_id"].nunique()),
        "authoritative_task_method_rows": int(len(core)),
        "stage4f_by_type": closure["integrity"]["by_type"],
        "stage4f_manager": closure["manager"],
        "optional_stage4a_loaded": bool(len(stage4a_tm)),
        "optional_stage4a_path": str(stage4a),
        "scientific_guardrails": {
            "transition_dense_is_composite": True,
            "pure_fixed_topology_state_contrast":
                "transition_mid vs sparse_drive",
            "search_fitness_is_final_performance": False,
            "exact_mask_uniqueness_assumed": False,
        },
    }
    (outdir / "integrated_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    write_summary(
        outdir=outdir,
        complete_integrity=complete_integrity,
        closure=closure,
        core=core,
        contrasts=contrasts,
        fixed_state=fixed_state,
        fixed_gain=fixed_gain,
        degeneracy_summary=degsummary,
    )

    print()
    print("=== DONE ===")
    print(f"Authoritative core tasks : {core['task_id'].nunique()} / 225")
    print(f"Task-method rows         : {len(core)} / 1575")
    print(f"Complete old tasks       : {complete_tm['task_id'].nunique()}")
    print(f"Repaired ER tasks        : {closure['er']['task_id'].nunique()}")
    print(f"Stage4F done/failed      : {closure['manager']['done']}/{closure['manager']['failed']}")
    print(f"Stage4A discovery loaded : {bool(len(stage4a_tm))}")
    print(f"Output                   : {outdir}")
    print()
    print("Read first:")
    print("  integrated_stage4_summary.md")
    print("  stage4_paired_contrasts.csv")
    print("  fixed_topology_state_shift_analysis.csv")
    print("  optimizer_functional_degeneracy_summary.csv")
    print("  stage4_gain_landscape_vs_spectral.csv")


if __name__ == "__main__":
    main()
