#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complexity–Leverage closure analysis atlas v1

Purpose
-------
Re-analyse and render a publication-style figure atlas for:
1) Stage 4 cross-state transfer + functional degeneracy closure;
2) Allen Visual Behavior lag-aware latent-dynamics refit.

This script intentionally keeps these closure results separate from the existing
Stage 1–5 integrated manuscript atlas. It produces many candidate figures first;
main/SI selection can be done later after the remaining biological datasets arrive.

Default project layout
----------------------
D:\\Research\\Neural Science
  bio data\\Allen_lag_aware_refit_20260820.tar.gz
  bio data\\ComplexityLeverage_ClosureCompute_v2_Turbo_results_20260820_154030.tar.gz
  plot\\analyze_closure_extensions_v1.py

Outputs
-------
plot\\ClosureExtensions_v1\\
  analysis\\
  source_data\\
  figures_stage4_transfer\\
  figures_stage4_degeneracy\\
  figures_allen_lagaware\\
  README.md

Dependencies: numpy, pandas, scipy, matplotlib
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import sys
import tarfile
import tempfile
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

VERSION = "1.0.0"
RNG_SEED = 20260820

# Restrained palette kept compatible with the previous project atlas.
C_BLUE = "#355F7F"
C_TEAL = "#2A9D8F"
C_ORANGE = "#C27628"
C_PURPLE = "#8E6CA5"
C_ROSE = "#B75C70"
C_GREY = "#7A8288"
C_LIGHT = "#BEC7D1"
C_DARK = "#26333F"
C_GRID = "#D9DDE1"
STATE_COLORS = {
    "sparse_drive": C_BLUE,
    "transition_mid": C_TEAL,
    "transition_dense": C_ORANGE,
    "familiar_active": C_BLUE,
    "novel_active": C_TEAL,
    "passive": C_ORANGE,
}


def parse_args():
    ap = argparse.ArgumentParser(description="Build Stage4 closure + Allen lag-aware analysis atlas.")
    ap.add_argument("--root", default=str(Path.cwd()))
    ap.add_argument("--allen-archive", default=None)
    ap.add_argument("--turbo-archive", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--force-extract", action="store_true")
    return ap.parse_args()


def set_style(root: Path):
    # If the project's custom style module exists, allow it to set rcParams.
    plot_dir = root / "plot"
    if plot_dir.exists() and str(plot_dir) not in sys.path:
        sys.path.insert(0, str(plot_dir))
    try:
        import plot_style_complexity as psc  # type: ignore
        for fn in ("apply_style", "set_style", "use_style"):
            if hasattr(psc, fn):
                try:
                    getattr(psc, fn)()
                    break
                except Exception:
                    pass
    except Exception:
        pass

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def clean_axes(ax, grid_axis=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=C_GRID, linewidth=0.65, alpha=0.6)
        ax.set_axisbelow(True)


def safe_extract_tar(archive: Path, dest: Path, force=False) -> Path:
    marker = dest / ".extracted_ok"
    if force and dest.exists():
        shutil.rmtree(dest)
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        base = dest.resolve()
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if os.path.commonpath([str(base), str(target)]) != str(base):
                raise RuntimeError(f"Unsafe archive path: {member.name}")
        tf.extractall(dest)
    marker.write_text("ok\n", encoding="utf-8")
    return dest


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    # Prefer analysis files over checkpoints if duplicated.
    hits.sort(key=lambda p: ("/analysis/" not in p.as_posix(), len(p.as_posix())))
    return hits[0]


def read_csv(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(find_one(root, name))


def read_json(root: Path, name: str) -> dict:
    return json.loads(find_one(root, name).read_text(encoding="utf-8"))


def bootstrap_ci(x, stat=np.mean, n=20000, rng=None):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    vals = np.empty(n, dtype=float)
    for i in range(n):
        vals[i] = stat(rng.choice(x, size=len(x), replace=True))
    return tuple(np.quantile(vals, [0.025, 0.975]))


def exact_signflip_p_greater(x):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n == 0:
        return np.nan
    obs = x.mean()
    # Exact enumeration is safe for n=9 here; fallback Monte Carlo if needed.
    if n <= 20:
        ge = 0
        total = 2 ** n
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            if np.mean(x * np.asarray(signs)) >= obs - 1e-15:
                ge += 1
        return ge / total
    rng = np.random.default_rng(RNG_SEED)
    m = 200000
    sims = np.mean(x[None, :] * rng.choice([-1.0, 1.0], size=(m, n)), axis=1)
    return (1 + np.sum(sims >= obs)) / (m + 1)


def safe_wilcoxon(x, y=None, alternative="two-sided"):
    try:
        r = stats.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return np.nan, np.nan


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    if len(pv) == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[np.where(ok)[0]] = tmp
    return out


def write_source(df: pd.DataFrame, source_dir: Path, fig_id: str) -> Path:
    p = source_dir / f"{fig_id}_source.csv"
    df.to_csv(p, index=False)
    return p


def save_fig(fig, fig_id: str, group_dir: Path, dpi: int):
    png = group_dir / f"{fig_id}.png"
    pdf = group_dir / f"{fig_id}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png, pdf


def add_catalog(catalog, fig_id, group, png, source, message, tier):
    catalog.append({
        "figure_id": fig_id,
        "group": group,
        "png": str(png),
        "source_data": str(source) if source else "",
        "scientific_message": message,
        "initial_triage": tier,
    })


def boxstrip(ax, groups, labels, colors, ylabel, zero=False):
    vals = [np.asarray(pd.Series(g).dropna(), dtype=float) for g in groups]
    bp = ax.boxplot(vals, positions=np.arange(1, len(vals)+1), widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color=C_DARK, linewidth=1.4),
                    whiskerprops=dict(color="black", linewidth=1),
                    capprops=dict(color="black", linewidth=1))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.78); patch.set_edgecolor("none")
    rng = np.random.default_rng(RNG_SEED)
    for i, (v, c) in enumerate(zip(vals, colors), start=1):
        jitter = rng.normal(0, 0.055, size=len(v))
        ax.scatter(np.full(len(v), i)+jitter, v, s=16, color=C_GREY, alpha=0.48,
                   edgecolors="none", zorder=3)
    ax.set_xticks(np.arange(1, len(labels)+1)); ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    if zero:
        ax.axhline(0, color=C_GREY, linestyle="--", linewidth=1)
    clean_axes(ax, "y")


def heatmap(ax, mat, xlabels, ylabels, cmap="RdBu_r", vmin=None, vmax=None, fmt=".2f", cbar_label=""):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=9,
                        color="white" if (vmax is not None and vmin is not None and abs(val) > 0.55*max(abs(vmin),abs(vmax))) else C_DARK)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label)
    return im


def main():
    args = parse_args()
    root = Path(args.root)
    allen_archive = Path(args.allen_archive) if args.allen_archive else root / "bio data" / "Allen_lag_aware_refit_20260820.tar.gz"
    turbo_archive = Path(args.turbo_archive) if args.turbo_archive else root / "bio data" / "ComplexityLeverage_ClosureCompute_v2_Turbo_results_20260820_154030.tar.gz"
    out = Path(args.out) if args.out else root / "plot" / "ClosureExtensions_v1"

    if not allen_archive.exists():
        raise FileNotFoundError(f"Allen archive not found: {allen_archive}")
    if not turbo_archive.exists():
        raise FileNotFoundError(f"Turbo archive not found: {turbo_archive}")

    set_style(root)
    analysis_out = out / "analysis"
    source_out = out / "source_data"
    transfer_out = out / "figures_stage4_transfer"
    degeneracy_out = out / "figures_stage4_degeneracy"
    allen_out = out / "figures_allen_lagaware"
    cache_out = out / "_extracted"
    for d in [analysis_out, source_out, transfer_out, degeneracy_out, allen_out, cache_out]:
        d.mkdir(parents=True, exist_ok=True)

    allen_ex = safe_extract_tar(allen_archive, cache_out / "allen", args.force_extract)
    turbo_ex = safe_extract_tar(turbo_archive, cache_out / "turbo", args.force_extract)

    # ---------- Load Stage4 closure ----------
    cross_key = read_json(turbo_ex, "cross_state_key_metrics.json")
    cross_anchor = read_csv(turbo_ex, "pure_state_crossover_by_anchor.csv")
    cross_rep = read_csv(turbo_ex, "pure_state_crossover_by_replicate.csv")
    transfer = read_csv(turbo_ex, "transfer_centered_by_anchor.csv")
    transfer_mean = read_csv(turbo_ex, "transfer_matrix_mean.csv")
    composite = read_csv(turbo_ex, "composite_dense_interactions.csv")

    dg_key = read_json(turbo_ex, "degeneracy_key_metrics.json")
    dg = read_csv(turbo_ex, "degeneracy_metrics_by_condition.csv")
    dg_ind = read_csv(turbo_ex, "degeneracy_metrics_independent_home.csv")
    pair_geom = read_csv(turbo_ex, "pairwise_mask_score_geometry.csv")
    pair_ind = read_csv(turbo_ex, "pairwise_geometry_independent_home.csv")
    dg_budget = read_csv(turbo_ex, "degeneracy_primary_by_budget.csv")
    dg_rep = read_json(turbo_ex, "independent_replication_summary.json")

    # ---------- Load Allen closure ----------
    allen_key = read_json(allen_ex, "lagaware_key_metrics.json")
    diag = read_csv(allen_ex, "allen_lagaware_experiment_diagnostics.csv")
    oldnew = read_csv(allen_ex, "old_vs_lagaware_r2.csv")
    raw = read_csv(allen_ex, "state_pairs_raw_by_container.csv")
    resid = read_csv(allen_ex, "state_pairs_residual_by_container.csv")
    raw_sum = read_csv(allen_ex, "state_pairs_raw_summary.csv")
    resid_sum = read_csv(allen_ex, "state_pairs_residual_summary.csv")
    qgate = read_csv(allen_ex, "quality_gated_state_pairs_summary.csv")
    long = read_csv(allen_ex, "allen_lagaware_leverage_long.csv")

    rng = np.random.default_rng(RNG_SEED)
    catalog = []

    # ================================================================
    # STAGE 4 CROSS-STATE TRANSFER
    # ================================================================
    states = ["sparse_drive", "transition_mid", "transition_dense"]
    pretty_state = {"sparse_drive":"Sparse", "transition_mid":"Transition-mid", "transition_dense":"Transition-dense"}

    # CT01 mean centered transfer matrix
    piv = transfer_mean.pivot(index="target_state", columns="source_state", values="delta_vs_target_home").reindex(index=states, columns=states)
    fig, ax = plt.subplots(figsize=(5.2,4.3))
    heatmap(ax, piv.values, [pretty_state[s] for s in states], [pretty_state[s] for s in states],
            cmap="RdBu_r", vmin=-0.14, vmax=0.14, fmt=".3f", cbar_label="Score difference vs target-state home mask")
    ax.set_xlabel("Source state of optimized allocation"); ax.set_ylabel("Evaluation state")
    source=write_source(transfer_mean,source_out,"CT01_transfer_matrix_centered")
    png,_=save_fig(fig,"CT01_transfer_matrix_centered",transfer_out,args.dpi)
    add_catalog(catalog,"CT01_transfer_matrix_centered","stage4_transfer",png,source,
                "Every off-diagonal transfer is worse than the target state's own optimized allocation.","MAIN_CANDIDATE")

    # CT02 crossover by budget, anchor points
    fig, ax = plt.subplots(figsize=(5.4,4.0))
    ks=sorted(cross_anchor.k.unique())
    for seed,g in cross_anchor.groupby("graph_seed"):
        g=g.sort_values("k")
        ax.plot(g.k,g.crossover_interaction,color=C_LIGHT,linewidth=1.0,alpha=.9)
        ax.scatter(g.k,g.crossover_interaction,color=C_GREY,s=25,alpha=.75)
    means=cross_anchor.groupby("k").crossover_interaction.mean().reindex(ks)
    ax.plot(ks,means.values,color=C_ROSE,linewidth=2.0,marker="o",markersize=6,zorder=4)
    ax.axhline(0,color=C_GREY,ls="--",lw=1)
    ax.set_xticks(ks); ax.set_xlabel("Complexity budget k"); ax.set_ylabel("Pure-state crossover interaction")
    clean_axes(ax,"y")
    source=write_source(cross_anchor,source_out,"CT02_crossover_by_budget")
    png,_=save_fig(fig,"CT02_crossover_by_budget",transfer_out,args.dpi)
    add_catalog(catalog,"CT02_crossover_by_budget","stage4_transfer",png,source,
                "Pure state-specific allocation advantage remains positive across graph seeds and budgets.","MAIN_CANDIDATE")

    # CT03 mid and sparse home advantage paired per anchor
    fig, ax = plt.subplots(figsize=(4.8,4.2))
    for _,r in cross_anchor.iterrows():
        ax.plot([0,1],[r.mid_home_advantage,r.sparse_home_advantage],color=C_LIGHT,lw=1,alpha=.8)
    ax.scatter(np.zeros(len(cross_anchor)),cross_anchor.mid_home_advantage,color=C_TEAL,s=34,zorder=3)
    ax.scatter(np.ones(len(cross_anchor)),cross_anchor.sparse_home_advantage,color=C_BLUE,s=34,zorder=3)
    ax.axhline(0,color=C_GREY,ls="--",lw=1)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Transition-mid home advantage","Sparse home advantage"],rotation=18,ha="right")
    ax.set_ylabel("Home-state score advantage"); clean_axes(ax,"y")
    source=write_source(cross_anchor,source_out,"CT03_home_advantage_paired")
    png,_=save_fig(fig,"CT03_home_advantage_paired",transfer_out,args.dpi)
    add_catalog(catalog,"CT03_home_advantage_paired","stage4_transfer",png,source,
                "Both sides of the matched pure-state contrast generally favour the home allocation.","MAIN_CANDIDATE")

    # CT04 distribution of all replicate crossover interactions by budget
    fig, ax = plt.subplots(figsize=(5.0,4.1))
    groups=[cross_rep.loc[cross_rep.k==k,"crossover_interaction"] for k in ks]
    boxstrip(ax,groups,[f"k={k}" for k in ks],[C_BLUE,C_TEAL,C_ORANGE],"Replicate-level crossover interaction",zero=True)
    source=write_source(cross_rep,source_out,"CT04_replicate_crossover_distribution")
    png,_=save_fig(fig,"CT04_replicate_crossover_distribution",transfer_out,args.dpi)
    add_catalog(catalog,"CT04_replicate_crossover_distribution","stage4_transfer",png,source,
                "Crossover is visible across independent optimizer replicates, not only after anchor averaging.","SI_STRONG")

    # CT05 off-diagonal transfer penalty distribution by source-target pair
    off=transfer[transfer.source_state!=transfer.target_state].copy()
    off["pair"]=[f"{pretty_state[a]} → {pretty_state[b]}" for a,b in zip(off.source_state,off.target_state)]
    pair_order=[f"{pretty_state[a]} → {pretty_state[b]}" for a in states for b in states if a!=b]
    fig, ax=plt.subplots(figsize=(6.2,4.2))
    data=[off.loc[off.pair==p,"delta_vs_target_home"] for p in pair_order]
    boxstrip(ax,data,pair_order,[STATE_COLORS[states[i%3]] for i in range(6)],"Transferred score − target-state home score",zero=True)
    source=write_source(off,source_out,"CT05_offdiagonal_transfer_penalties")
    png,_=save_fig(fig,"CT05_offdiagonal_transfer_penalties",transfer_out,args.dpi)
    add_catalog(catalog,"CT05_offdiagonal_transfer_penalties","stage4_transfer",png,source,
                "All directional cross-state transfers incur a penalty on average.","SI_STRONG")

    # CT06 graph-seed consistency of crossover
    fig, ax=plt.subplots(figsize=(5.2,4.0))
    seedtab=cross_anchor.pivot(index="k",columns="graph_seed",values="crossover_interaction")
    for seed in seedtab.columns:
        ax.plot(seedtab.index,seedtab[seed],marker="o",lw=1.4,label=str(seed))
    ax.axhline(0,color=C_GREY,ls="--",lw=1); ax.set_xticks(ks)
    ax.set_xlabel("Complexity budget k"); ax.set_ylabel("Crossover interaction")
    clean_axes(ax,"y"); ax.legend(frameon=False,title="Graph seed",fontsize=8)
    source=write_source(cross_anchor,source_out,"CT06_graph_seed_consistency")
    png,_=save_fig(fig,"CT06_graph_seed_consistency",transfer_out,args.dpi)
    add_catalog(catalog,"CT06_graph_seed_consistency","stage4_transfer",png,source,
                "The state-specific crossover is not driven by a single graph realization.","SI")

    # CT07 home vs foreign absolute score scatter from replicate table
    rows=[]
    for _,r in cross_rep.iterrows():
        rows.append({"graph_seed":r.graph_seed,"k":r.k,"side":"transition_mid","home":r.mid_in_mid,"foreign":r.sparse_in_mid})
        rows.append({"graph_seed":r.graph_seed,"k":r.k,"side":"sparse_drive","home":r.sparse_in_sparse,"foreign":r.mid_in_sparse})
    hf=pd.DataFrame(rows)
    fig,ax=plt.subplots(figsize=(4.7,4.4))
    for side,c in [("transition_mid",C_TEAL),("sparse_drive",C_BLUE)]:
        g=hf[hf.side==side]
        ax.scatter(g.foreign,g.home,s=26,alpha=.65,color=c,label=pretty_state[side])
    lo=min(hf[["home","foreign"]].min()); hi=max(hf[["home","foreign"]].max())
    ax.plot([lo,hi],[lo,hi],ls="--",color=C_GREY,lw=1)
    ax.set_xlabel("Foreign-state optimized allocation score"); ax.set_ylabel("Home-state optimized allocation score")
    clean_axes(ax,"both"); ax.legend(frameon=False)
    source=write_source(hf,source_out,"CT07_home_vs_foreign_scores")
    png,_=save_fig(fig,"CT07_home_vs_foreign_scores",transfer_out,args.dpi)
    add_catalog(catalog,"CT07_home_vs_foreign_scores","stage4_transfer",png,source,
                "Matched evaluations place most observations above the identity line.","MAIN_CANDIDATE")

    # ================================================================
    # STAGE 4 FUNCTIONAL DEGENERACY
    # ================================================================
    primary=dg[np.isclose(dg.epsilon,0.02)].copy()
    independent=dg_ind[np.isclose(dg_ind.epsilon,0.02)].copy()

    # DG01 pairwise geometry
    fig,ax=plt.subplots(figsize=(5.0,4.2))
    for k,c in zip(ks,[C_BLUE,C_TEAL,C_ORANGE]):
        g=pair_geom[pair_geom.k==k]
        ax.scatter(g.jaccard_distance,g.fresh_score_gap,s=24,alpha=.55,color=c,label=f"k={k}")
    ax.axhline(0.02,color=C_ROSE,ls="--",lw=1,label="ε=0.02")
    ax.set_xlabel("Allocation Jaccard distance"); ax.set_ylabel("Fresh validation score gap")
    clean_axes(ax,"both"); ax.legend(frameon=False,fontsize=8)
    source=write_source(pair_geom,source_out,"DG01_pairwise_geometry")
    png,_=save_fig(fig,"DG01_pairwise_geometry",degeneracy_out,args.dpi)
    add_catalog(catalog,"DG01_pairwise_geometry","stage4_degeneracy",png,source,
                "At low budget, structurally distant allocations can remain near-equivalent in function.","MAIN_CANDIDATE")

    metrics=[
        ("D_epsilon_mean_jaccard_distance","DG02_D_epsilon_by_budget","Sampled functional degeneracy Dε",[C_BLUE,C_TEAL,C_ORANGE],"MAIN_CANDIDATE"),
        ("near_optimal_count","DG03_near_optimal_count_by_budget","Near-optimal allocations among four searches",[C_BLUE,C_TEAL,C_ORANGE],"SI_STRONG"),
        ("union_expansion","DG04_union_expansion_by_budget","Union expansion |∪Eε| / k",[C_BLUE,C_TEAL,C_ORANGE],"MAIN_CANDIDATE"),
        ("core_fraction","DG05_core_fraction_by_budget","Core fraction |∩Eε| / k",[C_BLUE,C_TEAL,C_ORANGE],"MAIN_CANDIDATE"),
        ("selection_entropy_union","DG06_selection_entropy_by_budget","Selection entropy over union",[C_BLUE,C_TEAL,C_ORANGE],"SI_STRONG"),
    ]
    for col,fid,ylab,cols,tier in metrics:
        fig,ax=plt.subplots(figsize=(4.8,4.0))
        groups=[primary.loc[primary.k==k,col] for k in ks]
        boxstrip(ax,groups,[f"k={k}" for k in ks],cols,ylab,zero=False)
        source=write_source(primary[["graph_seed","regime","k",col]],source_out,fid)
        png,_=save_fig(fig,fid,degeneracy_out,args.dpi)
        msg={
            "D_epsilon_mean_jaccard_distance":"Functional degeneracy is strongest when intrinsic-complexity budget is scarce.",
            "near_optimal_count":"Low-budget conditions more often admit multiple near-optimal searched allocations.",
            "union_expansion":"Near-optimal low-budget solutions span a substantially larger union of candidate nodes.",
            "core_fraction":"High-budget near-optimal solutions converge on a large common allocation core.",
            "selection_entropy_union":"Node-selection uncertainty is concentrated at low budget.",
        }[col]
        add_catalog(catalog,fid,"stage4_degeneracy",png,source,msg,tier)

    # DG07 epsilon sensitivity lines by budget
    sens=dg.groupby(["epsilon","k"],as_index=False).agg(mean_D=("D_epsilon_mean_jaccard_distance","mean"),mean_near=("near_optimal_count","mean"))
    fig,ax=plt.subplots(figsize=(5.0,4.0))
    for k,c in zip(ks,[C_BLUE,C_TEAL,C_ORANGE]):
        g=sens[sens.k==k].sort_values("epsilon")
        ax.plot(g.epsilon,g.mean_D,marker="o",lw=1.7,color=c,label=f"k={k}")
    ax.set_xlabel("Near-optimal tolerance ε"); ax.set_ylabel("Mean sampled Dε")
    clean_axes(ax,"y"); ax.legend(frameon=False)
    source=write_source(sens,source_out,"DG07_epsilon_sensitivity")
    png,_=save_fig(fig,"DG07_epsilon_sensitivity",degeneracy_out,args.dpi)
    add_catalog(catalog,"DG07_epsilon_sensitivity","stage4_degeneracy",png,source,
                "The budget dependence of sampled degeneracy persists across ε sensitivity values.","SI_STRONG")

    # DG08 independent replication scatter for D
    keys=["graph_seed","regime","k"]
    dd=primary[keys+["D_epsilon_mean_jaccard_distance"]].merge(
        independent[keys+["D_epsilon_mean_jaccard_distance"]],on=keys,suffixes=("_fresh","_independent"))
    fig,ax=plt.subplots(figsize=(4.5,4.3))
    ax.scatter(dd.D_epsilon_mean_jaccard_distance_fresh,dd.D_epsilon_mean_jaccard_distance_independent,
               color=C_BLUE,s=28,alpha=.7)
    ax.plot([0,1],[0,1],ls="--",color=C_GREY,lw=1)
    rho,p=stats.spearmanr(dd.D_epsilon_mean_jaccard_distance_fresh,dd.D_epsilon_mean_jaccard_distance_independent)
    ax.text(.04,.94,f"Spearman ρ = {rho:.2f}\np = {p:.3g}",transform=ax.transAxes,va="top")
    ax.set_xlabel("D₀.₀₂, fresh Stage4F evaluation"); ax.set_ylabel("D₀.₀₂, independent home evaluation")
    clean_axes(ax,"both")
    source=write_source(dd,source_out,"DG08_independent_replication_D")
    png,_=save_fig(fig,"DG08_independent_replication_D",degeneracy_out,args.dpi)
    add_catalog(catalog,"DG08_independent_replication_D","stage4_degeneracy",png,source,
                "The degeneracy geometry replicates under an independent evaluation set.","MAIN_CANDIDATE")

    # DG09 regime x budget heatmap of D
    h=primary.pivot_table(index="regime",columns="k",values="D_epsilon_mean_jaccard_distance",aggfunc="mean").reindex(index=states,columns=ks)
    fig,ax=plt.subplots(figsize=(4.6,3.7))
    heatmap(ax,h.values,[f"k={k}" for k in ks],[pretty_state[s] for s in states],cmap="viridis",vmin=0,vmax=1,fmt=".2f",cbar_label="Mean D₀.₀₂")
    ax.set_xlabel("Complexity budget"); ax.set_ylabel("Collective state")
    source=write_source(h.reset_index(),source_out,"DG09_state_budget_degeneracy_heatmap")
    png,_=save_fig(fig,"DG09_state_budget_degeneracy_heatmap",degeneracy_out,args.dpi)
    add_catalog(catalog,"DG09_state_budget_degeneracy_heatmap","stage4_degeneracy",png,source,
                "Functional degeneracy depends on budget and collective state, with the dominant gradient at low budget.","SI")

    # ================================================================
    # ALLEN LAG-AWARE REFIT
    # ================================================================
    state_order=["familiar_active","novel_active","passive"]
    pretty_allen={"familiar_active":"Familiar active","novel_active":"Novel active","passive":"Passive"}

    # AL01 lag distribution overall
    lag_counts=diag.groupby(["chosen_lag_seconds","state"]).size().reset_index(name="count")
    total_lag=diag.chosen_lag_seconds.value_counts().sort_index().rename_axis("chosen_lag_seconds").reset_index(name="count")
    fig,ax=plt.subplots(figsize=(4.7,3.9))
    ax.bar(total_lag.chosen_lag_seconds.astype(str),total_lag["count"],color=C_PURPLE,alpha=.85)
    ax.set_xlabel("Selected lag (s)"); ax.set_ylabel("Experiments")
    clean_axes(ax,"y")
    source=write_source(lag_counts,source_out,"AL01_selected_lag_distribution")
    png,_=save_fig(fig,"AL01_selected_lag_distribution",allen_out,args.dpi)
    add_catalog(catalog,"AL01_selected_lag_distribution","allen_lagaware",png,source,
                "The blocked-validation procedure most often selects the shortest tested lag (0.25 s).","SI")

    # AL02 lag x state heatmap counts
    lagtab=lag_counts.pivot(index="state",columns="chosen_lag_seconds",values="count").fillna(0).reindex(index=state_order)
    lags=sorted(diag.chosen_lag_seconds.unique())
    lagtab=lagtab.reindex(columns=lags,fill_value=0)
    fig,ax=plt.subplots(figsize=(5.0,3.5))
    heatmap(ax,lagtab.values,[str(x) for x in lags],[pretty_allen[s] for s in state_order],cmap="Purples",vmin=0,vmax=max(1,lagtab.values.max()),fmt=".0f",cbar_label="Experiments")
    ax.set_xlabel("Selected lag (s)"); ax.set_ylabel("State")
    source=write_source(lagtab.reset_index(),source_out,"AL02_lag_by_state_heatmap")
    png,_=save_fig(fig,"AL02_lag_by_state_heatmap",allen_out,args.dpi)
    add_catalog(catalog,"AL02_lag_by_state_heatmap","allen_lagaware",png,source,
                "Lag selection is not confined to a single behavioral state.","DIAGNOSTIC")

    # AL03 held-out test R2 by state
    fig,ax=plt.subplots(figsize=(5.0,4.0))
    groups=[diag.loc[diag.state==s,"test_r2"] for s in state_order]
    boxstrip(ax,groups,[pretty_allen[s] for s in state_order],[C_BLUE,C_TEAL,C_ORANGE],"Held-out latent-dynamics R²",zero=True)
    source=write_source(diag[["experiment_id","container_id","state","test_r2"]],source_out,"AL03_test_r2_by_state")
    png,_=save_fig(fig,"AL03_test_r2_by_state",allen_out,args.dpi)
    add_catalog(catalog,"AL03_test_r2_by_state","allen_lagaware",png,source,
                "Absolute held-out latent-dynamics predictability is modest and heterogeneous.","MAIN_BOUNDARY")

    # AL04 delta vs persistence by state
    fig,ax=plt.subplots(figsize=(5.0,4.0))
    groups=[diag.loc[diag.state==s,"test_delta_vs_persistence"] for s in state_order]
    boxstrip(ax,groups,[pretty_allen[s] for s in state_order],[C_BLUE,C_TEAL,C_ORANGE],"R² gain vs persistence baseline",zero=True)
    source=write_source(diag[["experiment_id","container_id","state","test_delta_vs_persistence"]],source_out,"AL04_delta_vs_persistence")
    png,_=save_fig(fig,"AL04_delta_vs_persistence",allen_out,args.dpi)
    add_catalog(catalog,"AL04_delta_vs_persistence","allen_lagaware",png,source,
                "Despite low absolute R², lag-aware latent dynamics beat a persistence baseline in most experiments.","MAIN_CANDIDATE")

    # AL05 raw vs residual state stability paired
    paired=raw.merge(resid,on=["container_id","state_a","state_b"],suffixes=("_raw","_residual"))
    fig,ax=plt.subplots(figsize=(4.6,4.1))
    for _,r in paired.iterrows():
        ax.plot([0,1],[r.rho_raw,r.rho_residual],color=C_LIGHT,lw=.8,alpha=.55)
    ax.scatter(np.zeros(len(paired)),paired.rho_raw,color=C_BLUE,s=20,alpha=.65)
    ax.scatter(np.ones(len(paired)),paired.rho_residual,color=C_TEAL,s=20,alpha=.65)
    ax.axhline(0,color=C_GREY,ls="--",lw=1)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Raw leverage rank","Activity/loading residual"],rotation=18,ha="right")
    ax.set_ylabel("Same-cell state-pair Spearman ρ"); clean_axes(ax,"y")
    source=write_source(paired,source_out,"AL05_raw_vs_residual_state_stability")
    png,_=save_fig(fig,"AL05_raw_vs_residual_state_stability",allen_out,args.dpi)
    add_catalog(catalog,"AL05_raw_vs_residual_state_stability","allen_lagaware",png,source,
                "Removing activity/loading structure markedly reduces state-to-state leverage-rank stability.","MAIN_CANDIDATE")

    # AL06 raw state-pair median matrix (3 unique pairs)
    mat=np.eye(3)
    for _,r in raw_sum.iterrows():
        i=state_order.index(r.state_a); j=state_order.index(r.state_b); mat[i,j]=mat[j,i]=r.median_rho
    fig,ax=plt.subplots(figsize=(4.4,3.9))
    heatmap(ax,mat,[pretty_allen[s] for s in state_order],[pretty_allen[s] for s in state_order],cmap="RdBu_r",vmin=-1,vmax=1,fmt=".2f",cbar_label="Median within-container Spearman ρ")
    source=write_source(raw_sum,source_out,"AL06_raw_state_pair_heatmap")
    png,_=save_fig(fig,"AL06_raw_state_pair_heatmap",allen_out,args.dpi)
    add_catalog(catalog,"AL06_raw_state_pair_heatmap","allen_lagaware",png,source,
                "Raw leverage rankings retain moderate cross-state ordering.","SI")

    # AL07 residual state-pair median matrix
    mat=np.eye(3)
    for _,r in resid_sum.iterrows():
        i=state_order.index(r.state_a); j=state_order.index(r.state_b); mat[i,j]=mat[j,i]=r.median_rho
    fig,ax=plt.subplots(figsize=(4.4,3.9))
    heatmap(ax,mat,[pretty_allen[s] for s in state_order],[pretty_allen[s] for s in state_order],cmap="RdBu_r",vmin=-1,vmax=1,fmt=".2f",cbar_label="Median residual Spearman ρ")
    source=write_source(resid_sum,source_out,"AL07_residual_state_pair_heatmap")
    png,_=save_fig(fig,"AL07_residual_state_pair_heatmap",allen_out,args.dpi)
    add_catalog(catalog,"AL07_residual_state_pair_heatmap","allen_lagaware",png,source,
                "After residualization, state-pair rank correlations fall to weak-to-moderate levels.","MAIN_CANDIDATE")

    # AL08 quality gate summary
    gate_order=["beats_persistence","r2_gt_0","r2_gt_005","r2_gt0_and_beats_persistence"]
    qgate=qgate.copy(); qgate["pair"]=[f"{pretty_allen[a]}–{pretty_allen[b]}" for a,b in zip(qgate.state_a,qgate.state_b)]
    pairnames=list(dict.fromkeys(qgate["pair"]))
    x=np.arange(len(gate_order)); width=.24
    fig,ax=plt.subplots(figsize=(6.2,4.1))
    for j,pair in enumerate(pairnames):
        g=qgate[qgate["pair"]==pair].set_index("gate").reindex(gate_order)
        ax.bar(x+(j-1)*width,g.median_rho,width=width,label=pair,alpha=.82)
    ax.set_xticks(x); ax.set_xticklabels(["Beats persistence","R² > 0","R² > 0.05","R² > 0 + beats\npersistence"],rotation=22,ha="right")
    ax.set_ylabel("Median same-cell state-pair ρ"); clean_axes(ax,"y"); ax.legend(frameon=False,fontsize=8)
    source=write_source(qgate,source_out,"AL08_quality_gated_state_stability")
    png,_=save_fig(fig,"AL08_quality_gated_state_stability",allen_out,args.dpi)
    add_catalog(catalog,"AL08_quality_gated_state_stability","allen_lagaware",png,source,
                "Cross-state ordering persists in quality-gated subsets, but should be interpreted conditional on model quality.","MAIN_CANDIDATE")

    # AL09 old vs lag-aware test R2
    fig,ax=plt.subplots(figsize=(4.5,4.3))
    ax.scatter(oldnew.old_r2,oldnew.new_r2,s=28,color=C_BLUE,alpha=.62)
    lo=min(oldnew[["old_r2","new_r2"]].min()); hi=max(oldnew[["old_r2","new_r2"]].max())
    ax.plot([lo,hi],[lo,hi],ls="--",color=C_GREY,lw=1)
    ax.set_xlabel("Previous held-out R²"); ax.set_ylabel("Lag-aware held-out R²")
    clean_axes(ax,"both")
    source=write_source(oldnew,source_out,"AL09_old_vs_lagaware_r2")
    png,_=save_fig(fig,"AL09_old_vs_lagaware_r2",allen_out,args.dpi)
    add_catalog(catalog,"AL09_old_vs_lagaware_r2","allen_lagaware",png,source,
                "Lag-aware refitting does not materially improve absolute held-out R² over the previous formulation.","MAIN_BOUNDARY")

    # AL10 new-old R2 difference distribution
    fig,ax=plt.subplots(figsize=(4.7,3.8))
    ax.hist(oldnew.delta_r2_new_minus_old,bins=20,color=C_PURPLE,alpha=.82,edgecolor="white",linewidth=.5)
    ax.axvline(0,color=C_GREY,ls="--",lw=1)
    ax.axvline(oldnew.delta_r2_new_minus_old.median(),color=C_ROSE,lw=1.4)
    ax.set_xlabel("Lag-aware R² − previous R²"); ax.set_ylabel("Experiments"); clean_axes(ax,"y")
    source=write_source(oldnew,source_out,"AL10_r2_improvement_distribution")
    png,_=save_fig(fig,"AL10_r2_improvement_distribution",allen_out,args.dpi)
    add_catalog(catalog,"AL10_r2_improvement_distribution","allen_lagaware",png,source,
                "The refit acts as a methodological stress test rather than a predictive-performance upgrade.","SI_STRONG")

    # AL11 raw and residual rho by state pair
    pair_order=[("familiar_active","novel_active"),("familiar_active","passive"),("novel_active","passive")]
    labels=[f"{pretty_allen[a]}–{pretty_allen[b]}" for a,b in pair_order]
    fig,ax=plt.subplots(figsize=(6.0,4.0))
    xpos=np.arange(3)
    raw_med=[];res_med=[]
    for a,b in pair_order:
        g=paired[(paired.state_a==a)&(paired.state_b==b)]
        raw_med.append(g.rho_raw.median());res_med.append(g.rho_residual.median())
    width=.34
    ax.bar(xpos-width/2,raw_med,width,color=C_BLUE,label="Raw")
    ax.bar(xpos+width/2,res_med,width,color=C_TEAL,label="Residual")
    ax.set_xticks(xpos);ax.set_xticklabels(labels,rotation=20,ha="right")
    ax.set_ylabel("Median within-container Spearman ρ");clean_axes(ax,"y");ax.legend(frameon=False)
    source=write_source(paired,source_out,"AL11_state_pair_raw_residual")
    png,_=save_fig(fig,"AL11_state_pair_raw_residual",allen_out,args.dpi)
    add_catalog(catalog,"AL11_state_pair_raw_residual","allen_lagaware",png,source,
                "Residualization reduces stability for every behavioral-state pair.","MAIN_CANDIDATE")

    # AL12 experiment quality fractions by state
    rows=[]
    for s,g in diag.groupby("state"):
        rows.append({"state":s,"positive_r2_fraction":float((g.test_r2>0).mean()),"beats_persistence_fraction":float((g.test_delta_vs_persistence>0).mean()),"n":len(g)})
    qfrac=pd.DataFrame(rows).set_index("state").reindex(state_order).reset_index()
    fig,ax=plt.subplots(figsize=(5.3,3.9))
    x=np.arange(3);w=.34
    ax.bar(x-w/2,qfrac.positive_r2_fraction,w,color=C_PURPLE,label="R² > 0")
    ax.bar(x+w/2,qfrac.beats_persistence_fraction,w,color=C_TEAL,label="Beats persistence")
    ax.set_xticks(x);ax.set_xticklabels([pretty_allen[s] for s in state_order],rotation=20,ha="right")
    ax.set_ylim(0,1.05);ax.set_ylabel("Experiment fraction");clean_axes(ax,"y");ax.legend(frameon=False)
    source=write_source(qfrac,source_out,"AL12_model_quality_fractions")
    png,_=save_fig(fig,"AL12_model_quality_fractions",allen_out,args.dpi)
    add_catalog(catalog,"AL12_model_quality_fractions","allen_lagaware",png,source,
                "Relative-to-persistence quality is much more robust than positive absolute R².","SI_STRONG")

    # AL13 chosen lag vs held-out R2
    fig,ax=plt.subplots(figsize=(4.8,4.0))
    for s,c in zip(state_order,[C_BLUE,C_TEAL,C_ORANGE]):
        g=diag[diag.state==s]
        jitter=rng.normal(0,.025,len(g))
        ax.scatter(g.chosen_lag_seconds+jitter,g.test_r2,s=25,alpha=.6,color=c,label=pretty_allen[s])
    ax.axhline(0,color=C_GREY,ls="--",lw=1)
    ax.set_xscale("log",base=2);ax.set_xticks([.25,.5,1,2]);ax.set_xticklabels(["0.25","0.5","1","2"])
    ax.set_xlabel("Selected lag (s)");ax.set_ylabel("Held-out R²");clean_axes(ax,"y");ax.legend(frameon=False,fontsize=8)
    source=write_source(diag[["experiment_id","state","chosen_lag_seconds","test_r2"]],source_out,"AL13_lag_vs_test_r2")
    png,_=save_fig(fig,"AL13_lag_vs_test_r2",allen_out,args.dpi)
    add_catalog(catalog,"AL13_lag_vs_test_r2","allen_lagaware",png,source,
                "Longer selected lags do not visibly guarantee better held-out dynamics prediction.","DIAGNOSTIC")

    # AL14 model quality vs residual state stability, container aggregated
    cquality=diag.groupby("container_id",as_index=False).agg(median_test_r2=("test_r2","median"),median_delta_persist=("test_delta_vs_persistence","median"))
    cres=resid.groupby("container_id",as_index=False).agg(median_residual_rho=("rho","median"))
    qr=cquality.merge(cres,on="container_id")
    fig,ax=plt.subplots(figsize=(4.6,4.1))
    ax.scatter(qr.median_test_r2,qr.median_residual_rho,s=32,color=C_TEAL,alpha=.7)
    rho,p=stats.spearmanr(qr.median_test_r2,qr.median_residual_rho,nan_policy="omit")
    ax.text(.04,.95,f"ρ = {rho:.2f}\np = {p:.3g}",transform=ax.transAxes,va="top")
    ax.axvline(0,color=C_GREY,ls="--",lw=1);ax.axhline(0,color=C_GREY,ls="--",lw=1)
    ax.set_xlabel("Container median held-out R²");ax.set_ylabel("Container median residual state-pair ρ");clean_axes(ax,"both")
    source=write_source(qr,source_out,"AL14_quality_vs_residual_stability")
    png,_=save_fig(fig,"AL14_quality_vs_residual_stability",allen_out,args.dpi)
    add_catalog(catalog,"AL14_quality_vs_residual_stability","allen_lagaware",png,source,
                "Diagnostic test of whether residual state stability is driven by latent-model quality.","DIAGNOSTIC")

    # AL15 cell-level activity vs raw/residual leverage rank
    sample=long.copy()
    fig,ax=plt.subplots(figsize=(4.7,4.1))
    ax.scatter(sample.mean_activity,sample.L_primary_rank,s=9,color=C_BLUE,alpha=.18,edgecolors="none",label="Raw rank")
    ax.scatter(sample.mean_activity,sample.L_resid_rank,s=9,color=C_TEAL,alpha=.12,edgecolors="none",label="Residual rank")
    ax.set_xlabel("Mean activity");ax.set_ylabel("Within-experiment leverage rank");clean_axes(ax,"both");ax.legend(frameon=False,fontsize=8)
    source=write_source(sample[["experiment_id","container_id","experiment_state","cell_specimen_id","mean_activity","L_primary_rank","L_resid_rank"]],source_out,"AL15_activity_vs_leverage_rank")
    png,_=save_fig(fig,"AL15_activity_vs_leverage_rank",allen_out,args.dpi)
    add_catalog(catalog,"AL15_activity_vs_leverage_rank","allen_lagaware",png,source,
                "Residualization explicitly separates leverage ranking from simple activity amplitude.","SI")

    # ================================================================
    # Inferential closure + claim adjudication
    # ================================================================
    cross_ci=bootstrap_ci(cross_anchor.crossover_interaction,stat=np.mean,n=args.bootstrap,rng=rng)
    cross_p=exact_signflip_p_greater(cross_anchor.crossover_interaction)
    mid_pos=int((cross_anchor.mid_home_advantage>0).sum())
    sparse_pos=int((cross_anchor.sparse_home_advantage>0).sum())
    offdiag_all_negative=bool((transfer_mean.loc[transfer_mean.source_state!=transfer_mean.target_state,"delta_vs_target_home"]<0).all())

    pri8=primary[primary.k==8].set_index(["graph_seed","regime"])
    pri64=primary[primary.k==64].set_index(["graph_seed","regime"])
    common=pri8.index.intersection(pri64.index)
    d8minus64=(pri8.loc[common,"D_epsilon_mean_jaccard_distance"]-pri64.loc[common,"D_epsilon_mean_jaccard_distance"]).values
    d_p=exact_signflip_p_greater(d8minus64)
    d_ci=bootstrap_ci(d8minus64,stat=np.mean,n=args.bootstrap,rng=rng)

    rho_rep,p_rep=stats.spearmanr(dd.D_epsilon_mean_jaccard_distance_fresh,dd.D_epsilon_mean_jaccard_distance_independent)

    W_rawres,p_rawres=safe_wilcoxon(paired.rho_raw,paired.rho_residual,alternative="greater")
    rawres_delta=paired.rho_raw-paired.rho_residual
    rawres_ci=bootstrap_ci(rawres_delta,stat=np.median,n=args.bootstrap,rng=rng)
    W_oldnew,p_oldnew=safe_wilcoxon(oldnew.new_r2,oldnew.old_r2,alternative="two-sided")
    binom_p=float(stats.binomtest(int((diag.test_delta_vs_persistence>0).sum()),len(diag),0.5,alternative="greater").pvalue)

    # State-pair raw vs residual tests, FDR corrected.
    pair_stats=[]
    for (a,b),g in paired.groupby(["state_a","state_b"]):
        w,p=safe_wilcoxon(g.rho_raw,g.rho_residual,alternative="greater")
        pair_stats.append({"state_a":a,"state_b":b,"n":len(g),"median_raw":g.rho_raw.median(),"median_residual":g.rho_residual.median(),
                           "median_drop":np.median(g.rho_raw-g.rho_residual),"wilcoxon_W":w,"p_raw_gt_residual":p})
    pair_stats=pd.DataFrame(pair_stats)
    pair_stats["q_BH"]=bh_fdr(pair_stats.p_raw_gt_residual)
    pair_stats.to_csv(analysis_out/"allen_raw_vs_residual_pair_tests.csv",index=False)

    metrics={
        "version":VERSION,
        "stage4_cross_state":{
            "n_anchors":int(len(cross_anchor)),
            "mean_crossover_interaction":float(cross_anchor.crossover_interaction.mean()),
            "bootstrap95_mean":list(map(float,cross_ci)),
            "exact_signflip_p_greater":float(cross_p),
            "positive_anchors":int((cross_anchor.crossover_interaction>0).sum()),
            "mid_home_positive_anchors":mid_pos,
            "sparse_home_positive_anchors":sparse_pos,
            "all_six_mean_offdiagonal_transfers_negative":offdiag_all_negative,
            "dense_contrast_guardrail":"transition_dense differs in connection probability and remains topology+dynamics composite. Primary pure-state contrast is transition_mid vs sparse_drive."
        },
        "stage4_functional_degeneracy":{
            "definition":dg_key.get("definition"),
            "epsilon":0.02,
            "mean_D":float(primary.D_epsilon_mean_jaccard_distance.mean()),
            "median_D":float(primary.D_epsilon_mean_jaccard_distance.median()),
            "mean_D_k8":float(pri8.D_epsilon_mean_jaccard_distance.mean()),
            "mean_D_k64":float(pri64.D_epsilon_mean_jaccard_distance.mean()),
            "mean_D_k8_minus_k64":float(np.mean(d8minus64)),
            "bootstrap95_k8_minus_k64":list(map(float,d_ci)),
            "exact_signflip_p_k8_gt_k64":float(d_p),
            "independent_replication_spearman":float(rho_rep),
            "independent_replication_p":float(p_rep),
            "guardrail":"Four optimizer replicates per condition sample an empirical lower bound of the near-optimal equivalence class; they do not enumerate the full combinatorial solution volume."
        },
        "allen_lagaware":{
            "experiments":int(len(diag)),
            "containers":int(diag.container_id.nunique()),
            "cells_rows":int(len(long)),
            "median_test_r2":float(diag.test_r2.median()),
            "positive_r2_fraction":float((diag.test_r2>0).mean()),
            "median_delta_vs_persistence":float(diag.test_delta_vs_persistence.median()),
            "beats_persistence_fraction":float((diag.test_delta_vs_persistence>0).mean()),
            "binomial_p_beats_persistence_gt_half":binom_p,
            "selected_lag_counts":{str(k):int(v) for k,v in diag.chosen_lag_seconds.value_counts().sort_index().items()},
            "median_raw_state_pair_rho":float(paired.rho_raw.median()),
            "median_residual_state_pair_rho":float(paired.rho_residual.median()),
            "median_raw_minus_residual_rho":float(np.median(rawres_delta)),
            "bootstrap95_median_raw_minus_residual":list(map(float,rawres_ci)),
            "wilcoxon_p_raw_gt_residual":float(p_rawres),
            "median_new_minus_old_r2":float(oldnew.delta_r2_new_minus_old.median()),
            "wilcoxon_p_new_vs_old":float(p_oldnew),
            "guardrail":"Absolute held-out R² remains modest. Lag-aware refit should be framed as a leakage-controlled stress test; dynamical-leverage claims should retain quality gates and the persistence baseline."
        }
    }
    (analysis_out/"closure_key_metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf-8")

    claims=pd.DataFrame([
        ["Collective network state reshapes the dynamical-leverage landscape.","SUPPORTED","Pure same-topology state contrast: all 9 anchor crossover interactions are positive; exact one-sided sign-flip p=%.4g."%cross_p],
        ["A state-optimized allocation is generally portable to another state.","REJECTED","All six mean off-diagonal transfers are negative relative to the target-state home allocation."],
        ["Functionally near-equivalent allocations can be structurally distinct.","SUPPORTED_WITH_GUARDRAIL","At ε=0.02 sampled D is high at k=8 and falls sharply by k=64; independent evaluation replicates the geometry."],
        ["The full near-optimal solution-space volume has been enumerated.","NOT_SUPPORTED","Only four optimizer solutions per condition are sampled."],
        ["Lag-aware Allen refitting materially improves absolute predictive R².","NOT_SUPPORTED","Median new-old R² = %.4f; paired Wilcoxon p=%.3g."%(oldnew.delta_r2_new_minus_old.median(),p_oldnew)],
        ["Lag-aware latent dynamics outperform a persistence baseline.","SUPPORTED","%.1f%% of experiments beat persistence; median ΔR²=%.3f."%(100*(diag.test_delta_vs_persistence>0).mean(),diag.test_delta_vs_persistence.median())],
        ["Allen leverage ranking contains state-dependent structure beyond simple activity/loading.","SUPPORTED_WITH_QUALITY_GATE","Median raw state-pair ρ %.3f falls to residual %.3f; paired p=%.3g."%(paired.rho_raw.median(),paired.rho_residual.median(),p_rawres)],
        ["Allen lag-aware dynamics are uniformly well predicted.","REJECTED","Median held-out R² is only %.4f; positive R² in %.1f%% of experiments."%(diag.test_r2.median(),100*(diag.test_r2>0).mean())],
    ],columns=["claim","verdict","evidence"])
    claims.to_csv(analysis_out/"claim_adjudication.csv",index=False)

    # Figure catalog
    catalog_df=pd.DataFrame(catalog)
    catalog_df.to_csv(analysis_out/"figure_catalog.csv",index=False)

    # Inventory of critical files actually used
    inv=[]
    for arc,label in [(allen_archive,"allen_archive"),(turbo_archive,"turbo_archive")]:
        inv.append({"role":label,"path":str(arc),"bytes":arc.stat().st_size})
    for p in sorted(analysis_out.glob("*")):
        if p.is_file(): inv.append({"role":"generated_analysis","path":str(p),"bytes":p.stat().st_size})
    pd.DataFrame(inv).to_csv(analysis_out/"source_and_output_inventory.csv",index=False)

    report=f"""# Complexity–Leverage closure atlas v{VERSION}

## Executive conclusion

This closure analysis strengthens two different parts of the manuscript without yet merging them into the Stage 1–5 atlas.

### Stage 4: state-specific dynamical leverage is directly supported

The primary pure-state contrast compares `transition_mid` with `sparse_drive`, which share the same topology density. Across {len(cross_anchor)} graph×budget anchors, the crossover interaction is positive in {(cross_anchor.crossover_interaction>0).sum()}/{len(cross_anchor)} anchors. Mean crossover = {cross_anchor.crossover_interaction.mean():.4f}; bootstrap 95% CI [{cross_ci[0]:.4f}, {cross_ci[1]:.4f}]; exact one-sided sign-flip p = {cross_p:.6f}. This is the cleanest current computational evidence that the identity of valuable allocation sites depends on collective state rather than topology alone.

All six mean off-diagonal source→target transfers are below the target-state home allocation. `transition_dense` remains a topology+dynamics composite because its connection probability differs, so it should be presented only as an auxiliary robustness contrast.

### Stage 4: functional degeneracy is now a formal, measurable object

At ε=0.02, sampled mean Dε = {primary.D_epsilon_mean_jaccard_distance.mean():.4f}. Mean Dε is {pri8.D_epsilon_mean_jaccard_distance.mean():.4f} at k=8 versus {pri64.D_epsilon_mean_jaccard_distance.mean():.4f} at k=64. The paired k8−k64 difference is {np.mean(d8minus64):.4f}, bootstrap 95% CI [{d_ci[0]:.4f}, {d_ci[1]:.4f}], exact one-sided sign-flip p = {d_p:.6f}. Independent-home evaluation reproduces the condition-level geometry (Spearman ρ={rho_rep:.3f}, p={p_rep:.4g}).

Interpretation: when complexity is scarce, there can be multiple structurally distinct allocations with near-equivalent network function; at higher budget the solutions converge toward a larger shared core. This is an empirical sampled equivalence class, not a full enumeration of combinatorial solution volume.

### Allen lag-aware refit: methodological closure, not a performance upgrade

The refit contains {len(diag)} experiments from {diag.container_id.nunique()} containers and {len(long):,} cell rows. Median held-out R² is {diag.test_r2.median():.4f}, with {(diag.test_r2>0).mean()*100:.1f}% positive. Therefore absolute latent-dynamics predictability remains modest.

However, {(diag.test_delta_vs_persistence>0).mean()*100:.1f}% of experiments beat the matched persistence baseline, with median ΔR² = {diag.test_delta_vs_persistence.median():.4f} (one-sided binomial p={binom_p:.3g}). Lag-aware refitting does not improve the previous model's absolute R²: median new−old = {oldnew.delta_r2_new_minus_old.median():.4f}, paired Wilcoxon p={p_oldnew:.3g}. Thus the correct role of this analysis is a leakage-controlled stress test, not a claim of improved forecasting.

Most importantly, same-cell state-pair leverage ordering drops from median raw ρ={paired.rho_raw.median():.3f} to residual ρ={paired.rho_residual.median():.3f} after removing activity/loading structure. The median paired reduction is {np.median(rawres_delta):.3f}, 95% bootstrap CI [{rawres_ci[0]:.3f}, {rawres_ci[1]:.3f}], paired one-sided Wilcoxon p={p_rawres:.3g}. This supports state-dependent reconfiguration beyond simple activity/loading, but manuscript claims must retain model-quality gates because absolute R² is limited.

## Recommended current wording

**Computational:** “Collective network state reshapes the dynamical-leverage landscape: allocations optimized for one state lose value when transferred to another, even under a matched topology.”

**Allocation geometry:** “Finite complexity admits state- and budget-dependent functional degeneracy: under scarce budgets, structurally distinct allocations can be functionally near-equivalent, whereas larger budgets reveal a more stable shared allocation core.”

**Allen:** “State-dependent leverage reconfiguration survives a lag-aware, leakage-controlled latent-dynamics refit and cannot be reduced to activity amplitude or latent loading alone; however, absolute predictive quality is modest, motivating explicit quality-gated interpretation.”

## Boundaries that should remain explicit

1. Do not use `transition_dense` as the primary pure-state transfer proof; it changes topology density.
2. Do not call four optimizer replicates the full solution-space volume.
3. Do not claim that lag-aware refitting improves absolute Allen prediction.
4. Do not describe all Allen experiments as high-quality dynamical models; use persistence and R² quality gates.
5. Do not infer biological intrinsic complexity from Allen Visual Behavior alone; this analysis concerns state-resolved leverage.

## Files to inspect first

- `analysis/claim_adjudication.csv`
- `analysis/closure_key_metrics.json`
- `analysis/figure_catalog.csv`
- `figures_stage4_transfer/CT01_transfer_matrix_centered.png`
- `figures_stage4_transfer/CT02_crossover_by_budget.png`
- `figures_stage4_degeneracy/DG01_pairwise_geometry.png`
- `figures_stage4_degeneracy/DG08_independent_replication_D.png`
- `figures_allen_lagaware/AL05_raw_vs_residual_state_stability.png`
- `figures_allen_lagaware/AL04_delta_vs_persistence.png`
"""
    (analysis_out/"closure_analysis_report.md").write_text(report,encoding="utf-8")

    readme=f"""# ClosureExtensions_v1

Run from PowerShell at the project root:

```powershell
python .\\plot\\analyze_closure_extensions_v1.py `
  --root "D:\\Research\\Neural Science"
```

Explicit archives if needed:

```powershell
python .\\plot\\analyze_closure_extensions_v1.py `
  --root "D:\\Research\\Neural Science" `
  --allen-archive "D:\\Research\\Neural Science\\bio data\\Allen_lag_aware_refit_20260820.tar.gz" `
  --turbo-archive "D:\\Research\\Neural Science\\bio data\\ComplexityLeverage_ClosureCompute_v2_Turbo_results_20260820_154030.tar.gz"
```

Output: `D:\\Research\\Neural Science\\plot\\ClosureExtensions_v1`

PNG is rendered at {args.dpi} dpi; PDF is vector. Every plotted result has a source-data CSV and the atlas contains a figure catalog plus claim-adjudication table.
"""
    (out/"README.md").write_text(readme,encoding="utf-8")

    print("=== Complexity–Leverage closure atlas complete ===")
    print(json.dumps({
        "status":"COMPLETE",
        "version":VERSION,
        "output_root":str(out),
        "figure_count_png":len(catalog_df),
        "stage4_mean_crossover":metrics["stage4_cross_state"]["mean_crossover_interaction"],
        "stage4_crossover_p":metrics["stage4_cross_state"]["exact_signflip_p_greater"],
        "degeneracy_k8_minus_k64":metrics["stage4_functional_degeneracy"]["mean_D_k8_minus_k64"],
        "allen_median_test_r2":metrics["allen_lagaware"]["median_test_r2"],
        "allen_residual_state_pair_rho":metrics["allen_lagaware"]["median_residual_state_pair_rho"],
    },indent=2,ensure_ascii=False))
    print("Read first:")
    print(analysis_out/"closure_analysis_report.md")
    print(analysis_out/"claim_adjudication.csv")
    print(analysis_out/"figure_catalog.csv")

if __name__ == "__main__":
    main()
