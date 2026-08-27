#!/usr/bin/env python3
"""Build standardized leaf panels and fixed-width manuscript reference figures.

This is an assembly/redraw layer.  It never mutates the frozen V5.3 atlas.
Scientific values are read only from the V5.3 per-panel source-data tables.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib import font_manager
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageOps


HERE = Path(__file__).resolve().parent
PLOT_ROOT = HERE.parent
ATLAS = PLOT_ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "01_MAIN"
OUT = HERE / "output"
LEAF = OUT / "leaf_panels"
LEGENDS = OUT / "shared_legends"
REF = OUT / "reference_figures"
META = OUT / "metadata"

DPI = 600
LEAF_SIZE = (3.35, 2.45)  # standardized redraw canvas, inches
REFERENCE_WIDTH_MM = 180.0


def resolve_font(name: str, environment_variable: str) -> Path | None:
    explicit = os.environ.get(environment_variable)
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate
    try:
        candidate = Path(font_manager.findfont(name, fallback_to_default=False))
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


ARIAL = resolve_font("Arial", "ARIAL_FONT")
ARIAL_BOLD = resolve_font("Arial:style=Bold", "ARIAL_BOLD_FONT")

BLUE, MID, RED = "#2166AC", "#F7F7F7", "#B2182B"
HEATMAP_CMAP = LinearSegmentedColormap.from_list("ct01_blue_white_red", [BLUE, MID, RED], N=256)

COL = {
    "sparse_drive": "#355F7F", "transition_mid": "#2A9D8F", "transition_dense": "#C27628",
    "frequency_transition": "#355F7F", "heterogeneity_optimum": "#2A9D8F", "sparse_complexity": "#C27628",
    "high_frequency": "#4F789D", "high_intermediate": "#3D8B7A", "intermediate": "#C39A46",
    "low_frequency": "#A94F63", "weak_or_irregular": "#8174A8",
    "random": "#7A7A7A", "coverage_greedy": "#355F7F", "cycle_proxy": "#B75C70",
    "feedback_hub": "#5E6C84", "high_degree": "#7568A9", "module_bridge": "#73835C",
    "gpu_surrogate": "#2A9D8F", "spectral": "#2166AC", "none": "#C8C8C8",
    "primary_shared": "#2A9D8F", "protocol_conservative": "#C27628",
}

LABEL = {
    "sparse_drive": "Sparse drive", "transition_mid": "Transition mid", "transition_dense": "Transition dense",
    "frequency_transition": "Frequency transition", "heterogeneity_optimum": "Heterogeneity optimum",
    "sparse_complexity": "Sparse complexity", "high_frequency": "High frequency",
    "high_intermediate": "High-intermediate", "intermediate": "Intermediate",
    "low_frequency": "Low frequency", "weak_or_irregular": "Weak / irregular",
    "random": "Random", "coverage_greedy": "Coverage greedy", "cycle_proxy": "Cycle proxy",
    "feedback_hub": "Feedback hub", "high_degree": "High degree", "module_bridge": "Module bridge",
    "gpu_surrogate": "GPU surrogate", "spectral": "Spectral", "none": "None",
    "primary_shared": "Primary shared", "protocol_conservative": "Protocol conservative",
    "scale_free": "Scale-free", "erdos_renyi": "Erdos-Renyi", "er": "ER",
    "modular": "Modular", "small_world": "Small-world",
}

PRETTY_ENDPOINT = {
    "connection": "Connection", "psp_amplitude": "PSP amplitude", "psc_amplitude": "PSC amplitude",
    "stp_induction_50hz": "STP induction, 50 Hz", "variability_resting_state": "Resting variability",
    "G_norm_50hz": "Normalized gain, 50 Hz", "G_norm_5hz": "Normalized gain, 5 Hz",
    "G_norm_10hz": "Normalized gain, 10 Hz", "G_norm_20hz": "Normalized gain, 20 Hz",
}


def setup_style() -> None:
    if ARIAL is not None:
        font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD is not None:
        font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7.0, "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 6.2,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def source_csv(panel_dir: Path, token: str) -> Path:
    hits = sorted((panel_dir / "source_data").glob(f"*{token}*.csv"))
    if not hits:
        raise FileNotFoundError(f"No source CSV for {token} in {panel_dir}")
    return hits[0]


def save_leaf(fig, fig_id: str, leaf_id: str, source_paths: list[Path], role: str, manifest: list[dict]):
    outdir = LEAF / fig_id
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{leaf_id}.png"
    pdf = outdir / f"{leaf_id}.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    with Image.open(png) as im:
        px = im.size
    manifest.append({
        "figure": fig_id, "leaf_id": leaf_id, "role": role, "generation": "source-data redraw",
        "png": str(png.relative_to(HERE)), "pdf": str(pdf.relative_to(HERE)),
        "pixel_width": px[0], "pixel_height": px[1], "dpi": DPI,
        "standard_canvas_width_in": LEAF_SIZE[0], "standard_canvas_height_in": LEAF_SIZE[1],
        "source_data": [str(p.relative_to(PLOT_ROOT)) for p in source_paths],
    })


def save_legend(fig_id: str, name: str, handles, labels, ncol: int):
    LEGENDS.mkdir(parents=True, exist_ok=True)
    width = max(3.0, min(7.0, 0.82 * len(labels)))
    fig = plt.figure(figsize=(width, 0.42))
    fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False,
               handlelength=1.9, columnspacing=1.15, handletextpad=0.45)
    for ext in ("png", "pdf"):
        kwargs = {"dpi": DPI} if ext == "png" else {}
        fig.savefig(LEGENDS / f"{fig_id}__{name}.{ext}", bbox_inches="tight", pad_inches=0.03, **kwargs)
    plt.close(fig)


def line_handles(keys):
    return [Line2D([0], [0], color=COL[k], lw=1.6, marker="o", ms=3.5,
                   ls="--" if k == "random" else "-") for k in keys]


def redraw_fig2(manifest):
    figroot = ATLAS / "Fig02"
    # Fig2c: two frequency-state compositions, no local legend.
    pdir = figroot / "Fig02_c__Heterogeneous_and_sparse_complexity_regimes"
    states = ["high_frequency", "high_intermediate", "intermediate", "low_frequency", "weak_or_irregular"]
    for idx, token in enumerate(["heterogeneity_optimum", "sparse_complexity"], 1):
        src = source_csv(pdir, token)
        d = pd.read_csv(src)
        tab = d.pivot_table(index="complexity_fraction", columns="frequency_state", values="fraction", aggfunc="sum", fill_value=0).sort_index()
        tab = tab.reindex(columns=[s for s in states if s in tab.columns], fill_value=0)
        fig, ax = plt.subplots(figsize=LEAF_SIZE)
        ax.stackplot(tab.index, *[tab[s] for s in tab.columns], colors=[COL[s] for s in tab.columns], alpha=.9, linewidth=0)
        ax.set(xlabel="High-complexity fraction", ylabel="State fraction", ylim=(0, 1))
        clean_axis(ax); fig.tight_layout()
        save_leaf(fig, "Fig02", f"Fig02_c_{idx:02d}", [src], "legend-free frequency-state composition", manifest)
    save_legend("Fig02", "frequency_states", [mpl.patches.Patch(color=COL[s]) for s in states], [LABEL[s] for s in states], 5)

    # Fig2d: two metrics sharing the same transition-window legend.
    pdir = figroot / "Fig02_d__Marginal_resource_efficiency"
    windows = ["frequency_transition", "heterogeneity_optimum", "sparse_complexity"]
    specs = [("complexity_efficiency", "Complexity efficiency"), ("emergence_gain", "Emergence gain")]
    for idx, (metric, ylabel) in enumerate(specs, 1):
        src = source_csv(pdir, metric)
        d = pd.read_csv(src)
        fig, ax = plt.subplots(figsize=LEAF_SIZE)
        for k in windows:
            g = d[d.window.eq(k)].sort_values("complexity_fraction")
            if g.empty: continue
            x, y, ci = g.complexity_fraction.to_numpy(float), g[f"{metric}_mean"].to_numpy(float), g[f"{metric}_ci95"].to_numpy(float)
            ax.plot(x, y, color=COL[k], lw=1.5, marker="o", ms=3.2)
            ax.fill_between(x, y-ci, y+ci, color=COL[k], alpha=.12, lw=0)
        ax.set(xlabel="High-complexity fraction", ylabel=ylabel); clean_axis(ax); fig.tight_layout()
        save_leaf(fig, "Fig02", f"Fig02_d_{idx:02d}", [src], "legend-free Stage1B metric", manifest)
    save_legend("Fig02", "stage1b_windows", line_handles(windows), [LABEL[k] for k in windows], 3)

    # Stage2 method triplet.
    stage2 = ["random", "coverage_greedy", "cycle_proxy", "feedback_hub", "high_degree", "module_bridge", "none"]
    jobs = [
        ("Fig02_e__Placement_gain_relative_to_random", "placement_gain_vs_random", "Fig02_e_01", "Placement gain vs random"),
        ("Fig02_f__Complexity_economy", "gain_per_complex_unit", "Fig02_f_01", "Gain per high-complexity unit"),
        ("Fig02_g__Allocation_landscape", "fraction_of_condition_best", "Fig02_g_02", "Fraction of conditions best"),
    ]
    for dirname, token, leaf_id, ylabel in jobs:
        src = source_csv(figroot / dirname, token)
        d = pd.read_csv(src)
        fig, ax = plt.subplots(figsize=LEAF_SIZE)
        for k in stage2:
            g = d[d.placement.eq(k)].sort_values("complexity_fraction")
            if g.empty: continue
            x, y, ci = g.complexity_fraction.to_numpy(float), g["mean"].to_numpy(float), g["ci95"].to_numpy(float)
            ax.plot(x, y, color=COL[k], lw=1.35, marker="o", ms=2.9, ls="--" if k == "random" else "-")
            ax.fill_between(x, y-ci, y+ci, color=COL[k], alpha=.09, lw=0)
        if token == "placement_gain_vs_random": ax.axhline(0, color="#888", lw=.7, ls=":")
        ax.set(xlabel="High-complexity fraction", ylabel=ylabel); clean_axis(ax); fig.tight_layout()
        save_leaf(fig, "Fig02", leaf_id, [src], "legend-free Stage2 method curve", manifest)
    save_legend("Fig02", "stage2_allocation_methods", line_handles(stage2), [LABEL[k] for k in stage2], 4)

    # Fig2h: four topology panels sharing placement methods.
    pdir = figroot / "Fig02_h__Budget-response_curves_across_network_architectures"
    methods = ["cycle_proxy", "feedback_hub", "gpu_surrogate", "high_degree", "module_bridge", "random", "spectral"]
    for idx, topo in enumerate(["er", "modular", "scale_free", "small_world"], 1):
        src = source_csv(pdir, f"score_vs_budget_{topo}")
        d = pd.read_csv(src)
        fig, ax = plt.subplots(figsize=LEAF_SIZE)
        for k in methods:
            g = d[d.placement.eq(k)].sort_values("complexity_fraction")
            if g.empty: continue
            x, y, ci = g.complexity_fraction.to_numpy(float), g["mean"].to_numpy(float), g["ci95"].to_numpy(float)
            ax.plot(x, y, color=COL[k], lw=1.25, marker="o", ms=2.7, ls="--" if k == "random" else "-")
            ax.fill_between(x, y-ci, y+ci, color=COL[k], alpha=.08, lw=0)
        ax.set(xlabel="High-complexity fraction", ylabel="Collective-dynamics score")
        ax.text(.02, .98, LABEL[topo], transform=ax.transAxes, va="top", ha="left", fontsize=7.2, weight="bold")
        clean_axis(ax); fig.tight_layout()
        save_leaf(fig, "Fig02", f"Fig02_h_{idx:02d}", [src], "legend-free Stage3 topology curve", manifest)
    save_legend("Fig02", "stage3_placement_methods", line_handles(methods), [LABEL[k] for k in methods], 4)


def heatmap_norm(data):
    finite = np.asarray(data, float)[np.isfinite(data)]
    if finite.size == 0: return Normalize(0, 1), "sequential use of CT01 palette"
    if finite.min() < 0 < finite.max() or finite.min() < 0:
        vmax = max(abs(float(finite.min())), abs(float(finite.max())), 1e-12)
        return TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), "zero-centered diverging"
    vmax = max(float(finite.max()), 1e-12)
    # Keeping the symmetric CT01 normalization makes zero exactly neutral
    # white while the observed nonnegative data occupy only the red half.
    return TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), "nonnegative white-to-red half of CT01 palette"


def draw_heatmap(d, rows, cols, values, xlabel, ylabel, cbar_label, fig_id, leaf_id, src, manifest, fmt=".2f"):
    tab = d.pivot_table(index=rows, columns=cols, values=values, aggfunc="mean") if isinstance(rows, str) else d
    data = tab.to_numpy(float)
    norm, mode = heatmap_norm(data)
    fig, ax = plt.subplots(figsize=LEAF_SIZE)
    im = ax.imshow(data, aspect="auto", cmap=HEATMAP_CMAP, norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(tab.shape[1]), [LABEL.get(str(x), str(x).replace("_", " ").title()) for x in tab.columns], rotation=32, ha="right")
    ax.set_yticks(np.arange(tab.shape[0]), [LABEL.get(str(x), str(x).replace("_", " ").title()) for x in tab.index])
    ax.set(xlabel=xlabel, ylabel=ylabel)
    scale = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1e-12)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v): ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=5.6, color="white" if abs(v) > .58*scale else "#243746")
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.035); cb.set_label(cbar_label, fontsize=6.5); cb.ax.tick_params(labelsize=5.6)
    fig.tight_layout()
    save_leaf(fig, fig_id, leaf_id, [src], f"CT01 palette heatmap ({mode})", manifest)


def redraw_heatmaps(manifest):
    # Fig1b CT01.
    pdir = ATLAS / "Fig01" / "Fig01_b__Cross-state_transfer_matrix"
    src = source_csv(pdir, "CT01")
    d = pd.read_csv(src)
    order = ["sparse_drive", "transition_mid", "transition_dense"]
    tab = d.pivot_table(index="target_state", columns="source_state", values="delta_vs_target_home").reindex(index=order, columns=order)
    draw_heatmap(tab, None, None, None, "Source state of optimized allocation", "Evaluation state", "Score difference vs target-state home mask", "Fig01", "Fig01_b_01", src, manifest, ".3f")

    pdir = ATLAS / "Fig02" / "Fig02_g__Allocation_landscape"
    src = source_csv(pdir, "winner_atlas")
    d = pd.read_csv(src)
    tab = d.pivot_table(index="row_label", columns="placement", values="win_fraction", aggfunc="sum", fill_value=0)
    draw_heatmap(tab, None, None, None, "Winning allocation rule", "Network condition", "Winner fraction across budgets", "Fig02", "Fig02_g_01", src, manifest)

    pdir = ATLAS / "Fig02" / "Fig02_i__Cross-topology_synthesis"
    src = source_csv(pdir, "topology_by_allocation_gain")
    d = pd.read_csv(src)
    draw_heatmap(d, "topology", "placement", "placement_gain_vs_random_mean", "Allocation rule", "Topology", "Gain vs random", "Fig02", "Fig02_i_01", src, manifest)

    pdir = ATLAS / "Fig04" / "Fig04_g__State_×_budget_degeneracy_landscape"
    src = source_csv(pdir, "DG09")
    d = pd.read_csv(src).set_index("regime")
    draw_heatmap(d, None, None, None, "Complexity budget k", "Collective regime", "Sampled functional degeneracy", "Fig04", "Fig04_g_01", src, manifest)


def redraw_fig3e(manifest):
    pdir = ATLAS / "Fig03" / "Fig03_e__Gain_landscape_across_topology_and_collective_regime"
    src = source_csv(pdir, "gain_landscape")
    d = pd.read_csv(src); d["budget_fraction"] = d["k"] / d["N"]
    topologies = ["scale_free", "erdos_renyi", "modular", "small_world"]
    regimes = ["sparse_drive", "transition_mid", "transition_dense"]
    for idx, topo in enumerate(topologies, 1):
        fig, ax = plt.subplots(figsize=LEAF_SIZE)
        sub = d[d.topology.eq(topo)]
        for reg in regimes:
            g = sub[sub.regime.eq(reg)].sort_values("budget_fraction")
            if g.empty: continue
            x = g.budget_fraction.to_numpy(float); y = g.mean_gain.to_numpy(float)
            ax.plot(x, y, color=COL[reg], lw=1.45, marker="o", ms=3.0)
            ax.fill_between(x, g.bootstrap95_low.to_numpy(float), g.bootstrap95_high.to_numpy(float), color=COL[reg], alpha=.13, lw=0)
        ax.axhline(0, color="#888", lw=.7, ls="--")
        ax.set(xlabel="High-complexity fraction (k / N)", ylabel="Dynamics-aware gain vs spectral")
        ax.text(.02, .98, LABEL[topo], transform=ax.transAxes, va="top", ha="left", fontsize=7.2, weight="bold")
        clean_axis(ax); fig.tight_layout()
        save_leaf(fig, "Fig03", f"Fig03_e_{idx:02d}", [src], "split legend-free Fig3E topology panel", manifest)
    save_legend("Fig03", "collective_regimes", line_handles(regimes), [LABEL[k] for k in regimes], 3)


def redraw_fig4_shared(manifest):
    figroot = ATLAS / "Fig04"
    pdir = figroot / "Fig04_a__Pairwise_geometry_of_functionally_degenerate_allocations"
    src = source_csv(pdir, "DG01"); d = pd.read_csv(src)
    ks = [8, 32, 64]; kcols = {8:"#355F7F", 32:"#2A9D8F", 64:"#C27628"}
    fig, ax = plt.subplots(figsize=LEAF_SIZE)
    for k in ks:
        g=d[d.k.eq(k)]; ax.scatter(g.jaccard_distance, g.fresh_score_gap, s=8, alpha=.5, color=kcols[k])
    ax.axhline(.02, color="#B75C70", ls="--", lw=.8); ax.text(.98,.94, "epsilon = 0.02", transform=ax.transAxes, ha="right", va="top", fontsize=6, color="#B75C70")
    ax.set(xlabel="Allocation Jaccard distance", ylabel="Fresh validation score gap"); clean_axis(ax); fig.tight_layout()
    save_leaf(fig,"Fig04","Fig04_a_01",[src],"legend-free k geometry; epsilon directly annotated",manifest)

    pdir = figroot / "Fig04_e__Epsilon_robustness_of_functional_degeneracy"
    src = source_csv(pdir, "DG07"); d = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=LEAF_SIZE)
    for k in ks:
        g=d[d.k.eq(k)].sort_values("epsilon"); ax.plot(g.epsilon,g.mean_D,color=kcols[k],lw=1.5,marker="o",ms=3.2)
    ax.set(xlabel="Near-optimal tolerance epsilon",ylabel="Mean sampled degeneracy");clean_axis(ax);fig.tight_layout()
    save_leaf(fig,"Fig04","Fig04_e_01",[src],"legend-free epsilon sensitivity",manifest)
    save_legend("Fig04","complexity_budget",line_handles([])+[Line2D([0],[0],color=kcols[k],lw=1.6,marker="o",ms=3.5) for k in ks],[f"k={k}" for k in ks],3)


def redraw_fig5_shared(manifest):
    figroot = ATLAS / "Fig05"
    defs=["primary_shared","protocol_conservative"]; offsets={defs[0]:-.12,defs[1]:.12}
    jobs=[
        ("Fig05_h__Identity_absorption_and_within_between_decomposition","SYN15","Fig05_h_01","absorption"),
        ("Fig05_i__Local_dynamic_boundary","SYN20","Fig05_i_01","continuous"),
        ("Fig05_i__Local_dynamic_boundary","SYN16","Fig05_i_02","forest"),
        ("Fig05_i__Local_dynamic_boundary","SYN17","Fig05_i_03","frequency"),
        ("Fig05_j__Integrated_local-circuit_adjudication","SYN27","Fig05_j_01","forest"),
    ]
    for dirname,token,leaf_id,kind in jobs:
        src=source_csv(figroot/dirname,token);d=pd.read_csv(src);fig,ax=plt.subplots(figsize=LEAF_SIZE)
        if kind=="frequency":
            for definition in defs:
                g=d[d.definition.eq(definition)].sort_values("freq")
                ax.errorbar(g.freq,g.beta_after_identity,yerr=1.96*g.se_after_identity,color=COL[definition],marker="o",ms=3.2,lw=1.2,capsize=2)
            ax.axhline(0,color="#777",lw=.7);ax.set_xscale("log");ax.set_xticks([5,10,20,50],["5","10","20","50"])
            ax.set(xlabel="Standardized presynaptic train frequency (Hz)",ylabel="Conditional complexity effect")
        elif kind=="continuous":
            endpoints=["psp_amplitude","psc_amplitude","G_norm_50hz"];x=np.arange(len(endpoints))
            for definition in defs:
                g=d[d.definition.eq(definition)].set_index("endpoint").reindex(endpoints)
                ax.scatter(x+offsets[definition],g.fold_local_increment_after,s=22,color=COL[definition])
            ax.axhline(0,color="#777",lw=.7);ax.set_xticks(x,[PRETTY_ENDPOINT[e] for e in endpoints],rotation=24,ha="right")
            ax.set_ylabel("Fold-local grouped delta CV R2")
        elif kind=="absorption":
            endpoints=list(dict.fromkeys(d.endpoint.tolist())); y=np.arange(len(endpoints))
            for definition in defs:
                g=d[d.definition.eq(definition)].set_index("endpoint").reindex(endpoints)
                ax.scatter(g.absorption_fraction,y+offsets[definition],s=20,color=COL[definition])
            ax.axvline(1,color="#777",lw=.7,ls="--");ax.set_yticks(y,[PRETTY_ENDPOINT.get(e,e.replace("_"," ")) for e in endpoints])
            ax.set_xlabel("Identity-absorption fraction of marginal delta R2")
        else:
            endpoints=list(dict.fromkeys(d.endpoint.tolist()));y=np.arange(len(endpoints))
            for definition in defs:
                g=d[d.definition.eq(definition)].set_index("endpoint").reindex(endpoints)
                ax.errorbar(g.beta_after_identity,y+offsets[definition],xerr=1.96*g.se_after_identity,fmt="o",capsize=2,ms=3.2,color=COL[definition])
            ax.axvline(0,color="#777",lw=.7);ax.set_yticks(y,[PRETTY_ENDPOINT.get(e,e.replace("_"," ")) for e in endpoints]);ax.set_xlabel("Identity-conditioned complexity effect (95% CI)")
        clean_axis(ax);fig.tight_layout();save_leaf(fig,"Fig05",leaf_id,[src],"legend-free shared-definition panel",manifest)
    save_legend("Fig05","transfer_definitions",line_handles(defs),[LABEL[k] for k in defs],2)


def leaf_id_from_paths(figdir: Path, paneldir: Path, asset: Path):
    m=re.match(r"(Fig\d{2})_([a-z])",paneldir.name)
    if not m:return None
    idx=re.match(r"(\d+)_",asset.name)
    return f"{m.group(1)}_{m.group(2)}_{int(idx.group(1)) if idx else 1:02d}"


def copy_unmodified(manifest):
    existing={(r["figure"],r["leaf_id"]) for r in manifest}
    for figdir in sorted(ATLAS.glob("Fig??")):
        outdir=LEAF/figdir.name;outdir.mkdir(parents=True,exist_ok=True)
        for paneldir in sorted(figdir.glob("Fig??_*")):
            for png in sorted(paneldir.glob("*.png")):
                leaf_id=leaf_id_from_paths(figdir,paneldir,png)
                if not leaf_id or (figdir.name,leaf_id) in existing:continue
                pdf=png.with_suffix(".pdf")
                outpng=outdir/f"{leaf_id}.png";outpdf=outdir/f"{leaf_id}.pdf"
                shutil.copy2(png,outpng)
                if pdf.exists():shutil.copy2(pdf,outpdf)
                sources=sorted((paneldir/"source_data").glob("*")) if (paneldir/"source_data").exists() else []
                with Image.open(outpng) as im:px=im.size
                manifest.append({"figure":figdir.name,"leaf_id":leaf_id,"role":"unmodified scientific leaf panel","generation":"copied from frozen V5.3","png":str(outpng.relative_to(HERE)),"pdf":str(outpdf.relative_to(HERE)) if outpdf.exists() else "","pixel_width":px[0],"pixel_height":px[1],"dpi":DPI,"standard_canvas_width_in":"source native","standard_canvas_height_in":"source native","source_data":[str(p.relative_to(PLOT_ROOT)) for p in sources]})


def placeholder_image(text, size=(1200,900)):
    fig,ax=plt.subplots(figsize=(4,3),dpi=300);ax.axis("off");ax.add_patch(mpl.patches.Rectangle((.04,.05),.92,.9,fill=False,ls="--",lw=1,color="#999"));ax.text(.5,.5,text,ha="center",va="center",fontsize=9,color="#666",wrap=True);fig.canvas.draw();arr=np.asarray(fig.canvas.buffer_rgba()).copy();plt.close(fig);return Image.fromarray(arr).convert("RGB")


def trim_white(im):
    rgb=im.convert("RGB");bg=Image.new("RGB",rgb.size,"white");diff=ImageChops.difference(rgb,bg).convert("L");box=diff.point(lambda p: 255 if p>8 else 0).getbbox()
    return rgb.crop(box) if box else rgb


def make_references(manifest):
    byfig={}
    for row in manifest:byfig.setdefault(row["figure"],[]).append(row)
    placeholders={"Fig02":[("Fig02_a_00","a  schematic placeholder")],"Fig03":[("Fig03_a_00","a  schematic placeholder")],"Fig06":[("Fig06_j_00","j  schematic placeholder")]}
    legend_files={
        "Fig02":["Fig02__frequency_states.png","Fig02__stage1b_windows.png","Fig02__stage2_allocation_methods.png","Fig02__stage3_placement_methods.png"],
        "Fig03":["Fig03__collective_regimes.png"],"Fig04":["Fig04__complexity_budget.png"],"Fig05":["Fig05__transfer_definitions.png"]}
    # A common four-column grid keeps the physical leaf-panel scale constant
    # across all six 180-mm main figures. Empty cells are intentional.
    cols={f"Fig{i:02d}":4 for i in range(1,7)}
    panel_letters={}
    for fig_id,rows in byfig.items():
        cards=[]
        for leaf_id,text in placeholders.get(fig_id,[]):cards.append((leaf_id,placeholder_image(text),text.split()[0]))
        for r in sorted(rows,key=lambda x:x["leaf_id"]):
            im=trim_white(Image.open(HERE/r["png"])); letter=r["leaf_id"].split("_")[1]; cards.append((r["leaf_id"],im,letter))
        cards.sort(key=lambda x:x[0])
        ncol=cols[fig_id];nrow=math.ceil(len(cards)/ncol);cell_w=1000;cell_h=790;gap=42;top=90;legend_h=0
        legends=[]
        for name in legend_files.get(fig_id,[]):
            p=LEGENDS/name
            if p.exists():legends.append(trim_white(Image.open(p)))
        if legends:legend_h=sum(max(80,int(im.height*min((ncol*cell_w)/im.width,1)))+24 for im in legends)+25
        canvas=Image.new("RGB",(ncol*cell_w+(ncol+1)*gap,nrow*cell_h+(nrow+1)*gap+top+legend_h),"white")
        from PIL import ImageDraw,ImageFont
        title_font=ImageFont.truetype(str(ARIAL_BOLD),28) if ARIAL_BOLD.exists() else ImageFont.load_default()
        label_font=ImageFont.truetype(str(ARIAL_BOLD),25) if ARIAL_BOLD.exists() else ImageFont.load_default()
        draw=ImageDraw.Draw(canvas);draw.text((gap,22),f"{fig_id} - fixed-width script reference (180 mm)",fill="#222",font=title_font)
        for q,(leaf_id,im,letter) in enumerate(cards):
            rr,cc=divmod(q,ncol);x=gap+cc*(cell_w+gap);y=top+gap+rr*(cell_h+gap)
            scale=min((cell_w-20)/im.width,(cell_h-30)/im.height);nw=max(1,int(im.width*scale));nh=max(1,int(im.height*scale));res=im.resize((nw,nh),Image.Resampling.LANCZOS)
            canvas.paste(res,(x+(cell_w-nw)//2,y+(cell_h-nh)//2));draw.text((x,y),letter,fill="#111",font=label_font,stroke_width=0)
        y=nrow*cell_h+(nrow+1)*gap+top
        for im in legends:
            scale=min((ncol*cell_w)/im.width,1);nw=int(im.width*scale);nh=int(im.height*scale);res=im.resize((nw,nh),Image.Resampling.LANCZOS);canvas.paste(res,((canvas.width-nw)//2,y));y+=nh+24
        width_in=REFERENCE_WIDTH_MM/25.4;height_in=width_in*canvas.height/canvas.width
        png=REF/f"{fig_id}_reference.png";pdf=REF/f"{fig_id}_reference.pdf";canvas.save(png,dpi=(300,300),compress_level=6)
        fig=plt.figure(figsize=(width_in,height_in));ax=fig.add_axes([0,0,1,1]);ax.imshow(canvas);ax.axis("off");fig.savefig(pdf,dpi=300);plt.close(fig)


def write_metadata(manifest):
    META.mkdir(parents=True,exist_ok=True)
    with (META/"leaf_panel_manifest.json").open("w",encoding="utf-8") as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    fields=sorted({k for r in manifest for k in r})
    with (META/"leaf_panel_manifest.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in manifest:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()})
    palette={"name":"CT01 blue-white-red","colors":[BLUE,MID,RED],"reference":str((PLOT_ROOT/'Manuscript_Figures_Nature_DataFirst_V5_2/01_MAIN/Fig01/Fig01_b__Cross-state_transfer_matrix/01_CT01_transfer_matrix_centered.png').relative_to(PLOT_ROOT)),"rule":"Signed heatmaps are symmetric around zero; nonnegative heatmaps use zero=white and maximum=red."}
    (META/"heatmap_palette.json").write_text(json.dumps(palette,indent=2),encoding="utf-8")
    (OUT/"README.md").write_text("""# Manuscript figure assembly V5.3\n\n- `leaf_panels/`: one scientific asset per PNG/PDF; repeated-legend groups are redrawn without local legends.\n- `shared_legends/`: standalone shared legend strips.\n- `reference_figures/`: fixed-width (180 mm) script reference layouts for Fig1-Fig6.\n- `metadata/`: source-data lineage and heatmap palette rules.\n\nThe frozen V5.3 atlas is never modified. Fig3E is split into four topology-specific leaves. Heatmaps use the CT01 blue-white-red palette.\n""",encoding="utf-8")


def main():
    setup_style()
    for d in [LEAF,LEGENDS,REF,META]:d.mkdir(parents=True,exist_ok=True)
    manifest=[]
    redraw_fig2(manifest);redraw_heatmaps(manifest);redraw_fig3e(manifest);redraw_fig4_shared(manifest);redraw_fig5_shared(manifest)
    copy_unmodified(manifest);manifest.sort(key=lambda r:(r["figure"],r["leaf_id"]))
    make_references(manifest);write_metadata(manifest)
    print(json.dumps({"status":"COMPLETE","leaf_panels":len(manifest),"reference_figures":6,"output":str(OUT)},indent=2))


if __name__ == "__main__":main()
