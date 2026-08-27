#!/usr/bin/env python3
"""Build the compact four-microplot-per-row Fig2 reference from V5.3 source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


MM_PER_INCH = 25.4
DPI = 600
ROOT = Path(__file__).resolve().parents[3]
ATLAS = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "01_MAIN" / "Fig02"
ED03 = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "02_EXTENDED_DATA" / "ED03"
DEFAULT_OUT = Path(__file__).resolve().parent / "output_full_v1"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE, WHITE, RED = "#2166AC", "#F7F7F7", "#B2182B"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "ct01_blue_white_red", [BLUE, WHITE, RED], N=256
)

COL = {
    "dose": "#2A9D8F",
    "frequency_transition": "#38598C",
    "heterogeneity_optimum": "#2A9D8F",
    "sparse_complexity": "#C27628",
    "high_frequency": "#4F789D",
    "high_intermediate": "#3D8B7A",
    "intermediate": "#C39A46",
    "low_frequency": "#A94F63",
    "weak_or_irregular": "#8174A8",
    "random": "#737C82",
    "coverage_greedy": "#38598C",
    "cycle_proxy": "#D06F82",
    "feedback_hub": "#5875A4",
    "high_degree": "#8B7DAA",
    "module_bridge": "#6F9D63",
    "gpu_surrogate": "#2A9D8F",
    "spectral": "#2378B8",
}

LABEL = {
    "frequency_transition": "Frequency transition",
    "heterogeneity_optimum": "Heterogeneity optimum",
    "sparse_complexity": "Sparse complexity",
    "high_frequency": "High frequency",
    "high_intermediate": "High-intermediate",
    "intermediate": "Intermediate",
    "low_frequency": "Low frequency",
    "weak_or_irregular": "Weak / irregular",
    "random": "Random",
    "coverage_greedy": "Coverage greedy",
    "cycle_proxy": "Cycle proxy",
    "feedback_hub": "Feedback hub",
    "high_degree": "High degree",
    "module_bridge": "Module bridge",
    "gpu_surrogate": "GPU surrogate",
    "spectral": "Spectral",
    "er": "ER",
    "modular": "Modular",
    "scale_free": "Scale-free",
    "small_world": "Small-world",
}

STATES = [
    "high_frequency", "high_intermediate", "intermediate",
    "low_frequency", "weak_or_irregular",
]
WINDOWS = ["frequency_transition", "heterogeneity_optimum", "sparse_complexity"]
METHODS = [
    "random", "coverage_greedy", "cycle_proxy", "feedback_hub",
    "high_degree", "module_bridge", "gpu_surrogate", "spectral",
]
STAGE2_METHODS = [
    "random", "coverage_greedy", "cycle_proxy", "feedback_hub",
    "high_degree", "module_bridge",
]
STAGE3_METHODS = [
    "random", "cycle_proxy", "feedback_hub", "gpu_surrogate",
    "high_degree", "module_bridge", "spectral",
]
TOPOLOGY_ORDER = ["scale_free", "er", "modular", "small_world"]
TOPOLOGY_COLORS = {
    "scale_free": "#355F7F", "er": "#8BA36A",
    "modular": "#8174A8", "small_world": "#C27628",
}


def setup_style() -> None:
    if ARIAL.exists():
        font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists():
        font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 5.2,
        "axes.labelsize": 5.2,
        "xtick.labelsize": 4.6,
        "ytick.labelsize": 4.6,
        "legend.fontsize": 4.4,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def add_axes_mm(fig, fw, fh, x, y, w, h):
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def draw_resource_allocation_schematic(ax):
    """Draw the approved 34 x 19 mm Fig.2a controlled-design schematic."""
    teal, blue = "#2A9D8F", "#38598C"
    ink, mid, light, pale = "#263238", "#7B858A", "#D9DEE1", "#F2F4F5"
    ax.set_xlim(0, 34); ax.set_ylim(0, 19); ax.axis("off")
    ax.plot([17, 17], [4.15, 15.25], color=light, lw=.48)

    ax.text(8.5, 14.55, r"Total budget, $B_C$", ha="center", va="center",
            fontsize=4.15, fontweight="bold", color=ink)
    ax.text(25.5, 14.55, r"Allocation, $A$", ha="center", va="center",
            fontsize=4.15, fontweight="bold", color=ink)

    def network(cx, cy, high_nodes):
        radius = 1.72
        angles = np.deg2rad([90, 30, -30, -90, -150, 150])
        pts = np.c_[cx + radius*np.cos(angles), cy + radius*np.sin(angles)]
        edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,3),(1,4),(2,5)]
        for i, j in edges:
            ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]],
                    color=light, lw=.48, zorder=1, solid_capstyle="round")
        for i, (x, y) in enumerate(pts):
            high = i in high_nodes
            ax.add_patch(Circle((x, y), .36, facecolor=teal if high else pale,
                                edgecolor=teal if high else mid,
                                linewidth=.48, zorder=3))

    def schematic_arrow(x1, x2, y, double=False):
        ax.add_patch(FancyArrowPatch((x1,y), (x2,y),
                                    arrowstyle="<->" if double else "->",
                                    mutation_scale=5.0, lw=.58, color=mid,
                                    shrinkA=0, shrinkB=0))

    network(4.7, 10.25, {0}); network(12.3, 10.25, {0,1,5})
    schematic_arrow(7.0, 10.0, 10.25)
    ax.text(4.7, 7.45, "Low", ha="center", va="center", fontsize=3.55, color=mid)
    ax.text(12.3, 7.45, "High", ha="center", va="center", fontsize=3.55, color=mid)
    ax.text(8.5, 5.55, r"$B_C$ varies;  $A$ fixed", ha="center", va="center",
            fontsize=3.35, color=mid)

    network(21.7, 10.25, {0,2,4}); network(29.3, 10.25, {1,3,5})
    schematic_arrow(24.0, 27.0, 10.25, double=True)
    ax.text(21.7, 7.45, r"$A_1$", ha="center", va="center", fontsize=3.55, color=mid)
    ax.text(29.3, 7.45, r"$A_2$", ha="center", va="center", fontsize=3.55, color=mid)
    ax.text(25.5, 5.55, r"$A$ varies;  $B_C$ fixed", ha="center", va="center",
            fontsize=3.35, color=mid)

    ax.text(14.5, 2.15, "Controlled intervention", ha="right", va="center",
            fontsize=3.55, color=ink)
    ax.add_patch(FancyArrowPatch((15.0,2.15), (19.0,2.15), arrowstyle="->",
                                mutation_scale=5.2, lw=.72, color=blue,
                                shrinkA=0, shrinkB=0))
    ax.text(19.5, 2.15, "Collective response", ha="left", va="center",
            fontsize=3.55, color=ink)


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.5)


def source_csv(panel: str, token: str) -> Path:
    hits = sorted((ATLAS / panel / "source_data").glob(f"*{token}*.csv"))
    if not hits:
        raise FileNotFoundError(f"No source CSV for {panel}: {token}")
    return hits[0]


def ed03_csv(panel_glob: str, token: str) -> Path:
    panels = sorted(ED03.glob(panel_glob))
    if not panels:
        raise FileNotFoundError(f"No ED03 panel for {panel_glob}")
    hits = sorted((panels[0] / "source_data").glob(f"*{token}*.csv"))
    if not hits:
        raise FileNotFoundError(f"No ED03 source CSV for {token}")
    return hits[0]


def plot_b_single(ax, token, mean_col, ci_col, ylabel, color):
    p = source_csv("Fig02_b__Collective_dose_response", token)
    d = pd.read_csv(p).sort_values("complexity_fraction")
    x = d.complexity_fraction.to_numpy(float)
    y = d[mean_col].to_numpy(float)
    ci = d[ci_col].to_numpy(float)
    ax.plot(x, y, color=color, lw=1.0, marker="o", ms=2.2)
    ax.fill_between(x, y-ci, y+ci, color=color, alpha=.14, lw=0)
    ax.set(xlabel="High-complexity fraction", ylabel=ylabel)
    clean_axis(ax)
    return p


def plot_b_rate_frequency(ax):
    specs = [
        ("mean_rate", "mean_rate_hz_mean", "mean_rate_hz_ci95",
         "Mean firing rate", COL["dose"]),
        ("frequency", "dominant_frequency_hz_mean", "dominant_frequency_hz_ci95",
         "Dominant frequency", COL["frequency_transition"]),
    ]
    paths = []
    for token, mean_col, ci_col, label, color in specs:
        p = source_csv("Fig02_b__Collective_dose_response", token)
        paths.append(p)
        d = pd.read_csv(p).sort_values("complexity_fraction")
        x = d.complexity_fraction.to_numpy(float)
        y = d[mean_col].to_numpy(float)
        ci = d[ci_col].to_numpy(float)
        ax.plot(x, y, color=color, lw=1.0, marker="o", ms=2.1, label=label)
        ax.fill_between(x, y-ci, y+ci, color=color, alpha=.12, lw=0)
    ax.set(xlabel="High-complexity fraction", ylabel="Rate / frequency (Hz)")
    ax.legend(loc="lower right", frameon=False, fontsize=4.1, handlelength=1.15,
              labelspacing=.18, handletextpad=.35, borderaxespad=.25)
    clean_axis(ax)
    return paths


def plot_c(ax, token, title, show_ylabel):
    p = source_csv("Fig02_c__Heterogeneous_and_sparse_complexity_regimes", token)
    d = pd.read_csv(p)
    tab = d.pivot_table(index="complexity_fraction", columns="frequency_state",
                        values="fraction", aggfunc="sum", fill_value=0).sort_index()
    tab = tab.reindex(columns=STATES, fill_value=0)
    ax.stackplot(tab.index, *[tab[s] for s in STATES],
                 colors=[COL[s] for s in STATES], alpha=.9, linewidth=0)
    ax.set(xlabel="High-complexity fraction", ylim=(0, 1))
    ax.set_ylabel("State fraction" if show_ylabel else "")
    ax.text(.88, .05, title, transform=ax.transAxes, va="bottom", ha="right",
            fontsize=4.8, fontweight="bold", color="white")
    clean_axis(ax)
    return p


def plot_d(ax, token, metric, ylabel):
    p = source_csv("Fig02_d__Marginal_resource_efficiency", token)
    d = pd.read_csv(p)
    for key in WINDOWS:
        g = d[d.window.eq(key)].sort_values("complexity_fraction")
        x = g.complexity_fraction.to_numpy(float)
        y = g[f"{metric}_mean"].to_numpy(float)
        ci = g[f"{metric}_ci95"].to_numpy(float)
        ax.plot(x, y, color=COL[key], lw=.9, marker="o", ms=1.8)
        ax.fill_between(x, y-ci, y+ci, color=COL[key], alpha=.10, lw=0)
    ax.set(xlabel="High-complexity fraction", ylabel=ylabel)
    clean_axis(ax)
    return p


def plot_method_curve(ax, panel, token, ylabel, zero=False):
    p = source_csv(panel, token)
    d = pd.read_csv(p)
    for key in METHODS:
        if key not in set(d.placement):
            continue
        g = d[d.placement.eq(key)].sort_values("complexity_fraction")
        x = g.complexity_fraction.to_numpy(float)
        y = g["mean"].to_numpy(float)
        ci = g["ci95"].to_numpy(float)
        ax.plot(x, y, color=COL[key], lw=.8, marker="o", ms=1.6,
                ls="--" if key == "random" else "-")
        ax.fill_between(x, y-ci, y+ci, color=COL[key], alpha=.065, lw=0)
    if zero:
        ax.axhline(0, color="#858C91", lw=.6, ls="--", dashes=(3, 2))
    ax.set(xlabel="High-complexity fraction", ylabel=ylabel)
    clean_axis(ax)
    return p


def plot_forest_f2(ax):
    p = source_csv("Fig02_f__Complexity_economy", "minimum_complexity")
    d = pd.read_csv(p)
    order = [k for k in METHODS if k in set(d.placement)]
    d = d.set_index("placement").reindex(order)
    y = np.arange(len(order))[::-1]
    for yy, key in zip(y, order):
        row = d.loc[key]
        ax.errorbar(row["mean"], yy, xerr=row["ci95"], fmt="o", ms=2.3,
                    color=COL[key], ecolor=COL[key], elinewidth=.65, capsize=1.5)
    ax.set_yticks(y, [LABEL[k] for k in order], fontsize=3.7)
    ax.set_xlabel("Minimum fraction for 80% of best")
    clean_axis(ax)
    return p


def heatmap_norm(values, nonnegative=False):
    vmax = max(float(np.nanmax(np.abs(values))), 1e-12)
    if nonnegative:
        return TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    return TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)


def plot_g1(ax, fig, fw, fh, cbar_x, cbar_y):
    p = source_csv("Fig02_g__Allocation_landscape", "winner_atlas")
    d = pd.read_csv(p)
    tab = d.pivot_table(index="row_label", columns="placement", values="win_fraction",
                        aggfunc="sum", fill_value=0)
    preferred = [k for k in METHODS if k in tab.columns]
    tab = tab.reindex(columns=preferred)
    data = tab.to_numpy(float)
    im = ax.imshow(data, aspect="auto", cmap=HEATMAP_CMAP,
                   norm=heatmap_norm(data, nonnegative=True), interpolation="nearest")
    ax.set_xticks(np.arange(tab.shape[1]), [LABEL[k] for k in tab.columns],
                  rotation=42, ha="right", fontsize=3.3)
    ax.set_yticks(np.arange(tab.shape[0]),
                  [str(x).replace(" | ", " |\n") for x in tab.index], fontsize=3.25)
    ax.set(xlabel="Winning allocation rule", ylabel="Network condition")
    scale = max(float(np.nanmax(data)), 1e-12)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=2.7,
                    color="white" if v > .58*scale else "#243746")
    cax = add_axes_mm(fig, fw, fh, cbar_x, cbar_y, 1.25, 20)
    cb = fig.colorbar(im, cax=cax, ticks=[0, scale])
    cb.ax.tick_params(labelsize=3.5, length=1.3, pad=1)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb.ax.set_title("Winner\nfraction", fontsize=3.7, pad=1)
    return p


def plot_i(ax, fig, fw, fh, cbar_x, cbar_y):
    p = source_csv("Fig02_i__Cross-topology_synthesis", "topology_by_allocation_gain")
    d = pd.read_csv(p)
    topo_order = ["er", "modular", "scale_free", "small_world"]
    place_order = [k for k in METHODS if k in set(d.placement)]
    tab = d.pivot_table(index="topology", columns="placement",
                        values="placement_gain_vs_random_mean").reindex(
                            index=topo_order, columns=place_order)
    data = tab.to_numpy(float)
    vmax = max(float(np.nanmax(np.abs(data))), 1e-12)
    im = ax.imshow(data, aspect="auto", cmap=HEATMAP_CMAP,
                   norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                   interpolation="nearest")
    ax.set_xticks(np.arange(tab.shape[1]), [LABEL[k] for k in tab.columns],
                  rotation=42, ha="right", fontsize=3.2)
    ax.set_yticks(np.arange(tab.shape[0]), [LABEL[k] for k in tab.index],
                  fontsize=3.7, rotation=45, ha="right", va="center",
                  rotation_mode="anchor")
    ax.set(xlabel="Allocation rule", ylabel="Topology")
    ax.yaxis.labelpad = .5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=2.8,
                        color="white" if abs(v) > .58*vmax else "#243746")
    cax = add_axes_mm(fig, fw, fh, cbar_x, cbar_y, 1.25, 20)
    cb = fig.colorbar(im, cax=cax, ticks=[-vmax, 0, vmax])
    cb.ax.tick_params(labelsize=3.4, length=1.3, pad=1)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb.ax.set_title("Gain vs\nrandom", fontsize=3.7, pad=1)
    return p


def plot_h(ax, topo, show_ylabel, shared_ylim):
    p = source_csv("Fig02_h__Budget-response_curves_across_network_architectures",
                   f"score_vs_budget_{topo}")
    d = pd.read_csv(p)
    for key in METHODS:
        if key not in set(d.placement):
            continue
        g = d[d.placement.eq(key)].sort_values("complexity_fraction")
        x, y = g.complexity_fraction.to_numpy(float), g["mean"].to_numpy(float)
        ci = g["ci95"].to_numpy(float)
        ax.plot(x, y, color=COL[key], lw=.78, marker="o", ms=1.5,
                ls="--" if key == "random" else "-")
        ax.fill_between(x, y-ci, y+ci, color=COL[key], alpha=.055, lw=0)
    ax.set(xlabel="High-complexity fraction", ylabel="Collective score",
           ylim=shared_ylim)
    ax.text(.02, .96, LABEL[topo], transform=ax.transAxes, va="top", ha="left",
            fontsize=4.8, fontweight="bold")
    clean_axis(ax)
    return p


def plot_saving_distribution(ax):
    p = ed03_csv("ED03_c__*", "complexity_saving_by_topology")
    d = pd.read_csv(p)
    groups = [d.loc[d.topology.eq(k), "complexity_saving_fraction"].dropna().to_numpy(float)
              for k in TOPOLOGY_ORDER]
    bp = ax.boxplot(groups, positions=np.arange(4), widths=.55, patch_artist=True,
                    showfliers=False, medianprops={"color": "#263238", "lw": .8},
                    whiskerprops={"color": "#4E585E", "lw": .6},
                    capprops={"color": "#4E585E", "lw": .6},
                    boxprops={"color": "#4E585E", "lw": .6})
    for patch, key in zip(bp["boxes"], TOPOLOGY_ORDER):
        patch.set_facecolor(TOPOLOGY_COLORS[key]); patch.set_alpha(.48)
    for idx, (vals, key) in enumerate(zip(groups, TOPOLOGY_ORDER)):
        jitter = np.linspace(-.16, .16, len(vals))
        ax.scatter(idx+jitter, vals, s=1.4, color=TOPOLOGY_COLORS[key], alpha=.08,
                   edgecolor="none")
    ax.axhline(0, color="#777", lw=.6, ls="--", dashes=(3, 2))
    ax.set_xticks(np.arange(4), [LABEL[k] for k in TOPOLOGY_ORDER],
                  rotation=28, ha="right", fontsize=3.8)
    ax.set_xlabel("Network topology")
    ax.set_ylabel("Saving fraction")
    handles = [Line2D([0], [0], linestyle="none", marker="o", ms=2.8,
                      markerfacecolor=TOPOLOGY_COLORS[k], markeredgecolor="none")
               for k in TOPOLOGY_ORDER]
    ax.legend(handles, [LABEL[k] for k in TOPOLOGY_ORDER], loc="lower left",
              ncol=1, frameon=False, fontsize=3.15, handlelength=.7,
              labelspacing=.12, handletextpad=.25, borderaxespad=.25)
    clean_axis(ax)
    return p


def plot_positive_saving(ax):
    p = ed03_csv("ED03_d__*", "positive_saving_fraction")
    d = pd.read_csv(p).set_index("topology").reindex(TOPOLOGY_ORDER)
    x = np.arange(4)
    for xx, key in zip(x, TOPOLOGY_ORDER):
        y = float(d.loc[key, "positive_fraction"])
        ax.vlines(xx, 0, y, color=TOPOLOGY_COLORS[key], lw=2.0, alpha=.65)
        ax.scatter(xx, y, s=22, facecolor="white", edgecolor=TOPOLOGY_COLORS[key],
                   linewidth=.9, zorder=3)
    ax.set_xticks(x, [LABEL[k] for k in TOPOLOGY_ORDER], rotation=28,
                  ha="right", fontsize=3.8)
    ax.set_xlabel("Network topology")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Positive fraction")
    handles = [Line2D([0], [0], linestyle="none", marker="o", ms=2.8,
                      markerfacecolor="white", markeredgecolor=TOPOLOGY_COLORS[k],
                      markeredgewidth=.8) for k in TOPOLOGY_ORDER]
    ax.legend(handles, [LABEL[k] for k in TOPOLOGY_ORDER], loc="upper right",
              ncol=1, frameon=False, fontsize=3.15, handlelength=.7,
              labelspacing=.12, handletextpad=.25, borderaxespad=.25)
    clean_axis(ax)
    return p


def add_shared_legends(fig):
    state_handles = [Patch(facecolor=COL[k], edgecolor="none") for k in STATES]
    window_handles = [Line2D([0], [0], color=COL[k], lw=1.0, marker="o", ms=2.2)
                      for k in WINDOWS]
    shared_methods = [k for k in METHODS if k in set(STAGE2_METHODS + STAGE3_METHODS)]
    method_handles = [Line2D([0], [0], color=COL[k], lw=1.0, marker="o", ms=2.0,
                             ls="--" if k == "random" else "-") for k in shared_methods]
    common = dict(frameon=False, mode=None, handlelength=1.2, handletextpad=.3,
                  columnspacing=.8, borderaxespad=0)
    fig.legend(state_handles, [LABEL[k] for k in STATES], ncol=5,
               loc="lower center", bbox_to_anchor=(49/183, 129.2/170),
               fontsize=3.9, **common)
    fig.legend(window_handles, [LABEL[k] for k in WINDOWS], ncol=3,
               loc="lower center", bbox_to_anchor=(137/183, 129.2/170),
               fontsize=4.0, **common)
    fig.legend(method_handles, [LABEL[k] for k in shared_methods], ncol=9,
               loc="lower center", bbox_to_anchor=(.5, 67.0/170),
               fontsize=3.75, **common)


def build(out: Path):
    fw, fh = 183.0, 170.0
    fig = plt.figure(figsize=(fw/MM_PER_INCH, fh/MM_PER_INCH))
    xs = [10.0, 54.0, 98.0, 142.0]
    ys = [141.0, 110.0, 79.0, 46.0, 13.0]
    aw, ah = 34.0, 19.0
    sources = []

    # Row 1: a and the three b microplots use four equal normal-sized slots.
    axa = add_axes_mm(fig, fw, fh, xs[0], ys[0], aw, ah)
    draw_resource_allocation_schematic(axa)
    sources.append(ATLAS / "Fig02_a__Controlled_resource-versus-allocation_design" /
                   "00_SCHEMATIC_BRIEF.txt")

    axb1 = add_axes_mm(fig, fw, fh, xs[1], ys[0], aw, ah)
    sources.extend(plot_b_rate_frequency(axb1))
    axb2 = add_axes_mm(fig, fw, fh, xs[2], ys[0], aw, ah)
    sources.append(plot_b_single(axb2, "synchrony", "synchrony_proxy_mean",
                                 "synchrony_proxy_ci95", "Synchrony proxy",
                                 "#8B7DAA"))
    axb3 = add_axes_mm(fig, fw, fh, xs[3], ys[0], aw, ah)
    sources.append(plot_b_single(axb3, "rhythm_score", "rhythm_score_mean",
                                 "rhythm_score_ci95", "Rhythm score",
                                 "#2A9D8F"))

    # Row 2: c pair followed by d pair; neither pair wraps.
    for x, token, title, show_y in [
        (xs[0], "heterogeneity_optimum", "Heterogeneity optimum", True),
        (xs[1], "sparse_complexity", "Sparse complexity", False),
    ]:
        ax = add_axes_mm(fig, fw, fh, x, ys[1], aw, ah)
        sources.append(plot_c(ax, token, title, show_y))
    for x, token, metric, ylabel in [
        (xs[2], "complexity_efficiency", "complexity_efficiency", "Complexity efficiency"),
        (xs[3], "emergence_gain", "emergence_gain", "Emergence gain"),
    ]:
        ax = add_axes_mm(fig, fw, fh, x, ys[1], aw, ah)
        sources.append(plot_d(ax, token, metric, ylabel))

    # Row 3: e, the f pair, and the g curve form a four-wide Stage2 strip.
    axe = add_axes_mm(fig, fw, fh, xs[0], ys[2], aw, ah)
    sources.append(plot_method_curve(
        axe, "Fig02_e__Placement_gain_relative_to_random",
        "placement_gain_vs_random", "Placement gain vs random", zero=True))
    axf1 = add_axes_mm(fig, fw, fh, xs[1], ys[2], aw, ah)
    sources.append(plot_method_curve(
        axf1, "Fig02_f__Complexity_economy", "gain_per_complex_unit",
        "Gain per high-complexity unit"))
    axf2 = add_axes_mm(fig, fw, fh, xs[2], ys[2], aw, ah)
    sources.append(plot_forest_f2(axf2))
    axg2 = add_axes_mm(fig, fw, fh, xs[3], ys[2], aw, ah)
    sources.append(plot_method_curve(
        axg2, "Fig02_g__Allocation_landscape", "fraction_of_condition_best",
        "Fraction best"))

    # Row 4: h remains an indivisible four-microplot Stage3 strip.
    h_paths = [source_csv("Fig02_h__Budget-response_curves_across_network_architectures",
                          f"score_vs_budget_{t}") for t in
               ["er", "modular", "scale_free", "small_world"]]
    h_max = 0.0
    for p in h_paths:
        d = pd.read_csv(p)
        h_max = max(h_max, float(np.nanmax(d["mean"] + d["ci95"])))
    h_ylim = (0, np.ceil(h_max*10)/10)
    h_xs, h_aw = [10.0, 54.0, 98.0, 142.0], 35.5
    for idx, (x, topo) in enumerate(zip(h_xs, ["er", "modular", "scale_free", "small_world"])):
        ax = add_axes_mm(fig, fw, fh, x, ys[3], h_aw, ah)
        sources.append(plot_h(ax, topo, True, h_ylim))

    # Row 5: four-part allocation synthesis (two main heatmaps + two ED03 summaries).
    axg1 = add_axes_mm(fig, fw, fh, xs[0], ys[4], aw-3.0, ah)
    sources.append(plot_g1(axg1, fig, fw, fh, xs[0]+30.0, ys[4]))
    axi = add_axes_mm(fig, fw, fh, xs[1], ys[4], aw-3.0, ah)
    sources.append(plot_i(axi, fig, fw, fh, xs[1]+31.8, ys[4]))
    synthesis_aw = 32.0
    axsave = add_axes_mm(fig, fw, fh, xs[2]+1.0, ys[4], synthesis_aw, ah)
    sources.append(plot_saving_distribution(axsave))
    axprev = add_axes_mm(fig, fw, fh, xs[3]+1.0, ys[4], synthesis_aw, ah)
    sources.append(plot_positive_saving(axprev))

    panel_labels = [
        ("a", 3, ys[0]+20.5), ("b", 47, ys[0]+20.5),
        ("c", 3, ys[1]+20.5), ("d", 91, ys[1]+20.5),
        ("e", 3, ys[2]+20.5), ("f", 47, ys[2]+20.5), ("g", 135, ys[2]+20.5),
        ("h", 3, ys[3]+20.5),
        ("i", 3, ys[4]+20.5), ("j", 91, ys[4]+20.5), ("k", 135, ys[4]+20.5),
    ]
    for label, x, y in panel_labels:
        fig.text(x/fw, y/fh, label, ha="left", va="bottom", fontsize=8,
                 fontweight="bold")

    add_shared_legends(fig)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "Fig02_full_nature_trial"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"), dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})

    geometry = {
        "figure_size_mm": [fw, fh],
        "standard_micro_axis_mm": [aw, ah],
        "b_micro_axis_mm": [aw, ah],
        "b_internal_gap_mm": 10.0,
        "b_microplot_count": 3,
        "h_micro_axis_mm": [h_aw, ah],
        "h_internal_gap_mm": 8.5,
        "grid_columns": 4,
        "row_order": ["a+b1+b2+b3", "c1+c2+d1+d2", "e+f1+f2+g_curve",
                      "h1+h2+h3+h4", "i1+i2+i3+i4"],
        "no_wrapped_panels": ["b", "c", "d", "f", "h", "i"],
    }
    (out / "axis_geometry_mm.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    (out / "source_data_manifest.json").write_text(json.dumps({
        "figure": "Fig02", "source_data": sorted({str(p.relative_to(ROOT)) for p in sources})
    }, indent=2), encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"status": "COMPLETE", "output": str(out)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    setup_style()
    build(args.out)


if __name__ == "__main__":
    main()
