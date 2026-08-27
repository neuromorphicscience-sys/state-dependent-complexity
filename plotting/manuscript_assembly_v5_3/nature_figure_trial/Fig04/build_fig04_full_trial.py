#!/usr/bin/env python3
"""Build the source-data-first Nature-width Fig.4 reference figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy import stats


MM_PER_INCH = 25.4
DPI = 600
ROOT = Path(__file__).resolve().parents[3]
ATLAS = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "01_MAIN" / "Fig04"
ED05 = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "02_EXTENDED_DATA" / "ED05"
DEFAULT_OUT = Path(__file__).resolve().parent / "output_full_v1"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

COL = {8: "#38598C", 32: "#2A9D8F", 64: "#C27628"}
BLUE, WHITE, RED = "#2166AC", "#F7F7F7", "#B2182B"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "ct01_blue_white_red", [BLUE, WHITE, RED], N=256)
STATE_LABEL = {
    "sparse_drive": "Sparse",
    "transition_mid": "Transition-mid",
    "transition_dense": "Transition-dense",
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
        "legend.fontsize": 4.0,
        "axes.linewidth": .6,
        "xtick.major.width": .6,
        "ytick.major.width": .6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def add_axes_mm(fig, fw, fh, x, y, w, h):
    return fig.add_axes([x/fw, y/fh, w/fw, h/fh])


def clean_axis(ax, grid=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.5)
    if grid:
        ax.grid(axis=grid, color="#D9DEE1", lw=.35, alpha=.5)
        ax.set_axisbelow(True)


def source_csv(panel: str, token: str) -> Path:
    hits = sorted((ATLAS / panel / "source_data").glob(f"*{token}*.csv"))
    if not hits:
        raise FileNotFoundError(f"No source CSV for {panel}: {token}")
    return hits[0]


def plot_pairwise(ax):
    p = source_csv("Fig04_a__Pairwise_geometry_of_functionally_degenerate_allocations",
                   "pairwise_geometry")
    d = pd.read_csv(p)
    for k in [8, 32, 64]:
        g = d[d.k == k]
        ax.scatter(g.jaccard_distance, g.fresh_score_gap, s=8,
                   color=COL[k], alpha=.55, edgecolor="none", label=f"k={k}")
    ax.axhline(.02, color="#D06F82", lw=.7, ls="--", dashes=(3, 2),
               label="epsilon=0.02")
    ax.set(xlabel="Allocation Jaccard distance",
           ylabel="Fresh validation score gap")
    ax.legend(loc="upper left", frameon=False, fontsize=3.7,
              handlelength=1.1, labelspacing=.16, handletextpad=.35,
              borderaxespad=.25)
    clean_axis(ax)
    return p


def plot_budget_distribution(ax, panel, token, value_col, ylabel, source_path=None):
    p = Path(source_path) if source_path is not None else source_csv(panel, token)
    d = pd.read_csv(p)
    positions = np.arange(3)
    groups = [d.loc[d.k == k, value_col].dropna().to_numpy(float)
              for k in [8, 32, 64]]
    bp = ax.boxplot(groups, positions=positions, widths=.56, patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#263238", "lw": .75},
                    whiskerprops={"color": "#4E585E", "lw": .6},
                    capprops={"color": "#4E585E", "lw": .6},
                    boxprops={"color": "#4E585E", "lw": .6})
    for patch, k in zip(bp["boxes"], [8, 32, 64]):
        patch.set_facecolor(COL[k]); patch.set_alpha(.68)
    for x, (vals, k) in enumerate(zip(groups, [8, 32, 64])):
        offsets = np.linspace(-.12, .12, len(vals))
        ax.scatter(x+offsets, vals, s=5.5, color="#8A9297", alpha=.42,
                   edgecolor="none", zorder=3)
    ax.set_xticks(positions, ["k=8", "k=32", "k=64"], rotation=25,
                  ha="right", fontsize=4.2)
    ax.set_xlabel("Complexity budget")
    ax.set_ylabel(ylabel)
    clean_axis(ax, grid="y")
    return p


def plot_selection_entropy(ax):
    p = (ED05 / "ED05_b__Node-selection_entropy_versus_budget" / "source_data" /
         "01_DG06_selection_entropy_by_budget__DG06_selection_entropy_by_budget_source.csv")
    if not p.exists():
        raise FileNotFoundError(p)
    return plot_budget_distribution(
        ax, None, None, "selection_entropy_union", "Selection entropy",
        source_path=p)


def plot_epsilon(ax):
    p = source_csv("Fig04_e__Epsilon_robustness_of_functional_degeneracy",
                   "epsilon_sensitivity")
    d = pd.read_csv(p)
    for k in [8, 32, 64]:
        g = d[d.k == k].sort_values("epsilon")
        ax.plot(g.epsilon, g.mean_D, color=COL[k], lw=.9, marker="o",
                ms=2.2, label=f"k={k}")
    ax.set(xlabel="Near-optimal tolerance epsilon", ylabel="Mean sampled D")
    ax.legend(loc="upper left", frameon=False, fontsize=3.8,
              handlelength=1.1, labelspacing=.18, handletextpad=.35,
              borderaxespad=.25)
    clean_axis(ax, grid="y")
    return p


def plot_replication(ax):
    p = source_csv("Fig04_f__Independent_replication_of_degeneracy",
                   "independent_replication")
    d = pd.read_csv(p)
    x = d.D_epsilon_mean_jaccard_distance_fresh.to_numpy(float)
    y = d.D_epsilon_mean_jaccard_distance_independent.to_numpy(float)
    ax.scatter(x, y, s=10, color=COL[8], alpha=.7, edgecolor="none")
    ax.plot([0, 1], [0, 1], color="#858E93", lw=.65,
            ls="--", dashes=(3, 2))
    rho, pval = stats.spearmanr(x, y)
    ax.text(.04, .96, f"Spearman rho = {rho:.2f}\np = {pval:.3g}",
            transform=ax.transAxes, ha="left", va="top", fontsize=4.1)
    ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03),
           xlabel="D(0.02), fresh evaluation",
           ylabel="D(0.02), independent evaluation")
    clean_axis(ax)
    return p


def plot_heatmap(ax, fig, fw, fh, cbar_x, cbar_y, cbar_h):
    p = source_csv("Fig04_g__State_×_budget_degeneracy_landscape",
                   "state_budget_degeneracy")
    d = pd.read_csv(p).set_index("regime")
    order = ["sparse_drive", "transition_mid", "transition_dense"]
    data = d.reindex(order)[["8", "32", "64"]].to_numpy(float)
    im = ax.imshow(data, aspect="equal", cmap=HEATMAP_CMAP,
                   norm=Normalize(vmin=0, vmax=1), interpolation="nearest")
    ax.set_xticks(np.arange(3), ["k=8", "k=32", "k=64"],
                  rotation=28, ha="right", fontsize=4.0)
    short_state = ["Sparse", "Trans.-mid", "Trans.-dense"]
    ax.set_yticks(np.arange(3), short_state, fontsize=3.9)
    ax.set(xlabel="Complexity budget", ylabel="Collective state")
    for i in range(3):
        for j in range(3):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=3.7, color="white" if v < .18 or v > .78 else "#243746")
    cax = add_axes_mm(fig, fw, fh, cbar_x, cbar_y, 1.3, cbar_h)
    cb = fig.colorbar(im, cax=cax, ticks=[0, .5, 1])
    cb.ax.tick_params(labelsize=3.7, length=1.3, pad=1)
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cb.ax.set_title("Mean D", fontsize=3.8, pad=1)
    return p


def build(out: Path):
    fw, fh = 183.0, 82.0
    fig = plt.figure(figsize=(fw/MM_PER_INCH, fh/MM_PER_INCH))
    xs = [10.0, 54.0, 98.0, 142.0]
    aw, ah = 34.0, 27.0
    y1, y2 = 48.0, 8.0
    sources = []

    sources.append(plot_pairwise(add_axes_mm(fig, fw, fh, xs[0], y1, aw, ah)))
    sources.append(plot_budget_distribution(
        add_axes_mm(fig, fw, fh, xs[1], y1, aw, ah),
        "Fig04_b__Degeneracy_index_versus_complexity_budget", "D_epsilon_by_budget",
        "D_epsilon_mean_jaccard_distance", "Functional degeneracy D"))
    sources.append(plot_budget_distribution(
        add_axes_mm(fig, fw, fh, xs[2], y1, aw, ah),
        "Fig04_c__Near-optimal_union_expansion", "union_expansion_by_budget",
        "union_expansion", "Union expansion"))
    sources.append(plot_budget_distribution(
        add_axes_mm(fig, fw, fh, xs[3], y1, aw, ah),
        "Fig04_d__Emergent_shared-core_fraction", "core_fraction_by_budget",
        "core_fraction", "Core fraction"))

    sources.append(plot_epsilon(add_axes_mm(fig, fw, fh, xs[0], y2, aw, ah)))
    sources.append(plot_replication(add_axes_mm(fig, fw, fh, xs[1], y2, aw, ah)))
    sources.append(plot_selection_entropy(add_axes_mm(fig, fw, fh, xs[2], y2, aw, ah)))
    sources.append(plot_heatmap(add_axes_mm(fig, fw, fh, 145.0, y2, 27.0, ah),
                                fig, fw, fh, 175.0, y2, ah))

    panel_labels = [
        ("a", 3, y1+28.5), ("b", 47, y1+28.5),
        ("c", 91, y1+28.5), ("d", 135, y1+28.5),
        ("e", 3, y2+28.5), ("f", 47, y2+28.5),
        ("g", 91, y2+28.5), ("h", 138, y2+28.5),
    ]
    for label, x, y in panel_labels:
        fig.text(x/fw, y/fh, label, ha="left", va="bottom",
                 fontsize=8, fontweight="bold")

    out.mkdir(parents=True, exist_ok=True)
    stem = out / "Fig04_full_nature_trial"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"), dpi=DPI,
                pil_kwargs={"compression": "tiff_lzw"})
    (out / "axis_geometry_mm.json").write_text(json.dumps({
        "figure_size_mm": [fw, fh],
        "standard_axis_height_mm": ah,
        "row_order": ["a+b+c+d", "e+f+g+h"],
        "first_row_axis_mm": [aw, ah],
        "second_row_cartesian_axis_mm": [aw, ah],
        "heatmap_axis_mm": [27.0, ah],
        "no_wrapped_panels": ["a", "b", "c", "d", "e", "f", "g", "h"],
    }, indent=2), encoding="utf-8")
    (out / "source_data_manifest.json").write_text(json.dumps({
        "figure": "Fig04",
        "source_data": sorted({str(p.relative_to(ROOT)) for p in sources}),
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
