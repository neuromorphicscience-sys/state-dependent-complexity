#!/usr/bin/env python3
"""Nature-style source-data redraw trial for Fig1a-b."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


MM_PER_INCH = 25.4
DPI = 600

# Frozen semantic colors requested for this manuscript.
STATE_COLORS = {
    "sparse_drive": "#2A9D8F",
    "transition_dense": "#38598C",
    "transition_mid": "#8B7DAA",
}
STATE_LABELS = {
    "sparse_drive": "Sparse drive",
    "transition_dense": "Transition dense",
    "transition_mid": "Transition mid",
}
STATE_SHORT = {
    "sparse_drive": "Sparse",
    "transition_dense": "Transition\ndense",
    "transition_mid": "Transition\nmid",
}
STATE_ORDER = ["sparse_drive", "transition_dense", "transition_mid"]
MATRIX_ORDER = ["sparse_drive", "transition_mid", "transition_dense"]

HEATMAP_BLUE = "#2166AC"
HEATMAP_MID = "#F7F7F7"
HEATMAP_RED = "#B2182B"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "ct01_blue_white_red", [HEATMAP_BLUE, HEATMAP_MID, HEATMAP_RED], N=256
)


def parse_args():
    here = Path(__file__).resolve().parent
    plot_root = here.parents[2]
    default_atlas = plot_root / "Manuscript_Figures_Nature_DataFirst_V5_3" / "01_MAIN" / "Fig01"
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig01-root", type=Path, default=default_atlas)
    ap.add_argument("--out", type=Path, default=here / "output")
    return ap.parse_args()


def setup_style():
    for font_path in [Path("/mnt/c/Windows/Fonts/arial.ttf"), Path("/mnt/c/Windows/Fonts/arialbd.ttf")]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    # Mandatory editable-text and publication-size rules.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update({
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 2.7,
        "ytick.major.size": 2.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def add_panel_label(ax, label):
    ax.text(-0.12, 1.055, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.0, fontweight="bold", clip_on=False)


def load_sources(fig01_root: Path):
    a_hits = sorted((fig01_root / "Fig01_a__State_×_complexity-budget_dependence" / "source_data").glob("*Stage4A_c_state_budget*.csv"))
    b_hits = sorted((fig01_root / "Fig01_b__Cross-state_transfer_matrix" / "source_data").glob("*CT01_transfer_matrix_centered*.csv"))
    if not a_hits or not b_hits:
        raise FileNotFoundError("Required frozen Fig1a/Fig1b source-data tables were not found")
    return a_hits[0], b_hits[0], pd.read_csv(a_hits[0]), pd.read_csv(b_hits[0])


def plot_state_budget(ax, d: pd.DataFrame, show_label=True):
    for state in STATE_ORDER:
        g = d[d["regime"].eq(state)].sort_values("k")
        x = g["k"].to_numpy(float)
        y = g["mean_gain"].to_numpy(float)
        lo = g["ci95_low"].to_numpy(float)
        hi = g["ci95_high"].to_numpy(float)
        color = STATE_COLORS[state]
        ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0, zorder=1)
        ax.plot(x, y, color=color, linewidth=1.65, marker="o", markersize=3.8,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.35,
                label=STATE_LABELS[state], zorder=2)
    ax.axhline(0, color="#8C8C8C", linewidth=0.8, linestyle="--", dashes=(3, 2), zorder=0)
    ax.set_xlabel("Complexity budget, k")
    ax.set_ylabel("Gain vs spectral")
    ax.set_xticks([8, 16, 24, 32, 48, 64])
    ax.set_xlim(5.5, 66.5)
    ax.set_ylim(-0.31, 0.52)
    ax.legend(loc="upper left", ncol=1, borderaxespad=0.15, handlelength=1.8,
              labelspacing=0.32, handletextpad=0.55)
    clean_axis(ax)
    if show_label:
        add_panel_label(ax, "a")


def text_color_for_value(value, norm):
    r, g, b, _ = HEATMAP_CMAP(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.53 else "#243746"


def plot_transfer_heatmap(ax, d: pd.DataFrame, show_label=True):
    tab = d.pivot_table(index="target_state", columns="source_state",
                        values="delta_vs_target_home", aggfunc="mean")
    tab = tab.reindex(index=MATRIX_ORDER, columns=MATRIX_ORDER)
    matrix = tab.to_numpy(float)
    max_abs = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))))
    vmax = math.ceil(max_abs * 100.0) / 100.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(matrix, cmap=HEATMAP_CMAP, norm=norm, aspect="equal", interpolation="nearest")
    labels = [STATE_SHORT[x] for x in MATRIX_ORDER]
    ax.set_xticks(np.arange(3), labels)
    ax.set_yticks(np.arange(3), labels)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xlabel("Source state")
    ax.set_ylabel("Evaluation state")
    for i in range(3):
        for j in range(3):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.3,
                    color=text_color_for_value(value, norm))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
    if show_label:
        add_panel_label(ax, "b")
    return im, vmax


def save_bundle(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=DPI)
    fig.savefig(base.with_suffix(".tiff"), dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def build_individuals(out: Path, da: pd.DataFrame, db: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(105 / MM_PER_INCH, 70 / MM_PER_INCH))
    fig.subplots_adjust(left=.15, right=.98, bottom=.18, top=.95)
    plot_state_budget(ax, da)
    save_bundle(fig, out / "Fig01_a_state_budget")

    fig, ax = plt.subplots(figsize=(70 / MM_PER_INCH, 70 / MM_PER_INCH))
    fig.subplots_adjust(left=.27, right=.80, bottom=.18, top=.89)
    im, vmax = plot_transfer_heatmap(ax, db)
    cax = fig.add_axes([.835, .18, .030, .71])
    cb = fig.colorbar(im, cax=cax, ticks=np.linspace(-vmax, vmax, 5))
    cb.ax.set_title("Δ score\nvs home", fontsize=5.8, pad=3)
    cb.ax.tick_params(labelsize=5.8, length=2.2)
    save_bundle(fig, out / "Fig01_b_transfer_matrix")


def build_combined(out: Path, da: pd.DataFrame, db: pd.DataFrame):
    fig = plt.figure(figsize=(180 / MM_PER_INCH, 70 / MM_PER_INCH))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], left=.075, right=.92,
                          bottom=.18, top=.90, wspace=.43)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    plot_state_budget(axa, da)
    im, vmax = plot_transfer_heatmap(axb, db)
    cb = fig.colorbar(im, ax=axb, fraction=.047, pad=.045, ticks=np.linspace(-vmax, vmax, 5))
    cb.ax.set_title("Δ score\nvs home", fontsize=5.8, pad=3)
    cb.ax.tick_params(labelsize=5.8, length=2.2)
    save_bundle(fig, out / "Fig01_ab_nature_trial")


def write_manifest(out: Path, a_src: Path, b_src: Path):
    manifest = {
        "figure": "Fig01a-b nature-figure trial",
        "backend": "Python/Matplotlib",
        "final_combined_size_mm": [180, 70],
        "palette": {
            "states": STATE_COLORS,
            "heatmap": [HEATMAP_BLUE, HEATMAP_MID, HEATMAP_RED],
            "heatmap_center": 0.0,
        },
        "source_data": {
            "Fig01a": str(a_src),
            "Fig01b": str(b_src),
        },
        "statistics": {
            "Fig01a": "Archived mean_gain and archived ci95_low/ci95_high; no recomputation",
            "Fig01b": "Archived delta_vs_target_home",
        },
        "outputs": [
            "Fig01_a_state_budget", "Fig01_b_transfer_matrix", "Fig01_ab_nature_trial"
        ],
    }
    (out / "source_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    setup_style()
    a_src, b_src, da, db = load_sources(args.fig01_root)
    args.out.mkdir(parents=True, exist_ok=True)
    build_individuals(args.out, da, db)
    build_combined(args.out, da, db)
    write_manifest(args.out, a_src, b_src)
    print(json.dumps({"status": "COMPLETE", "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
