#!/usr/bin/env python3
"""Source-data redraw of the complete six-panel Fig1 trial."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from build_fig01_ab_trial import (
    DPI,
    HEATMAP_CMAP,
    MATRIX_ORDER,
    MM_PER_INCH,
    STATE_COLORS,
    STATE_LABELS,
    clean_axis,
    save_bundle,
    setup_style,
    text_color_for_value,
)


STATE_TICKS = {
    "sparse_drive": "Sparse",
    "transition_mid": "Transition\nmid",
    "transition_dense": "Transition\ndense",
}
INTERACTION = "#C97C7C"
PAIR_LINE = "#C7CFD8"
NEUTRAL_DARK = "#65727C"
BUDGET_COLORS = ["#D9E1E8", "#9FB0BE", "#667D8C"]


def parse_args():
    here = Path(__file__).resolve().parent
    plot_root = here.parents[2]
    default_root = plot_root / "Manuscript_Figures_Nature_DataFirst_V5_3" / "01_MAIN" / "Fig01"
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig01-root", type=Path, default=default_root)
    parser.add_argument("--out", type=Path, default=here / "output_full_v2")
    return parser.parse_args()


def add_panel_label(ax, label, x=-0.12, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.0, fontweight="bold", clip_on=False)


def find_source(panel_dir: Path, token: str) -> Path:
    hits = sorted((panel_dir / "source_data").glob(f"*{token}*.csv"))
    if not hits:
        raise FileNotFoundError(f"Missing source data for {panel_dir.name}: {token}")
    return hits[0]


def load_all_sources(root: Path):
    specs = {
        "a": ("Fig01_a__State_×_complexity-budget_dependence", "Stage4A_c_state_budget"),
        "b": ("Fig01_b__Cross-state_transfer_matrix", "CT01_transfer_matrix_centered"),
        "c": ("Fig01_c__Matched-topology_crossover_across_budgets", "CT02_crossover_by_budget"),
        "d": ("Fig01_d__Home_allocation_advantage", "CT03_home_advantage_paired"),
        "e": ("Fig01_e__Home-versus-foreign_score_geometry", "CT07_home_vs_foreign_scores"),
        "f": ("Fig01_f__Replicate-level_validation_of_the_state_crossover", "CT04_replicate_crossover_distribution"),
    }
    paths, frames = {}, {}
    for key, (folder, token) in specs.items():
        paths[key] = find_source(root / folder, token)
        frames[key] = pd.read_csv(paths[key])
    return paths, frames


def plot_a(ax, d, label=True):
    order = ["sparse_drive", "transition_dense", "transition_mid"]
    for state in order:
        g = d[d["regime"].eq(state)].sort_values("k")
        x = g["k"].to_numpy(float)
        y = g["mean_gain"].to_numpy(float)
        lo = g["ci95_low"].to_numpy(float)
        hi = g["ci95_high"].to_numpy(float)
        color = STATE_COLORS[state]
        ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=1)
        ax.plot(x, y, color=color, linewidth=0.95, marker="o", markersize=3.6,
                markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.45,
                label={"sparse_drive": "Sparse", "transition_dense": "Transition dense",
                       "transition_mid": "Transition mid"}[state], zorder=2)
    ax.axhline(0, color="#858C91", linewidth=0.75, linestyle="--", dashes=(3, 2), zorder=0)
    ax.set(xlabel="Complexity budget, k", ylabel="Gain vs spectral", xlim=(5.5, 66.5), ylim=(-0.31, 0.52))
    ax.set_xticks([8, 16, 24, 32, 48, 64])
    ax.legend(loc="upper left", ncol=3, borderaxespad=0.15, handlelength=1.35,
              columnspacing=0.75, handletextpad=0.35, fontsize=5.25)
    clean_axis(ax)
    if label:
        add_panel_label(ax, "a")


def plot_b(ax, d, label=True):
    tab = d.pivot_table(index="target_state", columns="source_state",
                        values="delta_vs_target_home", aggfunc="mean")
    matrix = tab.reindex(index=MATRIX_ORDER, columns=MATRIX_ORDER).to_numpy(float)
    max_abs = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))))
    vmax = math.ceil(max_abs * 100) / 100
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap=HEATMAP_CMAP, norm=norm, aspect="equal", interpolation="nearest")
    labels = ["Sparse", "T-mid", "T-dense"]
    ax.set_xticks(np.arange(3), labels, rotation=28, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(3), labels)
    ax.tick_params(axis="both", length=0, labelsize=5.3)
    ax.set_xlabel("Source state")
    ax.set_ylabel("Evaluation state")
    for i in range(3):
        for j in range(3):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=5.8,
                    color=text_color_for_value(value, norm))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
    if label:
        add_panel_label(ax, "b", x=-0.25)
    return im, vmax


def plot_c(ax, d, label=True):
    for _, g in d.sort_values("k").groupby("graph_seed"):
        ax.plot(g["k"], g["crossover_interaction"], color=PAIR_LINE, linewidth=0.65,
                marker="o", markersize=2.8, markerfacecolor="#A8B0B6",
                markeredgecolor="#7F888E", markeredgewidth=0.3, zorder=1)
    mean = d.groupby("k", as_index=False)["crossover_interaction"].mean().sort_values("k")
    ax.plot(mean["k"], mean["crossover_interaction"], color=INTERACTION, linewidth=0.95,
            marker="o", markersize=3.6, markeredgecolor=INTERACTION, markeredgewidth=0.45,
            zorder=3)
    ax.axhline(0, color="#858C91", linewidth=0.75, linestyle="--", dashes=(3, 2), zorder=0)
    ax.set(xlabel="Complexity budget, k", ylabel="Crossover",
           xlim=(5.5, 66.5), ylim=(-0.004, 0.093))
    ax.set_xticks([8, 32, 64])
    clean_axis(ax)
    if label:
        add_panel_label(ax, "c")


def plot_d(ax, d, label=True):
    for _, row in d.iterrows():
        ys = [row["mid_home_advantage"], row["sparse_home_advantage"]]
        ax.plot([0, 1], ys, color=PAIR_LINE, linewidth=0.65, zorder=1)
    ax.scatter(np.zeros(len(d)), d["mid_home_advantage"], s=18,
               color=STATE_COLORS["transition_mid"], edgecolor="white", linewidth=0.35, zorder=2)
    ax.scatter(np.ones(len(d)), d["sparse_home_advantage"], s=18,
               color=STATE_COLORS["sparse_drive"], edgecolor="white", linewidth=0.35, zorder=2)
    ax.axhline(0, color="#858C91", linewidth=0.75, linestyle="--", dashes=(3, 2), zorder=0)
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.015, 0.145)
    ax.set_xticks([0, 1], ["Transition-mid", "Sparse"])
    ax.set_xlabel("Collective state")
    ax.set_ylabel("Home-state advantage")
    clean_axis(ax)
    if label:
        add_panel_label(ax, "d")


def plot_e(ax, d, label=True):
    for state in ["sparse_drive", "transition_mid"]:
        g = d[d["side"].eq(state)]
        ax.scatter(g["foreign"], g["home"], s=15, color=STATE_COLORS[state], alpha=0.80,
                   edgecolor=STATE_COLORS[state], linewidth=0.35, zorder=2)
    lo = min(float(d["foreign"].min()), float(d["home"].min())) - 0.015
    hi = max(float(d["foreign"].max()), float(d["home"].max())) + 0.015
    ax.plot([lo, hi], [lo, hi], color="#7E878D", linewidth=0.9,
            linestyle="--", dashes=(4, 2), zorder=1)
    ax.set(xlabel="Foreign-state score", ylabel="Home-state score",
           xlim=(lo, hi), ylim=(lo, hi))
    clean_axis(ax)
    if label:
        add_panel_label(ax, "e")


def plot_f(ax, d, label=True):
    budgets = [8, 32, 64]
    data = [d.loc[d["k"].eq(k), "crossover_interaction"].to_numpy(float) for k in budgets]
    boxes = ax.boxplot(data, positions=np.arange(3), widths=0.55, patch_artist=True,
                       showfliers=False, medianprops={"color": "#24323A", "linewidth": 1.0},
                       boxprops={"color": "#3F4A50", "linewidth": 0.8},
                       whiskerprops={"color": "#3F4A50", "linewidth": 0.8},
                       capprops={"color": "#3F4A50", "linewidth": 0.8})
    for patch, color in zip(boxes["boxes"], BUDGET_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    for i, values in enumerate(data):
        jitter = np.linspace(-0.11, 0.11, len(values))
        ax.scatter(i + jitter, values, s=11, color="#727B81", alpha=0.62,
                   edgecolor="none", zorder=3)
    ax.axhline(0, color="#858C91", linewidth=0.75, linestyle="--", dashes=(3, 2), zorder=0)
    ax.set_xticks(np.arange(3), [str(k) for k in budgets])
    ax.set_xlabel("Complexity budget, k")
    ax.set_ylabel("Replicate crossover")
    ax.set_ylim(-0.09, 0.225)
    clean_axis(ax)
    if label:
        add_panel_label(ax, "f")


def add_axes_mm(fig, fig_width_mm, fig_height_mm, x_mm, y_mm, width_mm, height_mm):
    return fig.add_axes([
        x_mm / fig_width_mm,
        y_mm / fig_height_mm,
        width_mm / fig_width_mm,
        height_mm / fig_height_mm,
    ])


def build_combined(out: Path, frames):
    fig_width_mm, fig_height_mm = 183.0, 118.0
    fig = plt.figure(figsize=(fig_width_mm / MM_PER_INCH, fig_height_mm / MM_PER_INCH))
    axa = add_axes_mm(fig, fig_width_mm, fig_height_mm, 12, 65.5, 45, 36)
    axb = add_axes_mm(fig, fig_width_mm, fig_height_mm, 70, 65.5, 36, 36)
    axc = add_axes_mm(fig, fig_width_mm, fig_height_mm, 128, 65.5, 45, 36)
    axd = add_axes_mm(fig, fig_width_mm, fig_height_mm, 12, 16.5, 45, 36)
    axe = add_axes_mm(fig, fig_width_mm, fig_height_mm, 70, 16.5, 45, 36)
    axf = add_axes_mm(fig, fig_width_mm, fig_height_mm, 128, 16.5, 45, 36)
    plot_a(axa, frames["a"], label=False)
    im, vmax = plot_b(axb, frames["b"], label=False)
    plot_c(axc, frames["c"], label=False)
    plot_d(axd, frames["d"], label=False)
    plot_e(axe, frames["e"], label=False)
    plot_f(axf, frames["f"], label=False)
    handles, labels = axa.get_legend_handles_labels()
    if axa.legend_ is not None:
        axa.legend_.remove()
    axa.legend(handles, labels, loc="upper left", ncol=1, frameon=False,
               borderaxespad=0.15, handlelength=1.25, labelspacing=0.22,
               handletextpad=0.35, fontsize=5.3)
    cax = add_axes_mm(fig, fig_width_mm, fig_height_mm, 109, 65.5, 2.5, 36)
    cb = fig.colorbar(im, cax=cax, ticks=[-vmax, 0, vmax])
    cb.ax.set_title("Δ vs\nhome", fontsize=5.0, pad=2)
    cb.ax.tick_params(labelsize=5.2, length=1.8, pad=1.5)
    seed_handle = Line2D([0], [0], color=PAIR_LINE, linewidth=0.65, marker="o",
                         markersize=3.2, markerfacecolor="#A8B0B6",
                         markeredgecolor="#7F888E", markeredgewidth=0.35)
    mean_handle = Line2D([0], [0], color=INTERACTION, linewidth=0.95, marker="o",
                         markersize=3.4, markerfacecolor=INTERACTION,
                         markeredgecolor=INTERACTION, markeredgewidth=0.45)
    mid_handle = Line2D([0], [0], linestyle="none", marker="o", markersize=3.6,
                        markerfacecolor=STATE_COLORS["transition_mid"],
                        markeredgecolor=STATE_COLORS["transition_mid"])
    sparse_handle = Line2D([0], [0], linestyle="none", marker="o", markersize=3.6,
                           markerfacecolor=STATE_COLORS["sparse_drive"],
                           markeredgecolor=STATE_COLORS["sparse_drive"])
    inside_legend_kw = dict(ncol=1, frameon=False, fontsize=5.2,
                            handlelength=1.15, labelspacing=0.22,
                            handletextpad=0.35, borderaxespad=0.25)
    axc.legend([seed_handle, mean_handle], ["Graph seed", "Mean"],
               loc="lower right", bbox_to_anchor=(1.0, 0.045),
               **inside_legend_kw)
    axd.legend([mid_handle, sparse_handle], ["Transition-mid", "Sparse"],
               loc="upper left", **inside_legend_kw)
    axe.legend([mid_handle, sparse_handle], ["Transition-mid", "Sparse"],
               loc="upper left", **inside_legend_kw)
    for panel, x_mm, y_mm in [
        ("a", 4, 103.5), ("b", 62, 103.5), ("c", 120, 103.5),
        ("d", 4, 54.5), ("e", 62, 54.5), ("f", 120, 54.5),
    ]:
        fig.text(x_mm / fig_width_mm, y_mm / fig_height_mm, panel,
                 ha="left", va="bottom", fontsize=8.0, fontweight="bold")
    fig.canvas.draw()
    geometry = {}
    for panel, ax in [("a", axa), ("b", axb), ("c", axc),
                      ("d", axd), ("e", axe), ("f", axf)]:
        pos = ax.get_position()
        geometry[panel] = {
            "axis_width_mm": round(pos.width * fig_width_mm, 3),
            "axis_height_mm": round(pos.height * fig_height_mm, 3),
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / "axis_geometry_mm.json").write_text(
        json.dumps(geometry, indent=2), encoding="utf-8"
    )
    save_bundle(fig, out / "Fig01_full_nature_trial")


def build_leaf_panels(out: Path, frames):
    leaf = out / "leaf_panels"

    fig_width_mm, fig_height_mm = 90.0, 58.0
    fig = plt.figure(figsize=(fig_width_mm / MM_PER_INCH, fig_height_mm / MM_PER_INCH))
    ax = add_axes_mm(fig, fig_width_mm, fig_height_mm, 16, 11, 70, 42)
    plot_a(ax, frames["a"], label=False)
    if ax.legend_ is not None:
        ax.legend_.remove()
    save_bundle(fig, leaf / "Fig01_a_state_budget")

    fig_width_mm, fig_height_mm = 82.0, 66.0
    fig = plt.figure(figsize=(fig_width_mm / MM_PER_INCH, fig_height_mm / MM_PER_INCH))
    ax = add_axes_mm(fig, fig_width_mm, fig_height_mm, 19, 15, 46, 46)
    im, vmax = plot_b(ax, frames["b"], label=False)
    cax = add_axes_mm(fig, fig_width_mm, fig_height_mm, 68, 15, 3, 46)
    cb = fig.colorbar(im, cax=cax, ticks=[-vmax, 0, vmax])
    cb.set_label("Δ score vs home", fontsize=6.0, labelpad=3)
    cb.ax.tick_params(labelsize=5.2, length=1.8, pad=1.5)
    save_bundle(fig, leaf / "Fig01_b_transfer_matrix")

    specs = [
        ("c", plot_c, "Fig01_c_crossover_by_budget"),
        ("d", plot_d, "Fig01_d_home_advantage_paired"),
        ("e", plot_e, "Fig01_e_home_vs_foreign"),
        ("f", plot_f, "Fig01_f_replicate_validation"),
    ]
    for key, func, stem in specs:
        fig_width_mm, fig_height_mm = 78.0, 58.0
        fig = plt.figure(figsize=(fig_width_mm / MM_PER_INCH, fig_height_mm / MM_PER_INCH))
        ax = add_axes_mm(fig, fig_width_mm, fig_height_mm, 16, 11, 58, 42)
        func(ax, frames[key], label=False)
        save_bundle(fig, leaf / stem)


def write_manifest(out: Path, paths):
    manifest = {
        "figure": "Full Fig01 nature-figure trial",
        "backend": "Python/Matplotlib only",
        "final_size_mm": [183, 118],
        "axes_contract_mm": {
            "a_wide": [70, 42],
            "b_heatmap": [46, 46],
            "c_to_f_standard": [58, 42],
            "combined_standard_fit": [45, 36],
            "combined_heatmap_fit": [36, 36],
            "combined_colorbar": [2.5, 36],
        },
        "panel_order": ["a", "b", "c", "d", "e", "f"],
        "palette": {
            "states": STATE_COLORS,
            "heatmap": ["#2166AC", "#F7F7F7", "#B2182B"],
            "heatmap_center": 0,
            "interaction_mean": INTERACTION,
            "budgets": BUDGET_COLORS,
        },
        "source_data": {key: str(path) for key, path in paths.items()},
        "display_computations": {
            "a": "Archived mean_gain and archived ci95 bounds; no recomputation",
            "b": "Archived delta_vs_target_home",
            "c": "Archived graph-seed values plus descriptive mean by budget",
            "d": "All nine archived matched pairs",
            "e": "All 72 archived observations",
            "f": "All 36 archived observations plus standard descriptive box summaries",
        },
        "exports": ["SVG", "PDF", "PNG 600 dpi", "TIFF 600 dpi"],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    setup_style()
    plt.rcParams.update({
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.4,
    })
    paths, frames = load_all_sources(args.fig01_root)
    args.out.mkdir(parents=True, exist_ok=True)
    build_combined(args.out, frames)
    build_leaf_panels(args.out, frames)
    write_manifest(args.out, paths)
    print(json.dumps({"status": "COMPLETE", "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
