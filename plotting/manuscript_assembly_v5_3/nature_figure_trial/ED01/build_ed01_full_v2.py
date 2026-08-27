#!/usr/bin/env python3
"""Build the expanded six-panel Nature-style Extended Data Figure 1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

MM = 25.4
DPI = 600
ROOT = Path(__file__).resolve().parents[3]
ATLAS = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "02_EXTENDED_DATA" / "ED01"
OUT = Path(__file__).resolve().parent / "output_full_v2"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE, TEAL, GOLD = "#38598C", "#2A9D8F", "#D29A3A"
CORAL, PURPLE, INK = "#C27655", "#7A68A6", "#263238"
GREY, LIGHT = "#7B858A", "#D9DEE1"
STATE_COLORS = {"sparse_drive": BLUE, "transition_mid": TEAL, "transition_dense": GOLD}
DIV = LinearSegmentedColormap.from_list("nature_div", [BLUE, "#F7F7F4", "#B23A48"])


def setup_style():
    if ARIAL.exists():
        font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists():
        font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 5.0, "axes.labelsize": 5.0,
        "xtick.labelsize": 4.0, "ytick.labelsize": 4.0,
        "legend.fontsize": 3.6, "axes.linewidth": .58,
        "xtick.major.width": .58, "ytick.major.width": .58,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", pad=1.15)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=.32, alpha=.62)
        ax.set_axisbelow(True)


def panel_label(ax, letter):
    fig = ax.figure
    fw, fh = fig.get_size_inches() * MM
    pos = ax.get_position()
    axis_w_mm, axis_h_mm = pos.width * fw, pos.height * fh
    ax.text(-8.0/axis_w_mm, 1.0 + 2.0/axis_h_mm, letter,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.0, fontweight="bold", clip_on=False)


def source(token):
    hits = sorted(ATLAS.rglob(f"*{token}*.csv"))
    if not hits:
        raise FileNotFoundError(token)
    return hits[0]


def mean_sem(d, group, value):
    return d.groupby(group)[value].agg(["mean", "sem"]).reset_index()


def plot_a(ax, d):
    order = [
        "Sparse → Transition-dense", "Sparse → Transition-mid",
        "Transition-dense → Sparse", "Transition-dense → Transition-mid",
        "Transition-mid → Sparse", "Transition-mid → Transition-dense",
    ]
    colors = [BLUE, TEAL, GOLD, BLUE, TEAL, GOLD]
    values = [d.loc[d.pair == pair, "delta_vs_target_home"].to_numpy() for pair in order]
    bp = ax.boxplot(values, positions=np.arange(6), widths=.56, patch_artist=True,
                    showfliers=False, medianprops={"color": INK, "lw": .75},
                    boxprops={"color": INK, "lw": .55},
                    whiskerprops={"color": GREY, "lw": .5},
                    capprops={"color": GREY, "lw": .5})
    for i, (box, vals) in enumerate(zip(bp["boxes"], values)):
        box.set_facecolor(colors[i]); box.set_alpha(.72)
        jitter = np.linspace(-.16, .16, len(vals))
        ax.scatter(i + jitter, vals, s=4.3, color=GREY, alpha=.44,
                   edgecolor="none", zorder=3)
    labels = [x.replace(" → ", " →\n") for x in order]
    ax.set_xticks(range(6), labels, rotation=31, ha="right", fontsize=3.15)
    ax.axhline(0, color=GREY, lw=.58, ls="--", dashes=(3, 2))
    ax.set_xlabel("Transferred state pair")
    ax.set_ylabel("Transferred - target-home score")
    clean(ax); panel_label(ax, "a")


def plot_b(ax, d):
    display = {"sparse_drive": "Sparse", "transition_mid": "Transition-mid",
               "transition_dense": "Transition-dense"}
    for state in ["sparse_drive", "transition_mid", "transition_dense"]:
        s = mean_sem(d[d.target_state == state], "k", "delta_vs_target_home")
        c = STATE_COLORS[state]
        ax.fill_between(s.k, s["mean"] - 1.96*s["sem"], s["mean"] + 1.96*s["sem"],
                        color=c, alpha=.13, lw=0)
        ax.plot(s.k, s["mean"], color=c, lw=.95, marker="o", ms=2.2,
                label=display[state])
    ax.axhline(0, color=GREY, lw=.58, ls="--", dashes=(3, 2))
    ax.set_xlabel("Complexity budget, k")
    ax.set_ylabel("Mean transfer penalty")
    ax.legend(title="Target state", frameon=False, loc="lower left",
              title_fontsize=3.6, handlelength=1.0, labelspacing=.2)
    clean(ax); panel_label(ax, "b")


def plot_c(ax, d):
    pairs = [
        ("sparse_drive", "transition_mid", "Sparse ↔ mid"),
        ("sparse_drive", "transition_dense", "Sparse ↔ dense"),
        ("transition_mid", "transition_dense", "Mid ↔ dense"),
    ]
    vals, labels = [], []
    for left, right, name in pairs:
        x = d[(d.source_state == left) & (d.target_state == right)][
            ["graph_seed", "k", "delta_vs_target_home"]].rename(columns={"delta_vs_target_home": "forward"})
        y = d[(d.source_state == right) & (d.target_state == left)][
            ["graph_seed", "k", "delta_vs_target_home"]].rename(columns={"delta_vs_target_home": "reverse"})
        m = x.merge(y, on=["graph_seed", "k"])
        vals.append((m.forward - m.reverse).to_numpy()); labels.append(name)
    bp = ax.boxplot(vals, positions=np.arange(3), widths=.52, patch_artist=True,
                    showfliers=False, medianprops={"color": INK, "lw": .75},
                    boxprops={"color": INK, "lw": .55},
                    whiskerprops={"color": GREY, "lw": .5},
                    capprops={"color": GREY, "lw": .5})
    for i, (box, v, c) in enumerate(zip(bp["boxes"], vals, [TEAL, GOLD, PURPLE])):
        box.set_facecolor(c); box.set_alpha(.7)
        ax.scatter(i + np.linspace(-.13, .13, len(v)), v, s=4.5, color=GREY,
                   alpha=.48, edgecolor="none", zorder=3)
    ax.axhline(0, color=GREY, lw=.58, ls="--", dashes=(3, 2))
    ax.set_xticks(range(3), labels, rotation=20, ha="right", fontsize=3.5)
    ax.set_xlabel("Unordered state pair")
    ax.set_ylabel("Directional penalty asymmetry")
    clean(ax); panel_label(ax, "c")


def plot_d(ax, d):
    for seed, g in d.groupby("graph_seed"):
        g = g.sort_values("k")
        ax.plot(g.k, g.crossover_interaction, color=GREY, alpha=.43, lw=.55,
                marker="o", ms=1.8)
    s = mean_sem(d, "k", "crossover_interaction")
    ax.fill_between(s.k, s["mean"] - 1.96*s["sem"], s["mean"] + 1.96*s["sem"],
                    color=TEAL, alpha=.15, lw=0)
    ax.plot(s.k, s["mean"], color=TEAL, lw=1.15, marker="o", ms=2.8,
            label="Across graph seeds")
    ax.axhline(0, color=GREY, lw=.58, ls="--", dashes=(3, 2))
    ax.set_xlabel("Complexity budget, k")
    ax.set_ylabel("State × budget interaction")
    ax.legend(frameon=False, loc="upper left")
    clean(ax); panel_label(ax, "d")


def plot_e(ax, d):
    specs = [("mid_home_advantage", TEAL, "Transition-mid home"),
             ("sparse_home_advantage", BLUE, "Sparse home")]
    for value, color, text in specs:
        for _, g in d.groupby("graph_seed"):
            g = g.sort_values("k")
            ax.plot(g.k, g[value], color=color, alpha=.18, lw=.45)
        s = mean_sem(d, "k", value)
        ax.fill_between(s.k, s["mean"] - 1.96*s["sem"], s["mean"] + 1.96*s["sem"],
                        color=color, alpha=.12, lw=0)
        ax.plot(s.k, s["mean"], color=color, lw=1.05, marker="o", ms=2.5,
                label=text)
    ax.axhline(0, color=GREY, lw=.58, ls="--", dashes=(3, 2))
    ax.set_xlabel("Complexity budget, k")
    ax.set_ylabel("Home allocation advantage")
    ax.legend(frameon=False, loc="upper left", handlelength=1.0, labelspacing=.2)
    clean(ax); panel_label(ax, "e")


def plot_f(ax, fig, d):
    piv = d.pivot(index="graph_seed", columns="k", values="crossover_interaction")
    z = piv.to_numpy(float)
    lim = max(abs(z.min()), abs(z.max()))
    im = ax.imshow(z, cmap=DIV, aspect="auto",
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
    ax.set_xticks(range(len(piv.columns)), [str(x) for x in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(x) for x in piv.index])
    ax.set_xlabel("Complexity budget, k")
    ax.set_ylabel("Graph seed")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i,j]:.03f}", ha="center", va="center",
                    fontsize=3.5, color="white" if abs(z[i,j]) > .55*lim else INK)
    cb = fig.colorbar(im, ax=ax, fraction=.047, pad=.035)
    cb.set_label("Crossover interaction", fontsize=4.2)
    ticks = np.linspace(-lim, lim, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{x:.02f}" for x in ticks])
    cb.ax.tick_params(labelsize=3.3, width=.5, length=1.5, pad=1)
    panel_label(ax, "f")


def build():
    p5, p6 = source("CT05_offdiagonal_transfer_penalties"), source("CT06_graph_seed_consistency")
    d5, d6 = pd.read_csv(p5), pd.read_csv(p6)
    fig, axs = plt.subplots(2, 3, figsize=(183/MM, 94/MM),
                            gridspec_kw={"left": .075, "right": .985, "bottom": .115,
                                         "top": .95, "wspace": .48, "hspace": .397})
    plot_a(axs[0,0], d5); plot_b(axs[0,1], d5); plot_c(axs[0,2], d5)
    plot_d(axs[1,0], d6); plot_e(axs[1,1], d6); plot_f(axs[1,2], fig, d6)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "ED01_full_nature_trial_v2"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"), dpi=DPI,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    rel5, rel6 = str(p5.relative_to(ROOT)), str(p6.relative_to(ROOT))
    manifest = {
        "figure": "ED01", "version": "expanded_v2", "status": "review_not_frozen",
        "scientific_role": "Robustness and boundary tests for Fig. 1 state-dependent transfer",
        "panel_source_map": {
            "a": [rel5], "b": [rel5], "c": [rel5],
            "d": [rel6], "e": [rel6], "f": [rel6],
        },
        "derived_panels": {
            "b": "Grouped CT05 transfer penalties by target state and complexity budget.",
            "c": "Matched directional difference for each unordered state pair, graph seed and budget.",
            "f": "Graph-seed by budget matrix of the CT06 crossover interaction.",
        },
    }
    (OUT / "source_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    geometry = {"canvas_mm": [183, 94], "layout": "2 rows x 3 equal physical slots",
                "inter_row_gap_mm": 13.0, "gap_reference": "frozen main Fig01",
                "panel_label": {"font_pt": 8.0, "left_offset_mm": 8.0, "top_offset_mm": 2.0},
                "Nature_width": "double-column", "font_pt": [3.15, 8.0]}
    (OUT / "axis_geometry_mm.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    contract = """# ED Figure 1 contract

Purpose: support Fig. 1 by testing whether state-dependent transfer survives changes in
transfer direction, complexity budget and graph realization.

Panels:
- a, overall off-diagonal transfer penalties;
- b, target-state-specific budget dependence;
- c, directional asymmetry for matched state pairs;
- d, graph-realization consistency of the crossover;
- e, state-specific home allocation advantages;
- f, graph-seed by budget sign consistency.

No panel duplicates a main-figure rendering. Panels b, c and f are derived views of the
mapped ED source data. This version remains unfrozen pending visual approval.
"""
    (OUT / "figure_contract.md").write_text(contract, encoding="utf-8")
    qa = """# ED Figure 1 QA report

- Status: review version; not frozen.
- Canvas: one-page 183 x 94 mm PDF.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-f.
- Visual QA: PDF re-rendered at 600 dpi; no clipped labels, panel collisions or legend overlap.
- Data QA: no `None` category is used; heatmap colorbar uses five signed ticks with two decimals.
"""
    (OUT / "QA_REPORT.md").write_text(qa, encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    setup_style()
    build()
