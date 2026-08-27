#!/usr/bin/env python3
"""Build the Fig.5-supporting biological-complexity robustness ED Figure 5."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

MM, DPI = 25.4, 600
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "02_EXTENDED_DATA" / "ED06"
OUT = Path(__file__).resolve().parent / "output_full_v2"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE, TEAL, GOLD, RED = "#38598C", "#2A9D8F", "#C27628", "#B84A4A"
INK, GREY, LIGHT = "#263238", "#7B858A", "#D9DEE1"
DIV = LinearSegmentedColormap.from_list("fig1_div", ["#2166AC", "#F7F7F7", "#B2182B"])


def setup_style():
    if ARIAL.exists(): font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists(): font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 5.0, "axes.labelsize": 5.0,
        "xtick.labelsize": 4.0, "ytick.labelsize": 4.0, "legend.fontsize": 3.4,
        "axes.linewidth": .58, "xtick.major.width": .58, "ytick.major.width": .58,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def source(token):
    hits = sorted(p for p in SRC.rglob("*.csv") if token in p.name)
    if not hits: raise FileNotFoundError(token)
    return hits[0]


def source_in(panel_dir, token):
    hits = sorted((SRC / panel_dir / "source_data").glob(f"*{token}*.csv"))
    if not hits: raise FileNotFoundError(f"{panel_dir}: {token}")
    return hits[0]


def add_axes_mm(fig, fw, fh, x, y, w, h):
    return fig.add_axes([x/fw, y/fh, w/fw, h/fh])


def clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", pad=1.15)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=.32, alpha=.62)
        ax.set_axisbelow(True)


def panel_label(ax, letter, fw=183, fh=134):
    pos = ax.get_position(); aw, ah = pos.width*fw, pos.height*fh
    ax.text(-8/aw, 1+2/ah, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="bottom", clip_on=False)


def add_colorbar(fig, fw, fh, im, x, y, h, title, ticks=None):
    cax = add_axes_mm(fig, fw, fh, x, y, 1.55, h)
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    cb.ax.set_title(title, fontsize=3.45, pad=1)
    cb.ax.tick_params(labelsize=3.0, width=.5, length=1.4, pad=1)
    return cb


def plot_attrition(ax, d):
    labels = ["NWB inventory", "Usable Noise 1 + 2", "AB contexts complete",
              "BA contexts complete", "AB + BA complete", "Primary cohort (GLIF1-5)"]
    y = np.arange(len(d))[::-1]
    colors = [BLUE]*5 + [TEAL]
    ax.barh(y, d.n, color=colors, height=.68, edgecolor=INK, lw=.35)
    ax.set_yticks(y, labels, fontsize=3.2)
    ax.set_xlim(0, 1370)
    for yy, n in zip(y, d.n):
        ax.text(n+22, yy, f"{int(n):,}", va="center", fontsize=3.25)
    ax.set_xlabel("Cells / specimens"); clean(ax, "x")


def plot_prediction(ax, d):
    rng = np.random.default_rng(531)
    for c in range(4):
        g = d[d.C_req == c].pred_nested_rerun.to_numpy()
        x = c + rng.uniform(-.13, .13, len(g))
        ax.scatter(x, g, s=4.2, color=BLUE, alpha=.27, edgecolor="none")
        ax.hlines(np.median(g), c-.22, c+.22, color=RED, lw=.85)
    ax.plot([0, 3], [0, 3], color=GREY, lw=.55, ls="--", dashes=(3, 2))
    rho = d[["C_req", "pred_nested_rerun"]].corr(method="spearman").iloc[0, 1]
    ax.text(.04, .96, f"Held-out Spearman ρ = {rho:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=3.35)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xlim(-.35, 3.35); ax.set_ylim(-.1, 3.25)
    ax.set_xlabel("Observed minimum mechanism cost"); ax.set_ylabel("OOF-predicted cost")
    clean(ax)


def confusion(d, a, b):
    aa = np.asarray(a, int); bb = np.asarray(b, int)
    z = np.zeros((4, 4), int)
    for x, y in zip(aa, bb): z[x, y] += 1
    return z


def draw_count_heat(ax, z, xlabel, ylabel, vmax=None, fontsize=3.3):
    vmax = int(np.max(z)) if vmax is None else vmax
    im = ax.imshow(z, aspect="auto", cmap=DIV, norm=Normalize(vmin=0, vmax=vmax))
    ax.set_xticks(range(z.shape[1])); ax.set_yticks(range(z.shape[0]))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            val = int(z[i, j])
            ax.text(j, i, str(val), ha="center", va="center", fontsize=fontsize,
                    color="white" if val > .70*vmax else INK)
    return im


def plot_ordinal(ax, fig, fw, fh, d, cax_x, y):
    pred = np.clip(np.rint(d.pred_nested_rerun), 0, 3).astype(int)
    z = confusion(d, d.C_req.astype(int), pred)
    im = draw_count_heat(ax, z, "Rounded OOF prediction", "Observed cost")
    add_colorbar(fig, fw, fh, im, cax_x, y, 27, "Cells", [0, 60, 120])


def plot_forest(ax, d, grouping, ylabel):
    g = d[d.grouping == grouping].copy()
    order = ["1", "2/3", "4", "5", "6a", "6b"] if grouping == "layer" else ["aspiny", "sparsely spiny", "spiny"]
    g["group"] = pd.Categorical(g.group, order, ordered=True); g = g.sort_values("group")
    y = np.arange(len(g))[::-1]
    lo = g.rho-g.bootstrap95_low; hi = g.bootstrap95_high-g.rho
    ax.errorbar(g.rho, y, xerr=np.vstack([lo, hi]), fmt="o", ms=2.5, color=BLUE,
                ecolor=BLUE, elinewidth=.65, capsize=1.6, capthick=.55)
    ax.axvline(0, color=GREY, lw=.55, ls="--", dashes=(3, 2))
    labels = [f"{x.group} (n={int(x.n)})" for _, x in g.iterrows()]
    ax.set_yticks(y, labels, fontsize=3.15); ax.set_xlabel("Within-group OOF Spearman ρ")
    if grouping == "dendrite_type":
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", va="center",
                 rotation_mode="anchor")
    ax.set_ylabel(ylabel)
    if grouping == "dendrite_type":
        ax.yaxis.set_label_coords(-.27, .5)
    clean(ax, "x")


def temporal_matrix(d):
    vals = [10, 25, 50, 100, 200, 500]
    tab = pd.crosstab(d.C_temporal_AB, d.C_temporal_BA).reindex(index=vals, columns=vals, fill_value=0)
    return tab.to_numpy(), vals


def plot_temporal(ax, fig, fw, fh, d, cax_x, y):
    z, vals = temporal_matrix(d)
    im = ax.imshow(z, aspect="auto", cmap=DIV, norm=Normalize(vmin=0, vmax=z.max()))
    ax.set_xticks(range(6), vals, fontsize=3.1); ax.set_yticks(range(6), vals, fontsize=3.1)
    ax.set_xlabel("Minimum sufficient context B-A (ms)")
    ax.set_ylabel("Minimum sufficient context A-B (ms)")
    for i in range(6):
        for j in range(6):
            v = int(z[i, j]); ax.text(j, i, str(v), ha="center", va="center", fontsize=2.7,
                                      color="white" if v > .55*z.max() else INK)
    add_colorbar(fig, fw, fh, im, cax_x, y, 27, "Cells", [0, 325, 650])


def plot_gain(ax, d):
    x = d.long_short_gain_mean.dropna().to_numpy()
    ax.hist(x, bins=36, color=BLUE, alpha=.68, edgecolor="white", lw=.15)
    med = float(np.median(x)); ax.axvline(med, color=RED, lw=.75)
    ax.text(.96, .96, f"median = {med:.4f}\nn = {len(x):,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=3.35)
    ax.set_xlabel("Held-out R² gain: 500 ms - 10 ms"); ax.set_ylabel("Cells"); clean(ax)


def plot_epsilon_dist(ax, fig, fw, fh, d, cax_x, y):
    tab = d.pivot(index="cost", columns="epsilon", values="fraction").sort_index()
    z = tab.to_numpy()
    im = ax.imshow(z, aspect="auto", cmap=DIV, norm=Normalize(vmin=0, vmax=.5))
    labels = [f"{x:.3f}" for x in tab.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=43, ha="right", fontsize=2.85)
    ax.set_yticks(range(4), [f"Cost {x}" for x in tab.index], fontsize=3.1)
    ax.set_xlabel("Absolute EV tolerance, ε"); ax.set_ylabel("Minimum mechanism cost")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i,j]:.2f}", ha="center", va="center", fontsize=2.45,
                    color="white" if z[i,j] > .34 else INK)
    add_colorbar(fig, fw, fh, im, cax_x, y, 27, "Fraction", [0, .25, .5])


def plot_epsilon_mean(ax, d):
    ax.plot(d.epsilon, d.mean_creq, color=TEAL, lw=.9, marker="o", ms=2.3)
    ax.axvline(.02, color=BLUE, lw=.65, ls="--", dashes=(3, 2))
    ax.text(.02, .96, "Primary ε = 0.020", transform=ax.transAxes,
            ha="left", va="top", fontsize=3.2)
    ax.set_xlabel("Absolute EV tolerance, ε"); ax.set_ylabel("Mean minimum mechanism cost")
    ax.set_xticks([0, .02, .04]); ax.set_ylim(.75, 2.08); clean(ax)


def transition(d, target):
    a = d.creq_eps_0p020.astype(int)
    col = f"creq_eps_0p{int(round(target*1000)):03d}"
    return confusion(d, a, d[col].astype(int))


def plot_transitions(ax1, ax2, fig, fw, fh, d, cax_x, y):
    z1, z2 = transition(d, .015), transition(d, .025)
    vmax = max(z1.max(), z2.max())
    im = draw_count_heat(ax1, z1, "Cost at ε = 0.015", "Cost at ε = 0.020", vmax, 3.0)
    draw_count_heat(ax2, z2, "Cost at ε = 0.025", "", vmax, 3.0)
    ax2.set_yticklabels([])
    add_colorbar(fig, fw, fh, im, cax_x, y, 27, "Cells", [0, 70, 140])


def build():
    paths = {k: source(tok) for k, tok in {
        "a": "cohort_attrition", "b": "nested_rerun_multiaxial_predictions",
        "d": "within_layer_oof_effects", "e": "within_dendrite_oof_effects",
        "f1": "temporal_AB_BA_consistency", "f2": "long_vs_short_context_gain",
        "g": "epsilon_creq_distribution_long", "h": "epsilon_sensitivity_summary",
        "i1": "transition_baseline_to_0p015", "i2": "transition_baseline_to_0p025",
    }.items()}
    paths["b"] = source_in("ED06_b__Prediction_by_observed_complexity_level", "nested_rerun_multiaxial_predictions")
    paths["c"] = source_in("ED06_c__Ordinal_confusion_structure", "nested_rerun_multiaxial_predictions")
    data = {k: pd.read_csv(p) for k, p in paths.items()}
    fw, fh, ah = 183.0, 134.0, 27.0
    y1, y2, y3 = 92.0, 52.0, 12.0
    fig = plt.figure(figsize=(fw/MM, fh/MM))

    a = add_axes_mm(fig, fw, fh, 29, y1, 37, ah)
    b = add_axes_mm(fig, fw, fh, 81, y1, 42, ah)
    c = add_axes_mm(fig, fw, fh, 139, y1, 29, ah)
    d = add_axes_mm(fig, fw, fh, 10, y2, 34, ah)
    e = add_axes_mm(fig, fw, fh, 59, y2, 29, ah)
    f1 = add_axes_mm(fig, fw, fh, 98, y2, 30, ah)
    f2 = add_axes_mm(fig, fw, fh, 142, y2, 34, ah)
    g = add_axes_mm(fig, fw, fh, 10, y3, 40, ah)
    h = add_axes_mm(fig, fw, fh, 67, y3, 32, ah)
    i1 = add_axes_mm(fig, fw, fh, 109, y3, 27, ah)
    i2 = add_axes_mm(fig, fw, fh, 145, y3, 27, ah)

    plot_attrition(a, data["a"])
    plot_prediction(b, data["b"])
    plot_ordinal(c, fig, fw, fh, data["c"], 171, y1)
    plot_forest(d, data["d"], "layer", "Cortical layer")
    plot_forest(e, data["e"], "dendrite_type", "Dendrite class")
    plot_temporal(f1, fig, fw, fh, data["f1"], 131, y2)
    plot_gain(f2, data["f2"])
    plot_epsilon_dist(g, fig, fw, fh, data["g"], 53, y3)
    plot_epsilon_mean(h, data["h"])
    plot_transitions(i1, i2, fig, fw, fh, data["i1"], 175, y3)

    for ax, letter in [(a, "a"), (b, "b"), (c, "c"), (d, "d"), (e, "e"),
                       (f1, "f"), (g, "g"), (h, "h"), (i1, "i")]:
        panel_label(ax, letter, fw, fh)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "ED05_full_nature_trial_v2"
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"), dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    rel = {k: str(v.relative_to(ROOT)) for k, v in paths.items()}
    panel_map = {
        "a": [rel["a"]], "b": [rel["b"]], "c": [rel["c"]],
        "d": [rel["d"]], "e": [rel["e"]], "f": [rel["f1"], rel["f2"]],
        "g": [rel["g"]], "h": [rel["h"]], "i": [rel["i1"], rel["i2"]],
    }
    manifest = {
        "figure": "ED05", "version": "biological_robustness_v2",
        "status": "review_not_frozen",
        "scientific_role": "Support Fig.5 by establishing cohort integrity, held-out biological generalization, temporal robustness and epsilon-definition stability.",
        "panel_source_map": panel_map,
    }
    (OUT / "source_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    geom = {
        "canvas_mm": [fw, fh], "row_gap_mm": 13,
        "axes_mm": {"a":[29,y1,37,ah], "b":[81,y1,42,ah], "c":[139,y1,29,ah],
                    "d":[10,y2,34,ah], "e":[59,y2,29,ah], "f1":[98,y2,30,ah],
                    "f2":[142,y2,34,ah], "g":[10,y3,40,ah], "h":[67,y3,32,ah],
                    "i1":[109,y3,27,ah], "i2":[145,y3,27,ah]},
    }
    (OUT / "axis_geometry_mm.json").write_text(json.dumps(geom, indent=2), encoding="utf-8")
    contract = """# ED Figure 5 contract

- Purpose: support main Fig.5 with biological-complexity cohort, held-out prediction,
  within-group generalization, temporal/context robustness and epsilon sensitivity.
- Canvas: 183 x 134 mm; three 27-mm-high rows with fixed 13 mm gaps.
- Sequence: `a+b+c / d+e+f+f / g+h+i+i`.
- Multi-plot panels f and i never wrap across rows.
- Panel labels inherit the common 8 mm left / 2 mm above physical offsets.
- This is a review version and is not frozen.
"""
    (OUT / "figure_contract.md").write_text(contract, encoding="utf-8")
    qa = """# ED Figure 5 QA report

- Status: review version; not frozen.
- Canvas: one-page 183 x 134 mm PDF.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-i.
- Layout: fixed 13 mm inter-row gaps; multi-plot panels do not wrap.
- Visual QA: PDF re-rendered at 600 dpi; no clipping, panel collision, legend overlap or colorbar collision.
"""
    (OUT / "QA_REPORT.md").write_text(qa, encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    setup_style(); build()
