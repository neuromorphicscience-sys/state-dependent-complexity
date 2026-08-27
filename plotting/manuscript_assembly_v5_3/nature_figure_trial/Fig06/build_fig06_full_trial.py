#!/usr/bin/env python3
"""Build the source-data-first Nature-width Fig.6 reference figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

MM_PER_INCH = 25.4
DPI = 600
ROOT = Path(__file__).resolve().parents[3]
ATLAS = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "01_MAIN" / "Fig06"
ED07 = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "02_EXTENDED_DATA" / "ED07"
DEFAULT_OUT = Path(__file__).resolve().parent / "output_full_v1"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE = "#38598C"
TEAL = "#2A9D8F"
CORAL = "#C27655"
GOLD = "#D29A3A"
PURPLE = "#7A68A6"
INK = "#263238"
GREY = "#7B858A"
LIGHT = "#D9DEE1"


def setup_style():
    if ARIAL.exists(): font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists(): font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 5.0, "axes.labelsize": 5.0,
        "xtick.labelsize": 4.1, "ytick.labelsize": 4.1,
        "legend.fontsize": 3.7, "axes.linewidth": .58,
        "xtick.major.width": .58, "ytick.major.width": .58,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def add_axes_mm(fig, fw, fh, x, y, w, h):
    return fig.add_axes([x/fw, y/fh, w/fw, h/fh])


def clean(ax, grid=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.25)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=.34, alpha=.58)
        ax.set_axisbelow(True)


def src(panel, token):
    hits = sorted((ATLAS/panel/"source_data").glob(f"*{token}*.csv"))
    if not hits: raise FileNotFoundError((panel, token))
    return hits[0]


def paired_summary(ax, raw, residual, labels=("Raw leverage", "Activity residual")):
    key = ["state_a", "state_b"]
    d = raw.merge(residual, on=key, suffixes=("_raw", "_res"))
    for _, r in d.iterrows():
        ax.plot([0, 1], [r.median_rho_raw, r.median_rho_res], color="#C8D0D4", lw=.65)
    ax.scatter(np.zeros(len(d)), d.median_rho_raw, color=BLUE, s=10, zorder=3)
    ax.scatter(np.ones(len(d)), d.median_rho_res, color=TEAL, s=10, zorder=3)
    ax.set_xticks([0, 1], labels, fontsize=3.7)
    ax.set_xlabel("Leverage definition")
    ax.set_ylabel("Median state-pair Spearman rho")
    clean(ax, "y")


def plot_a(ax1, ax2):
    p1 = src("Fig06_a__Fast-state_stable_backbone", "raw_summary")
    p2 = src("Fig06_a__Fast-state_stable_backbone", "residual_summary")
    paired_summary(ax1, pd.read_csv(p1), pd.read_csv(p2))
    p3 = src("Fig06_a__Fast-state_stable_backbone", "backbone_fraction")
    d = pd.read_csv(p3)
    bp = ax2.boxplot([d.backbone_fraction], positions=[0], widths=.38, patch_artist=True,
                     showfliers=False, medianprops={"color": INK, "lw": .75},
                     boxprops={"color": INK, "lw": .6},
                     whiskerprops={"color": GREY, "lw": .55},
                     capprops={"color": GREY, "lw": .55})
    bp["boxes"][0].set_facecolor(BLUE); bp["boxes"][0].set_alpha(.75)
    jitter = np.linspace(-.13, .13, len(d))
    ax2.scatter(jitter, d.backbone_fraction, s=4.5, color=GREY, alpha=.45, edgecolor="none")
    ax2.axhline(.5, color=GREY, lw=.55, ls="--", dashes=(3, 2))
    ax2.set_xticks([0], ["Steinmetz"])
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Rank-variance backbone fraction")
    clean(ax2, "y")
    return [p1, p2, p3]


def plot_b(ax):
    p1 = src("Fig06_b__Larger-context_residual_reconfiguration", "raw_summary")
    p2 = src("Fig06_b__Larger-context_residual_reconfiguration", "residual_summary")
    paired_summary(ax, pd.read_csv(p1), pd.read_csv(p2), ("Raw", "Residual"))
    return [p1, p2]


def plot_b_quality(ax):
    p = (ED07 / "ED07_h__Quality-gated_state_stability" / "source_data" /
         "01_S5BR_20_allen_r2_gated_state_stability__allen_primary_state_pairs_r2_gated_summary.csv")
    d = pd.read_csv(p)
    pairs = list(dict.fromkeys(zip(d.state_a, d.state_b)))
    for a, b in pairs:
        g = d[(d.state_a == a) & (d.state_b == b)].sort_values("r2_threshold")
        ax.plot(g.r2_threshold, g.median_rho, color="#C8D0D4", lw=.65)
        ax.scatter(g.r2_threshold, g.median_rho, color=CORAL, s=9, zorder=3)
    ax.set_xticks([0, .05], ["R² > 0", "R² > 0.05"])
    ax.set_xlabel("Minimum latent-model R²")
    ax.set_ylabel("Quality-gated state-pair rho")
    ax.set_ylim(.35, .64)
    clean(ax, "y")
    return [p]


def plot_c(ax1, ax2):
    panel = "Fig06_c__Cross-dataset_state_scale_and_backbone_fraction"
    ps = src(panel, "steinmetz_state_pairs_raw")
    pa = src(panel, "allen_primary_state_pairs_raw")
    par = src(panel, "allen_within_session_arousal_raw")
    ds, da, dar = pd.read_csv(ps), pd.read_csv(pa), pd.read_csv(par)
    vals = [ds.median_rho.median(), da.median_rho.median(),
            float(dar.loc[dar.contrast == "running", "median_rho"].iloc[0]),
            float(dar.loc[dar.contrast == "pupil", "median_rho"].iloc[0])]
    labs = ["Steinmetz\nfast epochs", "Allen\ncontexts", "Allen\nrunning", "Allen\npupil"]
    ax1.bar(range(4), vals, color=[BLUE, CORAL, TEAL, PURPLE], width=.68)
    ax1.set_xticks(range(4), labs, rotation=35, ha="right", fontsize=3.5)
    ax1.set_xlabel("Dataset / state contrast")
    ax1.set_ylabel("Median state-pair rho")
    ax1.set_ylim(0, 1.02)
    clean(ax1, "y")

    pbs = src(panel, "steinmetz_backbone_fraction")
    pba = src(panel, "allen_backbone_fraction")
    bs, ba = pd.read_csv(pbs), pd.read_csv(pba)
    bp = ax2.boxplot([bs.backbone_fraction, ba.backbone_fraction], positions=[0, 1],
                     widths=.55, patch_artist=True, showfliers=False,
                     medianprops={"color": INK, "lw": .75},
                     boxprops={"color": INK, "lw": .58},
                     whiskerprops={"color": GREY, "lw": .55},
                     capprops={"color": GREY, "lw": .55})
    for patch, color in zip(bp["boxes"], [BLUE, CORAL]):
        patch.set_facecolor(color); patch.set_alpha(.72)
    for x, d, color in [(0, bs, BLUE), (1, ba, CORAL)]:
        off = np.linspace(-.14, .14, len(d))
        ax2.scatter(x+off, d.backbone_fraction, s=4.5, color=color, alpha=.35, edgecolor="none")
    ax2.axhline(.5, color=GREY, lw=.55, ls="--", dashes=(3, 2))
    ax2.set_xticks([0, 1], ["Steinmetz", "Allen VBO"], rotation=25, ha="right")
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("Backbone fraction")
    clean(ax2, "y")
    return [ps, pa, par, pbs, pba]


def paired_mouse(ax, d, home, cross, ylabel, colors=(TEAL, CORAL)):
    for _, r in d.iterrows(): ax.plot([0, 1], [r[home], r[cross]], color="#CBD1D4", lw=.55)
    ax.scatter(np.zeros(len(d)), d[home], color=colors[0], s=7, zorder=3)
    ax.scatter(np.ones(len(d)), d[cross], color=colors[1], s=7, zorder=3)
    ax.plot([0, 1], [d[home].mean(), d[cross].mean()], color=INK, marker="o",
            markerfacecolor="white", ms=3.3, lw=.8, zorder=4)
    ax.axhline(.5, color=GREY, lw=.5, ls="--", dashes=(3, 2))
    ax.set_xticks([0, 1], ["Home", "Cross"])
    ax.set_xlabel("Transfer condition")
    ax.set_ylabel(ylabel)
    clean(ax, "y")


def lollipop_mouse(ax, d, col, ylabel, color=TEAL, baseline=0):
    x = np.arange(len(d))
    ax.vlines(x, baseline, d[col], color="#CAD0D3", lw=.55)
    ax.scatter(x, d[col], color=color, s=7, zorder=3, edgecolor="none")
    ax.axhline(baseline, color=GREY, lw=.55, ls="--" if baseline else "-", dashes=(3, 2))
    ax.axhline(d[col].mean(), color=INK, lw=.75)
    ax.set_xticks(x, [str(v) for v in d.subject], rotation=55, ha="right", fontsize=3.0)
    ax.set_xlabel("Mouse")
    ax.set_ylabel(ylabel)
    clean(ax, "y")


def plot_d(ax1, ax2):
    panel = "Fig06_d__State-matched_population_transfer"
    p1 = src(panel, "home_vs_cross_auc_paired")
    d1 = pd.read_csv(p1)
    paired_mouse(ax1, d1, "decoder_auc_home", "decoder_auc_cross", "Held-out decoder AUC")
    p2 = src(panel, "Rdecoder_by_mouse")
    d2 = pd.read_csv(p2)
    lollipop_mouse(ax2, d2, "decoder_auc_crossover", "Rdecoder", TEAL)
    return [p1, p2]


def single_ci(ax, d, ylabel, color=TEAL, null=0):
    lo, hi, val = float(d.bootstrap_ci_low.iloc[0]), float(d.bootstrap_ci_high.iloc[0]), float(d.mean_effect.iloc[0])
    ax.errorbar([0], [val], yerr=[[val-lo], [hi-val]], fmt="o", color=color,
                ms=3.3, lw=.8, capsize=1.8)
    ax.axhline(null, color=GREY, lw=.55, ls="--", dashes=(3, 2))
    ax.set_xticks([0], [f"{int(d.n_mouse_sessions.iloc[0])} mice"])
    ax.set_xlabel("Population")
    ax.set_ylabel(ylabel)
    clean(ax, "y")


def plot_e(ax1, ax2):
    panel = "Fig06_e__Formal_population-level_inference"
    p1 = src(panel, "formal_effect_ci")
    single_ci(ax1, pd.read_csv(p1), "Population Rdecoder", TEAL)
    p2 = src(panel, "high_precision_permutation")
    d = pd.read_csv(p2)
    ax2.hist(d.null, bins=35, color="#BCC5C9", edgecolor="white", lw=.25)
    obs = float(d.observed.iloc[0])
    pperm = (np.sum(d.null >= obs)+1)/(len(d)+1)
    ax2.axvline(obs, color=TEAL, lw=1.1)
    ax2.text(.96, .94, f"Observed = {obs:.3f}\np = {pperm:.4f}", transform=ax2.transAxes,
             ha="right", va="top", fontsize=3.8)
    ax2.set(xlabel="Fixed-split null Rdecoder", ylabel="Count")
    clean(ax2)
    return [p1, p2]


def plot_f(ax1, ax2):
    panel = "Fig06_f__Identical-input_state_imprint"
    # AUC excess is exactly AUC - 0.5, so it retains the complete information in M05.
    p_auc = src(panel, "same_image_state_auc_by_mouse")
    p1 = src(panel, "same_image_state_excess_by_mouse")
    d = pd.read_csv(p1)
    lollipop_mouse(ax1, d, "same_image_state_auc_excess_mean", "Same-image AUC excess", CORAL)
    p2 = src(panel, "same_image_formal_effect_ci")
    single_ci(ax2, pd.read_csv(p2), "AUC excess above chance", CORAL)
    return [p_auc, p1, p2]


def plot_g(ax):
    panel = "Fig06_g__Alternative-explanation_controls"
    paths = [src(panel, "state_time_confound"), src(panel, "running_confound"),
             src(panel, "stimulus_state_cramers")]
    cols = ["_abs_time", "_abs_run", "state_stimulus_cramers_v"]
    labels = ["Time", "Running", "Stimulus V"]
    colors = [BLUE, GOLD, PURPLE]
    ds = [pd.read_csv(p) for p in paths]
    merged = ds[0]
    for d in ds[1:]: merged = merged.merge(d, on="subject")
    x = np.arange(3)
    for _, r in merged.iterrows(): ax.plot(x, [r[c] for c in cols], color="#D4D9DB", lw=.45, alpha=.7)
    for i, (c, color) in enumerate(zip(cols, colors)):
        ax.scatter(np.full(len(merged), i), merged[c], color=color, s=7, alpha=.8, edgecolor="none")
        ax.hlines(merged[c].mean(), i-.18, i+.18, color=INK, lw=.85)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Potential confound")
    ax.set_ylabel("Confound association magnitude")
    clean(ax, "y")
    return paths


def plot_h(ax1, ax2):
    panel = "Fig06_h__Real-edge_specificity_boundary"
    p1 = src(panel, "real_edge_home_vs_cross_auc")
    d1 = pd.read_csv(p1)
    paired_mouse(ax1, d1, "decoder_auc_home", "decoder_auc_cross", "Real-edge decoder AUC", (TEAL, CORAL))
    p2 = src(panel, "real_edge_Rdecoder_by_mouse")
    d2 = pd.read_csv(p2)
    lollipop_mouse(ax2, d2, "decoder_auc_crossover", "Real-edge Rdecoder", PURPLE)
    return [p1, p2]


def plot_i(ax):
    p = src("Fig06_i__Formal_biological_synthesis", "formal_effect_summary_forest")
    d = pd.read_csv(p)
    y = np.arange(len(d))[::-1]
    ax.hlines(y, d.lo, d.hi, color=TEAL, lw=.8)
    ax.scatter(d.value, y, color=TEAL, s=10, edgecolor="none", zorder=3)
    ax.axvline(0, color=GREY, lw=.55, ls="--", dashes=(3, 2))
    labels = ["Decoder", "Landscape", "Top-20", "Activity-resid."]
    ax.set_yticks(y, labels, fontsize=3.65)
    ax.set_xlabel("Effect relative to null")
    clean(ax, "x")
    return [p]


def build(out):
    fw, fh = 183.0, 107.0
    fig = plt.figure(figsize=(fw/MM_PER_INCH, fh/MM_PER_INCH))
    ys = [83.0, 58.0, 33.0, 8.0]
    ah = 15.0
    sources = []

    xs4 = [10,54,98,142]
    # Row 1: four standard plots.
    sources += plot_a(add_axes_mm(fig,fw,fh,xs4[0],ys[0],34,ah),
                      add_axes_mm(fig,fw,fh,xs4[1],ys[0],34,ah))
    sources += plot_b(add_axes_mm(fig,fw,fh,xs4[2],ys[0],34,ah))
    sources += plot_b_quality(add_axes_mm(fig,fw,fh,xs4[3],ys[0],34,ah))
    # Rows 2-3: four standard plots.
    sources += plot_c(add_axes_mm(fig,fw,fh,xs4[0],ys[1],34,ah), add_axes_mm(fig,fw,fh,xs4[1],ys[1],34,ah))
    sources += plot_d(add_axes_mm(fig,fw,fh,xs4[2],ys[1],34,ah), add_axes_mm(fig,fw,fh,xs4[3],ys[1],34,ah))
    sources += plot_e(add_axes_mm(fig,fw,fh,xs4[0],ys[2],34,ah), add_axes_mm(fig,fw,fh,xs4[1],ys[2],34,ah))
    sources += plot_f(add_axes_mm(fig,fw,fh,xs4[2],ys[2],34,ah), add_axes_mm(fig,fw,fh,xs4[3],ys[2],34,ah))
    # Row 4: controls, real-edge boundary and formal synthesis.
    sources += plot_g(add_axes_mm(fig,fw,fh,xs4[0],ys[3],34,ah))
    sources += plot_h(add_axes_mm(fig,fw,fh,xs4[1],ys[3],34,ah),
                      add_axes_mm(fig,fw,fh,xs4[2],ys[3],34,ah))
    sources += plot_i(add_axes_mm(fig,fw,fh,xs4[3],ys[3],34,ah))

    labels = [("a",10.5,ys[0]+15.8),("b",98.5,ys[0]+15.8),
              ("c",10.5,ys[1]+15.8),("d",98.5,ys[1]+15.8),
              ("e",10.5,ys[2]+15.8),("f",98.5,ys[2]+15.8),
              ("g",10.5,ys[3]+15.8),("h",54.5,ys[3]+15.8),
              ("i",142.5,ys[3]+15.8)]
    for lab,x,y in labels:
        fig.text(x/fw,y/fh,lab,fontsize=8,fontweight="bold",ha="left",va="bottom")

    out.mkdir(parents=True, exist_ok=True)
    stem = out/"Fig06_full_nature_trial"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"), dpi=DPI, pil_kwargs={"compression":"tiff_lzw"})
    (out/"axis_geometry_mm.json").write_text(json.dumps({
        "figure_size_mm":[fw,fh], "standard_axis_height_mm":ah,
        "row_order":["a+a+b+b","c+c+d+d","e+e+f+f","g+h+h+i"],
        "wide_axis_mm":[48,ah], "standard_axis_mm":[34,ah],
        "merged_without_information_loss":{
            "f":"AUC excess is AUC minus 0.5; raw AUC remains in source manifest",
            "g":"three mouse-level confound metrics share one axis with all observations retained"
        }
    },indent=2),encoding="utf-8")
    (out/"source_data_manifest.json").write_text(json.dumps({
        "figure":"Fig06", "source_data":sorted({str(Path(p).relative_to(ROOT)) for p in sources})
    },indent=2),encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"status":"COMPLETE","output":str(out)},indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=ap.parse_args(); setup_style(); build(args.out)


if __name__ == "__main__": main()
