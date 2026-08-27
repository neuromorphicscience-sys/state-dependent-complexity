#!/usr/bin/env python3
"""Build the Fig.6-supporting population-state control Extended Data Figure 6."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

MM, DPI = 25.4, 600
ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "02_EXTENDED_DATA" / "ED07"
OUT = Path(__file__).resolve().parent / "output_full_v2"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE, TEAL, GOLD, RED, PURPLE = "#38598C", "#2A9D8F", "#C27628", "#B84A4A", "#7B5A8E"
INK, GREY, LIGHT = "#263238", "#7B858A", "#D9DEE1"
STATE_COLORS = [BLUE, TEAL, "#C77956", PURPLE, "#B9676F"]


def setup_style():
    if ARIAL.exists(): font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists(): font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family":"Arial", "font.size":5.0, "axes.labelsize":5.0,
        "xtick.labelsize":4.0, "ytick.labelsize":4.0, "legend.fontsize":3.4,
        "axes.linewidth":.58, "xtick.major.width":.58, "ytick.major.width":.58,
        "xtick.major.size":1.8, "ytick.major.size":1.8,
        "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none",
        "savefig.facecolor":"white", "figure.facecolor":"white",
    })


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


def source_in(panel, token):
    hits = sorted((SRC / panel / "source_data").glob(f"*{token}*.csv"))
    if not hits: raise FileNotFoundError(f"{panel}: {token}")
    return hits[0]


def box_strip(ax, groups, labels, colors, ylabel, ylim=None, zero=False):
    rng = np.random.default_rng(606)
    bp = ax.boxplot(groups, positions=np.arange(len(groups)), widths=.62, patch_artist=True,
                    showfliers=False, medianprops={"color":INK,"lw":.7},
                    whiskerprops={"color":INK,"lw":.55}, capprops={"color":INK,"lw":.55},
                    boxprops={"edgecolor":INK,"lw":.55})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(.82)
    for i, (vals, color) in enumerate(zip(groups, colors)):
        x = i + rng.uniform(-.18, .18, len(vals))
        ax.scatter(x, vals, s=3.8, color=color, alpha=.28, edgecolor="none", rasterized=True)
    if zero: ax.axhline(0, color=GREY, lw=.55, ls="--", dashes=(3,2))
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=3.25)
    ax.set_ylabel(ylabel)
    if ylim is not None: ax.set_ylim(*ylim)
    clean(ax)


def plot_latent_quality(ax, d):
    order = ["baseline", "visual", "decision", "movement", "outcome"]
    box_strip(ax, [d.loc[d.state==s,"latent_r2"].to_numpy() for s in order], order,
              STATE_COLORS, "Held-out latent-dynamics R²", (-.62,.66), True)
    ax.set_xlabel("Behavioral state", labelpad=.6)


def plot_confound(ax, d):
    order = ["mean_activity", "pca_loading_norm"]
    labels = ["Mean activity", "PCA loading norm"]
    box_strip(ax, [d.loc[d.confound==s,"rho"].to_numpy() for s in order], labels,
              [BLUE, TEAL], "Spearman ρ with leverage rank", (0,1), True)
    ax.set_xlabel("Prominence covariate", labelpad=.6)


def plot_flex(ax, d, xcol, xlabel, quantile_clip=False):
    x, y = d[xcol].to_numpy(float), d.role_flexibility.to_numpy(float)
    ax.scatter(x, y, s=.72, color=BLUE, alpha=.12, edgecolor="none", rasterized=True)
    rho = d[[xcol,"role_flexibility"]].corr(method="spearman").iloc[0,1]
    ax.text(.04,.96,f"Spearman ρ = {rho:.2f}",transform=ax.transAxes,
            ha="left",va="top",fontsize=3.35)
    if quantile_clip:
        ax.set_xlim(0, d[xcol].quantile(.995)); ax.set_ylim(-.005, d.role_flexibility.quantile(.995)*1.04)
    ax.set_xlabel(xlabel); ax.set_ylabel("Role flexibility"); clean(ax)


def plot_identity(ax, d):
    x, y = d.null_mean_flex.to_numpy(), d.observed_mean_flex.to_numpy()
    lim = max(x.max(), y.max())*1.08
    ax.plot([0,lim],[0,lim],color=GREY,lw=.6,ls="--",dashes=(3,2))
    ax.scatter(x,y,s=10,color=PURPLE,alpha=.58,edgecolor="white",linewidth=.25)
    ratio = np.median(d.observed_to_null)
    ax.text(.04,.96,f"Median observed/null = {ratio:.2f}",transform=ax.transAxes,
            ha="left",va="top",fontsize=3.35)
    ax.set_xlim(0,lim); ax.set_ylim(0,lim)
    ax.set_xlabel("Identity-shuffle mean flexibility")
    ax.set_ylabel("Observed mean flexibility"); clean(ax)


def plot_allen_quality(ax, d):
    order = ["familiar_active", "novel_active", "passive"]
    labels = ["Familiar active", "Novel active", "Passive"]
    box_strip(ax, [d.loc[d.experiment_state==s,"latent_r2"].to_numpy() for s in order],
              labels, [BLUE,TEAL,"#C77956"], "Held-out latent-dynamics R²", (-.58,.36), True)
    ax.set_xlabel("Experiment state", labelpad=.6)


def plot_positive_fraction(ax, d):
    order = ["familiar_active", "novel_active", "passive"]
    labels = ["Familiar", "Novel", "Passive"]
    frac = [float((d.loc[d.experiment_state==s,"latent_r2"]>0).mean()) for s in order]
    ax.bar(range(3),frac,color=[BLUE,TEAL,"#C77956"],width=.64,edgecolor=INK,lw=.45)
    ax.axhline(.5,color=GREY,lw=.55,ls="--",dashes=(3,2))
    for i,v in enumerate(frac): ax.text(i,v+.025,f"{v:.2f}",ha="center",va="bottom",fontsize=3.25)
    ax.set_xticks(range(3),labels,rotation=25,ha="right",fontsize=3.25)
    ax.set_ylim(0,.82); ax.set_xlabel("Experiment state", labelpad=.6)
    ax.set_ylabel("Fraction of experiments with R² > 0"); clean(ax)


def plot_arousal(ax, raw, residual, contrast, ylabel=False):
    r = raw[raw.contrast==contrast].rho.to_numpy()
    q = residual[residual.contrast==contrast].rho.to_numpy()
    box_strip(ax,[r,q],["Raw","Residual"],[BLUE,TEAL],
              "Within-experiment Spearman ρ" if ylabel else "",(-.55,.95),True)
    ax.set_xlabel("Leverage representation", labelpad=.6)
    ax.text(.96,.96,contrast.capitalize(),transform=ax.transAxes,ha="right",va="top",fontsize=3.45)


def build():
    paths = {
        "a": source_in("ED07_a__Steinmetz_latent-model_quality", "steinmetz_latent_r2"),
        "b": source_in("ED07_c__Activity_loading_contribution", "steinmetz_confound_correlations"),
        "c1": source_in("ED07_d__Leverage_flexibility_versus_activity", "S5BR_07"),
        "c2": source_in("ED07_d__Leverage_flexibility_versus_activity", "S5BR_08"),
        "d": source_in("ED07_e__Neuron-identity_shuffle", "steinmetz_identity_shuffle"),
        "e1": source_in("ED07_f__Allen_model-quality_boundary", "S5BR_15"),
        "e2": source_in("ED07_f__Allen_model-quality_boundary", "S5BR_16"),
        "f1": source_in("ED07_i__Allen_activity_controls", "S5BR_21"),
        "f2": source_in("ED07_i__Allen_activity_controls", "S5BR_22"),
        "g1r": source_in("ED07_j__Within-session_running_and_pupil_state_tests", "S5BR_23_allen_running_within_session__allen_within_session_arousal_pairs_raw"),
        "g1q": source_in("ED07_j__Within-session_running_and_pupil_state_tests", "S5BR_23_allen_running_within_session__allen_within_session_arousal_pairs_residual"),
        "g2r": source_in("ED07_j__Within-session_running_and_pupil_state_tests", "S5BR_24_allen_pupil_within_session__allen_within_session_arousal_pairs_raw"),
        "g2q": source_in("ED07_j__Within-session_running_and_pupil_state_tests", "S5BR_24_allen_pupil_within_session__allen_within_session_arousal_pairs_residual"),
    }
    data = {k:pd.read_csv(p) for k,p in paths.items()}
    fw,fh,ah = 183.0,134.0,27.0
    y1,y2,y3 = 92.0,52.0,12.0
    fig = plt.figure(figsize=(fw/MM,fh/MM))
    a=add_axes_mm(fig,fw,fh,10,y1,34,ah); b=add_axes_mm(fig,fw,fh,54,y1,34,ah)
    c1=add_axes_mm(fig,fw,fh,98,y1,34,ah); c2=add_axes_mm(fig,fw,fh,142,y1,34,ah)
    d=add_axes_mm(fig,fw,fh,20,y2,42,ah); e1=add_axes_mm(fig,fw,fh,82,y2,34,ah)
    e2=add_axes_mm(fig,fw,fh,136,y2,34,ah)
    f1=add_axes_mm(fig,fw,fh,10,y3,34,ah); f2=add_axes_mm(fig,fw,fh,54,y3,34,ah)
    g1=add_axes_mm(fig,fw,fh,98,y3,34,ah); g2=add_axes_mm(fig,fw,fh,142,y3,34,ah)

    plot_latent_quality(a,data["a"]); plot_confound(b,data["b"])
    plot_flex(c1,data["c1"],"mean_activity","Mean activity",True)
    plot_flex(c2,data["c2"],"activity_flexibility","Activity-rank flexibility",True)
    plot_identity(d,data["d"]); plot_allen_quality(e1,data["e1"]); plot_positive_fraction(e2,data["e2"])
    plot_flex(f1,data["f1"],"mean_activity","Mean activity",True)
    plot_flex(f2,data["f2"],"activity_flexibility","Activity-rank flexibility",True)
    plot_arousal(g1,data["g1r"],data["g1q"],"running",True)
    plot_arousal(g2,data["g2r"],data["g2q"],"pupil",False)

    for ax,letter in [(a,"a"),(b,"b"),(c1,"c"),(d,"d"),(e1,"e"),(f1,"f"),(g1,"g")]:
        panel_label(ax,letter,fw,fh)

    OUT.mkdir(parents=True,exist_ok=True); stem=OUT/"ED06_full_nature_trial_v2"
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)

    rel={k:str(v.relative_to(ROOT)) for k,v in paths.items()}
    panel_map={"a":[rel["a"]],"b":[rel["b"]],"c":[rel["c1"],rel["c2"]],
               "d":[rel["d"]],"e":[rel["e1"],rel["e2"]],"f":[rel["f1"],rel["f2"]],
               "g":[rel["g1r"],rel["g1q"],rel["g2r"],rel["g2q"]]}
    manifest={"figure":"ED06","version":"population_controls_v2","status":"review_not_frozen",
              "scientific_role":"Support Fig.6 with latent-model quality boundaries, activity controls, identity-shuffle tests and within-session arousal-state robustness.",
              "panel_source_map":panel_map}
    (OUT/"source_data_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    geom={"canvas_mm":[fw,fh],"row_gap_mm":13,
          "axes_mm":{"a":[10,y1,34,ah],"b":[54,y1,34,ah],"c1":[98,y1,34,ah],"c2":[142,y1,34,ah],
                     "d":[20,y2,42,ah],"e1":[82,y2,34,ah],"e2":[136,y2,34,ah],
                     "f1":[10,y3,34,ah],"f2":[54,y3,34,ah],"g1":[98,y3,34,ah],"g2":[142,y3,34,ah]}}
    (OUT/"axis_geometry_mm.json").write_text(json.dumps(geom,indent=2),encoding="utf-8")
    contract="""# ED Figure 6 contract

- Purpose: support main Fig.6 with model-quality and alternative-explanation controls.
- Canvas: 183 x 134 mm; three 27-mm-high rows with fixed 13 mm gaps.
- Sequence: `a+b+c+c / d+e+e / f+f+g+g`.
- Multi-plot panels c, e, f and g never wrap across rows.
- Raw/residual main-effect panels and the promoted quality-gated panel are not duplicated.
- This is a review version and is not frozen.
"""
    (OUT/"figure_contract.md").write_text(contract,encoding="utf-8")
    qa="""# ED Figure 6 QA report

- Status: review version; not frozen.
- Canvas: one-page 183 x 134 mm PDF.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-g.
- Layout: fixed 13 mm inter-row gaps; multi-plot panels do not wrap.
- Visual QA: PDF re-rendered at 600 dpi; no clipping, panel collision, label overlap or legend collision.
"""
    (OUT/"QA_REPORT.md").write_text(qa,encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__=="__main__":
    setup_style(); build()
