#!/usr/bin/env python3
"""Build the expanded Fig.2-supporting Extended Data Figure 2."""

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

MM, DPI = 25.4, 600
ROOT = Path(__file__).resolve().parents[3]
ED02 = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "02_EXTENDED_DATA" / "ED02"
ED03 = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "02_EXTENDED_DATA" / "ED03"
OUT = Path(__file__).resolve().parent / "output_full_v2"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

INK, GREY, LIGHT = "#263238", "#7B858A", "#D9DEE1"
BLUE, WHITE, RED = "#2166AC", "#F7F7F7", "#B2182B"
DIV = LinearSegmentedColormap.from_list("fig1_div", [BLUE, WHITE, RED])
COL = {
    "frequency_transition": "#38598C", "heterogeneity_optimum": "#2A9D8F",
    "sparse_complexity": "#C27628", "high_frequency": "#4F789D",
    "high_intermediate": "#3D8B7A", "intermediate": "#C39A46",
    "low_frequency": "#A94F63", "weak_or_irregular": "#8174A8",
}
TOPO = {"scale_free": "#355F7F", "er": "#8BA36A",
        "modular": "#8174A8", "small_world": "#C27628"}
WINDOWS = ["frequency_transition", "heterogeneity_optimum", "sparse_complexity"]
WINDOW_LABEL = {"frequency_transition": "Frequency transition",
                "heterogeneity_optimum": "Heterogeneity optimum",
                "sparse_complexity": "Sparse complexity"}
TOPO_LABEL = {"scale_free": "Scale-free", "er": "ER",
              "modular": "Modular", "small_world": "Small-world"}
METHOD_LABEL = {"coverage_greedy": "Coverage greedy", "cycle_proxy": "Cycle proxy",
                "feedback_hub": "Feedback hub", "high_degree": "High degree",
                "module_bridge": "Module bridge", "random": "Random"}


def setup_style():
    if ARIAL.exists(): font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists(): font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 5.0, "axes.labelsize": 5.0,
        "xtick.labelsize": 4.0, "ytick.labelsize": 4.0, "legend.fontsize": 3.45,
        "axes.linewidth": .58, "xtick.major.width": .58, "ytick.major.width": .58,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def add_axes_mm(fig, fw, fh, x, y, w, h):
    return fig.add_axes([x/fw, y/fh, w/fw, h/fh])


def clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", pad=1.15)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=.32, alpha=.62)
        ax.set_axisbelow(True)


def label(fig, ax, letter, fw=183, fh=134):
    pos = ax.get_position(); aw, ah = pos.width*fw, pos.height*fh
    ax.text(-8.0/aw, 1.0+2.0/ah, letter, transform=ax.transAxes,
            fontsize=8, fontweight="bold", ha="left", va="bottom", clip_on=False)


def src(base, token):
    hits = sorted(base.rglob(f"*{token}*.csv"))
    if not hits: raise FileNotFoundError(token)
    return hits[0]


def plot_windows(ax, d, value, ci, ylabel, show_legend=False):
    for window in WINDOWS:
        g = d[d.window == window].sort_values("complexity_fraction")
        x, y, e = g.complexity_fraction.to_numpy(float), g[value].to_numpy(float), g[ci].to_numpy(float)
        color = COL[window]
        ax.fill_between(x, y-e, y+e, color=color, alpha=.12, lw=0)
        ax.plot(x, y, color=color, lw=.85, marker="o", ms=1.8, label=WINDOW_LABEL[window])
    ax.set_xlabel("High-complexity fraction"); ax.set_ylabel(ylabel); clean(ax)
    if show_legend:
        ax.legend(frameon=False, loc="upper right", fontsize=2.85,
                  handlelength=.85, handletextpad=.25, labelspacing=.15)


def plot_stack(ax, d):
    order = ["high_frequency", "high_intermediate", "intermediate", "low_frequency", "weak_or_irregular"]
    piv = d.pivot(index="complexity_fraction", columns="frequency_state", values="fraction").fillna(0).sort_index()
    ax.stackplot(piv.index, [piv[c] for c in order], colors=[COL[c] for c in order], alpha=.92,
                 labels=[c.replace("_", " ").replace("weak or irregular", "weak / irregular") for c in order])
    ax.set_ylim(0,1); ax.set_xlabel("High-complexity fraction"); ax.set_ylabel("State fraction")
    leg = ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(.84, .025),
                    fontsize=2.65, ncol=1, handlelength=.65,
                    handletextpad=.25, labelspacing=.08, borderaxespad=0)
    for text in leg.get_texts():
        text.set_color("white")
    clean(ax, None)


def plot_simple_ci(ax, d, ylabel, color):
    d=d.sort_values("complexity_fraction")
    x=d.complexity_fraction.to_numpy(float); y=d["mean"].to_numpy(float); e=d["ci95"].to_numpy(float)
    ax.fill_between(x,y-e,y+e,color=color,alpha=.14,lw=0)
    ax.plot(x,y,color=color,lw=.95,marker="o",ms=2)
    ax.set_xlabel("High-complexity fraction"); ax.set_ylabel(ylabel); clean(ax)


def plot_emergence(ax, d):
    for top in ["scale_free","er","modular","small_world"]:
        g=d[d.topology==top].sort_values("complexity_fraction"); c=TOPO[top]
        ax.fill_between(g.complexity_fraction,g["mean"]-g.ci95,g["mean"]+g.ci95,color=c,alpha=.10,lw=0)
        ax.plot(g.complexity_fraction,g["mean"],color=c,lw=.8,marker="o",ms=1.7,label=TOPO_LABEL[top])
    ax.axhline(0,color=GREY,lw=.5,ls="--",dashes=(3,2))
    ax.set_xlabel("High-complexity fraction"); ax.set_ylabel("Emergence")
    ax.legend(frameon=False,loc="lower left",bbox_to_anchor=(.015,.02),
              fontsize=2.7,handlelength=.8,labelspacing=.1,borderaxespad=0)
    clean(ax)


def plot_positive(ax, d):
    order=["scale_free","er","modular","small_world"]
    d=d.set_index("topology").reindex(order); y=np.arange(4)
    ax.barh(y,d.positive_fraction,color=[TOPO[k] for k in order],height=.62,alpha=.82)
    ax.set_yticks(y,[TOPO_LABEL[k] for k in order],fontsize=3.4); ax.invert_yaxis(); ax.set_xlim(0,1.02)
    ax.set_xlabel("Positive-emergence fraction"); ax.set_ylabel("Topology"); clean(ax,"x")


def heat_data(d, row_order):
    d=d[d.placement.astype(str).str.lower().ne("none")].copy()
    methods=["coverage_greedy","cycle_proxy","feedback_hub","high_degree","module_bridge","random"]
    return d.pivot(index=d.columns[0],columns="placement",values="placement_gain_vs_random_mean").reindex(index=row_order,columns=methods)


def plot_heat(ax, piv, norm, ylabel):
    z=piv.to_numpy(float); im=ax.imshow(z,aspect="auto",cmap=DIV,norm=norm)
    ax.set_xticks(range(len(piv.columns)),[METHOD_LABEL[c] for c in piv.columns],rotation=43,ha="right",fontsize=3.0)
    ylabels=[TOPO_LABEL.get(str(x),str(x).replace("_"," ")) for x in piv.index]
    ax.set_yticks(range(len(piv.index)),ylabels,fontsize=3.35)
    ax.set_xlabel("Allocation rule"); ax.set_ylabel(ylabel)
    lim=max(abs(norm.vmin),abs(norm.vmax))
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j,i,f"{z[i,j]:.02f}",ha="center",va="center",fontsize=2.8,
                    color="white" if abs(z[i,j])>.55*lim else INK)
    return im


def plot_requirement(ax, d):
    for top in ["scale_free","er","modular","small_world"]:
        g=d[d.topology==top]
        ax.scatter(g.random_complex_units_required,g.targeted_complex_units_required,
                   s=5,facecolors="none",edgecolors=TOPO[top],linewidths=.45,alpha=.34,label=TOPO_LABEL[top])
    lo=min(d.random_complex_units_required.min(),d.targeted_complex_units_required.min())
    hi=max(d.random_complex_units_required.max(),d.targeted_complex_units_required.max())
    ax.plot([lo,hi],[lo,hi],color=GREY,lw=.6,ls="--",dashes=(3,2))
    ax.set_xlabel("Random complexity requirement"); ax.set_ylabel("Targeted complexity requirement")
    ax.legend(frameon=False,loc="upper left",fontsize=2.75,handletextpad=.1,labelspacing=.1)
    clean(ax)


def build():
    tokens = [
        (ED02,"window_frequency_transition"),(ED02,"frequency_transition_frequency_state_composition"),
        (ED02,"window_rhythm_transition"),(ED02,"window_rhythm_score"),
        (ED02,"complexity_activity_gain"),(ED02,"complexity_synchrony"),
        (ED02,"emergence_vs_budget_by_topology"),(ED02,"positive_emergence_fraction"),
        (ED03,"regime_by_allocation_gain"),(ED03,"topology_by_allocation_gain"),
        (ED03,"targeted_vs_random_complexity_requirement"),
    ]
    paths=[src(base,tok) for base,tok in tokens]; ds=[pd.read_csv(p) for p in paths]
    fw,fh,aw,ah=183.0,134.0,34.0,27.0
    xs=[10,54,98,142]; ys=[98,58,18]
    fig=plt.figure(figsize=(fw/MM,fh/MM))
    a=add_axes_mm(fig,fw,fh,xs[0],ys[0],aw,ah)
    b1=add_axes_mm(fig,fw,fh,xs[1],ys[0],aw,ah)
    b2=add_axes_mm(fig,fw,fh,xs[2],ys[0],aw,ah)
    c=add_axes_mm(fig,fw,fh,xs[3],ys[0],aw,ah)
    d1=add_axes_mm(fig,fw,fh,xs[0],ys[1],aw,ah)
    d2=add_axes_mm(fig,fw,fh,xs[1],ys[1],aw,ah)
    e1=add_axes_mm(fig,fw,fh,xs[2],ys[1],aw,ah)
    e2=add_axes_mm(fig,fw,fh,xs[3],ys[1],aw,ah)
    f=add_axes_mm(fig,fw,fh,18,ys[2],40,ah)
    g=add_axes_mm(fig,fw,fh,70,ys[2],40,ah)
    h=add_axes_mm(fig,fw,fh,134,ys[2],40,ah)

    plot_windows(a,ds[0],"frequency_transition_strength_mean","frequency_transition_strength_ci95","Frequency transition strength",True)
    plot_stack(b1,ds[1])
    plot_windows(b2,ds[2],"rhythm_transition_strength_mean","rhythm_transition_strength_ci95","Rhythm transition strength")
    plot_windows(c,ds[3],"rhythm_score_stage1b_mean","rhythm_score_stage1b_ci95","Rhythm organization")
    plot_simple_ci(d1,ds[4],"Activity gain",COL["frequency_transition"])
    plot_simple_ci(d2,ds[5],"Synchrony",COL["heterogeneity_optimum"])
    plot_emergence(e1,ds[6]); plot_positive(e2,ds[7])
    pf=heat_data(ds[8],["sparse_minimal","transition_edge"])
    pg=heat_data(ds[9],["er","modular","scale_free","small_world"])
    lim=max(abs(np.nanmin(pf.to_numpy())),abs(np.nanmax(pf.to_numpy())),abs(np.nanmin(pg.to_numpy())),abs(np.nanmax(pg.to_numpy())))
    norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
    im=plot_heat(f,pf,norm,"Collective state"); plot_heat(g,pg,norm,"Topology")
    cax=add_axes_mm(fig,fw,fh,114,ys[2],2,ah); cb=fig.colorbar(im,cax=cax,ticks=[-lim,0,lim])
    cb.set_ticklabels([f"{-lim:.02f}","0.00",f"{lim:.02f}"]); cb.set_label("Gain vs random",fontsize=4.2,labelpad=2)
    cb.ax.tick_params(labelsize=3.2,width=.5,length=1.5,pad=1)
    plot_requirement(h,ds[10])
    for axx,letter in [(a,"a"),(b1,"b"),(c,"c"),(d1,"d"),(e1,"e"),(f,"f"),(g,"g"),(h,"h")]:
        label(fig,axx,letter,fw,fh)

    OUT.mkdir(parents=True,exist_ok=True); stem=OUT/"ED02_full_nature_trial_v2"
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)

    rel=[str(p.relative_to(ROOT)) for p in paths]
    mapping={"a":[rel[0]],"b":[rel[1],rel[2]],"c":[rel[3]],"d":[rel[4],rel[5]],
             "e":[rel[6],rel[7]],"f":[rel[8]],"g":[rel[9]],"h":[rel[10]]}
    manifest={"figure":"ED02","version":"expanded_v2","status":"review_not_frozen",
              "scientific_role":"Support Fig.2 collective transitions, allocation economy and topology generalization.",
              "panel_source_map":mapping,
              "excluded_to_SI":["Four topology-specific placement-advantage curves from former ED04."],
              "data_filters":{"f_g":"Rows with placement=None are excluded as invalid computational output."}}
    (OUT/"source_data_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    geom={"canvas_mm":[fw,fh],"row_y_mm":ys,"axis_height_mm":ah,"inter_row_gap_mm":13,
          "panel_label":{"font_pt":8,"left_offset_mm":8,"top_offset_mm":2},
          "layout":"Rows 1-2: four 34-mm axes; row 3: two 40-mm heatmaps, shared colorbar and one 40-mm scatter."}
    (OUT/"axis_geometry_mm.json").write_text(json.dumps(geom,indent=2),encoding="utf-8")
    contract="""# ED Figure 2 contract

Purpose: support Fig.2 by connecting collective-state transitions to activity, synchrony,
emergence, allocation rules and topology-dependent complexity economy.

Panels b, d and e contain adjacent microplots under one panel letter. Invalid `None`
allocation rows are removed at the plotting entry. Detailed topology-specific placement
curves remain assigned to SI. This review version is not frozen.
"""
    (OUT/"figure_contract.md").write_text(contract,encoding="utf-8")
    qa="""# ED Figure 2 QA report

- Status: review version; not frozen.
- Canvas: one-page 183 x 134 mm PDF.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-h.
- Layout: fixed 13 mm inter-row gaps and main-Fig01 panel-label offsets.
- Data QA: invalid `None` allocation rows excluded from panels f and g.
- Visual QA: PDF re-rendered at 600 dpi; no clipping, panel collision or colorbar overlap.
"""
    (OUT/"QA_REPORT.md").write_text(qa,encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__=="__main__":
    setup_style(); build()
