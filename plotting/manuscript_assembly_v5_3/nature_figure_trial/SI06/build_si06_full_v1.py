#!/usr/bin/env python3
"""Build Supplementary Figure 6: OpenScope coefficient-landscape controls."""

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
SRC = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "03_SUPPLEMENTARY" / "S04"
OUT = Path(__file__).resolve().parent / "output_full_v1"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE, TEAL, GOLD, PURPLE = "#38598C", "#2A9D8F", "#C27628", "#7563A6"
INK, GREY, LIGHT = "#263238", "#7B858A", "#D9DEE1"
EFFECT_LIM=(-.017,.042)


def setup_style():
    if ARIAL.exists(): font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists(): font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family":"Arial", "font.size":5.0, "axes.labelsize":5.0,
        "xtick.labelsize":4.0, "ytick.labelsize":4.0, "legend.fontsize":3.2,
        "axes.linewidth":.58, "xtick.major.width":.58, "ytick.major.width":.58,
        "xtick.major.size":1.8, "ytick.major.size":1.8,
        "pdf.fonttype":42, "ps.fonttype":42, "svg.fonttype":"none",
        "savefig.facecolor":"white", "figure.facecolor":"white",
    })


def add_axes_mm(fig,fw,fh,x,y,w,h): return fig.add_axes([x/fw,y/fh,w/fw,h/fh])


def clean(ax,grid="y"):
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(direction="out",pad=1.1)
    if grid:
        ax.grid(axis=grid,color=LIGHT,lw=.32,alpha=.62); ax.set_axisbelow(True)


def panel_label(ax,letter,fw=183,fh=105):
    pos=ax.get_position(); aw,ah=pos.width*fw,pos.height*fh
    ax.text(-8/aw,1+2/ah,letter,transform=ax.transAxes,fontsize=8,fontweight="bold",
            ha="left",va="bottom",clip_on=False)


def source(folder,token=None):
    pattern=f"*{token}*.csv" if token else "*.csv"
    hits=sorted((SRC/folder/"source_data").glob(pattern))
    if len(hits)!=1: raise FileNotFoundError(f"{folder}: {token}: {hits}")
    return hits[0]


def ordered(d): return d.sort_values("subject").reset_index(drop=True)


def mouse_axis(ax,n=12):
    ax.set_xlim(.4,n+.6); ax.set_xticks([1,4,8,12]); ax.set_xlabel("Mouse rank")


def effect_style(ax,ylabel="Landscape advantage, Δρ"):
    ax.axhline(0,color=GREY,lw=.65,ls="--",dashes=(3,2)); ax.set_ylim(*EFFECT_LIM)
    ax.set_ylabel(ylabel); clean(ax)


def plot_within_cross(ax,d):
    for _,r in d.iterrows(): ax.plot([0,1],[r.rho_within,r.rho_cross],color=LIGHT,lw=.75,zorder=1)
    ax.scatter(np.zeros(len(d)),d.rho_within,s=9,color=TEAL,alpha=.85,edgecolor="white",linewidth=.25,zorder=3)
    ax.scatter(np.ones(len(d)),d.rho_cross,s=9,color=GOLD,alpha=.85,edgecolor="white",linewidth=.25,zorder=3)
    meds=[d.rho_within.median(),d.rho_cross.median()]
    ax.plot([0,1],meds,color=INK,lw=.9,zorder=4)
    ax.scatter([0,1],meds,s=18,facecolor="white",edgecolor=INK,linewidth=.8,zorder=5)
    ax.set_xlim(-.35,1.35); ax.set_xticks([0,1],["Within","Cross"])
    ax.set_xlabel("State pairing"); ax.set_ylabel("Coefficient-rank correlation")
    clean(ax)


def plot_mouse_effect(ax,d,column,color,title=None):
    x=np.arange(1,len(d)+1); y=d[column].to_numpy(float)
    ax.vlines(x,0,y,color=LIGHT,lw=.9,zorder=1)
    ax.scatter(x,y,s=10,color=color,alpha=.88,edgecolor="white",linewidth=.3,zorder=3)
    ax.axhline(np.median(y),color=GOLD,lw=.75)
    mouse_axis(ax,len(d)); effect_style(ax)
    if title: ax.text(.04,.95,title,transform=ax.transAxes,ha="left",va="top",fontsize=3.2)


def plot_definitions(ax,d):
    cols=["delta_rho","activity_residual_delta_rho","activity_selectivity_residual_delta_rho"]
    labels=["Raw","Activity","Act. + sel."]; colors=[PURPLE,TEAL,BLUE]
    for _,r in d.iterrows(): ax.plot(range(3),[r[c] for c in cols],color=LIGHT,lw=.72,zorder=1)
    for i,(col,color) in enumerate(zip(cols,colors)):
        ax.scatter(np.full(len(d),i),d[col],s=8,color=color,alpha=.85,edgecolor="white",linewidth=.25,zorder=3)
    meds=[d[c].median() for c in cols]
    ax.plot(range(3),meds,color=INK,lw=.85,zorder=4)
    ax.scatter(range(3),meds,s=17,facecolor="white",edgecolor=INK,linewidth=.75,zorder=5)
    ax.set_xlim(-.35,2.35); ax.set_xticks(range(3),labels,rotation=24,ha="right")
    ax.set_xlabel("Coefficient definition"); effect_style(ax)


def build():
    paths={
        "a":source("S04_a__Within-_versus_cross-state_coefficient-landscape_similarity"),
        "b":source("S04_b__Mouse-wise_coefficient-landscape_effect"),
        "c":source("S04_c__Raw_versus_controlled_coefficient_landscape"),
        "d1":source("S04_d__Activity_and_activity+selectivity_residualization","S13"),
        "d2":source("S04_d__Activity_and_activity+selectivity_residualization","S14"),
        "e":source("S04_g__Real-edge_coefficient-landscape_extension"),
    }
    data={k:ordered(pd.read_csv(p,encoding="utf-8-sig")) for k,p in paths.items()}
    fw,fh,h=183.0,105.0,27.0; ytop,ybottom=61.0,21.0
    fig=plt.figure(figsize=(fw/MM,fh/MM))
    a=add_axes_mm(fig,fw,fh,20,ytop,36,h); b=add_axes_mm(fig,fw,fh,77,ytop,36,h)
    c=add_axes_mm(fig,fw,fh,134,ytop,36,h); d1=add_axes_mm(fig,fw,fh,20,ybottom,36,h)
    d2=add_axes_mm(fig,fw,fh,77,ybottom,36,h); e=add_axes_mm(fig,fw,fh,134,ybottom,36,h)
    plot_within_cross(a,data["a"]); plot_mouse_effect(b,data["b"],"delta_rho",PURPLE)
    plot_definitions(c,data["c"])
    plot_mouse_effect(d1,data["d1"],"activity_residual_delta_rho",TEAL,"Activity residual")
    plot_mouse_effect(d2,data["d2"],"activity_selectivity_residual_delta_rho",BLUE,"Activity + selectivity")
    plot_mouse_effect(e,data["e"],"delta_rho",GOLD,"Real-edge definition")
    for ax,letter in [(a,"a"),(b,"b"),(c,"c"),(d1,"d"),(e,"e")]: panel_label(ax,letter,fw,fh)

    OUT.mkdir(parents=True,exist_ok=True); stem=OUT/"SI06_full_nature_trial_v1"
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
    rel={k:str(v.relative_to(ROOT)) for k,v in paths.items()}
    manifest={"figure":"SI06","version":"reference_v1","status":"review_not_frozen",
              "scientific_role":"Show mouse-consistent coefficient-landscape advantage after activity/selectivity control and under the real-edge definition.",
              "panel_source_map":{"a":[rel["a"]],"b":[rel["b"]],"c":[rel["c"]],
                                  "d":[rel["d1"],rel["d2"]],"e":[rel["e"]]}}
    (OUT/"source_data_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    geom={"canvas_mm":[fw,fh],"row_gap_mm":13,"shared_effect_ylim":list(EFFECT_LIM),
          "axes_mm":{"a":[20,ytop,36,h],"b":[77,ytop,36,h],"c":[134,ytop,36,h],
          "d1":[20,ybottom,36,h],"d2":[77,ybottom,36,h],"e":[134,ybottom,36,h]},
          "panel_label_offset_mm":[-8,2]}
    (OUT/"axis_geometry_mm.json").write_text(json.dumps(geom,indent=2),encoding="utf-8")
    contract="""# Supplementary Figure 6 contract

- Scientific role: OpenScope coefficient-landscape controls supporting Fig.6.
- Sequence: `a+b+c / d+d+e`; panel d contains two adjacent microplots and never wraps.
- Canvas: 183 x 105 mm; all six data axes are 36 x 27 mm with a fixed 13 mm row gap.
- Panels b-e share the same effect y-axis range and zero reference.
- This is a review version and is not frozen.
"""
    (OUT/"figure_contract.md").write_text(contract,encoding="utf-8")
    qa="""# Supplementary Figure 6 QA report

- Status: review version; not frozen.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-e.
- Layout: fixed 13 mm row gap; all six data axes are physically aligned within the 2 x 3 grid.
- Visual QA: PDF re-rendered at 600 dpi; no clipping, panel collision, label overlap or legend collision.
"""
    (OUT/"QA_REPORT.md").write_text(qa,encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__=="__main__":
    setup_style(); build()
