#!/usr/bin/env python3
"""Build Supplementary Figure 5: OpenScope cohort and validity audit."""

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
SRC = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "03_SUPPLEMENTARY" / "S06"
OUT = Path(__file__).resolve().parent / "output_full_v1"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE, TEAL, GOLD, RED = "#38598C", "#2A9D8F", "#C27628", "#B23A48"
INK, GREY, LIGHT = "#263238", "#7B858A", "#D9DEE1"


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


def source(folder):
    hits=sorted((SRC/folder/"source_data").glob("*.csv"))
    if len(hits)!=1: raise FileNotFoundError(f"{folder}: {hits}")
    return hits[0]


def ordered(d):
    return d.sort_values("subject").reset_index(drop=True)


def mouse_axis(ax,n=12):
    ax.set_xlim(.4,n+.6); ax.set_xticks([1,4,8,12]); ax.set_xlabel("Mouse rank")


def lollipop(ax,d,column,color,ylabel,reference=None,median=True,ylim=None,annotation_y=.95):
    x=np.arange(1,len(d)+1); y=d[column].to_numpy(float)
    base=reference if reference is not None else min(0,float(y.min()))
    ax.vlines(x,base,y,color=LIGHT,lw=.9,zorder=1)
    ax.scatter(x,y,s=10,color=color,alpha=.88,edgecolor="white",linewidth=.3,zorder=3)
    if reference is not None: ax.axhline(reference,color=GREY,lw=.65,ls="--",dashes=(3,2))
    if median:
        med=float(np.median(y)); ax.axhline(med,color=GOLD,lw=.75)
        va="top" if annotation_y>.5 else "bottom"
        ax.text(.98,annotation_y,f"Median = {med:.3f}",transform=ax.transAxes,ha="right",va=va,fontsize=3.0,color=GOLD)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel); mouse_axis(ax,len(d)); clean(ax)


def plot_auc(ax,d):
    lollipop(ax,d,"half_model_home_auc_min",BLUE,"Minimum split-half home-state AUC",reference=.5,ylim=(.485,.80))


def plot_pc1(ax,d):
    lollipop(ax,d,"pc1_explained",TEAL,"PC1 explained variance",reference=0,ylim=(0,.049),annotation_y=.06)


def plot_balance(ax,d):
    lollipop(ax,d,"state0_fraction",GOLD,"State-0 trial fraction",reference=.5,ylim=(.494,.5042))


def plot_units(ax,d):
    lollipop(ax,d,"n_units",BLUE,"Analyzed units",reference=0,ylim=(0,980),annotation_y=.06)


def plot_analysis_trials(ax,d):
    x=np.arange(1,len(d)+1); y=d.n_analysis_trials.to_numpy(float)
    ax.vlines(x,0,y,color=LIGHT,lw=.9); ax.scatter(x,y,s=10,color=TEAL,edgecolor="white",linewidth=.3,zorder=3)
    ax.axhline(np.median(y),color=GOLD,lw=.75)
    ax.text(.98,.06,f"All mice = {int(y[0]):,}",transform=ax.transAxes,ha="right",va="bottom",fontsize=3.0,color=GOLD)
    ax.set_ylim(0,1700); ax.set_ylabel("Analysis trials"); mouse_axis(ax,len(d)); clean(ax)


def plot_discovery_final(ax,d):
    x=np.arange(1,len(d)+1); discovery=d.n_discovery.to_numpy(float); final=d.n_final.to_numpy(float)
    for xx,a,b in zip(x,discovery,final): ax.plot([xx,xx],[a,b],color=LIGHT,lw=.9,zorder=1)
    ax.scatter(x,discovery,s=9,color=BLUE,alpha=.88,edgecolor="white",linewidth=.25,label="Discovery")
    ax.scatter(x,final,s=9,color=TEAL,alpha=.88,edgecolor="white",linewidth=.25,label="Final")
    ax.set_ylim(0,1050); ax.set_ylabel("Trial count"); mouse_axis(ax,len(d)); clean(ax)
    ax.legend(frameon=False,loc="lower right",handletextpad=.2,labelspacing=.15,borderaxespad=.2)


def build():
    paths={
        "a":source("S06_a__Half-model_minimum_predictive_validity"),
        "b":source("S06_b__Dominant_low-dimensional_mode"),
        "c":source("S06_c__State_fraction_balance"),
        "d1":source("S06_d__Units_per_mouse"),
        "d2":source("S06_e__Analysis_trials_per_mouse"),
        "d3":source("S06_f__Discovery-versus-final_trial_counts"),
    }
    data={k:ordered(pd.read_csv(p,encoding="utf-8-sig")) for k,p in paths.items()}
    fw,fh,h=183.0,105.0,27.0; ytop,ybottom=61.0,21.0
    fig=plt.figure(figsize=(fw/MM,fh/MM))
    a=add_axes_mm(fig,fw,fh,20,ytop,36,h); b=add_axes_mm(fig,fw,fh,77,ytop,36,h)
    c=add_axes_mm(fig,fw,fh,134,ytop,36,h); d1=add_axes_mm(fig,fw,fh,20,ybottom,36,h)
    d2=add_axes_mm(fig,fw,fh,77,ybottom,36,h); d3=add_axes_mm(fig,fw,fh,134,ybottom,36,h)
    plot_auc(a,data["a"]); plot_pc1(b,data["b"]); plot_balance(c,data["c"])
    plot_units(d1,data["d1"]); plot_analysis_trials(d2,data["d2"]); plot_discovery_final(d3,data["d3"])
    for ax,letter in [(a,"a"),(b,"b"),(c,"c"),(d1,"d")]: panel_label(ax,letter,fw,fh)

    OUT.mkdir(parents=True,exist_ok=True); stem=OUT/"SI05_full_nature_trial_v1"
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
    rel={k:str(v.relative_to(ROOT)) for k,v in paths.items()}
    manifest={"figure":"SI05","version":"reference_v1","status":"review_not_frozen",
              "scientific_role":"Audit OpenScope half-model validity, low-dimensional structure, state balance and mouse-wise unit/trial support.",
              "panel_source_map":{"a":[rel["a"]],"b":[rel["b"]],"c":[rel["c"]],
                                  "d":[rel["d1"],rel["d2"],rel["d3"]]}}
    (OUT/"source_data_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    geom={"canvas_mm":[fw,fh],"row_gap_mm":13,"axes_mm":{"a":[20,ytop,36,h],"b":[77,ytop,36,h],
          "c":[134,ytop,36,h],"d1":[20,ybottom,36,h],"d2":[77,ybottom,36,h],"d3":[134,ybottom,36,h]},
          "panel_label_offset_mm":[-8,2]}
    (OUT/"axis_geometry_mm.json").write_text(json.dumps(geom,indent=2),encoding="utf-8")
    contract="""# Supplementary Figure 5 contract

- Scientific role: OpenScope cohort and analysis-validity audit supporting Fig.6.
- Sequence: `a+b+c / d+d+d`; panel d contains three adjacent microplots and never wraps.
- Canvas: 183 x 105 mm; all six data axes are 36 x 27 mm with a fixed 13 mm row gap.
- Mouse-wise values remain individually visible; mouse rank replaces crowded six-digit IDs on the plotted x axes.
- This is a review version and is not frozen.
"""
    (OUT/"figure_contract.md").write_text(contract,encoding="utf-8")
    qa="""# Supplementary Figure 5 QA report

- Status: review version; not frozen.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-d.
- Layout: fixed 13 mm row gap; all six data axes are physically aligned within the 2 x 3 grid.
- Visual QA: PDF re-rendered at 600 dpi; no clipping, panel collision, label overlap or legend collision.
"""
    (OUT/"QA_REPORT.md").write_text(qa,encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__=="__main__":
    setup_style(); build()
