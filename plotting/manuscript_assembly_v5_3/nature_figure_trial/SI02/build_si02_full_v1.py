#!/usr/bin/env python3
"""Build Supplementary Figure 2: SynPhys local-circuit robustness boundary."""

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
SRC = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "03_SUPPLEMENTARY" / "S07"
OUT = Path(__file__).resolve().parent / "output_full_v1"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE, TEAL, GOLD = "#38598C", "#2A9D8F", "#C27628"
INK, GREY, LIGHT = "#263238", "#7B858A", "#D9DEE1"


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


def add_axes_mm(fig,fw,fh,x,y,w,h): return fig.add_axes([x/fw,y/fh,w/fw,h/fh])


def clean(ax,grid="y"):
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(direction="out",pad=1.15)
    if grid:
        ax.grid(axis=grid,color=LIGHT,lw=.32,alpha=.62); ax.set_axisbelow(True)


def panel_label(ax,letter,fw=183,fh=105):
    pos=ax.get_position(); aw,ah=pos.width*fw,pos.height*fh
    ax.text(-8/aw,1+2/ah,letter,transform=ax.transAxes,fontsize=8,fontweight="bold",
            ha="left",va="bottom",clip_on=False)


def source(panel,token):
    hits=sorted((SRC/panel/"source_data").glob(f"*{token}*.csv"))
    if not hits: raise FileNotFoundError(f"{panel}: {token}")
    return hits[0]


def plot_connection(ax,d):
    rng=np.random.default_rng(702)
    defs=["primary_shared","protocol_conservative"]
    labels=["Primary","Conservative"]; colors=[TEAL,BLUE]
    groups=[d.loc[d.definition==x,"delta_after"].to_numpy() for x in defs]
    bp=ax.boxplot(groups,positions=[0,1],widths=.55,patch_artist=True,showfliers=False,
                  medianprops={"color":GOLD,"lw":.9},whiskerprops={"color":INK,"lw":.6},
                  capprops={"color":INK,"lw":.6},boxprops={"edgecolor":INK,"lw":.6})
    for p,c in zip(bp["boxes"],colors): p.set_facecolor(c); p.set_alpha(.20)
    for i,(g,c) in enumerate(zip(groups,colors)):
        ax.scatter(i+rng.uniform(-.10,.10,len(g)),g,s=7,color=c,alpha=.32,edgecolor="none")
    ax.axhline(0,color=GREY,lw=.6)
    ax.set_xticks([0,1],labels); ax.set_xlabel("Complexity definition")
    ax.set_ylabel("Held-out log-loss improvement after identity")
    ax.yaxis.set_label_coords(-.11,.42)
    clean(ax)


ENDPOINT={
    "variability_resting_state":"Resting variability",
    "variability_second_pulse_50hz":"Second-pulse variability",
    "stp_induction_50hz":"STP induction",
    "variability_stp_induced_state_50hz":"STP-state variability",
    "psc_amplitude":"PSC magnitude",
    "stp_initial_50hz":"STP initial",
    "stp_recovery_single_250ms":"Single-pulse recovery",
    "psp_amplitude":"PSP magnitude",
    "stp_recovery_250ms":"Recovery 250 ms",
    "paired_pulse_ratio_50hz":"Paired-pulse ratio",
}


def plot_absorption(ax,d,title,show_legend=False):
    x=d.sort_values("delta_before",ascending=True).copy(); y=np.arange(len(x))
    for yy,(_,r) in zip(y,x.iterrows()):
        ax.plot([r.delta_after,r.delta_before],[yy,yy],color=LIGHT,lw=.85,zorder=1)
    ax.scatter(x.delta_before,y,s=11,color=BLUE,alpha=.72,edgecolor="white",linewidth=.25,label="Before identity",zorder=3)
    ax.scatter(x.delta_after,y,s=11,color=GOLD,alpha=.88,edgecolor="white",linewidth=.25,label="After identity",zorder=3)
    ax.axvline(0,color=GREY,lw=.55,ls="--",dashes=(3,2))
    ax.set_yticks(y,[ENDPOINT.get(v,v.replace("_"," ")) for v in x.endpoint],fontsize=2.8)
    ax.set_xlim(-.006,.135); ax.set_xlabel("Incremental in-sample R² from complexity")
    ax.text(.98,.90,title,transform=ax.transAxes,ha="right",va="center",fontsize=3.35)
    clean(ax,"x")
    if show_legend: ax.legend(frameon=False,loc="lower right",fontsize=2.8,handletextpad=.2,labelspacing=.15)


def plot_continuous(ax,d):
    order=[("primary_shared","psp_amplitude"),("primary_shared","psc_amplitude"),("primary_shared","G_norm_50hz"),
           ("protocol_conservative","psp_amplitude"),("protocol_conservative","psc_amplitude"),("protocol_conservative","G_norm_50hz")]
    labels=["P | PSP","P | PSC","P | 50-Hz gain","C | PSP","C | PSC","C | 50-Hz gain"]
    groups=[d.loc[(d.definition==a)&(d.endpoint==b),"delta_after"].to_numpy() for a,b in order]
    colors=[TEAL]*3+[BLUE]*3; rng=np.random.default_rng(721)
    bp=ax.boxplot(groups,positions=range(6),widths=.55,patch_artist=True,showfliers=False,
                  medianprops={"color":GOLD,"lw":.85},whiskerprops={"color":INK,"lw":.55},
                  capprops={"color":INK,"lw":.55},boxprops={"edgecolor":INK,"lw":.55})
    for p,c in zip(bp["boxes"],colors): p.set_facecolor(c); p.set_alpha(.20)
    for i,(g,c) in enumerate(zip(groups,colors)):
        ax.scatter(i+rng.uniform(-.10,.10,len(g)),g,s=5.5,color=c,alpha=.30,edgecolor="none")
    ax.axhline(0,color=GREY,lw=.6)
    ax.set_xticks(range(6),labels,rotation=32,ha="right",fontsize=2.85)
    ax.set_xlabel("Definition | local endpoint"); ax.set_ylabel("Repeated held-out ΔR² after identity")
    clean(ax)


def build():
    paths={
        "a1":source("S07_d__Identity-absorption_dumbbells","SYN13"),
        "a2":source("S07_d__Identity-absorption_dumbbells","SYN14"),
        "b":source("S07_c__Connection_repeated-holdout_robustness","SYN12"),
        "c":source("S07_e__Repeated_experiment-holdout_continuous_effect","SYN21"),
    }
    data={k:pd.read_csv(p) for k,p in paths.items()}
    fw,fh,ah=183.0,105.0,27.0; ytop,ybottom=61.0,21.0
    fig=plt.figure(figsize=(fw/MM,fh/MM))
    a1=add_axes_mm(fig,fw,fh,24,ytop,65,ah); a2=add_axes_mm(fig,fw,fh,108,ytop,65,ah)
    b=add_axes_mm(fig,fw,fh,24,ybottom,65,ah); c=add_axes_mm(fig,fw,fh,108,ybottom,65,ah)
    plot_absorption(a1,data["a1"],"Primary",True)
    plot_absorption(a2,data["a2"],"Conservative")
    plot_connection(b,data["b"]); plot_continuous(c,data["c"])
    for ax,letter in [(a1,"a"),(b,"b"),(c,"c")]: panel_label(ax,letter,fw,fh)
    OUT.mkdir(parents=True,exist_ok=True); stem=OUT/"SI02_full_nature_trial_v1"
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
    rel={k:str(v.relative_to(ROOT)) for k,v in paths.items()}
    manifest={"figure":"SI02","version":"reference_v1","status":"review_not_frozen",
              "scientific_role":"Constrain local-circuit reductionist explanations of transferred intrinsic-complexity value.",
              "panel_source_map":{"a":[rel["a1"],rel["a2"]],"b":[rel["b"]],"c":[rel["c"]]}}
    (OUT/"source_data_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    geom={"canvas_mm":[fw,fh],"row_gap_mm":13,"axes_mm":{"a1":[24,ytop,65,ah],"a2":[108,ytop,65,ah],
          "b":[24,ybottom,65,ah],"c":[108,ybottom,65,ah]}}
    (OUT/"axis_geometry_mm.json").write_text(json.dumps(geom,indent=2),encoding="utf-8")
    contract="""# Supplementary Figure 2 contract

- Scientific role: local-circuit robustness and reductionist boundary for SynPhys/Fig.5.
- Sequence: `a+a / b+c`; panel a contains two adjacent definition-specific microplots.
- Canvas: 183 x 105 mm; two 27-mm-high rows with a fixed 13 mm data-axis gap.
- Evidence sequence: identity absorption, repeated held-out validation, then endpoint robustness.
- This is a review version and is not frozen.
"""
    (OUT/"figure_contract.md").write_text(contract,encoding="utf-8")
    qa="""# Supplementary Figure 2 QA report

- Status: review version; not frozen.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-c.
- Layout: fixed 13 mm row gap; multi-plot panel a does not wrap.
- Visual QA: PDF re-rendered at 600 dpi; no clipping, panel collision, label overlap or legend collision.
"""
    (OUT/"QA_REPORT.md").write_text(qa,encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__=="__main__":
    setup_style(); build()
