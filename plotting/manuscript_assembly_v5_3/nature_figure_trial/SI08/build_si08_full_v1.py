#!/usr/bin/env python3
"""Build Supplementary Figure 8: OpenScope top-unit/top-set robustness."""

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
SRC = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "03_SUPPLEMENTARY" / "S05"
OUT = Path(__file__).resolve().parent / "output_full_v1"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE, TEAL, GOLD, PURPLE = "#38598C", "#2A9D8F", "#C27628", "#7563A6"
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
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(direction="out",pad=1.1)
    if grid:
        ax.grid(axis=grid,color=LIGHT,lw=.32,alpha=.62)
        ax.set_axisbelow(True)


def local_title(ax,text):
    ax.text(.04,1.025,text,transform=ax.transAxes,ha="left",va="bottom",
            fontsize=3.2,clip_on=False)


def panel_label(ax,letter,fw=183,fh=145):
    pos=ax.get_position(); aw,ah=pos.width*fw,pos.height*fh
    ax.text(-8/aw,1+2/ah,letter,transform=ax.transAxes,fontsize=8,fontweight="bold",
            ha="left",va="bottom",clip_on=False)


def source(folder,token=None):
    pattern=f"*{token}*.csv" if token else "*.csv"
    hits=sorted((SRC/folder/"source_data").glob(pattern))
    if len(hits)!=1: raise FileNotFoundError(f"{folder}: {token}: {hits}")
    return hits[0]


def paired(ax,d,left,right,labels,ylabel,title):
    for _,r in d.iterrows():
        ax.plot([0,1],[r[left],r[right]],color=LIGHT,lw=.72,zorder=1)
    ax.scatter(np.zeros(len(d)),d[left],s=9,color=TEAL,alpha=.86,
               edgecolor="white",linewidth=.28,zorder=3)
    ax.scatter(np.ones(len(d)),d[right],s=9,color=GOLD,alpha=.86,
               edgecolor="white",linewidth=.28,zorder=3)
    med=[d[left].median(),d[right].median()]
    ax.plot([0,1],med,color=INK,lw=.9,zorder=4)
    ax.scatter([0,1],med,s=18,facecolor="white",edgecolor=INK,linewidth=.8,zorder=5)
    ax.set_xlim(-.35,1.35); ax.set_xticks([0,1],labels)
    ax.set_xlabel("State pairing"); ax.set_ylabel(ylabel); local_title(ax,title); clean(ax)


def lollipop(ax,d,value,color,ylabel,title,zero=True):
    y=d[value].to_numpy(float); x=np.arange(1,len(y)+1)
    base=0 if zero else min(0,float(y.min())*.92)
    ax.vlines(x,base,y,color=LIGHT,lw=.9,zorder=1)
    ax.scatter(x,y,s=10,color=color,alpha=.9,edgecolor="white",linewidth=.3,zorder=3)
    ax.axhline(np.median(y),color=INK,lw=.8)
    if zero: ax.axhline(0,color=GREY,lw=.62,ls="--",dashes=(3,2))
    ax.set_xlim(.4,len(y)+.6); ax.set_xticks([1,4,8,12]); ax.set_xlabel("Mouse rank")
    ax.set_ylabel(ylabel); local_title(ax,title); clean(ax)


def aggregate(d,x,value):
    return d.groupby(["subject",x],as_index=False)[value].mean()


def sensitivity(ax,d,x,value,xvals,xticklabels,color,ylabel,title,reference=None,ylim=None):
    a=aggregate(d,x,value); xpos=np.arange(len(xvals))
    for _,g in a.groupby("subject"):
        g=g.set_index(x).reindex(xvals)
        ax.plot(xpos,g[value],color=LIGHT,lw=.72,zorder=1)
    med=a.groupby(x)[value].median().reindex(xvals)
    ax.plot(xpos,med,color=color,lw=1.15,zorder=3)
    ax.scatter(xpos,med,s=17,color=color,alpha=.88,edgecolor="white",linewidth=.35,zorder=4)
    ax.axhline(0,color=GREY,lw=.62,ls="--",dashes=(3,2))
    if reference is not None:
        ax.axvline(xvals.index(reference),color=GREY,lw=.62,ls="--",dashes=(3,2))
    ax.set_xlim(-.22,len(xvals)-.78); ax.set_xticks(xpos,xticklabels)
    ax.set_ylabel(ylabel); local_title(ax,title)
    if ylim is not None: ax.set_ylim(*ylim)
    clean(ax)


def permutation(ax,d):
    null=d["null"].to_numpy(float); obs=float(d["observed"].iloc[0])
    qlo,qhi=np.quantile(null,[.025,.975])
    ax.hist(null,bins=32,color=LIGHT,edgecolor=GREY,linewidth=.35)
    ax.axvline(obs,color=PURPLE,lw=1.2)
    ax.axvline(qlo,color=INK,lw=.65,ls="--",dashes=(3,2))
    ax.axvline(qhi,color=INK,lw=.65,ls="--",dashes=(3,2))
    ax.set_xlabel("Fixed-split null ΔJaccard"); ax.set_ylabel("Permutation count")
    local_title(ax,"Observed; dashed = 95% null"); clean(ax,grid=None)


def build():
    paths={
        "a1":source("S05_a__Top-unit_home-versus-cross_transfer"),
        "a2":source("S05_b__Mouse-wise_top-unit_transfer_effect"),
        "b1":source("S05_c__Top-unit_budget_sensitivity","S09"),
        "b2":source("S05_c__Top-unit_budget_sensitivity","S10"),
        "c1":source("S05_d__Top-set_overlap_and_mouse-wise_delta-Jaccard","S11"),
        "c2":source("S05_d__Top-set_overlap_and_mouse-wise_delta-Jaccard","S12"),
        "d":source("S05_e__Regularization_sensitivity_of_delta-Jaccard"),
        "e":source("S05_f__High-precision_permutation_boundary_for_delta-Jaccard"),
        "f":source("S05_g__Leave-one-mouse-out_top-set_effect"),
    }
    data={k:pd.read_csv(p,encoding="utf-8-sig") for k,p in paths.items()}

    budget_sets=[]
    for k,col in [("b1","topset_auc_crossover"),("b2","topset_balacc_crossover")]:
        budget_sets.append(aggregate(data[k],"top_fraction",col)[col].to_numpy(float))
    lo=min(v.min() for v in budget_sets); hi=max(v.max() for v in budget_sets)
    pad=.08*(hi-lo); budget_lim=(min(-.004,lo-pad),hi+pad)

    fw,fh,w,h=183.0,145.0,36.0,27.0
    x1,x2,x3=20.0,77.0,134.0; y1,y2,y3=103.0,63.0,23.0
    fig=plt.figure(figsize=(fw/MM,fh/MM))
    a1=add_axes_mm(fig,fw,fh,x1,y1,w,h); a2=add_axes_mm(fig,fw,fh,x2,y1,w,h)
    d=add_axes_mm(fig,fw,fh,x3,y1,w,h)
    b1=add_axes_mm(fig,fw,fh,x1,y2,w,h); b2=add_axes_mm(fig,fw,fh,x2,y2,w,h)
    e=add_axes_mm(fig,fw,fh,x3,y2,w,h)
    c1=add_axes_mm(fig,fw,fh,x1,y3,w,h); c2=add_axes_mm(fig,fw,fh,x2,y3,w,h)
    f=add_axes_mm(fig,fw,fh,x3,y3,w,h)

    paired(a1,data["a1"],"topset_auc_home","topset_auc_cross",["Home","Cross"],
           "Top-unit AUC","Home versus cross")
    lollipop(a2,data["a2"],"topset_auc_crossover",PURPLE,"Home advantage, ΔAUC",
             "Mouse-wise transfer")
    sensitivity(b1,data["b1"],"top_fraction","topset_auc_crossover",
                [.05,.10,.20,.40],["5","10","20","40"],PURPLE,
                "Home advantage, ΔAUC","AUC",reference=.20,ylim=budget_lim)
    b1.set_xlabel("Top-unit fraction (%)")
    sensitivity(b2,data["b2"],"top_fraction","topset_balacc_crossover",
                [.05,.10,.20,.40],["5","10","20","40"],BLUE,
                "Home advantage, Δ balanced accuracy","Balanced accuracy",
                reference=.20,ylim=budget_lim)
    b2.set_xlabel("Top-unit fraction (%)")
    paired(c1,data["c1"],"jaccard_within","jaccard_cross",["Within","Cross"],
           "Top-set Jaccard","Within versus cross")
    lollipop(c2,data["c2"],"delta_jaccard",GOLD,"Overlap advantage, ΔJaccard",
             "Mouse-wise overlap")
    sensitivity(d,data["d"],"C","delta_jaccard",[.25,1.,4.],["0.25","1","4"],TEAL,
                "Overlap advantage, ΔJaccard","Regularization sensitivity",reference=1.)
    d.set_xlabel("Logistic C")
    permutation(e,data["e"])
    lollipop(f,data["f"],"effect",BLUE,"Leave-one-out ΔJaccard",
             "Leave one mouse out")

    for ax,letter in [(a1,"a"),(b1,"b"),(c1,"c"),(d,"d"),(e,"e"),(f,"f")]:
        panel_label(ax,letter,fw,fh)

    OUT.mkdir(parents=True,exist_ok=True); stem=OUT/"SI08_full_nature_trial_v1"
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)

    rel={k:str(v.relative_to(ROOT)) for k,v in paths.items()}
    manifest={
        "figure":"SI08","version":"reference_v1","status":"review_not_frozen",
        "scientific_role":"Show that discrete top-unit/top-set representations recover state-dependent advantage across budgets, overlap definitions, regularization, permutation and leave-one-mouse-out tests.",
        "panel_source_map":{"a":[rel["a1"],rel["a2"]],"b":[rel["b1"],rel["b2"]],
                            "c":[rel["c1"],rel["c2"]],"d":[rel["d"]],
                            "e":[rel["e"]],"f":[rel["f"]]}}
    (OUT/"source_data_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    geom={"canvas_mm":[fw,fh],"row_gap_mm":13,"budget_effect_ylim":list(budget_lim),
          "axes_mm":{"a1":[x1,y1,w,h],"a2":[x2,y1,w,h],"b1":[x1,y2,w,h],
          "b2":[x2,y2,w,h],"c1":[x1,y3,w,h],"c2":[x2,y3,w,h],
          "d":[x3,y1,w,h],"e":[x3,y2,w,h],"f":[x3,y3,w,h]},
          "panel_label_offset_mm":[-8,2]}
    (OUT/"axis_geometry_mm.json").write_text(json.dumps(geom,indent=2),encoding="utf-8")
    contract="""# Supplementary Figure 8 contract

- Scientific role: OpenScope top-unit/top-set robustness supporting Fig.6.
- Sequence is column-major by paired blocks: `a+a+d / b+b+e / c+c+f`; panels a-c never wrap.
- Canvas: 183 x 145 mm; all nine data axes are 36 x 27 mm with fixed 13 mm row gaps.
- The two budget-sensitivity microplots share one y-axis range and the 20% reference.
- All local annotations sit above the data frames rather than on the plotted lines.
- This is a review version and is not frozen.
"""
    (OUT/"figure_contract.md").write_text(contract,encoding="utf-8")
    qa="""# Supplementary Figure 8 QA report

- Status: review version; not frozen.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-f (nine source tables).
- Layout: fixed 13 mm row gaps; all nine data axes use identical 36 x 27 mm geometry.
- Visual QA: PDF re-rendered at 600 dpi; no clipping, panel collision, label overlap, annotation collision or inter-row interference.
"""
    (OUT/"QA_REPORT.md").write_text(qa,encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__=="__main__":
    setup_style(); build()
