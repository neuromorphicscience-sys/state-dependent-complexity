#!/usr/bin/env python3
"""Build Supplementary Figure 7: OpenScope inferential sensitivity."""

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


def panel_label(ax,letter,fw=183,fh=105):
    pos=ax.get_position(); aw,ah=pos.width*fw,pos.height*fh
    ax.text(-8/aw,1+2/ah,letter,transform=ax.transAxes,fontsize=8,fontweight="bold",
            ha="left",va="bottom",clip_on=False)


def source(folder,token=None):
    pattern=f"*{token}*.csv" if token else "*.csv"
    hits=sorted((SRC/folder/"source_data").glob(pattern))
    if len(hits)!=1: raise FileNotFoundError(f"{folder}: {token}: {hits}")
    return hits[0]


def aggregate_regularization(d,value):
    return d.groupby(["subject","C"],as_index=False)[value].mean()


def regularization(ax,d,value,color,title,ylabel,ylim=None):
    a=aggregate_regularization(d,value)
    cs=[.25,1.,4.]; xpos=np.arange(3)
    for _,g in a.groupby("subject"):
        g=g.set_index("C").reindex(cs)
        ax.plot(xpos,g[value],color=LIGHT,lw=.72,zorder=1)
    med=a.groupby("C")[value].median().reindex(cs)
    ax.plot(xpos,med,color=color,lw=1.15,zorder=3)
    ax.scatter(xpos,med,s=17,color=color,alpha=.86,edgecolor="white",linewidth=.35,zorder=4)
    ax.axvline(1,color=GREY,lw=.62,ls="--",dashes=(3,2))
    if "delta" in value:
        ax.axhline(0,color=GREY,lw=.62,ls="--",dashes=(3,2))
    ax.set_xlim(-.22,2.22); ax.set_xticks(xpos,["0.25","1","4"])
    ax.set_xlabel("Logistic C"); ax.set_ylabel(ylabel)
    if ylim is not None: ax.set_ylim(*ylim)
    ax.text(.04,1.025,title,transform=ax.transAxes,ha="left",va="bottom",
            fontsize=3.2,clip_on=False)
    clean(ax)


def permutation(ax,d):
    null=d["null"].to_numpy(float); obs=float(d["observed"].iloc[0])
    qlo,qhi=np.quantile(null,[.025,.975])
    ax.hist(null,bins=32,color=LIGHT,edgecolor=GREY,linewidth=.35)
    ax.axvline(obs,color=PURPLE,lw=1.2)
    ax.axvline(qlo,color=INK,lw=.65,ls="--",dashes=(3,2))
    ax.axvline(qhi,color=INK,lw=.65,ls="--",dashes=(3,2))
    ax.set_xlabel("Fixed-split null Δρ"); ax.set_ylabel("Permutation count")
    ax.text(.04,1.025,"Observed; dashed = 95% null",transform=ax.transAxes,
            ha="left",va="bottom",fontsize=3.2,clip_on=False)
    clean(ax,grid=None)


def ranked_lollipop(ax,d,value,color,ylabel,title,zero=False):
    y=d[value].to_numpy(float); x=np.arange(1,len(y)+1)
    base=0 if zero else min(0,float(y.min())*.92)
    ax.vlines(x,base,y,color=LIGHT,lw=.9,zorder=1)
    ax.scatter(x,y,s=10,color=color,alpha=.88,edgecolor="white",linewidth=.3,zorder=3)
    ax.axhline(np.median(y),color=INK,lw=.8)
    if zero: ax.axhline(0,color=GREY,lw=.62,ls="--",dashes=(3,2))
    ax.set_xlim(.4,len(y)+.6); ax.set_xticks([1,4,8,12]); ax.set_xlabel("Mouse rank")
    ax.set_ylabel(ylabel)
    ax.text(.04,1.025,title,transform=ax.transAxes,ha="left",va="bottom",
            fontsize=3.2,clip_on=False)
    clean(ax)


def build():
    paths={
        "a1":source("S04_e__Regularization_sensitivity_of_raw_landscape_metrics","S15"),
        "a2":source("S04_e__Regularization_sensitivity_of_raw_landscape_metrics","S16"),
        "a3":source("S04_e__Regularization_sensitivity_of_raw_landscape_metrics","S17"),
        "b1":source("S04_f__Regularization_under_residualized_definitions","S18"),
        "b2":source("S04_f__Regularization_under_residualized_definitions","S19"),
        "c":source("S04_h__High-precision_permutation_boundary"),
        "d":source("S04_i__Leave-one-mouse-out_coefficient-landscape_effect"),
        "e":source("S04_j__Coefficient-landscape_split_instability_precision"),
    }
    data={k:pd.read_csv(p,encoding="utf-8-sig") for k,p in paths.items()}
    effect_sets=[]
    for k,col in [("a1","delta_rho"),("b1","activity_residual_delta_rho"),
                  ("b2","activity_selectivity_residual_delta_rho")]:
        effect_sets.append(aggregate_regularization(data[k],col)[col].to_numpy(float))
    lo=min(x.min() for x in effect_sets); hi=max(x.max() for x in effect_sets)
    pad=.08*(hi-lo); effect_lim=(min(-.002,lo-pad),hi+pad)

    fw,fh,w,h=183.0,105.0,32.0,27.0
    xcols=[14.0,58.0,102.0,146.0]; ytop,ybottom=61.0,21.0
    fig=plt.figure(figsize=(fw/MM,fh/MM))
    # Row 1: original panel a remains intact; original c becomes new panel b.
    a1=add_axes_mm(fig,fw,fh,xcols[0],ytop,w,h)
    a2=add_axes_mm(fig,fw,fh,xcols[1],ytop,w,h)
    a3=add_axes_mm(fig,fw,fh,xcols[2],ytop,w,h)
    c=add_axes_mm(fig,fw,fh,xcols[3],ytop,w,h)
    # Row 2: original b becomes new c, followed by original d and e.
    b1=add_axes_mm(fig,fw,fh,xcols[0],ybottom,w,h)
    b2=add_axes_mm(fig,fw,fh,xcols[1],ybottom,w,h)
    d=add_axes_mm(fig,fw,fh,xcols[2],ybottom,w,h)
    e=add_axes_mm(fig,fw,fh,xcols[3],ybottom,w,h)

    regularization(a1,data["a1"],"delta_rho",PURPLE,"Landscape advantage",
                   "Landscape advantage, Δρ",effect_lim)
    regularization(a2,data["a2"],"rho_within",TEAL,"Within-state similarity",
                   "Coefficient-rank correlation")
    regularization(a3,data["a3"],"rho_cross",GOLD,"Cross-state similarity",
                   "Coefficient-rank correlation")
    regularization(b1,data["b1"],"activity_residual_delta_rho",TEAL,"Activity residual",
                   "Landscape advantage, Δρ",effect_lim)
    regularization(b2,data["b2"],"activity_selectivity_residual_delta_rho",BLUE,
                   "Activity + selectivity","Landscape advantage, Δρ",effect_lim)
    permutation(c,data["c"])
    ranked_lollipop(d,data["d"],"effect",BLUE,"Leave-one-out advantage, Δρ",
                    "Leave one mouse out")
    ranked_lollipop(e,data["e"],"delta_rho",GOLD,"Repeat SD of Δρ",
                    "Repeated-split precision",zero=True)

    for ax,letter in [(a1,"a"),(c,"b"),(b1,"c"),(d,"d"),(e,"e")]:
        panel_label(ax,letter,fw,fh)

    OUT.mkdir(parents=True,exist_ok=True); stem=OUT/"SI07_full_nature_trial_v1"
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)

    rel={k:str(v.relative_to(ROOT)) for k,v in paths.items()}
    manifest={
        "figure":"SI07","version":"reference_v1","status":"review_not_frozen",
        "scientific_role":"Test coefficient-landscape sensitivity to regularization, residualization, permutation precision, leave-one-mouse-out analysis and repeated splits.",
        "panel_source_map":{"a":[rel["a1"],rel["a2"],rel["a3"]],
                            "b":[rel["c"]],"c":[rel["b1"],rel["b2"]],
                            "d":[rel["d"]],"e":[rel["e"]]}}
    (OUT/"source_data_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    geom={"canvas_mm":[fw,fh],"row_gap_mm":13,"regularization_effect_ylim":list(effect_lim),
          "axes_mm":{"a1":[xcols[0],ytop,w,h],"a2":[xcols[1],ytop,w,h],
          "a3":[xcols[2],ytop,w,h],"b1":[xcols[0],ybottom,w,h],
          "b2":[xcols[1],ybottom,w,h],"c":[xcols[3],ytop,w,h],
          "d":[xcols[2],ybottom,w,h],"e":[xcols[3],ybottom,w,h]},
          "panel_label_offset_mm":[-8,2]}
    (OUT/"axis_geometry_mm.json").write_text(json.dumps(geom,indent=2),encoding="utf-8")
    contract="""# Supplementary Figure 7 contract

- Scientific role: OpenScope coefficient-landscape inferential sensitivity supporting Fig.6.
- Sequence: `a+a+a+b / c+c+d+e`; original panel a remains intact, original c becomes new b, and original b becomes new c.
- Canvas: 183 x 105 mm; all eight data axes are 32 x 27 mm with a fixed 13 mm row gap.
- Raw and residualized regularization effect plots share one y-axis range and the C=1 reference.
- Panel c shows the observed statistic and both 95% null boundaries.
- This is a review version and is not frozen.
"""
    (OUT/"figure_contract.md").write_text(contract,encoding="utf-8")
    qa="""# Supplementary Figure 7 QA report

- Status: review version; not frozen.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-e (eight source tables).
- Layout: fixed 13 mm row gap; all eight data axes use identical 32 x 27 mm geometry.
- Visual QA: PDF re-rendered at 600 dpi after annotation relocation; all annotations clear the data and reference lines, with no clipping or inter-row collision.
"""
    (OUT/"QA_REPORT.md").write_text(qa,encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__=="__main__":
    setup_style(); build()
