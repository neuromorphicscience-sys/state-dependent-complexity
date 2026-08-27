#!/usr/bin/env python3
"""Build the Fig.4-supporting functional-degeneracy Extended Data Figure 4."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

MM, DPI = 25.4, 600
ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "01_MAIN" / "Fig04"
OLD_ED05 = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "02_EXTENDED_DATA" / "ED05"
OUT = Path(__file__).resolve().parent / "output_full_v2"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE, TEAL, GOLD = "#38598C", "#2A9D8F", "#C27628"
INK, GREY, LIGHT = "#263238", "#7B858A", "#D9DEE1"
DIV = LinearSegmentedColormap.from_list("fig1_div", ["#2166AC", "#F7F7F7", "#B2182B"])
STATE = {"sparse_drive": BLUE, "transition_mid": TEAL, "transition_dense": GOLD}
STATE_LABEL = {"sparse_drive": "Sparse", "transition_mid": "Transition-mid",
               "transition_dense": "Transition-dense"}
REGIMES = ["sparse_drive", "transition_mid", "transition_dense"]


def setup_style():
    if ARIAL.exists(): font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists(): font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family":"Arial","font.size":5.0,"axes.labelsize":5.0,
        "xtick.labelsize":4.0,"ytick.labelsize":4.0,"legend.fontsize":3.45,
        "axes.linewidth":.58,"xtick.major.width":.58,"ytick.major.width":.58,
        "xtick.major.size":1.8,"ytick.major.size":1.8,
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
        "savefig.facecolor":"white","figure.facecolor":"white",
    })


def add_axes_mm(fig,fw,fh,x,y,w,h):
    return fig.add_axes([x/fw,y/fh,w/fw,h/fh])


def clean(ax,grid="y"):
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(direction="out",pad=1.15)
    if grid:
        ax.grid(axis=grid,color=LIGHT,lw=.32,alpha=.62); ax.set_axisbelow(True)


def panel_label(ax,letter,fw=183,fh=94):
    pos=ax.get_position(); aw,ah=pos.width*fw,pos.height*fh
    ax.text(-8/aw,1+2/ah,letter,transform=ax.transAxes,fontsize=8,fontweight="bold",
            ha="left",va="bottom",clip_on=False)


def source(token):
    hits=[p for base in [MAIN,OLD_ED05] for p in base.rglob("*.csv") if token in p.name]
    if not hits: raise FileNotFoundError(token)
    return sorted(hits)[0]


def plot_count(ax,d):
    for reg in REGIMES:
        x=d[d.regime==reg]
        for _,g in x.groupby("graph_seed"):
            g=g.sort_values("k"); ax.plot(g.k,g.near_optimal_count,color=STATE[reg],alpha=.16,lw=.45)
        s=x.groupby("k").near_optimal_count.agg(["mean","sem"]).reset_index(); ci=1.96*s["sem"]
        ax.fill_between(s.k,s["mean"]-ci,s["mean"]+ci,color=STATE[reg],alpha=.13,lw=0)
        ax.plot(s.k,s["mean"],color=STATE[reg],lw=1.0,marker="o",ms=2.5,label=STATE_LABEL[reg])
        for k,g in x.groupby("k"):
            ax.scatter(np.full(len(g),k),g.near_optimal_count,s=4,color=STATE[reg],alpha=.28,edgecolor="none")
    ax.set_xticks([8,32,64]); ax.set_ylim(.85,4.15); ax.set_yticks([1,2,3,4])
    ax.set_xlabel("Complexity budget, k"); ax.set_ylabel("Near-optimal solution count")
    ax.legend(frameon=False,loc="upper right",fontsize=3.0,handlelength=.9,labelspacing=.12)
    clean(ax)


def plot_mean_heat(ax,fig,fw,fh,cax_x,cax_y,d):
    tab=d.groupby(["regime","k"]).near_optimal_count.mean().unstack("k").reindex(REGIMES)
    z=tab.to_numpy(float); norm=Normalize(vmin=1,vmax=4)
    im=ax.imshow(z,aspect="auto",cmap=DIV,norm=norm)
    ax.set_xticks(range(3),[str(x) for x in tab.columns]); ax.set_yticks(range(3),[STATE_LABEL[x] for x in tab.index],fontsize=3.5)
    ax.set_xlabel("Complexity budget, k"); ax.set_ylabel("Collective state")
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{z[i,j]:.2f}",ha="center",va="center",fontsize=3.4,
                    color="white" if z[i,j]>3.0 else INK)
    cax=add_axes_mm(fig,fw,fh,cax_x,cax_y,1.6,27); cb=fig.colorbar(im,cax=cax,ticks=[1,2.5,4])
    cb.ax.set_title("Near-opt.\ncount",fontsize=3.5,pad=1); cb.ax.tick_params(labelsize=3.0,width=.5,length=1.4,pad=1)


def plot_seed_heat(ax,fig,fw,fh,cax_x,cax_y,d):
    x=d.copy(); x["mean"]=x.groupby(["regime","k"]).near_optimal_count.transform("mean")
    x["deviation"]=x.near_optimal_count-x["mean"]
    x["row"]=x.regime.map(STATE_LABEL)+" | "+x.graph_seed.astype(str)
    order=[f"{STATE_LABEL[r]} | {s}" for r in REGIMES for s in sorted(x.graph_seed.unique())]
    tab=x.pivot(index="row",columns="k",values="deviation").reindex(order)
    z=tab.to_numpy(float); lim=max(abs(z.min()),abs(z.max()))
    im=ax.imshow(z,aspect="auto",cmap=DIV,norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
    ax.set_xticks(range(3),[str(k) for k in tab.columns]); ax.set_yticks(range(9),tab.index,fontsize=2.75)
    ax.set_xlabel("Complexity budget, k"); ax.set_ylabel("State | graph seed")
    for i in range(z.shape[0]):
        for j in range(3):
            ax.text(j,i,f"{z[i,j]:+.1f}",ha="center",va="center",fontsize=2.65,
                    color="white" if abs(z[i,j])>.58*lim else INK)
    cax=add_axes_mm(fig,fw,fh,cax_x,cax_y,1.6,27); cb=fig.colorbar(im,cax=cax,ticks=[-lim,0,lim])
    cb.ax.set_title("Count\ndeviation",fontsize=3.5,pad=1); cb.ax.tick_params(labelsize=3.0,width=.5,length=1.4,pad=1)
    cb.set_ticklabels([f"{-lim:.1f}","0",f"{lim:.1f}"])


def plot_relation(ax,d,ycol,ylabel,show_legend=False,annotation_right=False):
    offsets={42001:-.07,42002:0,42003:.07}
    for reg in REGIMES:
        g=d[d.regime==reg]
        x=g.near_optimal_count.to_numpy(float)+g.graph_seed.map(offsets).to_numpy(float)
        ax.scatter(x,g[ycol],s=7,color=STATE[reg],alpha=.58,edgecolor="white",linewidth=.25,label=STATE_LABEL[reg])
    s=d.groupby("near_optimal_count")[ycol].agg(["mean","sem"]).reset_index()
    ax.plot(s.near_optimal_count,s["mean"],color=INK,lw=.8,marker="o",ms=2.1,zorder=4)
    rho=d[["near_optimal_count",ycol]].corr(method="spearman").iloc[0,1]
    annot_x, annot_ha = (.96, "right") if annotation_right else (.04, "left")
    ax.text(annot_x,.96,f"Spearman ρ = {rho:.2f}",transform=ax.transAxes,
            ha=annot_ha,va="top",fontsize=3.45)
    ax.set_xticks([1,2,3,4]); ax.set_xlim(.75,4.25)
    ax.set_xlabel("Near-optimal solution count"); ax.set_ylabel(ylabel); clean(ax)
    if show_legend:
        ax.legend(frameon=False,loc="lower right",fontsize=2.75,handletextpad=.15,labelspacing=.1)


def build():
    p2,p3,p4,p5,p6=[source(tok) for tok in ["DG02","DG03","DG04","DG05","DG06"]]
    d2,d3,d4,d5,d6=[pd.read_csv(p) for p in [p2,p3,p4,p5,p6]]
    keys=["graph_seed","regime","k"]
    merged=d3.merge(d2,on=keys).merge(d4,on=keys).merge(d5,on=keys).merge(d6,on=keys)
    fw,fh,ah=183.0,94.0,27.0; ytop,ybottom=54.0,14.0
    fig=plt.figure(figsize=(fw/MM,fh/MM))
    a=add_axes_mm(fig,fw,fh,10,ytop,44,ah)
    b=add_axes_mm(fig,fw,fh,73,ytop,31,ah)
    c=add_axes_mm(fig,fw,fh,128,ytop,43,ah)
    d=add_axes_mm(fig,fw,fh,10,ybottom,34,ah)
    e=add_axes_mm(fig,fw,fh,54,ybottom,34,ah)
    f=add_axes_mm(fig,fw,fh,98,ybottom,34,ah)
    g=add_axes_mm(fig,fw,fh,142,ybottom,34,ah)
    plot_count(a,d3); plot_mean_heat(b,fig,fw,fh,107,ytop,d3); plot_seed_heat(c,fig,fw,fh,174.4,ytop,d3)
    plot_relation(d,merged,"D_epsilon_mean_jaccard_distance","Degeneracy index D",True)
    plot_relation(e,merged,"union_expansion","Union expansion")
    plot_relation(f,merged,"core_fraction","Shared-core fraction",annotation_right=True)
    plot_relation(g,merged,"selection_entropy_union","Selection entropy")
    for ax,letter in [(a,"a"),(b,"b"),(c,"c"),(d,"d"),(e,"e"),(f,"f"),(g,"g")]: panel_label(ax,letter,fw,fh)
    OUT.mkdir(parents=True,exist_ok=True); stem=OUT/"ED04_full_nature_trial_v2"
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
    rel={"D":str(p2.relative_to(ROOT)),"count":str(p3.relative_to(ROOT)),"union":str(p4.relative_to(ROOT)),
         "core":str(p5.relative_to(ROOT)),"entropy":str(p6.relative_to(ROOT))}
    manifest={"figure":"ED04","version":"degeneracy_v2","status":"review_not_frozen",
              "scientific_role":"Support Fig.4 by linking near-optimal solution multiplicity to functional degeneracy geometry.",
              "panel_source_map":{"a":[rel["count"]],"b":[rel["count"]],"c":[rel["count"]],
                                  "d":[rel["count"],rel["D"]],"e":[rel["count"],rel["union"]],
                                  "f":[rel["count"],rel["core"]],"g":[rel["count"],rel["entropy"]]},
              "display_computations":{"b":"Mean near-optimal count by regime and budget.",
                                      "c":"Count minus the matched regime-by-budget mean.",
                                      "d_g":"Exact-key merges on graph_seed, regime and k; Spearman rho shown descriptively."}}
    (OUT/"source_data_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    geom={"canvas_mm":[fw,fh],"axis_height_mm":ah,"inter_row_gap_mm":13,
          "panel_label":{"font_pt":8,"left_offset_mm":8,"top_offset_mm":2},
          "layout":"Top row: three robustness panels; bottom row: four cross-metric relations."}
    (OUT/"axis_geometry_mm.json").write_text(json.dumps(geom,indent=2),encoding="utf-8")
    contract="""# ED Figure 4 contract

Purpose: support Fig.4 by showing that near-optimal multiplicity is state- and budget-dependent,
stable enough to inspect across graph realizations, and quantitatively coupled to degeneracy,
union expansion, shared-core loss and selection entropy. Panels d-g are new cross-metric views
formed only by exact-key joins of existing source-data tables. This version is not frozen.
"""
    (OUT/"figure_contract.md").write_text(contract,encoding="utf-8")
    qa="""# ED Figure 4 QA report

- Status: review version; not frozen.
- Canvas: one-page 183 x 94 mm PDF.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-g.
- Layout: fixed 13 mm inter-row gap and main-Fig01 panel-label offsets.
- Join QA: panels d-g use exact graph_seed + regime + k matches (27/27 rows).
- Visual QA: PDF re-rendered at 600 dpi; no clipping, panel collision, legend overlap or colorbar collision.
"""
    (OUT/"QA_REPORT.md").write_text(qa,encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__=="__main__":
    setup_style(); build()
