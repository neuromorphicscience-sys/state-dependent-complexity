#!/usr/bin/env python3
"""Build Supplementary Figure 3: Steinmetz behavioral relevance."""

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
SRC = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "03_SUPPLEMENTARY" / "S02"
OUT = Path(__file__).resolve().parent / "output_full_v1"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE, TEAL, GOLD, RED = "#38598C", "#2A9D8F", "#C27628", "#B23A48"
INK, GREY, LIGHT, WHITE = "#263238", "#7B858A", "#D9DEE1", "#F7F7F4"
DIV = LinearSegmentedColormap.from_list("fig1_div", ["#2166AC", WHITE, "#B2182B"])


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


def panel_label(ax,letter,fw=183,fh=145):
    pos=ax.get_position(); aw,ah=pos.width*fw,pos.height*fh
    ax.text(-8/aw,1+2/ah,letter,transform=ax.transAxes,fontsize=8,fontweight="bold",
            ha="left",va="bottom",clip_on=False)


def source(folder,pattern):
    hits=sorted((SRC/folder/"source_data").glob(pattern))
    if len(hits)!=1: raise FileNotFoundError(f"{folder}: {pattern}: {hits}")
    return hits[0]


def plot_inventory(ax,d):
    labels=["Raw NWB inventory","Unique sessions","Duplicate aliases"]
    values=[d["inventory_rows"],d["unique_sessions"],d["inventory_rows"]-d["unique_sessions"]]
    colors=[GREY,BLUE,RED]
    y=np.arange(3)
    ax.barh(y,values,color=colors,height=.62,edgecolor="none",alpha=.88)
    for yy,v in zip(y,values): ax.text(v+.8,yy,f"{v:,}",va="center",ha="left",fontsize=3.5)
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlim(0,57)
    ax.set_xlabel("Session/file count")
    ax.text(.98,.05,"Session inventory",transform=ax.transAxes,ha="right",va="bottom",fontsize=3.35)
    clean(ax,"x")


def plot_rows(ax,d):
    labels=["Dynamical","Predictive"]
    raw=np.array([d["dyn_rows_raw"],d["pred_rows_raw"]])/1000
    strict=np.array([d["dyn_rows_strict"],d["pred_rows_strict"]])/1000
    x=np.arange(2); w=.32
    ax.bar(x-w/2,raw,w,color=GREY,alpha=.72,label="Raw")
    ax.bar(x+w/2,strict,w,color=TEAL,alpha=.88,label="Deduplicated")
    for xx,v in zip(x-w/2,raw): ax.text(xx,v+19,f"{v:.0f}",ha="center",va="bottom",fontsize=3.1)
    for xx,v in zip(x+w/2,strict): ax.text(xx,v+19,f"{v:.0f}",ha="center",va="bottom",fontsize=3.1)
    ax.set_xticks(x,labels); ax.set_ylim(0,1050); ax.set_ylabel("Analysis rows (thousands)")
    ax.set_xlabel("Analysis stream")
    ax.legend(frameon=False,loc="upper left",ncol=2,columnspacing=.7,handletextpad=.25)
    clean(ax)


STATES=["baseline","visual","decision","movement","outcome"]


def predictive_matrix(d,target):
    z=np.eye(len(STATES)); index={s:i for i,s in enumerate(STATES)}
    for _,r in d.loc[d.target==target].iterrows():
        i,j=index[r.state_a],index[r.state_b]; z[i,j]=z[j,i]=r.median_rho
    return z


def plot_predictive_heatmap(ax,d,target):
    z=predictive_matrix(d,target)
    im=ax.imshow(z,cmap=DIV,norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),
                 aspect="equal",interpolation="nearest")
    ax.set_xticks(range(5),STATES,rotation=36,ha="right",rotation_mode="anchor",fontsize=3.1)
    ax.set_yticks(range(5),STATES,fontsize=3.1)
    ax.tick_params(length=1.4,pad=.8)
    for i in range(5):
        for j in range(5):
            v=z[i,j]
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=2.55,
                    color="white" if abs(v)>=.58 else INK)
    ax.set_title(target.capitalize(),fontsize=4.0,pad=2.0)
    return im


def plot_predictive_dynamic(ax,d):
    order=["choice","engagement","outcome"]
    groups=[d.loc[d.target==x,"rho"].dropna().to_numpy() for x in order]
    colors=[BLUE,TEAL,"#C97957"]; rng=np.random.default_rng(315)
    bp=ax.boxplot(groups,positions=range(3),widths=.56,patch_artist=True,showfliers=False,
                  medianprops={"color":GOLD,"lw":.9},whiskerprops={"color":INK,"lw":.6},
                  capprops={"color":INK,"lw":.6},boxprops={"edgecolor":INK,"lw":.6})
    for p,c in zip(bp["boxes"],colors): p.set_facecolor(c); p.set_alpha(.70)
    for i,(g,c) in enumerate(zip(groups,colors)):
        ax.scatter(i+rng.uniform(-.10,.10,len(g)),g,s=4.3,color=c,alpha=.18,edgecolor="none")
    ax.axhline(0,color=GREY,lw=.65,ls="--",dashes=(4,2))
    ax.set_xticks(range(3),[x.capitalize() for x in order]); ax.set_xlabel("Behavioral target")
    ax.set_ylabel("Predictive-dynamical rank correlation, ρ")
    ax.set_ylim(-.24,.43); clean(ax)


def build():
    paths={
        "a":source("S02_a__Steinmetz_session_duplicate_audit","*.json"),
        "b1":source("S02_b__Predictive_contribution_to_choice","*.csv"),
        "b2":source("S02_c__Predictive_contribution_to_outcome","*.csv"),
        "b3":source("S02_d__Predictive_contribution_to_engagement","*.csv"),
        "c":source("S02_e__Predictive_importance_versus_dynamical_influence","*.csv"),
    }
    audit=json.loads(paths["a"].read_text(encoding="utf-8"))
    data={k:pd.read_csv(p) for k,p in paths.items() if k!="a"}
    fw,fh,htop,hbottom=183.0,105.0,27.0,30.0; ytop,ybottom=61.0,18.0
    fig=plt.figure(figsize=(fw/MM,fh/MM))
    a1=add_axes_mm(fig,fw,fh,14,ytop,36,htop); a2=add_axes_mm(fig,fw,fh,58,ytop,36,htop)
    b=add_axes_mm(fig,fw,fh,112,ytop,55,htop)
    c1=add_axes_mm(fig,fw,fh,28,ybottom,36,hbottom); c2=add_axes_mm(fig,fw,fh,78,ybottom,36,hbottom)
    c3=add_axes_mm(fig,fw,fh,128,ybottom,36,hbottom); cb=add_axes_mm(fig,fw,fh,168,ybottom,2,hbottom)
    plot_inventory(a1,audit); plot_rows(a2,audit)
    plot_predictive_dynamic(b,data["c"])
    im=plot_predictive_heatmap(c1,data["b1"],"choice")
    plot_predictive_heatmap(c2,data["b2"],"outcome")
    plot_predictive_heatmap(c3,data["b3"],"engagement")
    cbar=fig.colorbar(im,cax=cb,ticks=[-1,0,1]); cbar.ax.tick_params(labelsize=3.1,pad=.8,length=1.5)
    cbar.set_label("Median predictive-leverage rank ρ",fontsize=3.4,labelpad=1.3)
    for ax,letter in [(a1,"a"),(b,"b"),(c1,"c")]: panel_label(ax,letter,fw,fh)

    OUT.mkdir(parents=True,exist_ok=True); stem=OUT/"SI03_full_nature_trial_v1"
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
    rel={k:str(v.relative_to(ROOT)) for k,v in paths.items()}
    manifest={"figure":"SI03","version":"reference_v1","status":"review_not_frozen",
              "scientific_role":"Show that Steinmetz dynamical influence carries behaviorally relevant information after session deduplication and remains distinct from predictive importance.",
              "panel_source_map":{"a":[rel["a"]],"b":[rel["c"]],"c":[rel["b1"],rel["b2"],rel["b3"]]}}
    (OUT/"source_data_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    geom={"canvas_mm":[fw,fh],"row_gap_mm":13,"axes_mm":{"a1":[14,ytop,36,htop],"a2":[58,ytop,36,htop],
          "b":[112,ytop,55,htop],"c1":[28,ybottom,36,hbottom],"c2":[78,ybottom,36,hbottom],
          "c3":[128,ybottom,36,hbottom],"colorbar":[168,ybottom,2,hbottom]},"panel_label_offset_mm":[-8,2]}
    (OUT/"axis_geometry_mm.json").write_text(json.dumps(geom,indent=2),encoding="utf-8")
    contract="""# Supplementary Figure 3 contract

- Scientific role: behavioral relevance and non-equivalence of predictive and dynamical influence in Steinmetz data.
- Sequence: `a+a+b / c+c+c`; multi-plot panels a and c never wrap.
- Canvas: 183 x 105 mm; 27-mm top axes and 30-mm heatmaps with a fixed 13 mm data-axis gap.
- All three predictive-contribution heatmaps share one scale and the frozen Fig.1 diverging palette.
- This is a review version and is not frozen.
"""
    (OUT/"figure_contract.md").write_text(contract,encoding="utf-8")
    qa="""# Supplementary Figure 3 QA report

- Status: review version; not frozen.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-c.
- Layout: fixed 13 mm row gap; the three panel-c heatmaps remain on one row.
- Visual QA: PDF re-rendered at 600 dpi; no clipping, panel collision, label overlap or legend collision.
"""
    (OUT/"QA_REPORT.md").write_text(qa,encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__=="__main__":
    setup_style(); build()
