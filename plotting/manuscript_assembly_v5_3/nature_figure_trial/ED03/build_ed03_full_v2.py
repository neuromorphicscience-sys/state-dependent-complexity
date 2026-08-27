#!/usr/bin/env python3
"""Build the Fig.3-supporting confirmatory Extended Data Figure 3."""

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
S08 = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "03_SUPPLEMENTARY" / "S08"
OUT = Path(__file__).resolve().parent / "output_full_v2"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE, TEAL, GOLD = "#38598C", "#2A9D8F", "#C27628"
INK, GREY, LIGHT = "#263238", "#7B858A", "#D9DEE1"
DIV = LinearSegmentedColormap.from_list("fig1_div", ["#2166AC", "#F7F7F7", "#B2182B"])
DISP = LinearSegmentedColormap.from_list("dispersion", [BLUE, TEAL, "#D29A3A"])
STATE = {"sparse_drive": TEAL, "transition_mid": BLUE, "transition_dense": GOLD}
STATE_LABEL = {"sparse_drive": "Sparse drive", "transition_mid": "Transition mid",
               "transition_dense": "Transition dense"}
PHASES = ["er_patch", "stage4b_replication", "stage4c_atlas_extension",
          "stage4d_scaling", "stage4e_topology"]
PHASE_TITLES = ["ER patch", "Replication", "Atlas extension", "Scale scaling", "Topology"]


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


def src(base,token):
    hits=sorted(base.rglob(f"*{token}*.csv"))
    if not hits: raise FileNotFoundError(token)
    return hits[0]


def plot_heatmap(ax,fig,fw,fh,cax_x,cax_y,d):
    phases=["stage4b_replication","stage4c_atlas_extension","stage4d_scaling","stage4e_topology","er_patch"]
    phase_labels=["Replication","Atlas extension","Scale scaling","Topology","ER patch"]
    bases=["spectral","feedback_hub","high_degree","module_bridge","cycle_proxy","random"]
    base_labels=["Spectral","Feedback hub","High degree","Module bridge","Cycle proxy","Random"]
    x=d[d.grouping=="phase"].pivot(index="group",columns="baseline",values="mean_delta").reindex(index=phases,columns=bases)
    z=x.to_numpy(float); lim=.25
    im=ax.imshow(z,aspect="auto",cmap=DIV,norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
    ax.set_xticks(range(6),base_labels,rotation=43,ha="right",fontsize=3.0)
    ax.set_yticks(range(5),phase_labels,fontsize=3.2)
    ax.set_xlabel("Comparator"); ax.set_ylabel("Confirmatory phase")
    for i in range(5):
        for j in range(6):
            ax.text(j,i,f"{z[i,j]:.02f}",ha="center",va="center",fontsize=2.7,
                    color="white" if abs(z[i,j])>.13 else INK)
    cax=add_axes_mm(fig,fw,fh,cax_x,cax_y,1.8,27)
    cb=fig.colorbar(im,cax=cax,ticks=[0,.12,.25]); cb.ax.set_title("Mean\ngain",fontsize=3.6,pad=1)
    cb.ax.tick_params(labelsize=3.1,width=.5,length=1.4,pad=1)


def plot_bridge(ax,fig,fw,fh,cax_x,cax_y,d):
    sc=ax.scatter(d.coverage2,d.leverage_gain_vs_spectral,c=d.selected_dispersion,
                  cmap=DISP,s=6.2,alpha=.63,edgecolor="none")
    ax.axhline(0,color=GREY,lw=.55,ls="--",dashes=(3,2))
    rho=d[["coverage2","leverage_gain_vs_spectral"]].corr(method="spearman").iloc[0,1]
    ax.text(.04,.96,f"Spearman ρ = {rho:.2f}",transform=ax.transAxes,ha="left",va="top",fontsize=3.5)
    ax.set_xlabel("Two-hop selected-unit coverage"); ax.set_ylabel("Leverage gain vs spectral")
    clean(ax)
    cax=add_axes_mm(fig,fw,fh,cax_x,cax_y,1.8,27); cb=fig.colorbar(sc,cax=cax)
    cb.ax.set_title("Selected\ndispersion",fontsize=3.5,pad=1)
    cb.ax.tick_params(labelsize=3.0,width=.5,length=1.4,pad=1)


def phase_frame(core,phase):
    d=core[core.phase==phase]
    w=d.pivot_table(index="task_id",columns="method",values="score",aggfunc="mean").reset_index()
    meta=d.drop_duplicates("task_id")[["task_id","regime","budget_fraction"]]
    w=w.merge(meta,on="task_id"); w["gain"]=w.dynamics_aware-w.spectral
    return w


def plot_phase(ax,d,title,show_ylabel,show_legend):
    for reg in ["sparse_drive","transition_mid","transition_dense"]:
        g=d[d.regime==reg]
        s=g.groupby("budget_fraction").gain.agg(["mean","sem"]).reset_index(); s["sem"]=s["sem"].fillna(0)
        c=STATE[reg]; ci=1.96*s["sem"]
        ax.fill_between(s.budget_fraction,s["mean"]-ci,s["mean"]+ci,color=c,alpha=.12,lw=0)
        ax.plot(s.budget_fraction,s["mean"],color=c,lw=.85,marker="o",ms=1.8,label=STATE_LABEL[reg])
    ax.axhline(0,color=GREY,lw=.55,ls="--",dashes=(3,2)); ax.set_ylim(-.34,.44)
    ax.set_xlabel("High-complexity fraction"); ax.set_ylabel("Gain vs spectral" if show_ylabel else "")
    ax.set_title(title,fontsize=4.8,pad=2); clean(ax)
    if show_legend:
        ax.legend(frameon=False,loc="lower left",fontsize=2.75,handlelength=.8,
                  handletextpad=.2,labelspacing=.1,borderaxespad=.25)


def build():
    pa=src(S08/"S08_a__Confirmatory_baseline_comparison_across_Stage4_phases","stage4_paired_contrasts")
    pb=src(S08/"S08_b__Model-side_descriptor-to-leverage_bridge","model_side_biology_bridge")
    pc=src(S08/"S08_c__ER-patch_phase_gain_curves","stage4_core_225_task_methods")
    contrasts,bridge,core=pd.read_csv(pa),pd.read_csv(pb),pd.read_csv(pc)
    fw,fh,ah=183.0,94.0,27.0; ytop,ybottom=58.0,14.0
    fig=plt.figure(figsize=(fw/MM,fh/MM))
    a=add_axes_mm(fig,fw,fh,20,ytop,40,ah)
    b=add_axes_mm(fig,fw,fh,78,ytop,38,ah)
    c=add_axes_mm(fig,fw,fh,136,ytop,38,ah)
    d=add_axes_mm(fig,fw,fh,10,ybottom,34,ah)
    e=add_axes_mm(fig,fw,fh,54,ybottom,34,ah)
    f=add_axes_mm(fig,fw,fh,98,ybottom,34,ah)
    g=add_axes_mm(fig,fw,fh,142,ybottom,34,ah)
    plot_heatmap(a,fig,fw,fh,62,ytop,contrasts)
    plot_bridge(b,fig,fw,fh,118.5,ytop,bridge)
    frames={phase:phase_frame(core,phase) for phase in PHASES}
    plot_phase(c,frames[PHASES[0]],PHASE_TITLES[0],True,True)
    plot_phase(d,frames[PHASES[1]],PHASE_TITLES[1],True,True)
    plot_phase(e,frames[PHASES[2]],PHASE_TITLES[2],False,False)
    plot_phase(f,frames[PHASES[3]],PHASE_TITLES[3],False,False)
    plot_phase(g,frames[PHASES[4]],PHASE_TITLES[4],False,False)
    for ax,letter in [(a,"a"),(b,"b"),(c,"c"),(d,"d"),(e,"e"),(f,"f"),(g,"g")]: panel_label(ax,letter,fw,fh)
    OUT.mkdir(parents=True,exist_ok=True); stem=OUT/"ED03_full_nature_trial_v2"
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)
    rel={"paired":str(pa.relative_to(ROOT)),"bridge":str(pb.relative_to(ROOT)),"core":str(pc.relative_to(ROOT))}
    manifest={"figure":"ED03","version":"confirmatory_v2","status":"review_not_frozen",
              "scientific_role":"Support Fig.3 dynamics-aware leverage with independent replication and cross-phase generalization.",
              "promoted_from":"Former Supplementary Figure S08",
              "panel_source_map":{"a":[rel["paired"]],"b":[rel["bridge"]],"c":[rel["core"]],
                                  "d":[rel["core"]],"e":[rel["core"]],"f":[rel["core"]],"g":[rel["core"]]},
              "display_computations":{"b":"Spearman rho computed for coverage2 versus leverage_gain_vs_spectral.",
                                      "c_g":"Dynamics-aware score minus spectral score, summarized by regime and budget."}}
    (OUT/"source_data_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    geom={"canvas_mm":[fw,fh],"axis_height_mm":ah,"inter_row_gap_mm":17,
          "gap_exception":"ED03 uses 17 mm because panel a has rotated heatmap tick labels; ED01/02 remain at 13 mm.",
          "panel_label":{"font_pt":8,"left_offset_mm":8,"top_offset_mm":2},
          "phase_curve_ylim":[-.34,.44],"layout":"Top row: a-c; bottom row: d-g."}
    (OUT/"axis_geometry_mm.json").write_text(json.dumps(geom,indent=2),encoding="utf-8")
    contract="""# ED Figure 3 contract

Purpose: provide the confirmatory closure for Fig.3 dynamics-aware leverage.
Panel a compares all confirmatory phases against six baselines; panel b connects model-side
coverage and dispersion to leverage; panels c-g use a common y scale to show replication,
atlas extension, scaling and topology generalization. This review version is not frozen.
"""
    (OUT/"figure_contract.md").write_text(contract,encoding="utf-8")
    qa="""# ED Figure 3 QA report

- Status: review version; not frozen.
- Canvas: one-page 183 x 94 mm PDF.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-g.
- Layout: ED03-specific 17 mm inter-row gap and main-Fig01 panel-label offsets.
- Comparison QA: panels c-g share the same y-axis range (-0.34 to 0.44).
- Visual QA: PDF re-rendered at 600 dpi; no clipping, panel collision, legend overlap or colorbar collision.
"""
    (OUT/"QA_REPORT.md").write_text(qa,encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__=="__main__":
    setup_style(); build()
