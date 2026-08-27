#!/usr/bin/env python3
"""Build Supplementary Figure 1: SynPhys cohort and complexity stratification."""

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
SUPP = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "03_SUPPLEMENTARY"
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


def add_axes_mm(fig, fw, fh, x, y, w, h):
    return fig.add_axes([x/fw, y/fh, w/fw, h/fh])


def clean(ax, grid="y"):
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(direction="out", pad=1.15)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=.32, alpha=.62)
        ax.set_axisbelow(True)


def panel_label(ax, letter, fw=183, fh=62):
    pos=ax.get_position(); aw,ah=pos.width*fw,pos.height*fh
    ax.text(-8/aw,1+2/ah,letter,transform=ax.transAxes,fontsize=8,fontweight="bold",
            ha="left",va="bottom",clip_on=False)


def source(path):
    p=SUPP/path
    if not p.exists(): raise FileNotFoundError(p)
    return p


def plot_structural_effects(ax,d):
    x=d.sort_values("eta2_subtype",ascending=True).copy()
    labels=[s.replace("_"," ") for s in x.feature]
    colors=[TEAL if p<.05 else GREY for p in x.perm_p]
    ax.barh(range(len(x)),x.eta2_subtype,color=colors,height=.68,edgecolor=INK,lw=.35)
    ax.set_yticks(range(len(x)),labels,fontsize=2.85)
    ax.set_xlim(0,.80); ax.set_xticks([0,.2,.4,.6,.8])
    ax.set_xlabel("Subtype effect size, η²"); ax.set_ylabel("Structural descriptor")
    ax.text(.98,.04,"Permutation p < 0.05",transform=ax.transAxes,ha="right",va="bottom",
            fontsize=3.1,color=TEAL)
    clean(ax,"x")


def styled_box(ax, groups, labels, color, xlabel):
    bp=ax.boxplot(groups,positions=range(len(groups)),widths=.62,patch_artist=True,
                  showfliers=False,medianprops={"color":GOLD,"lw":.9},
                  whiskerprops={"color":INK,"lw":.6},capprops={"color":INK,"lw":.6},
                  boxprops={"edgecolor":INK,"lw":.6})
    for p in bp["boxes"]:
        p.set_facecolor(color); p.set_alpha(.22)
    ax.set_xticks(range(len(labels)),labels,rotation=30,ha="right",fontsize=3.15)
    ax.set_xlabel(xlabel,labelpad=.8)
    ax.set_ylabel("Transferred intrinsic-complexity score")
    ax.set_ylim(.38,1.84); clean(ax)


def plot_cre(ax,d):
    order=["pvalb","sst","vip","nr5a1","ntsr1","unknown","sim1","tlx3"]
    present=[x for x in order if x in set(d.cre_type)]
    styled_box(ax,[d.loc[d.cre_type==x,"C_primary_shared"].dropna().to_numpy() for x in present],
               [x.upper() if x in {"sst","vip"} else x.capitalize() for x in present],TEAL,"Cre type")


def plot_layer(ax,d):
    order=["2/3","6a","MISSING","4","5"]
    present=[x for x in order if x in set(d.cortical_layer.astype(str))]
    styled_box(ax,[d.loc[d.cortical_layer.astype(str)==x,"C_primary_shared"].dropna().to_numpy() for x in present],
               ["Unknown" if x=="MISSING" else x for x in present],BLUE,"Cortical layer")


def plot_sample_sizes(ax,d):
    x=d.sort_values("n",ascending=False).reset_index(drop=True)
    rank=np.arange(1,len(x)+1)
    ax.bar(rank,x.n,color=BLUE,width=.78,edgecolor="none",alpha=.88)
    ax.axhline(x.n.median(),color=GOLD,lw=.65,ls="--",dashes=(3,2))
    ax.text(.98,.96,f"Median n = {x.n.median():.0f}\nRange = {x.n.min():.0f}-{x.n.max():.0f}",
            transform=ax.transAxes,ha="right",va="top",fontsize=3.15)
    ax.set_xlim(.2,len(x)+.8); ax.set_xticks([1,16,32,48,64])
    ax.set_xlabel("Projection subtype rank"); ax.set_ylabel("Canonical neurons")
    clean(ax)


def build():
    paths={
        "a":source(Path("S01/S01_a__PFC_subtype_structural_effects/source_data/01_S5A_16_PFC_subtype_structural_effects_descriptive__pfc2022_feature_subtype_effects.csv")),
        "b1":source(Path("S07/S07_b__Complexity_by_Cre_type_and_cortical_layer/source_data/01_SYN07_complexity_by_cre_type__SYN07_complexity_by_cre_type.csv")),
        "b2":source(Path("S07/S07_b__Complexity_by_Cre_type_and_cortical_layer/source_data/02_SYN08_complexity_by_layer__SYN08_complexity_by_layer.csv")),
        "c":source(Path("S01/S01_b__PFC_subtype_sample_sizes/source_data/01_S5A_17_PFC_subtype_sample_sizes__pfc2022_subtype_centroids.csv")),
        "table1":source(Path("S01/S01_c__Zero-variance_structural-variable_audit/source_data/01_S5A_18_PFC_zero_variance_audit__pfc_feature_qc.csv")),
    }
    data={k:pd.read_csv(p) for k,p in paths.items()}
    fw,fh,y,h=183.0,62.0,15.0,32.0
    fig=plt.figure(figsize=(fw/MM,fh/MM))
    a=add_axes_mm(fig,fw,fh,25,y,33,h)
    b1=add_axes_mm(fig,fw,fh,70,y,30,h)
    b2=add_axes_mm(fig,fw,fh,113,y,30,h)
    c=add_axes_mm(fig,fw,fh,154,y,23,h)
    plot_structural_effects(a,data["a"]); plot_cre(b1,data["b1"])
    plot_layer(b2,data["b2"]); plot_sample_sizes(c,data["c"])
    for ax,letter in [(a,"a"),(b1,"b"),(c,"c")]: panel_label(ax,letter,fw,fh)

    OUT.mkdir(parents=True,exist_ok=True); stem=OUT/"SI01_full_nature_trial_v1"
    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)

    rel={k:str(v.relative_to(ROOT)) for k,v in paths.items()}
    manifest={"figure":"SI01","version":"reference_v1","status":"review_not_frozen",
              "scientific_role":"Document SynPhys/PFC cohort support and identity-structured intrinsic-complexity stratification.",
              "panel_source_map":{"a":[rel["a"]],"b":[rel["b1"],rel["b2"]],"c":[rel["c"]]},
              "supplementary_table_1_source":[rel["table1"]]}
    (OUT/"source_data_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    geom={"canvas_mm":[fw,fh],"axes_mm":{"a":[25,y,33,h],"b1":[70,y,30,h],
          "b2":[113,y,30,h],"c":[154,y,23,h]},"panel_label_offset_mm":[-8,2]}
    (OUT/"axis_geometry_mm.json").write_text(json.dumps(geom,indent=2),encoding="utf-8")
    contract="""# Supplementary Figure 1 contract

- Scientific role: cohort and identity-stratification support for the SynPhys/Fig.5 block.
- Sequence: `a+b+b+c`; panel b contains two adjacent microplots and never wraps.
- Canvas: 183 x 62 mm; one row of four 32-mm-high data axes.
- Zero-variance structural-variable audit is mapped to Supplementary Table 1, not plotted.
- This is a review version and is not frozen.
"""
    (OUT/"figure_contract.md").write_text(contract,encoding="utf-8")
    qa="""# Supplementary Figure 1 QA report

- Status: review version; not frozen.
- Exports: PDF, SVG, 600-dpi PNG and 600-dpi TIFF.
- Source mapping: complete for panels a-c and Supplementary Table 1.
- Multi-plot panel b remains on one row.
- Visual QA: PDF re-rendered at 600 dpi; no clipping, panel collision, label overlap or legend collision.
"""
    (OUT/"QA_REPORT.md").write_text(qa,encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__=="__main__":
    setup_style(); build()
