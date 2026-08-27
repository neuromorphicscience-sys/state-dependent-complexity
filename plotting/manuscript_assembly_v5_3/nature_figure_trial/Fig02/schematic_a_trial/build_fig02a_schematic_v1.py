#!/usr/bin/env python3
"""Draw a standalone Fig.2a controlled resource-allocation schematic."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

MM, DPI = 25.4, 600
HERE = Path(__file__).resolve().parent
OUT = HERE / "output_v1"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

TEAL = "#2A9D8F"
BLUE = "#38598C"
INK = "#263238"
MID = "#7B858A"
LIGHT = "#D9DEE1"
PALE = "#F2F4F5"


def setup_style():
    if ARIAL.exists(): font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists(): font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family":"Arial",
        "pdf.fonttype":42,
        "ps.fonttype":42,
        "svg.fonttype":"none",
        "savefig.facecolor":"white",
        "figure.facecolor":"white",
    })


def draw_network(ax,cx,cy,high_nodes):
    radius=1.72
    angles=np.deg2rad([90,30,-30,-90,-150,150])
    pts=np.c_[cx+radius*np.cos(angles),cy+radius*np.sin(angles)]
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,3),(1,4),(2,5)]
    for i,j in edges:
        ax.plot([pts[i,0],pts[j,0]],[pts[i,1],pts[j,1]],
                color=LIGHT,lw=.48,zorder=1,solid_capstyle="round")
    for i,(x,y) in enumerate(pts):
        high=i in high_nodes
        ax.add_patch(Circle((x,y),.36,facecolor=TEAL if high else PALE,
                            edgecolor=TEAL if high else MID,
                            linewidth=.48,zorder=3))


def arrow(ax,x1,x2,y,double=False):
    style="<->" if double else "->"
    ax.add_patch(FancyArrowPatch((x1,y),(x2,y),arrowstyle=style,
                                mutation_scale=5.0,lw=.58,color=MID,
                                shrinkA=0,shrinkB=0))


def build():
    fw,fh=34.0,19.0
    fig=plt.figure(figsize=(fw/MM,fh/MM))
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,fw); ax.set_ylim(0,fh); ax.axis("off")

    ax.plot([17,17],[4.15,15.25],color=LIGHT,lw=.48)

    ax.text(8.5,14.55,r"Total budget, $B_C$",ha="center",va="center",
            fontsize=4.15,fontweight="bold",color=INK)
    ax.text(25.5,14.55,r"Allocation, $A$",ha="center",va="center",
            fontsize=4.15,fontweight="bold",color=INK)

    draw_network(ax,4.7,10.25,{0})
    draw_network(ax,12.3,10.25,{0,1,5})
    arrow(ax,7.0,10.0,10.25)
    ax.text(4.7,7.45,"Low",ha="center",va="center",fontsize=3.55,color=MID)
    ax.text(12.3,7.45,"High",ha="center",va="center",fontsize=3.55,color=MID)
    ax.text(8.5,5.55,r"$B_C$ varies;  $A$ fixed",
            ha="center",va="center",fontsize=3.35,color=MID)

    draw_network(ax,21.7,10.25,{0,2,4})
    draw_network(ax,29.3,10.25,{1,3,5})
    arrow(ax,24.0,27.0,10.25,double=True)
    ax.text(21.7,7.45,r"$A_1$",ha="center",va="center",fontsize=3.55,color=MID)
    ax.text(29.3,7.45,r"$A_2$",ha="center",va="center",fontsize=3.55,color=MID)
    ax.text(25.5,5.55,r"$A$ varies;  $B_C$ fixed",
            ha="center",va="center",fontsize=3.35,color=MID)

    ax.text(14.5,2.15,"Controlled intervention",ha="right",va="center",
            fontsize=3.55,color=INK)
    ax.add_patch(FancyArrowPatch((15.0,2.15),(19.0,2.15),arrowstyle="->",
                                mutation_scale=5.2,lw=.72,color=BLUE,
                                shrinkA=0,shrinkB=0))
    ax.text(19.5,2.15,"Collective response",ha="left",va="center",
            fontsize=3.55,color=INK)

    OUT.mkdir(parents=True,exist_ok=True)
    stem=OUT/"Fig02a_controlled_resource_allocation_v1"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"),dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"),dpi=DPI,
                pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)

    metadata={
        "panel":"Fig02a",
        "status":"standalone_review_not_integrated",
        "final_size_mm":[fw,fh],
        "scientific_message":"Independently vary total intrinsic-complexity budget and its allocation while topology, collective state and drive remain fixed.",
        "encoding":{"high_complexity_node":TEAL,"ordinary_node":PALE,
                    "fixed_topology_edges":LIGHT},
    }
    (OUT/"schematic_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(stem.with_suffix(".pdf"))


if __name__=="__main__":
    setup_style(); build()
