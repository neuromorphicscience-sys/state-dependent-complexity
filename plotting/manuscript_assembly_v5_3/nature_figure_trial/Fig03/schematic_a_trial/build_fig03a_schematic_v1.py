#!/usr/bin/env python3
"""Draw the standalone Fig.3a static-prominence versus dynamical-leverage schematic."""

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

PURPLE = "#7563A6"
TEAL = "#2A9D8F"
BLUE = "#38598C"
INK = "#263238"
MID = "#7B858A"
LIGHT = "#D9DEE1"
PALE = "#F2F4F5"


def setup_style() -> None:
    if ARIAL.exists():
        font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists():
        font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


NODE_OFFSETS = np.array([
    [-2.15, 0.00],   # high-degree hub
    [-3.25, 1.72],
    [-3.25, -1.72],
    [-1.55, 2.05],
    [-1.55, -2.05],
    [0.05, 0.00],    # bridge / high-leverage node
    [2.10, 1.38],
    [2.10, -1.38],
])

EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (5, 6), (5, 7), (6, 7),
]


def draw_network(ax, cx: float, cy: float, highlight: int, color: str,
                 halo: bool = False) -> np.ndarray:
    pts = NODE_OFFSETS + np.array([cx, cy])
    for i, j in EDGES:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                color=LIGHT, lw=.54, zorder=1, solid_capstyle="round")
    if halo:
        x, y = pts[highlight]
        ax.add_patch(Circle((x, y), .78, facecolor=color, edgecolor="none",
                            alpha=.14, zorder=2))
    for i, (x, y) in enumerate(pts):
        selected = i == highlight
        ax.add_patch(Circle(
            (x, y), .34,
            facecolor=color if selected else PALE,
            edgecolor=color if selected else MID,
            linewidth=.52,
            zorder=3,
        ))
    return pts


def draw_response_glyph(ax, x0: float, y0: float) -> None:
    """Draw a compact set of population-response traces."""
    ax.add_patch(FancyArrowPatch(
        (x0 - 2.55, y0), (x0 - .35, y0), arrowstyle="->",
        mutation_scale=4.6, lw=.52, color=BLUE, shrinkA=0, shrinkB=0,
    ))
    t = np.linspace(0.0, 4.25, 100)
    profiles = (
        .42 * np.exp(-((t - 1.45) / .43) ** 2)
        - .14 * np.exp(-((t - 2.38) / .52) ** 2),
        -.18 * np.exp(-((t - 1.10) / .40) ** 2)
        + .38 * np.exp(-((t - 2.18) / .58) ** 2),
        .14 * np.exp(-((t - .82) / .34) ** 2)
        + .34 * np.exp(-((t - 2.72) / .62) ** 2),
    )
    for offset, profile in zip((.78, 0.0, -.78), profiles):
        baseline = y0 + offset
        ax.plot([x0, x0 + 4.25], [baseline, baseline],
                color=LIGHT, lw=.38, zorder=1, solid_capstyle="round")
        ax.plot(x0 + t, baseline + profile, color=BLUE, lw=.66,
                zorder=2, solid_capstyle="round", solid_joinstyle="round")


def draw_schematic(ax, fw: float = 34.0, fh: float = 20.5) -> None:
    """Draw the schematic into an existing axis using millimetre coordinates."""
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.axis("off")

    left_x, right_x, net_y = 7.2, 22.2, 11.35

    ax.text(left_x, 19.0, "Static prominence", ha="center", va="center",
            fontsize=4.35, fontweight="bold", color=INK)
    ax.text(25.0, 19.0, "Dynamical leverage", ha="center", va="center",
            fontsize=4.35, fontweight="bold", color=INK)

    draw_network(ax, left_x, net_y, highlight=0, color=PURPLE)
    right_pts = draw_network(ax, right_x, net_y, highlight=5, color=TEAL, halo=True)

    ax.text(14.75, net_y + .05, r"$\ne$", ha="center", va="center",
            fontsize=8.0, color=MID)
    ax.text(14.75, 15.65, r"same $G$", ha="center", va="center",
            fontsize=3.15, color=MID)

    bridge_x, bridge_y = right_pts[5]
    ax.text(bridge_x, 15.55, "+c", ha="center", va="center",
            fontsize=3.35, fontweight="bold", color=TEAL)
    ax.add_patch(FancyArrowPatch(
        (bridge_x, 14.92), (bridge_x, bridge_y + .52), arrowstyle="->",
        mutation_scale=5.0, lw=.62, color=TEAL, shrinkA=0, shrinkB=0,
    ))

    draw_response_glyph(ax, 28.65, net_y)
    ax.text(30.78, 9.15, r"collective $\Delta$", ha="center", va="center",
            fontsize=3.05, color=BLUE)

    ax.text(left_x, 6.70, "degree / centrality", ha="center", va="center",
            fontsize=3.35, color=PURPLE)
    ax.text(24.8, 6.70, "validated marginal gain", ha="center", va="center",
            fontsize=3.35, color=TEAL)

    footer_y = 1.70
    ax.text(5.6, footer_y, "rank nodes", ha="center", va="center",
            fontsize=3.45, color=INK)
    ax.add_patch(FancyArrowPatch(
        (9.25, footer_y), (11.45, footer_y), arrowstyle="->",
        mutation_scale=4.8, lw=.58, color=MID, shrinkA=0, shrinkB=0,
    ))
    ax.text(16.7, footer_y, "place complexity", ha="center", va="center",
            fontsize=3.45, color=INK)
    ax.add_patch(FancyArrowPatch(
        (21.85, footer_y), (24.05, footer_y), arrowstyle="->",
        mutation_scale=4.8, lw=.58, color=MID, shrinkA=0, shrinkB=0,
    ))
    ax.text(29.0, footer_y, "fresh validation", ha="center", va="center",
            fontsize=3.45, color=INK)


def build() -> None:
    fw, fh = 34.0, 20.5
    fig = plt.figure(figsize=(fw / MM, fh / MM))
    ax = fig.add_axes([0, 0, 1, 1])
    draw_schematic(ax, fw, fh)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "Fig03a_dynamical_vs_static_v1"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"), dpi=DPI,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    metadata = {
        "panel": "Fig03a",
        "status": "integrated_into_frozen_fig03",
        "final_size_mm": [fw, fh],
        "scientific_message": (
            "In the same topology, structural prominence and independently "
            "validated dynamical leverage need not identify the same node."
        ),
        "encoding": {
            "static_prominence": PURPLE,
            "dynamical_leverage": TEAL,
            "collective_response": BLUE,
            "ordinary_nodes": PALE,
            "fixed_topology_edges": LIGHT,
        },
    }
    (OUT / "schematic_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    setup_style()
    build()
