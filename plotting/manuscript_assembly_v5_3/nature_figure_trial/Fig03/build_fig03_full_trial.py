#!/usr/bin/env python3
"""Build the source-data-first Nature-width Fig.3 reference figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from schematic_a_trial.build_fig03a_schematic_v1 import draw_schematic


MM_PER_INCH = 25.4
DPI = 600
ROOT = Path(__file__).resolve().parents[3]
ATLAS = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "01_MAIN" / "Fig03"
DEFAULT_OUT = Path(__file__).resolve().parent / "output_full_v1"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

COL = {
    "dynamics_aware": "#2A9D8F",
    "spectral": "#38598C",
    "feedback_hub": "#5875A4",
    "high_degree": "#8B7DAA",
    "module_bridge": "#6F9D63",
    "cycle_proxy": "#D06F82",
    "random": "#737C82",
    "discovery": "#D06F82",
    "replication": "#38598C",
    "generalization": "#2A9D8F",
    "pooled": "#657178",
    "sparse_drive": "#2A9D8F",
    "transition_mid": "#38598C",
    "transition_dense": "#C27628",
}

METHOD_LABEL = {
    "dynamics_aware": "Dynamics-aware",
    "spectral": "Spectral",
    "feedback_hub": "Feedback hub",
    "high_degree": "High degree",
    "module_bridge": "Module bridge",
    "cycle_proxy": "Cycle proxy",
    "random": "Random",
}

DESC_LABEL = {
    "sel_total_degree": "Selected degree",
    "coverage1": "1-hop coverage",
    "coverage2": "2-hop coverage",
    "redundancy": "Redundancy",
    "dispersion": "Dispersion",
    "sel_spectral": "Spectral score",
    "sel_feedback": "Feedback score",
    "sel_cycle3": "3-cycle score",
    "sel_bridge": "Bridge score",
}

TOPOLOGY_LABEL = {
    "scale_free": "Scale-free",
    "erdos_renyi": "ER",
    "modular": "Modular",
    "small_world": "Small-world",
}

REGIME_LABEL = {
    "sparse_drive": "Sparse drive",
    "transition_mid": "Transition mid",
    "transition_dense": "Transition dense",
}


def setup_style() -> None:
    if ARIAL.exists():
        font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists():
        font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 5.2,
        "axes.labelsize": 5.2,
        "xtick.labelsize": 4.6,
        "ytick.labelsize": 4.6,
        "legend.fontsize": 4.1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def add_axes_mm(fig, fw, fh, x, y, w, h):
    return fig.add_axes([x/fw, y/fh, w/fw, h/fh])


def clean_axis(ax, xgrid=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.5)
    if xgrid:
        ax.grid(axis="x", color="#D9DEE1", lw=.35, alpha=.55)
        ax.set_axisbelow(True)


def source_csv(panel: str, token: str) -> Path:
    hits = sorted((ATLAS / panel / "source_data").glob(f"*{token}*.csv"))
    if not hits:
        raise FileNotFoundError(f"No source CSV for {panel}: {token}")
    return hits[0]


def plot_method_performance(ax):
    p = source_csv("Fig03_b__Validated_dynamics-aware_advantage", "method_performance")
    d = pd.read_csv(p)
    order = ["dynamics_aware", "spectral", "feedback_hub", "high_degree",
             "module_bridge", "cycle_proxy", "random"]
    d = d.set_index("method").reindex(order).reset_index()
    y = np.arange(len(d))[::-1]
    for yy, row in zip(y, d.itertuples(index=False)):
        c = COL[row.method]
        ax.hlines(yy, row.ci95_low, row.ci95_high, color=c, lw=.9)
        ax.scatter(row.mean_validated_score, yy, s=11, color=c, zorder=3)
    ax.set_yticks(y, [METHOD_LABEL[k] for k in order], fontsize=4.1)
    ax.set_xlabel("Validated score")
    clean_axis(ax, xgrid=True)
    return p


def plot_paired_gain(ax):
    p = source_csv("Fig03_b__Validated_dynamics-aware_advantage", "paired_gain")
    d = pd.read_csv(p)
    order = ["spectral", "feedback_hub", "high_degree", "module_bridge",
             "cycle_proxy", "random"]
    d = d.set_index("baseline").reindex(order).reset_index()
    y = np.arange(len(d))[::-1]
    ax.axvline(0, color="#8A9297", lw=.6, ls="--", dashes=(3, 2))
    for yy, row in zip(y, d.itertuples(index=False)):
        c = COL[row.baseline]
        ax.hlines(yy, row.ci95_low, row.ci95_high, color=c, lw=.9)
        ax.scatter(row.mean_gain, yy, s=11, color=c, zorder=3)
    ax.set_yticks(y, [METHOD_LABEL[k] for k in order], fontsize=4.1)
    ax.set_xlabel("Paired validated-score gain")
    clean_axis(ax, xgrid=True)
    return p


def plot_structural_shift(ax):
    p = source_csv("Fig03_c__Structural_departure_from_static_descriptors",
                   "structural_phenotype")
    d = pd.read_csv(p)
    order = ["sel_total_degree", "coverage1", "coverage2", "redundancy",
             "dispersion", "sel_spectral", "sel_feedback", "sel_cycle3",
             "sel_bridge"]
    d = d.set_index("descriptor").reindex(order).reset_index()
    y = np.arange(len(d))[::-1]
    ax.axvline(0, color="#8A9297", lw=.6, ls="--", dashes=(3, 2))
    for yy, row in zip(y, d.itertuples(index=False)):
        ax.hlines(yy, row.ci95_low, row.ci95_high, color=COL["spectral"], lw=.8)
        ax.scatter(row.mean_shift, yy, s=9, color=COL["dynamics_aware"], zorder=3)
    ax.set_yticks(y, [DESC_LABEL[k] for k in order], fontsize=4.0)
    ax.set_xlabel("Descriptor shift vs spectral")
    handles = [
        Line2D([0], [0], color=COL["dynamics_aware"], marker="o", lw=0, ms=2.5),
        Line2D([0], [0], color=COL["spectral"], lw=.9),
    ]
    ax.legend(handles, ["Mean shift", "95% CI"], loc="upper left",
              frameon=False, fontsize=3.6, handlelength=1.1,
              labelspacing=.18, handletextpad=.35, borderaxespad=.25)
    clean_axis(ax, xgrid=True)
    return p


def plot_descriptor_gain(ax):
    p = source_csv("Fig03_c__Structural_departure_from_static_descriptors",
                   "descriptor_gain")
    d = pd.read_csv(p).sort_values(
        "spearman_rho_descriptor_shift_vs_score_gain", ascending=False)
    y = np.arange(len(d))[::-1]
    rho = d["spearman_rho_descriptor_shift_vs_score_gain"].to_numpy(float)
    ax.axvline(0, color="#8A9297", lw=.6, ls="--", dashes=(3, 2))
    ax.hlines(y, 0, rho, color="#CCD2D6", lw=.8)
    ax.scatter(rho, y, s=10, color=COL["high_degree"], zorder=3)
    ax.set_yticks(y, [DESC_LABEL.get(k, k) for k in d.descriptor], fontsize=4.0)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman rho")
    handles = [
        Line2D([0], [0], color=COL["high_degree"], marker="o", lw=0, ms=2.5),
        Line2D([0], [0], color="#8A9297", lw=.6, ls="--", dashes=(3, 2)),
    ]
    ax.legend(handles, ["Descriptor-gain association", "Zero"],
              loc="upper left", frameon=False, fontsize=3.6,
              handlelength=1.1, labelspacing=.18, handletextpad=.35,
              borderaxespad=.25)
    clean_axis(ax, xgrid=True)
    return p


def plot_search_validation(ax):
    p = source_csv("Fig03_f__Search-versus-validation_separation", "search_validation")
    d = pd.read_csv(p)
    x = d.optimizer_search_best.to_numpy(float)
    y = d.final_score_mean.to_numpy(float)
    lo = min(float(np.nanmin(x)), float(np.nanmin(y)))
    hi = max(float(np.nanmax(x)), float(np.nanmax(y)))
    pad = .04 * (hi-lo)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color="#858E93", lw=.65,
            ls="--", dashes=(3, 2))
    ax.scatter(x, y, s=8, color=COL["dynamics_aware"], alpha=.82,
               edgecolor="none")
    ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad),
           xlabel="Optimizer search best", ylabel="Fresh validated score")
    clean_axis(ax)
    return p


def plot_replication_forest(ax):
    p = source_csv("Fig03_d__Independent_replication_and_cross-phase_generalization",
                   "paired_contrasts")
    discovery_p = source_csv("Fig03_b__Validated_dynamics-aware_advantage", "paired_gain")
    d = pd.read_csv(p)
    disc = pd.read_csv(discovery_p).query("baseline == 'spectral'").iloc[0]
    groups = [
        ("Discovery (4A)", disc.mean_gain, disc.ci95_low, disc.ci95_high,
         int(disc.n_paired_tasks), "discovery"),
    ]
    labels = {
        "scale_free_N256_replication_atlas": "Rep. + atlas",
        "ER_N256_repair": "ER repair",
        "modular_N256": "Modular",
        "small_world_N256": "Small-world",
        "scale_free_scaling_N128_N512": "SF scaling",
    }
    order = list(labels)
    sub = d[(d.baseline == "spectral") & (d.grouping == "evidence_stratum")]
    sub = sub.set_index("group").reindex(order)
    for key, row in sub.iterrows():
        family = "replication" if key == order[0] else "generalization"
        groups.append((labels[key], row.mean_delta, row.bootstrap95_low,
                       row.bootstrap95_high, int(row.n_tasks), family))
    pooled = d[(d.baseline == "spectral") & (d.grouping == "pooled")].iloc[0]
    groups.append(("Pooled core", pooled.mean_delta,
                   pooled.bootstrap95_low, pooled.bootstrap95_high,
                   int(pooled.n_tasks), "pooled"))

    y = np.arange(len(groups))[::-1]
    ax.axvline(0, color="#8A9297", lw=.6, ls="--", dashes=(3, 2))
    for yy, (label, mean, low, high, n, family) in zip(y, groups):
        c = COL[family]
        ax.hlines(yy, low, high, color=c, lw=1.0)
        ax.scatter(mean, yy, s=12, color=c, edgecolor="white", lw=.25, zorder=3)
        ax.text(high+.004, yy, f"n={n}", va="center", ha="left",
                fontsize=3.8, color="#4C555A")
    ax.set_yticks(y, [g[0] for g in groups], fontsize=3.9)
    ax.set_xlabel("Dynamics-aware gain vs spectral baseline")
    ax.set_ylim(-.35, 8.15)
    handles = [Line2D([0], [0], color=COL[k], lw=1, marker="o", ms=2.5)
               for k in ["discovery", "replication", "generalization", "pooled"]]
    ax.legend(handles, ["Discovery", "Replication", "Generalization", "Pooled"],
              ncol=2, loc="upper right", frameon=False, fontsize=3.45,
              handlelength=1.2, columnspacing=.7, labelspacing=.2,
              borderaxespad=.2)
    clean_axis(ax, xgrid=True)
    return [p, discovery_p]


def plot_gain_landscape(ax, topo, show_ylabel):
    p = source_csv("Fig03_e__Gain_landscape_across_topology_and_collective_regime",
                   "gain_landscape")
    d = pd.read_csv(p)
    d = d[d.topology == topo].copy()
    d["budget_fraction"] = d.k / d.N
    for regime in ["sparse_drive", "transition_mid", "transition_dense"]:
        r = d[d.regime == regime]
        first = True
        for _, g in r.groupby("phase", sort=False):
            g = g.sort_values("budget_fraction")
            x = g.budget_fraction.to_numpy(float)
            y = g.mean_gain.to_numpy(float)
            lo = g.bootstrap95_low.to_numpy(float)
            hi = g.bootstrap95_high.to_numpy(float)
            ax.plot(x, y, color=COL[regime], lw=.75, marker="o", ms=1.6,
                    label=REGIME_LABEL[regime] if first else None)
            ax.fill_between(x, lo, hi, color=COL[regime], alpha=.10, lw=0)
            first = False
    ax.axhline(0, color="#8A9297", lw=.55, ls="--", dashes=(3, 2))
    ax.set_xlabel("High-complexity fraction")
    ax.set_ylabel("Gain vs spectral" if show_ylabel else "")
    ax.text(.02, .96, TOPOLOGY_LABEL[topo], transform=ax.transAxes,
            ha="left", va="top", fontsize=4.8, fontweight="bold")
    clean_axis(ax)
    return p


def build(out: Path):
    fw, fh = 183.0, 108.0
    fig = plt.figure(figsize=(fw/MM_PER_INCH, fh/MM_PER_INCH))
    aw, ah = 34.0, 24.0
    xs = [10.0, 54.0, 98.0, 142.0]
    panel_sources = {}

    # Row 1: schematic, the indivisible b pair, and c.
    y1 = 76.0
    schematic_h = 20.5
    axa = add_axes_mm(fig, fw, fh, 6.5, y1 + (ah - schematic_h) / 2,
                      aw, schematic_h)
    draw_schematic(axa, aw, schematic_h)
    panel_sources["a"] = [
        ATLAS / "Fig03_a__Dynamical_leverage_versus_static_prominence" /
        "00_SCHEMATIC_BRIEF.txt",
        Path(__file__).resolve().parent / "schematic_a_trial" /
        "build_fig03a_schematic_v1.py",
    ]
    panel_sources["b"] = [
        plot_method_performance(add_axes_mm(fig, fw, fh, xs[1], y1, aw, ah)),
        plot_paired_gain(add_axes_mm(fig, fw, fh, xs[2], y1, aw, ah)),
    ]
    panel_sources["c"] = [
        plot_search_validation(add_axes_mm(fig, fw, fh, xs[3], y1, aw, ah))
    ]

    # Row 2: d pair and e occupy one continuous evidence strip.
    y2 = 42.0
    panel_sources["d"] = [
        plot_structural_shift(add_axes_mm(fig, fw, fh, 20.0, y2, 42.0, ah)),
        plot_descriptor_gain(add_axes_mm(fig, fw, fh, 82.0, y2, 42.0, ah)),
    ]
    panel_sources["e"] = list(
        plot_replication_forest(add_axes_mm(fig, fw, fh, 140.0, y2, 36.0, ah)))

    # Row 3: f is redrawn as four independent axes, never pasted as a 2x2 raster.
    y3 = 8.0
    e_paths = []
    for idx, (x, topo) in enumerate(zip(xs, ["scale_free", "erdos_renyi",
                                             "modular", "small_world"])):
        e_paths.append(plot_gain_landscape(
            add_axes_mm(fig, fw, fh, x, y3, aw, ah), topo, show_ylabel=True))
    panel_sources["f"] = e_paths

    panel_labels = [
        ("a", 3, y1+25.5), ("b", 47, y1+25.5), ("c", 135, y1+25.5),
        ("d", 3, y2+25.5), ("e", 133, y2+25.5), ("f", 3, y3+25.5),
    ]
    for label, x, y in panel_labels:
        fig.text(x/fw, y/fh, label, ha="left", va="bottom",
                 fontsize=8, fontweight="bold")

    regime_handles = [Line2D([0], [0], color=COL[k], lw=1, marker="o", ms=2.2)
                      for k in ["sparse_drive", "transition_mid", "transition_dense"]]
    fig.legend(regime_handles,
               [REGIME_LABEL[k] for k in ["sparse_drive", "transition_mid", "transition_dense"]],
               ncol=3, loc="lower center", bbox_to_anchor=(.5, 32.5/fh),
               frameon=False, fontsize=4.0, handlelength=1.2,
               columnspacing=.9, handletextpad=.35, borderaxespad=0)

    out.mkdir(parents=True, exist_ok=True)
    stem = out / "Fig03_full_nature_trial"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"), dpi=DPI,
                pil_kwargs={"compression": "tiff_lzw"})

    geometry = {
        "figure_size_mm": [fw, fh],
        "standard_axis_height_mm": ah,
        "standard_micro_axis_mm": [aw, ah],
        "panel_a_schematic_mm": [aw, schematic_h],
        "row_order": ["a+b1+b2+c", "d1+d2+e", "f1+f2+f3+f4"],
        "no_wrapped_panels": ["b", "d", "f"],
        "panel_f_redrawn_from_source_data": True,
    }
    (out / "axis_geometry_mm.json").write_text(
        json.dumps(geometry, indent=2), encoding="utf-8")
    flat_sources = [p for paths in panel_sources.values() for p in paths]
    (out / "source_data_manifest.json").write_text(json.dumps({
        "figure": "Fig03",
        "panel_source_map": {
            panel: list(dict.fromkeys(str(p.relative_to(ROOT)) for p in paths))
            for panel, paths in panel_sources.items()
        },
        "source_data": sorted({str(p.relative_to(ROOT)) for p in flat_sources}),
    }, indent=2), encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"status": "COMPLETE", "output": str(out)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    setup_style()
    build(args.out)


if __name__ == "__main__":
    main()
