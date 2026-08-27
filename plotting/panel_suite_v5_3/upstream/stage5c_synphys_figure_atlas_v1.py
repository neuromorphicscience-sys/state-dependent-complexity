#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Stage 5C — Synaptic Physiology Publication Figure Atlas v1
==========================================================

Purpose
-------
Generate a publication-oriented SynPhys figure atlas from the already-frozen
Stage5C integrated + narrative-closure outputs.

This script does NOT reopen the 249-GiB SQLite database and does NOT run new
hypothesis tests. It only re-organizes existing frozen results into a coherent
MAIN/SI/DIAGNOSTIC candidate atlas.

Scientific role of SynPhys in the paper
---------------------------------------
SynPhys is a local-circuit boundary test:

    intrinsic cellular complexity
        != simple structural hubness
        != local PSP/PSC/STP
        != standardized single-synapse temporal gain

This dataset should NOT be used to claim that local synaptic gain is the same
quantity as whole-network, state-dependent dynamical leverage.

Plotting contract
-----------------
- Use D:\Research\Neural Science\plot\plot_style_complexity.py when available.
- White background, Arial/sans-serif, project palette.
- No decorative titles inside scientific panels.
- 600-dpi PNG + vector PDF.
- Exact plot-source CSV for every figure.
- Metadata JSON for every figure.
- Figure catalog with MAIN / SI / DIAGNOSTIC candidate tags.
- Copy existing results; never modify or overwrite scientific input files.

Default inputs
--------------
D:\Research\Neural Science\plot\Stage5C_synphys_integrated_v1
D:\Research\Neural Science\plot\Stage5C_synphys_narrative_closure_v2

Default output
--------------
D:\Research\Neural Science\plot\Stage5C_SynPhys_FigureAtlas_v1

Formal run
----------
python .\plot\stage5c_synphys_figure_atlas_v1.py `
  --root "D:\Research\Neural Science"

Quick rendering check
---------------------
python .\plot\stage5c_synphys_figure_atlas_v1.py `
  --root "D:\Research\Neural Science" `
  --quick
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


DEFAULT_ROOT = Path.cwd()
SEED = 20260821

PRIMARY = "primary_shared"
CONSERVATIVE = "protocol_conservative"
PRIMARY_SCORE = "C_primary_shared"
CONSERVATIVE_SCORE = "C_protocol_conservative"

LOCAL_PHENOTYPES = [
    "psp_amplitude",
    "psc_amplitude",
    "paired_pulse_ratio_50hz",
    "stp_initial_50hz",
    "stp_induction_50hz",
    "stp_recovery_250ms",
    "stp_recovery_single_250ms",
    "variability_resting_state",
    "variability_second_pulse_50hz",
    "variability_stp_induced_state_50hz",
]

STP_ONLY = [
    "paired_pulse_ratio_50hz",
    "stp_initial_50hz",
    "stp_induction_50hz",
    "stp_recovery_250ms",
    "stp_recovery_single_250ms",
    "variability_resting_state",
    "variability_second_pulse_50hz",
    "variability_stp_induced_state_50hz",
]

PRETTY = {
    "connection": "Connection",
    "psp_amplitude": "PSP magnitude",
    "psc_amplitude": "PSC magnitude",
    "paired_pulse_ratio_50hz": "Paired-pulse ratio",
    "stp_initial_50hz": "STP initial",
    "stp_induction_50hz": "STP induction",
    "stp_recovery_250ms": "Recovery 250 ms",
    "stp_recovery_single_250ms": "Single-pulse recovery",
    "variability_resting_state": "Resting variability",
    "variability_second_pulse_50hz": "Second-pulse variability",
    "variability_stp_induced_state_50hz": "STP-state variability",
    "G_norm_5hz": "5 Hz",
    "G_norm_10hz": "10 Hz",
    "G_norm_20hz": "20 Hz",
    "G_norm_50hz": "50 Hz",
    "mouse_connection": "Mouse connection",
    "mouse_G_norm_50hz": "Mouse 50-Hz temporal gain",
    "human_psp_amplitude": "Human PSP",
}


# ---------------------------------------------------------------------
# CLI / I/O
# ---------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--integrated", type=Path, default=None)
    ap.add_argument("--closure", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-cre", type=int, default=10)
    ap.add_argument("--min-group-n", type=int, default=30)
    return ap.parse_args()


def sha256_file(path: Path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_rel(path: Path, root: Path):
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def read_csv(path: Path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return pd.read_csv(path, low_memory=False)


def read_json(path: Path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_style(root: Path):
    candidates = [
        root / "plot" / "plot_style_complexity.py",
        Path(__file__).resolve().parent / "plot_style_complexity.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("plot_style_complexity", p)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod, p
    return None, None


# ---------------------------------------------------------------------
# Project style helpers
# ---------------------------------------------------------------------

class FallbackStyle:
    # Only used if the project style file is unavailable.
    COLOR_COMPLEXITY_LOW = "#4F789D"
    COLOR_COMPLEXITY_HIGH = "#C7654C"
    COLOR_OPTIMIZED = "#168B8C"
    COLOR_SPECTRAL = "#7568A9"
    COLOR_HIGH_DEGREE = "#D49A3A"
    COLOR_GREY = "#9A9A9A"
    COLOR_GREY_DARK = "#666666"
    COLOR_GREY_LIGHT = "#D9D9D9"
    COLOR_AXIS = "#222222"

    @staticmethod
    def create_standard_figure():
        return plt.subplots(figsize=(10.72, 8.20))

    @staticmethod
    def apply_standard_axis_settings(ax, xlabel="", ylabel="", label_fontsize=28, tick_labelsize=22):
        for s in ax.spines.values():
            s.set_linewidth(1.5)
        ax.tick_params(direction="out", length=7, width=1.4, labelsize=tick_labelsize)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=14)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=14)


def configure_global_style(style):
    if hasattr(style, "init_global_style"):
        style.init_global_style()
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def standard_fig(style):
    fig, ax = style.create_standard_figure()
    return fig, ax


def apply_axes(style, ax, xlabel="", ylabel="", dense=False):
    lf = 25 if dense else 30
    tf = 20 if dense else 25
    style.apply_standard_axis_settings(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        label_fontsize=lf,
        tick_labelsize=tf,
    )


def remove_minor_y(ax):
    try:
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    except Exception:
        pass


def no_scientific_offset(ax, axis="y"):
    target = ax.yaxis if axis == "y" else ax.xaxis
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=False)
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    target.set_major_formatter(fmt)


# ---------------------------------------------------------------------
# Atlas writer
# ---------------------------------------------------------------------

class Atlas:
    def __init__(self, root, out, style, style_path, dpi):
        self.root = root
        self.out = out
        self.style = style
        self.style_path = style_path
        self.dpi = dpi
        self.fig_dir = out / "figures"
        self.src_dir = out / "source_data"
        self.meta_dir = out / "metadata"
        self.analysis_dir = out / "analysis"
        for p in [self.fig_dir, self.src_dir, self.meta_dir, self.analysis_dir]:
            p.mkdir(parents=True, exist_ok=True)
        self.catalog = []

    def save(
        self,
        fig_id,
        fig,
        source_df,
        *,
        tier,
        claim,
        description,
        input_paths,
        caveat="",
        source_note="Exact rows/aggregates used for rendering this panel.",
    ):
        png = self.fig_dir / f"{fig_id}.png"
        pdf = self.fig_dir / f"{fig_id}.pdf"
        csv = self.src_dir / f"{fig_id}.csv"
        meta = self.meta_dir / f"{fig_id}.json"

        fig.savefig(png, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        if source_df is None:
            source_df = pd.DataFrame()
        source_df.to_csv(csv, index=False, encoding="utf-8-sig")

        inputs = []
        for p in input_paths:
            p = Path(p)
            inputs.append({
                "path": safe_rel(p, self.root),
                "exists": p.exists(),
                "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
            })

        md = {
            "figure_id": fig_id,
            "candidate_tier": tier,
            "claim": claim,
            "description": description,
            "caveat": caveat,
            "plotting_standard": {
                "dpi_png": self.dpi,
                "vector_pdf": True,
                "white_background": True,
                "style_file": safe_rel(self.style_path, self.root) if self.style_path else "fallback",
                "source_csv": safe_rel(csv, self.root),
                "source_note": source_note,
            },
            "inputs": inputs,
        }
        meta.write_text(json.dumps(md, ensure_ascii=False, indent=2), encoding="utf-8")

        self.catalog.append({
            "figure_id": fig_id,
            "candidate_tier": tier,
            "claim": claim,
            "description": description,
            "caveat": caveat,
            "png": safe_rel(png, self.root),
            "pdf": safe_rel(pdf, self.root),
            "source_csv": safe_rel(csv, self.root),
            "metadata_json": safe_rel(meta, self.root),
        })

    def skip(self, fig_id, tier, claim, description, reason):
        self.catalog.append({
            "figure_id": fig_id,
            "candidate_tier": tier,
            "claim": claim,
            "description": description,
            "caveat": f"SKIPPED: {reason}",
            "png": "",
            "pdf": "",
            "source_csv": "",
            "metadata_json": "",
        })

    def finalize(self):
        cat = pd.DataFrame(self.catalog)
        cat.to_csv(self.analysis_dir / "figure_catalog.csv", index=False, encoding="utf-8-sig")
        return cat


# ---------------------------------------------------------------------
# Plot utilities
# ---------------------------------------------------------------------

def percentile_ci(x, lo=.025, hi=.975):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(x)), float(np.quantile(x, lo)), float(np.quantile(x, hi))


def boxplot_groups(ax, frame, category, value, categories, style):
    data = [
        pd.to_numeric(frame.loc[frame[category].astype(str).eq(str(c)), value], errors="coerce")
        .dropna().to_numpy()
        for c in categories
    ]
    bp = ax.boxplot(
        data,
        labels=[str(c) for c in categories],
        showfliers=False,
        patch_artist=False,
        widths=.58,
        medianprops={"linewidth": 2},
        whiskerprops={"linewidth": 1.4},
        capprops={"linewidth": 1.4},
        boxprops={"linewidth": 1.4},
    )
    return data


def binned_xy(x, y, n_bins=8):
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                      "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(d) < max(40, n_bins * 5):
        return pd.DataFrame(columns=["bin", "x", "y", "se", "n"])
    try:
        d["bin"] = pd.qcut(d["x"], n_bins, duplicates="drop")
    except Exception:
        return pd.DataFrame()
    out = (
        d.groupby("bin", observed=True)
        .agg(x=("x", "mean"), y=("y", "mean"), sd=("y", "std"), n=("y", "size"))
        .reset_index(drop=True)
    )
    out["se"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))
    out.insert(0, "bin", np.arange(1, len(out) + 1))
    return out[["bin", "x", "y", "se", "n"]]


def get_definition_colors(style):
    return {
        PRIMARY: getattr(style, "COLOR_OPTIMIZED", getattr(style, "COLOR_COMPLEXITY_LOW", None)),
        CONSERVATIVE: getattr(style, "COLOR_SPECTRAL", getattr(style, "COLOR_COMPLEXITY_HIGH", None)),
    }


def get_before_after_colors(style):
    return {
        "before": getattr(style, "COLOR_COMPLEXITY_LOW", None),
        "after": getattr(style, "COLOR_COMPLEXITY_HIGH", None),
    }


def create_source_row(label, **kwargs):
    r = {"label": label}
    r.update(kwargs)
    return r


# ---------------------------------------------------------------------
# Main figure builders
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    root = args.root.resolve()
    integrated = (
        args.integrated.resolve()
        if args.integrated
        else root / "plot" / "Stage5C_synphys_integrated_v1"
    )
    closure = (
        args.closure.resolve()
        if args.closure
        else root / "plot" / "Stage5C_synphys_narrative_closure_v2"
    )
    out = (
        args.out.resolve()
        if args.out
        else root / "plot" / "Stage5C_SynPhys_FigureAtlas_v1"
    )
    dpi = 180 if args.quick else args.dpi

    style, style_path = load_style(root)
    if style is None:
        style = FallbackStyle()
        style_path = None
        print("[WARN] plot_style_complexity.py not found; using fallback geometry.")
    configure_global_style(style)
    atlas = Atlas(root, out, style, style_path, dpi)

    IA = integrated / "analysis"
    IT = integrated / "tables"
    IS = integrated / "source_data"
    CA = closure / "analysis"

    required = [
        IA / "stage5c_integrated_report.json",
        IA / "harmonization_primary_shared.csv",
        IA / "transfer_feature_qc_primary_shared.csv",
        IA / "stage5a_oof_primary_shared.csv",
        IA / "stage5a_oof_protocol_conservative.csv",
        IT / "cell_intrinsic_atlas.csv",
        IT / "complexity_transfer_primary_shared.csv",
        IT / "complexity_transfer_protocol_conservative.csv",
        CA / "definition_robustness_models.csv",
        CA / "repeated_experiment_holdout.csv",
        CA / "mapper_bootstrap_key_effects.csv",
        CA / "species_psp_comparison.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing frozen Stage5C outputs:\n" + "\n".join(map(str, missing)) +
            "\nRun Stage5C integrated + narrative closure before the figure atlas."
        )

    report = read_json(IA / "stage5c_integrated_report.json")
    flat = read_csv(CA / "definition_robustness_models.csv")
    hold = read_csv(CA / "repeated_experiment_holdout.csv")
    boot = read_csv(CA / "mapper_bootstrap_key_effects.csv")
    species_psp = read_csv(CA / "species_psp_comparison.csv")
    cells = read_csv(IT / "cell_intrinsic_atlas.csv")
    pscore = read_csv(IT / "complexity_transfer_primary_shared.csv")
    cscore = read_csv(IT / "complexity_transfer_protocol_conservative.csv")
    qc_p = read_csv(IA / "transfer_feature_qc_primary_shared.csv")
    oof_p = read_csv(IA / "stage5a_oof_primary_shared.csv")
    oof_c = read_csv(IA / "stage5a_oof_protocol_conservative.csv")

    dcolors = get_definition_colors(style)
    bacolors = get_before_after_colors(style)
    grey = getattr(style, "COLOR_GREY", "#9A9A9A")
    grey_dark = getattr(style, "COLOR_GREY_DARK", "#666666")
    grey_light = getattr(style, "COLOR_GREY_LIGHT", "#D9D9D9")
    axis_col = getattr(style, "COLOR_AXIS", "#222222")
    low_col = getattr(style, "COLOR_COMPLEXITY_LOW", None)
    high_col = getattr(style, "COLOR_COMPLEXITY_HIGH", None)

    print("=" * 96)
    print("Stage5C SynPhys Publication Figure Atlas v1")
    print("Integrated:", integrated)
    print("Closure   :", closure)
    print("Output    :", out)
    print("DPI       :", dpi)
    print("=" * 96)

    # -----------------------------------------------------------------
    # SYN00 — conceptual boundary map
    # -----------------------------------------------------------------
    fig, ax = standard_fig(style)
    ax.set_axis_off()
    nodes = pd.DataFrame([
        ("Intrinsic multiaxial physiology", 0.08, 0.66, 0.23, 0.16, "input"),
        ("Transferred complexity Cᵢ", 0.38, 0.66, 0.20, 0.16, "complexity"),
        ("Local structure / PSP / STP", 0.69, 0.66, 0.23, 0.16, "local"),
        ("Collective network state s", 0.08, 0.25, 0.23, 0.16, "state"),
        ("Dynamical leverage Lᵢ(G,s,k)", 0.38, 0.25, 0.20, 0.16, "leverage"),
        ("Adaptive collective function", 0.69, 0.25, 0.23, 0.16, "function"),
    ], columns=["label","x","y","w","h","kind"])
    for r in nodes.itertuples():
        ec = axis_col
        lw = 1.8
        patch = FancyBboxPatch(
            (r.x, r.y), r.w, r.h,
            boxstyle="round,pad=0.012,rounding_size=0.01",
            transform=ax.transAxes, facecolor="white", edgecolor=ec, linewidth=lw
        )
        ax.add_patch(patch)
        ax.text(r.x+r.w/2, r.y+r.h/2, r.label, ha="center", va="center",
                transform=ax.transAxes, fontsize=20)
    arrows = [
        (0.31,0.74,0.38,0.74,"transfer"),
        (0.58,0.74,0.69,0.74,"boundary test"),
        (0.31,0.33,0.38,0.33,"collective context"),
        (0.58,0.33,0.69,0.33,"functional value"),
        (0.48,0.66,0.48,0.41,"not locally reducible"),
    ]
    for x0,y0,x1,y1,label in arrows:
        arr = FancyArrowPatch(
            (x0,y0),(x1,y1), transform=ax.transAxes,
            arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color=axis_col
        )
        ax.add_patch(arr)
        if label:
            ax.text((x0+x1)/2, (y0+y1)/2+0.035, label,
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=16, color=grey_dark)
    src = pd.DataFrame(arrows, columns=["x0","y0","x1","y1","edge_label"])
    atlas.save(
        "SYN00_conceptual_boundary_map", fig, src,
        tier="MAIN_CANDIDATE",
        claim="SynPhys constrains local/static explanations; the paper's leverage concept is collective and state-dependent.",
        description="Conceptual bridge showing the deliberately different roles of intrinsic complexity, local synaptic phenotypes, and collective dynamical leverage.",
        caveat="Conceptual schematic; arrows denote analysis logic, not causal claims.",
        input_paths=[IA/"00_STAGE5C_INTEGRATED_SUMMARY.md", CA/"00_STAGE5C_NARRATIVE_CLOSURE_SUMMARY.md"],
    )

    # -----------------------------------------------------------------
    # SYN01 — cohort flow
    # -----------------------------------------------------------------
    flow_path = IS / "F15_cohort_flow.csv"
    if flow_path.exists():
        flow = read_csv(flow_path)
    else:
        counts = report.get("counts", {})
        flow = pd.DataFrame([
            ("All cells", counts.get("cells", np.nan)),
            ("All directed pairs", counts.get("pairs", np.nan)),
            ("Structurally tested pairs", counts.get("tested_pairs", np.nan)),
            ("Mouse tested + C domain", counts.get("mouse_tested_primary", np.nan)),
            ("Mouse connected + C domain", counts.get("mouse_connected_primary", np.nan)),
            ("Mouse measured dynamics + C domain", counts.get("mouse_measured_dynamics_primary", np.nan)),
            ("Mouse modeled synapses + C domain", counts.get("mouse_standardized_models_primary", np.nan)),
        ], columns=["stage","n"])
    flow = flow.dropna()
    fig, ax = standard_fig(style)
    y = np.arange(len(flow))
    ax.barh(y, flow["n"].astype(float))
    ax.set_yticks(y, flow["stage"])
    ax.invert_yaxis()
    ax.set_xscale("log")
    for yi, n in zip(y, flow["n"]):
        ax.text(float(n)*1.05, yi, f"{int(n):,}", va="center", fontsize=18)
    apply_axes(style, ax, xlabel="Number of observations (log scale)", ylabel="", dense=True)
    remove_minor_y(ax)
    atlas.save(
        "SYN01_cohort_flow", fig, flow,
        tier="SI",
        claim="The SynPhys boundary analysis is based on large tested-pair cohorts and hundreds of measured/modelled synapses.",
        description="Cohort attrition from all cells/pairs to the transferred-complexity mouse analysis cohorts.",
        input_paths=[flow_path if flow_path.exists() else IA/"stage5c_integrated_report.json"],
    )

    # -----------------------------------------------------------------
    # SYN02 / SYN03 — Stage5A mapper OOF
    # -----------------------------------------------------------------
    for fig_id, df, label, mode in [
        ("SYN02_primary_mapper_oof", oof_p, "Primary shared-feature transfer", PRIMARY),
        ("SYN03_conservative_mapper_oof", oof_c, "Protocol-conservative transfer", CONSERVATIVE),
    ]:
        src = df.rename(columns={"C_req":"observed","OOF_pred":"predicted"}).copy()
        fig, ax = standard_fig(style)
        rng = np.random.default_rng(SEED + (0 if mode == PRIMARY else 1))
        xj = src["observed"].astype(float) + rng.normal(0, 0.035, len(src))
        ax.scatter(xj, src["predicted"], s=28, alpha=.32, color=dcolors[mode])
        g = src.groupby("observed")["predicted"].agg(["mean","sem","count"]).reset_index()
        ax.errorbar(g["observed"], g["mean"], yerr=g["sem"], fmt="o",
                    capsize=4, linewidth=1.7, markersize=8, color=axis_col)
        apply_axes(
            style, ax,
            xlabel="Stage5A minimum mechanistic complexity cost",
            ylabel="Held-out transferred-complexity prediction"
        )
        ax.set_xticks(sorted(src["observed"].dropna().unique()))
        atlas.save(
            fig_id, fig, src,
            tier="SI",
            claim="The SynPhys complexity score is transferred from an independently defined Stage5A phenotype.",
            description=f"Repeated/nested held-out Stage5A mapping used for {label}.",
            caveat="The mapper is intentionally modest in predictive accuracy; SynPhys conclusions are therefore phrased as detected/undetected associations, not proof of exact zero.",
            input_paths=[IA/f"stage5a_oof_{mode}.csv"],
        )

    # -----------------------------------------------------------------
    # SYN04 — transfer-definition agreement
    # -----------------------------------------------------------------
    cmp = (
        pscore[["cell_id", PRIMARY_SCORE]]
        .merge(cscore[["cell_id", CONSERVATIVE_SCORE]], on="cell_id", how="inner")
        .dropna()
    )
    fig, ax = standard_fig(style)
    ax.scatter(cmp[PRIMARY_SCORE], cmp[CONSERVATIVE_SCORE], s=20, alpha=.25,
               color=getattr(style, "COLOR_OPTIMIZED", low_col))
    lo = float(np.nanmin([cmp[PRIMARY_SCORE].min(), cmp[CONSERVATIVE_SCORE].min()]))
    hi = float(np.nanmax([cmp[PRIMARY_SCORE].max(), cmp[CONSERVATIVE_SCORE].max()]))
    ax.plot([lo,hi],[lo,hi], linewidth=1.3, linestyle="--", color=grey_dark)
    apply_axes(
        style, ax,
        xlabel="Primary shared-feature complexity",
        ylabel="Protocol-conservative complexity"
    )
    atlas.save(
        "SYN04_transfer_definition_agreement", fig, cmp,
        tier="SI",
        claim="Key SynPhys conclusions are checked against two frozen cross-dataset complexity definitions.",
        description="Cell-wise agreement between primary 7-feature and protocol-conservative 4-feature transferred complexity scores.",
        input_paths=[IT/"complexity_transfer_primary_shared.csv",
                     IT/"complexity_transfer_protocol_conservative.csv"],
    )

    # -----------------------------------------------------------------
    # SYN05 — cross-dataset feature shift in Stage5A-SD units
    # -----------------------------------------------------------------
    q = qc_p.copy()
    q["median_shift_stage5a_sd"] = (
        (pd.to_numeric(q["synphys_median"], errors="coerce") -
         pd.to_numeric(q["stage5a_median"], errors="coerce")) /
        pd.to_numeric(q["stage5a_sd"], errors="coerce").replace(0, np.nan)
    )
    q = q.replace([np.inf,-np.inf], np.nan).dropna(subset=["median_shift_stage5a_sd"])
    q = q.sort_values("median_shift_stage5a_sd")
    fig, ax = standard_fig(style)
    y = np.arange(len(q))
    ax.scatter(q["median_shift_stage5a_sd"], y, s=90, color=getattr(style, "COLOR_SPECTRAL", high_col))
    ax.axvline(0, linewidth=1.2, linestyle="--", color=grey_dark)
    ax.set_yticks(y, q["feature"])
    apply_axes(style, ax, xlabel="SynPhys median shift relative to Stage5A (Stage5A SD)", ylabel="", dense=True)
    remove_minor_y(ax)
    atlas.save(
        "SYN05_transfer_feature_domain_shift", fig, q,
        tier="SI",
        claim="Cross-dataset transfer is audited feature by feature rather than assumed to be in-domain.",
        description="Median SynPhys–Stage5A feature shift normalized by the Stage5A standard deviation.",
        caveat="This is a transfer-domain diagnostic, not a biological effect size.",
        input_paths=[IA/"transfer_feature_qc_primary_shared.csv"],
    )

    # -----------------------------------------------------------------
    # SYN06 / 07 / 08 — complexity organization across identity
    # -----------------------------------------------------------------
    score_merge = (
        cells.merge(pscore[["cell_id", PRIMARY_SCORE, PRIMARY_SCORE+"_domain"]],
                    on="cell_id", how="left")
    )
    m = (
        score_merge["species"].astype(str).str.lower().eq("mouse") &
        score_merge[PRIMARY_SCORE+"_domain"].fillna(False).astype(bool) &
        pd.to_numeric(score_merge[PRIMARY_SCORE], errors="coerce").notna()
    )
    cmouse = score_merge[m].copy()

    group_specs = [
        ("SYN06_complexity_by_cell_class", "cell_class", "Cell class", "SI", None),
        ("SYN07_complexity_by_cre_type", "cre_type", "Cre type", "SI", args.max_cre),
        ("SYN08_complexity_by_layer", "cortical_layer", "Cortical layer", "SI", None),
    ]
    for fig_id, col, xlabel, tier, maxn in group_specs:
        if col not in cmouse.columns:
            atlas.skip(fig_id, tier, "Biological-identity organization of transferred complexity.", f"Complexity score by {xlabel}.", f"missing column {col}")
            continue
        tmp = cmouse[[col, PRIMARY_SCORE]].copy()
        tmp[col] = tmp[col].fillna("MISSING").astype(str)
        counts = tmp[col].value_counts()
        cats = counts[counts >= args.min_group_n].index.tolist()
        if maxn is not None:
            cats = cats[:maxn]
        tmp = tmp[tmp[col].isin(cats)].copy()
        if not len(cats):
            atlas.skip(fig_id, tier, "Biological-identity organization of transferred complexity.", f"Complexity score by {xlabel}.", "no groups pass minimum n")
            continue
        # order by median score
        med = tmp.groupby(col)[PRIMARY_SCORE].median().sort_values()
        cats = med.index.tolist()
        fig, ax = standard_fig(style)
        data = boxplot_groups(ax, tmp, col, PRIMARY_SCORE, cats, style)
        ax.set_xticklabels(cats, rotation=28, ha="right")
        apply_axes(style, ax, xlabel=xlabel, ylabel="Transferred intrinsic-complexity score", dense=True)
        atlas.save(
            fig_id, fig, tmp,
            tier=tier,
            claim="Transferred intrinsic complexity is biologically identity-structured.",
            description=f"Distribution of mouse transferred complexity across {xlabel.lower()} groups.",
            caveat="Descriptive organization; not a causal identity effect.",
            input_paths=[IT/"cell_intrinsic_atlas.csv", IT/"complexity_transfer_primary_shared.csv"],
        )

    # -----------------------------------------------------------------
    # SYN09 — raw connection probability vs complexity
    # -----------------------------------------------------------------
    raw_conn_path = IS / "F04_connection_probability_binned.csv"
    if raw_conn_path.exists():
        b = read_csv(raw_conn_path)
        srcpaths = [raw_conn_path]
    else:
        tested_path = IT / "mouse_tested_pairs_primary.csv"
        tested = read_csv(tested_path, required=False)
        if tested is not None:
            b = binned_xy(tested[PRIMARY_SCORE], tested["connected"].astype(float), 8)
            srcpaths = [tested_path]
        else:
            b = pd.DataFrame()
            srcpaths = []
    if len(b):
        fig, ax = standard_fig(style)
        ax.errorbar(b["x"], b["y"], yerr=b["se"], fmt="o-", capsize=4,
                    linewidth=1.8, markersize=8, color=getattr(style, "COLOR_OPTIMIZED", low_col))
        apply_axes(style, ax, xlabel="Transferred intrinsic-complexity score", ylabel="Observed connection probability")
        atlas.save(
            "SYN09_raw_connection_probability", fig, b,
            tier="SI",
            claim="Raw local connection probability does not increase monotonically with transferred intrinsic complexity.",
            description="Observed connection probability across complexity quantiles.",
            caveat="Descriptive raw relationship; adjusted structural inference is shown separately.",
            input_paths=srcpaths,
        )
    else:
        atlas.skip("SYN09_raw_connection_probability","SI","Raw structural relation.","Observed connection probability across complexity bins.","source unavailable")

    # -----------------------------------------------------------------
    # SYN10 — adjusted connection OR across definitions
    # -----------------------------------------------------------------
    con = flat[flat["endpoint"].eq("connection")].copy()
    con["odds_ratio_plot"] = np.exp(con["beta_after_identity"])
    con["or_lo"] = np.exp(con["beta_after_identity"] - 1.96*con["se_after_identity"])
    con["or_hi"] = np.exp(con["beta_after_identity"] + 1.96*con["se_after_identity"])
    fig, ax = standard_fig(style)
    y = np.arange(len(con))
    for i, r in enumerate(con.itertuples()):
        ax.errorbar(
            r.odds_ratio_plot, i,
            xerr=np.array([[r.odds_ratio_plot-r.or_lo],[r.or_hi-r.odds_ratio_plot]]),
            fmt="o", capsize=5, markersize=10,
            color=dcolors.get(r.definition, axis_col)
        )
    ax.axvline(1, linewidth=1.2, linestyle="--", color=grey_dark)
    ax.set_yticks(y, [x.replace("_"," ") for x in con["definition"]])
    apply_axes(style, ax, xlabel="Connection odds ratio per 1 SD complexity", ylabel="", dense=True)
    remove_minor_y(ax)
    atlas.save(
        "SYN10_structural_decoupling_adjusted", fig, con,
        tier="MAIN_CANDIDATE",
        claim="Intrinsic complexity does not simply mark local structural hubness; adjusted connection association is directionally inverse under both transfer definitions.",
        description="Identity/context-adjusted mouse connection odds ratios under primary and protocol-conservative complexity transfer.",
        caveat="An inverse association is evidence against a simple hubness rule; it does not imply that lower degree is universally optimal.",
        input_paths=[CA/"definition_robustness_models.csv"],
    )

    # -----------------------------------------------------------------
    # SYN11 — connection bootstrap
    # -----------------------------------------------------------------
    bc = boot[boot["endpoint"].eq("mouse_connection")].copy()
    fig, ax = standard_fig(style)
    ax.hist(bc["beta"], bins=18, alpha=.85, color=getattr(style, "COLOR_OPTIMIZED", low_col))
    med, qlo, qhi = percentile_ci(bc["beta"])
    ax.axvline(0, linewidth=1.2, color=grey_dark)
    ax.axvline(med, linewidth=2.0, color=axis_col)
    ax.axvspan(qlo, qhi, alpha=.10, color=grey)
    apply_axes(style, ax, xlabel="Adjusted connection effect across Stage5A mapper draws", ylabel="Bootstrap-mapper count")
    atlas.save(
        "SYN11_connection_mapper_uncertainty", fig, bc,
        tier="SI",
        claim="The inverse structural direction is stable to uncertainty in the transferred Stage5A mapper.",
        description="Distribution of adjusted mouse connection coefficients across 100 frozen-mapper bootstrap realizations.",
        input_paths=[CA/"mapper_bootstrap_key_effects.csv"],
    )

    # -----------------------------------------------------------------
    # SYN12 — connection repeated experiment holdout
    # -----------------------------------------------------------------
    hc = hold[hold["endpoint"].eq("connection")].copy()
    fig, ax = standard_fig(style)
    defs = [d for d in [PRIMARY, CONSERVATIVE] if d in hc["definition"].unique()]
    data = [hc.loc[hc["definition"].eq(d), "delta_after"].dropna().to_numpy() for d in defs]
    ax.boxplot(data, labels=[d.replace("_"," ") for d in defs], showfliers=False,
               widths=.55, medianprops={"linewidth":2})
    rng = np.random.default_rng(SEED)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals),i)+rng.normal(0,.035,len(vals)), vals, s=24, alpha=.35,
                   color=dcolors[defs[i-1]])
    ax.axhline(0, linewidth=1.2, color=grey_dark)
    apply_axes(style, ax, xlabel="", ylabel="Held-out log-loss improvement after identity", dense=True)
    atlas.save(
        "SYN12_connection_repeated_holdout", fig, hc,
        tier="SI",
        claim="Structural decoupling has a stable direction, but its out-of-experiment predictive increment is small.",
        description="Thirty repeated experiment-group holdouts for the incremental connection prediction after identity/context controls.",
        input_paths=[CA/"repeated_experiment_holdout.csv"],
    )

    # -----------------------------------------------------------------
    # SYN13 / 14 — identity absorption dumbbells
    # -----------------------------------------------------------------
    local = flat[flat["endpoint"].isin(LOCAL_PHENOTYPES)].copy()
    local["absorption_fraction"] = np.where(
        pd.to_numeric(local["delta_before"], errors="coerce") > 0,
        1 - local["delta_after"] / local["delta_before"],
        np.nan
    )

    for fig_id, definition, tier in [
        ("SYN13_identity_absorption_dumbbell_primary", PRIMARY, "MAIN_CANDIDATE"),
        ("SYN14_identity_absorption_dumbbell_conservative", CONSERVATIVE, "SI"),
    ]:
        g = local[local["definition"].eq(definition)].copy()
        g = g[pd.to_numeric(g["delta_before"], errors="coerce").notna()]
        g = g.sort_values("delta_before")
        fig, ax = standard_fig(style)
        y = np.arange(len(g))
        for yi, r in zip(y, g.itertuples()):
            ax.plot([r.delta_after, r.delta_before], [yi, yi], linewidth=1.5, color=grey_light)
        ax.scatter(g["delta_before"], y, s=70, label="Before identity",
                   color=bacolors["before"])
        ax.scatter(g["delta_after"], y, s=70, label="After identity",
                   color=bacolors["after"])
        ax.axvline(0, linewidth=1.0, color=grey_dark)
        ax.set_yticks(y, [PRETTY.get(e,e) for e in g["endpoint"]])
        apply_axes(style, ax, xlabel="Incremental in-sample R² from complexity", ylabel="", dense=True)
        remove_minor_y(ax)
        ax.legend(loc="lower right", fontsize=18)
        atlas.save(
            fig_id, fig, g,
            tier=tier,
            claim="Marginal complexity–local-synapse associations largely collapse after biological identity/context is modeled.",
            description=f"Before-versus-after identity/context complexity increments for local strength/STP phenotypes ({definition}).",
            caveat="Identity absorption is descriptive organization, not causal mediation.",
            input_paths=[CA/"definition_robustness_models.csv"],
        )

    # -----------------------------------------------------------------
    # SYN15 — identity absorption fraction across both definitions
    # -----------------------------------------------------------------
    af = local[
        pd.to_numeric(local["delta_before"], errors="coerce").ge(0.005) &
        pd.to_numeric(local["absorption_fraction"], errors="coerce").notna()
    ].copy()
    fig, ax = standard_fig(style)
    endpoints = [e for e in LOCAL_PHENOTYPES if e in af["endpoint"].unique()]
    ybase = np.arange(len(endpoints))
    offsets = {PRIMARY:-.14, CONSERVATIVE:.14}
    for definition in [PRIMARY, CONSERVATIVE]:
        g = af[af["definition"].eq(definition)].set_index("endpoint").reindex(endpoints)
        ax.scatter(g["absorption_fraction"], ybase+offsets[definition], s=70,
                   label=definition.replace("_"," "), color=dcolors[definition])
    ax.axvline(1, linestyle="--", linewidth=1.1, color=grey_dark)
    ax.set_yticks(ybase, [PRETTY.get(e,e) for e in endpoints])
    apply_axes(style, ax, xlabel="Descriptive identity-absorption fraction of marginal ΔR²", ylabel="", dense=True)
    remove_minor_y(ax)
    ax.legend(loc="lower right", fontsize=17)
    atlas.save(
        "SYN15_identity_absorption_fraction", fig, af,
        tier="SI",
        claim="Local complexity–synaptic associations are predominantly identity-structured under both transfer definitions.",
        description="Fraction of marginal complexity explanatory value absorbed after adding biological identity/context.",
        caveat="Values near 1 indicate descriptive attenuation, not mediation.",
        input_paths=[CA/"definition_robustness_models.csv"],
    )

    # -----------------------------------------------------------------
    # SYN16 — measured STP conditioned forest
    # -----------------------------------------------------------------
    stp = flat[flat["endpoint"].isin(STP_ONLY)].copy()
    stp["lo"] = stp["beta_after_identity"] - 1.96*stp["se_after_identity"]
    stp["hi"] = stp["beta_after_identity"] + 1.96*stp["se_after_identity"]
    endpoints = [e for e in STP_ONLY if e in stp["endpoint"].unique()]
    fig, ax = standard_fig(style)
    ybase = np.arange(len(endpoints))
    offsets = {PRIMARY:-.14, CONSERVATIVE:.14}
    for definition in [PRIMARY, CONSERVATIVE]:
        g = stp[stp["definition"].eq(definition)].set_index("endpoint").reindex(endpoints)
        ax.errorbar(
            g["beta_after_identity"], ybase+offsets[definition],
            xerr=1.96*g["se_after_identity"], fmt="o", capsize=4,
            markersize=7, label=definition.replace("_"," "),
            color=dcolors[definition]
        )
    ax.axvline(0, linewidth=1.1, color=grey_dark)
    ax.set_yticks(ybase, [PRETTY.get(e,e) for e in endpoints])
    apply_axes(style, ax, xlabel="Identity-conditioned complexity effect (95% CI)", ylabel="", dense=True)
    remove_minor_y(ax)
    ax.legend(loc="lower right", fontsize=17)
    atlas.save(
        "SYN16_measured_STP_conditioned_forest", fig, stp,
        tier="SI",
        claim="Measured short-term synaptic plasticity does not form a reproducible independent within-identity complexity gradient.",
        description="Identity/context-conditioned effects across measured STP endpoints under both transfer definitions.",
        input_paths=[CA/"definition_robustness_models.csv"],
    )

    # -----------------------------------------------------------------
    # SYN17 — standardized temporal gain across frequency
    # -----------------------------------------------------------------
    gn = flat[
        flat["endpoint"].isin(["G_norm_5hz","G_norm_10hz","G_norm_20hz","G_norm_50hz"])
    ].copy()
    gn["freq"] = gn["endpoint"].str.extract(r"(\d+)").astype(float)
    fig, ax = standard_fig(style)
    for definition in [PRIMARY, CONSERVATIVE]:
        g = gn[gn["definition"].eq(definition)].sort_values("freq")
        ax.errorbar(g["freq"], g["beta_after_identity"],
                    yerr=1.96*g["se_after_identity"], marker="o", capsize=4,
                    linewidth=1.8, markersize=8,
                    label=definition.replace("_"," "), color=dcolors[definition])
    ax.axhline(0, linewidth=1.1, color=grey_dark)
    ax.set_xscale("log")
    ax.set_xticks([5,10,20,50], ["5","10","20","50"])
    apply_axes(style, ax, xlabel="Standardized presynaptic train frequency (Hz)",
               ylabel="Conditional complexity effect on local temporal gain")
    ax.legend(loc="upper left", fontsize=18)
    atlas.save(
        "SYN17_local_temporal_gain_frequency_boundary", fig, gn,
        tier="MAIN_CANDIDATE",
        claim="Across 5–50 Hz, standardized single-synapse temporal gain does not provide an independent complexity–leverage bridge.",
        description="Identity/context-conditioned complexity effects on standardized normalized synaptic temporal gain across train frequencies.",
        caveat="This is a local synaptic phenotype; it must not be relabeled as whole-network dynamical leverage.",
        input_paths=[CA/"definition_robustness_models.csv"],
    )

    # -----------------------------------------------------------------
    # SYN18 — raw Gnorm50 binned
    # -----------------------------------------------------------------
    gnb_path = IS / "F07_Gnorm50_binned.csv"
    if gnb_path.exists():
        gnb = read_csv(gnb_path)
        fig, ax = standard_fig(style)
        ax.errorbar(gnb["x"], gnb["y"], yerr=gnb["se"], fmt="o-", capsize=4,
                    linewidth=1.8, markersize=8, color=getattr(style, "COLOR_SPECTRAL", high_col))
        apply_axes(style, ax, xlabel="Transferred intrinsic-complexity score",
                   ylabel="Standardized 50-Hz temporal gain")
        atlas.save(
            "SYN18_raw_Gnorm50_binned", fig, gnb,
            tier="SI",
            claim="The local 50-Hz temporal phenotype shows little monotonic dependence on transferred complexity.",
            description="Raw binned relationship between transferred complexity and standardized 50-Hz temporal gain.",
            caveat="Raw/descriptive; adjusted and held-out results are the inferential analyses.",
            input_paths=[gnb_path],
        )
    else:
        atlas.skip("SYN18_raw_Gnorm50_binned","SI","Raw local temporal relation.","Binned Gnorm50 relationship.","integrated source-data file missing")

    # -----------------------------------------------------------------
    # SYN19 — standardized train trajectories by C quartile
    # -----------------------------------------------------------------
    train_path = IS / "F08_standardized_trains_by_C_quartile.csv"
    if train_path.exists():
        tr = read_csv(train_path)
        fig, ax = standard_fig(style)
        quartiles = sorted(tr["quartile"].dropna().astype(str).unique())
        # rely on project default color cycle; no new palette definitions
        for qv in quartiles:
            g = tr[tr["quartile"].astype(str).eq(qv)].sort_values("spike")
            ax.plot(g["spike"], g["mean"], marker="o", linewidth=1.7, markersize=6, label=qv)
            if "se" in g:
                ax.fill_between(g["spike"], g["mean"]-g["se"], g["mean"]+g["se"], alpha=.12)
        apply_axes(style, ax, xlabel="Spike number in standardized 50-Hz train", ylabel="Normalized synaptic amplitude |Aₖ/A₁|")
        ax.legend(title="Complexity quartile", fontsize=17, title_fontsize=17)
        atlas.save(
            "SYN19_standardized_train_profiles", fig, tr,
            tier="SI",
            claim="Standardized local train dynamics do not separate into a simple monotonic complexity hierarchy.",
            description="Mean standardized 12-spike train response by transferred-complexity quartile.",
            caveat="Descriptive local-synapse phenotype.",
            input_paths=[train_path],
        )
    else:
        atlas.skip("SYN19_standardized_train_profiles","SI","Local train dynamics.","Standardized train profiles by complexity quartile.","integrated source-data file missing")

    # -----------------------------------------------------------------
    # SYN20 — fold-local continuous endpoint increments
    # -----------------------------------------------------------------
    cont = flat[flat["endpoint"].isin(["psp_amplitude","psc_amplitude","G_norm_50hz"])].copy()
    fig, ax = standard_fig(style)
    endpoints = ["psp_amplitude","psc_amplitude","G_norm_50hz"]
    x = np.arange(len(endpoints))
    offsets = {PRIMARY:-.12, CONSERVATIVE:.12}
    for definition in [PRIMARY, CONSERVATIVE]:
        g = cont[cont["definition"].eq(definition)].set_index("endpoint").reindex(endpoints)
        ax.scatter(x+offsets[definition], g["fold_local_increment_after"], s=90,
                   label=definition.replace("_"," "), color=dcolors[definition])
    ax.axhline(0, linewidth=1.1, color=grey_dark)
    ax.set_xticks(x, [PRETTY.get(e,e) for e in endpoints])
    apply_axes(style, ax, xlabel="", ylabel="Fold-local experiment-grouped ΔCVR² after identity", dense=True)
    ax.legend(loc="lower right", fontsize=17)
    atlas.save(
        "SYN20_fold_local_continuous_generalization", fig, cont,
        tier="MAIN_CANDIDATE",
        claim="The local PSP/PSC/temporal-gain complexity increments do not robustly generalize out of experiment after biological identity is controlled.",
        description="Fully fold-local experiment-grouped predictive increments for continuous local synaptic phenotypes.",
        input_paths=[CA/"definition_robustness_models.csv"],
    )

    # -----------------------------------------------------------------
    # SYN21 — repeated experiment holdout continuous
    # -----------------------------------------------------------------
    hcont = hold[hold["endpoint"].isin(["psp_amplitude","psc_amplitude","G_norm_50hz"])].copy()
    keys = [
        (d,e) for d in [PRIMARY,CONSERVATIVE]
        for e in ["psp_amplitude","psc_amplitude","G_norm_50hz"]
        if len(hcont[(hcont["definition"].eq(d)) & (hcont["endpoint"].eq(e))])
    ]
    fig, ax = standard_fig(style)
    data = [
        hcont[(hcont["definition"].eq(d)) & (hcont["endpoint"].eq(e))]["delta_after"].dropna().to_numpy()
        for d,e in keys
    ]
    labels = [f"{'P' if d==PRIMARY else 'C'} | {PRETTY.get(e,e)}" for d,e in keys]
    ax.boxplot(data, labels=labels, showfliers=False, widths=.55, medianprops={"linewidth":2})
    rng = np.random.default_rng(SEED+9)
    for i, ((d,e),vals) in enumerate(zip(keys,data), start=1):
        ax.scatter(np.full(len(vals),i)+rng.normal(0,.035,len(vals)), vals,
                   s=20, alpha=.28, color=dcolors[d])
    ax.axhline(0, linewidth=1.1, color=grey_dark)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    apply_axes(style, ax, xlabel="", ylabel="Repeated held-out ΔR² after identity", dense=True)
    atlas.save(
        "SYN21_repeated_experiment_holdout_continuous", fig, hcont,
        tier="SI",
        claim="The absence of an independent local dynamic bridge is not a single-split artifact.",
        description="Thirty repeated experiment-group holdouts for PSP, PSC, and 50-Hz temporal gain under both transfer definitions.",
        input_paths=[CA/"repeated_experiment_holdout.csv"],
    )

    # -----------------------------------------------------------------
    # SYN22 — within/between identity decomposition
    # -----------------------------------------------------------------
    wb_path = IS / "F10_within_between_decomposition.csv"
    if wb_path.exists():
        wb = read_csv(wb_path)
    else:
        rows = []
        for ident in ["pre_cell_class","pre_cre_type"]:
            rr = report.get("within_between",{}).get(ident,{})
            if rr.get("status") == "ok":
                for term in ["C_between_z","C_within_z"]:
                    z = rr.get(term,{})
                    rows.append({
                        "identity":ident, "term":term,
                        "beta":z.get("beta"), "se":z.get("se_cluster"), "p":z.get("p_cluster")
                    })
        wb = pd.DataFrame(rows)
    if len(wb):
        wb["label"] = wb.apply(
            lambda r: f"{str(r['identity']).replace('pre_','').replace('_',' ')} | "
                      f"{'between' if 'between' in str(r['term']) else 'within'}",
            axis=1
        )
        fig, ax = standard_fig(style)
        y = np.arange(len(wb))
        ax.errorbar(wb["beta"], y, xerr=1.96*wb["se"], fmt="o", capsize=4,
                    markersize=8, color=getattr(style, "COLOR_HIGH_DEGREE", high_col))
        ax.axvline(0, linewidth=1.1, color=grey_dark)
        ax.set_yticks(y, wb["label"])
        apply_axes(style, ax, xlabel="Effect on standardized 50-Hz temporal gain (95% CI)", ylabel="", dense=True)
        remove_minor_y(ax)
        atlas.save(
            "SYN22_within_between_identity_decomposition", fig, wb,
            tier="SI",
            claim="Neither within-identity nor between-identity decomposition reveals a robust hidden local temporal-gain bridge.",
            description="Within/between broad cell-class and Cre-type complexity decomposition for standardized 50-Hz temporal gain.",
            caveat="Between-identity estimates have few effective identity groups and remain descriptive.",
            input_paths=[wb_path if wb_path.exists() else IA/"stage5c_integrated_report.json"],
        )
    else:
        atlas.skip("SYN22_within_between_identity_decomposition","SI","Identity decomposition.","Within/between decomposition.","source unavailable")

    # -----------------------------------------------------------------
    # SYN23 — Gnorm mapper bootstrap
    # -----------------------------------------------------------------
    bg = boot[boot["endpoint"].eq("mouse_G_norm_50hz")].copy()
    fig, ax = standard_fig(style)
    ax.hist(bg["beta"], bins=18, alpha=.85, color=getattr(style, "COLOR_SPECTRAL", high_col))
    med, qlo, qhi = percentile_ci(bg["beta"])
    ax.axvline(0, linewidth=1.2, color=grey_dark)
    ax.axvline(med, linewidth=2.0, color=axis_col)
    ax.axvspan(qlo, qhi, alpha=.10, color=grey)
    apply_axes(style, ax, xlabel="50-Hz temporal-gain effect across Stage5A mapper draws", ylabel="Bootstrap-mapper count")
    atlas.save(
        "SYN23_Gnorm50_mapper_uncertainty", fig, bg,
        tier="SI",
        claim="The local 50-Hz temporal-gain null is stable to Stage5A transfer uncertainty.",
        description="Distribution of adjusted 50-Hz local temporal-gain coefficients across 100 mapper bootstrap realizations.",
        input_paths=[CA/"mapper_bootstrap_key_effects.csv"],
    )

    # -----------------------------------------------------------------
    # SYN24 — mouse vs human exploratory PSP
    # -----------------------------------------------------------------
    sp = species_psp.copy()
    sp["lo"] = sp["beta"] - 1.96*sp["se"]
    sp["hi"] = sp["beta"] + 1.96*sp["se"]
    fig, ax = standard_fig(style)
    y = np.arange(len(sp))
    ax.errorbar(sp["beta"], y, xerr=1.96*sp["se"], fmt="o", capsize=5,
                markersize=11, color=getattr(style, "COLOR_HIGH_DEGREE", high_col))
    ax.axvline(0, linewidth=1.1, color=grey_dark)
    ax.set_yticks(y, sp["species"].str.replace("_"," "))
    apply_axes(style, ax, xlabel="Conditional PSP effect per 1 SD transferred complexity", ylabel="", dense=True)
    remove_minor_y(ax)
    atlas.save(
        "SYN24_species_PSP_exploratory", fig, sp,
        tier="SI",
        claim="An exploratory human PSP association contrasts with the mouse null and motivates direct human calibration.",
        description="Mouse versus exploratory human conditional PSP effects.",
        caveat="Human result is exploratory because the complexity mapper was trained in mouse.",
        input_paths=[CA/"species_psp_comparison.csv"],
    )

    # -----------------------------------------------------------------
    # SYN25 — human PSP bootstrap
    # -----------------------------------------------------------------
    bh = boot[boot["endpoint"].eq("human_psp_amplitude")].copy()
    fig, ax = standard_fig(style)
    ax.hist(bh["beta"], bins=18, alpha=.85, color=getattr(style, "COLOR_HIGH_DEGREE", high_col))
    med, qlo, qhi = percentile_ci(bh["beta"])
    ax.axvline(0, linewidth=1.2, color=grey_dark)
    ax.axvline(med, linewidth=2.0, color=axis_col)
    ax.axvspan(qlo, qhi, alpha=.10, color=grey)
    apply_axes(style, ax, xlabel="Exploratory human PSP effect across Stage5A mapper draws", ylabel="Bootstrap-mapper count")
    atlas.save(
        "SYN25_human_PSP_mapper_uncertainty", fig, bh,
        tier="SI",
        claim="The exploratory human PSP direction is stable across mapper bootstrap realizations.",
        description="Distribution of exploratory human PSP coefficients across Stage5A mapper bootstrap draws.",
        caveat="Not a primary cross-species conclusion; no human-trained intrinsic-complexity mapper is available.",
        input_paths=[CA/"mapper_bootstrap_key_effects.csv"],
    )

    # -----------------------------------------------------------------
    # SYN26 — primary vs conservative endpoint-effect agreement
    # -----------------------------------------------------------------
    continuous = flat[flat["effect_scale"].eq("standardized_beta")].copy()
    p = continuous[continuous["definition"].eq(PRIMARY)][["endpoint","beta_after_identity"]].rename(
        columns={"beta_after_identity":"primary_beta"}
    )
    c = continuous[continuous["definition"].eq(CONSERVATIVE)][["endpoint","beta_after_identity"]].rename(
        columns={"beta_after_identity":"conservative_beta"}
    )
    agree = p.merge(c, on="endpoint", how="inner")
    fig, ax = standard_fig(style)
    ax.scatter(agree["primary_beta"], agree["conservative_beta"], s=60, alpha=.65,
               color=getattr(style, "COLOR_OPTIMIZED", low_col))
    lo = float(np.nanmin([agree["primary_beta"].min(), agree["conservative_beta"].min()]))
    hi = float(np.nanmax([agree["primary_beta"].max(), agree["conservative_beta"].max()]))
    pad = max(0.02, (hi-lo)*.08)
    ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], linestyle="--", linewidth=1.1, color=grey_dark)
    ax.axhline(0, linewidth=.9, color=grey_light)
    ax.axvline(0, linewidth=.9, color=grey_light)
    apply_axes(style, ax, xlabel="Primary transfer: conditioned effect",
               ylabel="Protocol-conservative transfer: conditioned effect")
    atlas.save(
        "SYN26_transfer_definition_effect_agreement", fig, agree,
        tier="DIAGNOSTIC",
        claim="Continuous local-phenotype conclusions are not driven by a single transferred-complexity definition.",
        description="Endpoint-level agreement of identity-conditioned coefficients under primary and protocol-conservative transfer.",
        caveat="Diagnostic robustness plot; heterogeneous endpoints should not be interpreted as one biological effect family.",
        input_paths=[CA/"definition_robustness_models.csv"],
    )

    # -----------------------------------------------------------------
    # SYN27 — evidence hierarchy / narrative closure
    # -----------------------------------------------------------------
    # Data-backed summary: no invented score. Each row encodes the predeclared
    # role and observed robustness checks.
    evidence_rows = []
    for ep, role in [
        ("connection","Structural prominence"),
        ("psp_amplitude","Local synaptic strength"),
        ("psc_amplitude","Local synaptic strength"),
        ("stp_induction_50hz","Measured STP"),
        ("variability_resting_state","Measured synaptic variability"),
        ("G_norm_50hz","Standardized local temporal gain"),
    ]:
        for definition in [PRIMARY, CONSERVATIVE]:
            r = flat[(flat["endpoint"].eq(ep)) & (flat["definition"].eq(definition))]
            if r.empty:
                continue
            rr = r.iloc[0]
            evidence_rows.append({
                "endpoint":ep,
                "role":role,
                "definition":definition,
                "beta_after_identity":rr["beta_after_identity"],
                "p_after_identity":rr["p_after_identity"],
                "heldout_increment_after":rr["fold_local_increment_after"],
                "same_sign_as_primary":np.nan,
            })
    ev = pd.DataFrame(evidence_rows)
    fig, ax = standard_fig(style)
    # visualizes only beta with CI from flat; this is a compact closure forest
    ev2 = flat[
        flat["endpoint"].isin([
            "connection","psp_amplitude","psc_amplitude",
            "stp_induction_50hz","variability_resting_state","G_norm_50hz"
        ])
    ].copy()
    endpoints = list(dict.fromkeys(ev2["endpoint"].tolist()))
    ybase = np.arange(len(endpoints))
    offsets = {PRIMARY:-.14, CONSERVATIVE:.14}
    for definition in [PRIMARY,CONSERVATIVE]:
        g = ev2[ev2["definition"].eq(definition)].set_index("endpoint").reindex(endpoints)
        # connection uses log-odds and is therefore intentionally marked separately by shape
        ax.errorbar(
            g["beta_after_identity"], ybase+offsets[definition],
            xerr=1.96*g["se_after_identity"], fmt="o", capsize=4,
            markersize=7, label=definition.replace("_"," "),
            color=dcolors[definition]
        )
    ax.axvline(0, linewidth=1.1, color=grey_dark)
    ax.set_yticks(ybase, [PRETTY.get(e,e) for e in endpoints])
    apply_axes(style, ax, xlabel="Conditional coefficient (directional summary only)", ylabel="", dense=True)
    remove_minor_y(ax)
    ax.legend(loc="lower right", fontsize=17)
    atlas.save(
        "SYN27_narrative_closure_forest", fig, ev2,
        tier="DIAGNOSTIC",
        claim="SynPhys rules out several simple local/static reductions of the complexity-value principle.",
        description="Compact directional summary across structural, local-strength, STP, and local-temporal endpoints.",
        caveat="Connection is on a log-odds scale whereas continuous outcomes are standardized beta; magnitudes must not be compared across these scales.",
        input_paths=[CA/"definition_robustness_models.csv"],
    )

    # -----------------------------------------------------------------
    # Final catalog + atlas report
    # -----------------------------------------------------------------
    catalog = atlas.finalize()

    main_ids = catalog.loc[catalog["candidate_tier"].eq("MAIN_CANDIDATE"), "figure_id"].tolist()
    si_ids = catalog.loc[catalog["candidate_tier"].eq("SI"), "figure_id"].tolist()
    diag_ids = catalog.loc[catalog["candidate_tier"].eq("DIAGNOSTIC"), "figure_id"].tolist()
    skipped = catalog[catalog["caveat"].astype(str).str.startswith("SKIPPED")]

    report_lines = [
        "# Stage5C SynPhys Publication Figure Atlas v1",
        "",
        "## Frozen scientific role",
        "",
        "SynPhys is used as a local-circuit boundary test. It is not the positive network-level leverage validation.",
        "",
        "The target narrative is:",
        "",
        "> intrinsic cellular complexity is not reducible to simple structural hubness, local synaptic strength/STP, or standardized single-synapse temporal gain; the positive claim that functional value emerges in a collective state-dependent dynamical context is established by the model/network-state and population-state analyses elsewhere in the paper.",
        "",
        "## MAIN candidates",
        "",
    ]
    report_lines += [f"- `{x}`" for x in main_ids]
    report_lines += [
        "",
        "## SI candidates",
        "",
    ]
    report_lines += [f"- `{x}`" for x in si_ids]
    report_lines += [
        "",
        "## Diagnostic / audit candidates",
        "",
    ]
    report_lines += [f"- `{x}`" for x in diag_ids]
    if len(skipped):
        report_lines += ["", "## Skipped panels", ""]
        report_lines += [f"- `{r.figure_id}` — {r.caveat}" for r in skipped.itertuples()]
    report_lines += [
        "",
        "## Recommended manuscript use",
        "",
        "A single SynPhys main-text composite can later be assembled from approximately 6–8 panels, with the remaining plots placed in Extended Data / SI.",
        "",
        "Current strongest MAIN candidates:",
        "",
        "1. `SYN10_structural_decoupling_adjusted` — complexity is not simple hubness.",
        "2. `SYN13_identity_absorption_dumbbell_primary` — marginal local-synaptic associations collapse after identity/context control.",
        "3. `SYN17_local_temporal_gain_frequency_boundary` — local temporal gain is not the network leverage quantity.",
        "4. `SYN20_fold_local_continuous_generalization` — independent local bridge fails out of experiment.",
        "5. `SYN00_conceptual_boundary_map` — optional conceptual entry panel.",
        "",
        "The exploratory human PSP panels should remain SI/Extended Data unless direct human calibration is later added.",
        "",
        "## Output contract",
        "",
        "- every rendered panel: 600-dpi PNG (or quick-test DPI) + vector PDF",
        "- every rendered panel: exact source CSV",
        "- every rendered panel: metadata JSON",
        "- `analysis/figure_catalog.csv`: MAIN/SI/DIAGNOSTIC candidate triage",
        "",
        f"Generated figures: **{int((catalog['png']!='').sum())}**",
        f"Skipped figures: **{len(skipped)}**",
    ]
    (atlas.analysis_dir / "00_SYNPHYS_FIGURE_ATLAS_SUMMARY.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    manifest = {
        "status":"COMPLETE",
        "created":time.strftime("%Y-%m-%d %H:%M:%S"),
        "root":str(root),
        "integrated":str(integrated),
        "closure":str(closure),
        "output":str(out),
        "dpi":dpi,
        "style_file":str(style_path) if style_path else None,
        "n_catalog_rows":int(len(catalog)),
        "n_rendered":int((catalog["png"]!="").sum()),
        "n_skipped":int(len(skipped)),
        "main_candidates":main_ids,
        "si_candidates":si_ids,
        "diagnostic_candidates":diag_ids,
        "guardrails":[
            "No new hypothesis testing is performed.",
            "No SQLite access is required.",
            "Identity attenuation is not causal mediation.",
            "Inverse connection association is not a claim that lower degree is universally optimal.",
            "Local temporal gain is not whole-network dynamical leverage.",
            "Human PSP is exploratory because the transferred complexity mapper is mouse-trained.",
        ],
    }
    (atlas.analysis_dir / "atlas_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 96)
    print("SYNPHYS FIGURE ATLAS COMPLETE")
    print("Summary :", atlas.analysis_dir / "00_SYNPHYS_FIGURE_ATLAS_SUMMARY.md")
    print("Catalog :", atlas.analysis_dir / "figure_catalog.csv")
    print("Figures :", atlas.fig_dir)
    print("Source  :", atlas.src_dir)
    print("Metadata:", atlas.meta_dir)
    print("=" * 96)


if __name__ == "__main__":
    main()
