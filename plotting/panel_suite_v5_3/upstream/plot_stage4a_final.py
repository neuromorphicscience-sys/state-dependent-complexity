#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 4A publication plotting script FINAL (fixed-canvas)
=========================================

Purpose
-------
Create manuscript-ready Stage 4A figures from the analysis tables produced by
`analyze_stage4a_manuscript_v1_1.py`.

Scientific role of Stage 4A
---------------------------
Stage 4A is treated as the discovery bridge from static structural allocation
to dynamics-aware allocation / relative dynamical leverage.  This plotting
script does NOT rerun simulation or statistical analysis.  It only visualizes
already-computed Stage 4A analysis tables.

Default project layout
----------------------
D:\\Research\\Neural Science\\
    plot\\
        plot_style_v4.py
        plot_stage4a_final.py          <- this script
        Stage4A\\
            analysis\\              <- input CSVs
            figure\\                <- outputs from this script
                panels\\
                data\\
                Stage4A_main.*

Outputs
-------
Panel a  Overall validated score by allocation method
Panel b  Paired gain of dynamics-aware allocation vs matched baselines
Panel c  State x complexity-budget discovery landscape vs spectral baseline
Panel d  Structural phenotype shift vs spectral baseline
Panel e  Descriptor-shift / performance-gain association
Panel f  Search-stage score vs fresh validation score

Each panel is exported independently (PNG + PDF + tidy CSV) and a compact
2 x 3 composite figure is also exported.

Usage
-----
python plot\\plot_stage4a_final.py

Optional:
python plot\\plot_stage4a_final.py --root "D:\\Research\\Neural Science"
python plot\\plot_stage4a_final.py --dpi 600
python plot\\plot_stage4a_final.py --show

Notes
-----
- Requires `plot_style_v4.py` in ROOT/plot or next to this script.
- Requires the Stage4A manuscript analysis to have been run first.
- The script intentionally avoids panel titles; panel labels and axis labels
  carry the visual hierarchy, consistent with the project plotting standard.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path.cwd()


# -----------------------------------------------------------------------------
# Style loading
# -----------------------------------------------------------------------------

def load_project_style(root: Path):
    """Load plot_style_v4.py without requiring ROOT/plot on PYTHONPATH."""
    candidates = [
        root / "plot" / "plot_style_v4.py",
        Path(__file__).resolve().parent / "plot_style_v4.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("plot_style_v4", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    tried = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find plot_style_v4.py. Tried:\n  - " + tried
    )


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Stage4A analysis table not found:\n  {path}\n"
            "Run analyze_stage4a_manuscript_v1_1.py first."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Required table is empty: {path}")
    return df


def optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_columns(df: pd.DataFrame, cols: Iterable[str], source_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def clean_label(x: object) -> str:
    s = str(x)
    replacements = {
        "dynamics_aware": "Dynamics-aware",
        "spectral": "Spectral",
        "feedback_hub": "Feedback hub",
        "high_degree": "High degree",
        "module_bridge": "Module bridge",
        "cycle_proxy": "Cycle proxy",
        "random": "Random",
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
    return replacements.get(s, s.replace("_", " ").strip().title())


def regime_label(x: object) -> str:
    s = str(x).strip()
    return s.replace("_", " ").replace("-", " ").title()


def stable_method_order(df: pd.DataFrame) -> list[str]:
    preferred = [
        "dynamics_aware",
        "spectral",
        "feedback_hub",
        "high_degree",
        "module_bridge",
        "cycle_proxy",
        "random",
    ]
    methods = df["method"].dropna().astype(str).tolist() if "method" in df.columns else []
    seen = set(methods)
    return [m for m in preferred if m in seen] + sorted(seen - set(preferred))


def method_color(style, method: str) -> str:
    # Preserve project-level semantic meaning first.
    if method == "dynamics_aware":
        return style.COLORS["leverage"]
    if method == "spectral":
        return style.COLORS["complexity"]
    if method == "random":
        return style.COLORS["random"]
    if method == "high_degree":
        return style.COLORS["support"]
    if method == "feedback_hub":
        return style.COLORS["secondary"]
    if method == "module_bridge":
        return style.COLORS["accent"]
    if method == "cycle_proxy":
        return style.COLORS["rose"]
    return style.COLORS["dark_gray"]


def save_panel(style, fig, outstem: Path, data: pd.DataFrame, dpi: int) -> None:
    style.finalize_figure(fig, strip_titles=True)
    style.export_panel_bundle(
        fig,
        outstem,
        data=data,
        dpi=dpi,
        save_pdf=True,
        close=True,
    )


# -----------------------------------------------------------------------------
# Panel drawing functions
# Each function can draw either to a provided axis (composite) or a fresh fig.
# -----------------------------------------------------------------------------

def draw_panel_a(ax, df: pd.DataFrame, style, add_label: bool = True) -> pd.DataFrame:
    """Overall validated score by allocation method."""
    ensure_columns(
        df,
        ["method", "mean_validated_score", "ci95_low", "ci95_high"],
        "overall_validated_method_summary.csv",
    )
    d = numeric(df, ["mean_validated_score", "ci95_low", "ci95_high"])
    order = stable_method_order(d)
    d = d.set_index("method").reindex(order).reset_index()

    # Plot top-to-bottom with dynamics-aware first.
    y = np.arange(len(d))
    for yi, row in zip(y, d.itertuples(index=False)):
        method = str(row.method)
        mean = float(row.mean_validated_score)
        lo = float(row.ci95_low)
        hi = float(row.ci95_high)
        color = method_color(style, method)
        ax.hlines(yi, lo, hi, color=color, lw=1.35, zorder=2)
        ax.scatter(mean, yi, s=style.MARKER_SIZE, color=color, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(m) for m in d["method"]])
    ax.invert_yaxis()
    style.style_axes(
        ax,
        frame="open",
        xlabel="Validated score",
        ylabel=None,
        xgrid=True,
    )
    if add_label:
        style.add_panel_label(ax, "a")
    return d


def draw_panel_b(ax, df: pd.DataFrame, style, add_label: bool = True) -> pd.DataFrame:
    """Paired dynamics-aware gain over matched baselines."""
    ensure_columns(
        df,
        ["baseline", "mean_gain", "ci95_low", "ci95_high", "win_fraction"],
        "paired_dynamics_aware_vs_baselines.csv",
    )
    d = numeric(df, ["mean_gain", "ci95_low", "ci95_high", "win_fraction"])
    preferred = [
        "spectral",
        "feedback_hub",
        "high_degree",
        "module_bridge",
        "cycle_proxy",
        "random",
    ]
    present = set(d["baseline"].astype(str))
    order = [x for x in preferred if x in present] + sorted(present - set(preferred))
    d = d.set_index("baseline").reindex(order).reset_index()

    y = np.arange(len(d))
    style.add_zero_line(ax, orientation="v")
    for yi, row in zip(y, d.itertuples(index=False)):
        color = (
            style.COLORS["complexity"]
            if str(row.baseline) == "spectral"
            else style.COLORS["leverage"]
        )
        ax.hlines(yi, row.ci95_low, row.ci95_high, color=color, lw=1.35, zorder=2)
        ax.scatter(row.mean_gain, yi, s=style.MARKER_SIZE, color=color, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(x) for x in d["baseline"]])
    ax.invert_yaxis()
    style.style_axes(
        ax,
        frame="open",
        xlabel="Paired validated-score gain",
        ylabel=None,
        xgrid=True,
    )
    if add_label:
        style.add_panel_label(ax, "b")
    return d


def draw_panel_c(ax, df: pd.DataFrame, style, add_label: bool = True) -> pd.DataFrame:
    """State x budget discovery landscape relative to spectral baseline."""
    ensure_columns(
        df,
        ["regime", "k", "mean_gain", "ci95_low", "ci95_high"],
        "gain_by_regime_and_budget_vs_spectral.csv",
    )
    d = numeric(df, ["k", "mean_gain", "ci95_low", "ci95_high"])
    d = d.dropna(subset=["k", "mean_gain"]).copy()

    regimes = sorted(d["regime"].dropna().astype(str).unique().tolist())
    palette = [
        style.COLORS["leverage"],
        style.COLORS["complexity"],
        style.COLORS["support"],
        style.COLORS["biology"],
        style.COLORS["secondary"],
        style.COLORS["accent"],
        style.COLORS["rose"],
        style.COLORS["mismatch"],
    ]

    style.add_zero_line(ax, orientation="h")
    for i, regime in enumerate(regimes):
        g = d[d["regime"].astype(str).eq(regime)].sort_values("k")
        color = palette[i % len(palette)]
        x = g["k"].to_numpy(float)
        mean = g["mean_gain"].to_numpy(float)
        lo = g["ci95_low"].to_numpy(float)
        hi = g["ci95_high"].to_numpy(float)
        ax.plot(
            x,
            mean,
            color=color,
            lw=style.LINEWIDTH_MAIN,
            marker="o",
            ms=3.2,
            label=regime_label(regime),
            zorder=3,
        )
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(
                x,
                lo,
                hi,
                color=color,
                alpha=style.ALPHA_BAND,
                linewidth=0,
                zorder=2,
            )

    style.style_axes(
        ax,
        frame="open",
        xlabel="Complexity budget, k",
        ylabel="Gain vs spectral",
        ygrid=True,
    )
    if regimes:
        ax.legend(
            loc="best",
            frameon=False,
            fontsize=style.FONT_SIZE_LEGEND,
            handlelength=1.8,
            handletextpad=0.5,
        )
    if add_label:
        style.add_panel_label(ax, "c")
    return d


def draw_panel_d(ax, df: pd.DataFrame, style, add_label: bool = True) -> pd.DataFrame:
    """Overall structural phenotype shifts of dynamics-aware vs spectral."""
    ensure_columns(
        df,
        ["descriptor", "grouping", "mean_shift", "ci95_low", "ci95_high"],
        "structural_phenotype_shifts_vs_spectral.csv",
    )
    d = numeric(df, ["mean_shift", "ci95_low", "ci95_high"])
    d = d[d["grouping"].astype(str).eq("overall")].copy()
    if d.empty:
        raise ValueError(
            "No grouping='overall' rows found in structural_phenotype_shifts_vs_spectral.csv"
        )

    # Keep scientifically stable descriptor order.
    preferred = [
        "sel_total_degree",
        "coverage1",
        "coverage2",
        "redundancy",
        "dispersion",
        "sel_spectral",
        "sel_feedback",
        "sel_cycle3",
        "sel_bridge",
    ]
    present = set(d["descriptor"].astype(str))
    order = [x for x in preferred if x in present] + sorted(present - set(preferred))
    d = d.set_index("descriptor").reindex(order).reset_index()

    y = np.arange(len(d))
    style.add_zero_line(ax, orientation="v")
    ax.hlines(
        y,
        d["ci95_low"],
        d["ci95_high"],
        color=style.COLORS["complexity"],
        lw=1.35,
        zorder=2,
    )
    ax.scatter(
        d["mean_shift"],
        y,
        s=style.MARKER_SIZE,
        color=style.COLORS["leverage"],
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(x) for x in d["descriptor"]])
    ax.invert_yaxis()
    style.style_axes(
        ax,
        frame="open",
        xlabel="Descriptor shift vs spectral",
        ylabel=None,
        xgrid=True,
    )
    if add_label:
        style.add_panel_label(ax, "d")
    return d


def draw_panel_e(ax, df: pd.DataFrame, style, add_label: bool = True) -> pd.DataFrame:
    """Descriptor-shift / performance-gain Spearman associations."""
    ensure_columns(
        df,
        ["descriptor", "spearman_rho_descriptor_shift_vs_score_gain"],
        "descriptor_shift_vs_gain_correlations_spectral.csv",
    )
    rho_col = "spearman_rho_descriptor_shift_vs_score_gain"
    d = numeric(df, [rho_col, "spearman_p", "spearman_fdr_bh"])
    d = d.dropna(subset=[rho_col]).copy()
    d = d.sort_values(rho_col, ascending=True).reset_index(drop=True)

    y = np.arange(len(d))
    style.add_zero_line(ax, orientation="v")
    ax.hlines(
        y,
        0,
        d[rho_col],
        color=style.COLORS["light_gray"],
        lw=1.1,
        zorder=1,
    )
    ax.scatter(
        d[rho_col],
        y,
        s=style.MARKER_SIZE,
        color=style.COLORS["support"],
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([clean_label(x) for x in d["descriptor"]])
    style.style_axes(
        ax,
        frame="open",
        xlabel="Spearman ρ",
        ylabel=None,
        xgrid=True,
    )
    # Correlation range is intrinsically bounded.
    ax.set_xlim(-1.0, 1.0)
    if add_label:
        style.add_panel_label(ax, "e")
    return d


def draw_panel_f(ax, df: pd.DataFrame, style, add_label: bool = True) -> pd.DataFrame:
    """Search-stage score vs fresh validation score for dynamics-aware tasks."""
    ensure_columns(
        df,
        ["optimizer_search_best", "final_score_mean"],
        "search_vs_validation_task_audit.csv",
    )
    d = numeric(df, ["optimizer_search_best", "final_score_mean"])
    d = d.dropna(subset=["optimizer_search_best", "final_score_mean"]).copy()
    if d.empty:
        raise ValueError("No finite search/validation pairs found.")

    x = d["optimizer_search_best"].to_numpy(float)
    y = d["final_score_mean"].to_numpy(float)
    lo = float(np.nanmin(np.r_[x, y]))
    hi = float(np.nanmax(np.r_[x, y]))
    pad = max((hi - lo) * 0.05, 1e-6)

    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        ls="--",
        lw=style.LINEWIDTH_REFERENCE,
        color=style.COLORS["mid_gray"],
        zorder=1,
    )
    ax.scatter(
        x,
        y,
        s=style.MARKER_SIZE_SMALL,
        color=style.COLORS["leverage"],
        alpha=style.ALPHA_POINTS,
        edgecolors="none",
        zorder=3,
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    style.style_axes(
        ax,
        frame="open",
        xlabel="Optimizer search best",
        ylabel="Fresh validated score",
        xgrid=False,
        ygrid=False,
    )
    if add_label:
        style.add_panel_label(ax, "f")
    return d


# -----------------------------------------------------------------------------
# Standalone panels
# -----------------------------------------------------------------------------

def make_standalone_panels(tables: dict[str, pd.DataFrame], outdir: Path, style, dpi: int):
    panels_dir = outdir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    # a
    fig, ax = style.new_figure("s4a_method")
    da = draw_panel_a(ax, tables["overall"], style)
    fig.subplots_adjust(left=0.36, right=0.98, bottom=0.20, top=0.96)
    save_panel(style, fig, panels_dir / "Stage4A_a_method_performance", da, dpi)

    # b
    fig, ax = style.new_figure("s4a_gain")
    db = draw_panel_b(ax, tables["contrasts"], style)
    fig.subplots_adjust(left=0.35, right=0.98, bottom=0.20, top=0.96)
    save_panel(style, fig, panels_dir / "Stage4A_b_paired_gain", db, dpi)

    # c
    fig, ax = style.new_figure("s4a_budget")
    dc = draw_panel_c(ax, tables["state_budget"], style)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.96)
    save_panel(style, fig, panels_dir / "Stage4A_c_state_budget", dc, dpi)

    # d
    fig, ax = style.new_figure("s4a_descriptor")
    dd = draw_panel_d(ax, tables["phenotype"], style)
    fig.subplots_adjust(left=0.39, right=0.98, bottom=0.18, top=0.96)
    save_panel(style, fig, panels_dir / "Stage4A_d_structural_phenotype", dd, dpi)

    # e
    fig, ax = style.new_figure("s4a_correlation")
    de = draw_panel_e(ax, tables["correlations"], style)
    fig.subplots_adjust(left=0.39, right=0.98, bottom=0.18, top=0.96)
    save_panel(style, fig, panels_dir / "Stage4A_e_descriptor_gain", de, dpi)

    # f
    fig, ax = style.new_figure("s4a_validation")
    df = draw_panel_f(ax, tables["search_validation"], style)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.20, top=0.96)
    save_panel(style, fig, panels_dir / "Stage4A_f_search_validation", df, dpi)


# -----------------------------------------------------------------------------
# Composite manuscript figure
# -----------------------------------------------------------------------------

def make_composite(tables: dict[str, pd.DataFrame], outdir: Path, style, dpi: int):
    """Create a compact double-column 2 x 3 Stage4A figure."""
    # 180 mm x 126 mm works well as a dense six-panel manuscript figure.
    fig, axs = style.subplots(
        kind="s4a_composite_2x3",
        nrows=2,
        ncols=3,
        constrained_layout=False,
    )

    draw_panel_a(axs[0, 0], tables["overall"], style, add_label=True)
    draw_panel_b(axs[0, 1], tables["contrasts"], style, add_label=True)
    draw_panel_c(axs[0, 2], tables["state_budget"], style, add_label=True)
    draw_panel_d(axs[1, 0], tables["phenotype"], style, add_label=True)
    draw_panel_e(axs[1, 1], tables["correlations"], style, add_label=True)
    draw_panel_f(axs[1, 2], tables["search_validation"], style, add_label=True)

    # Small-panel tick labels should stay readable after final-size reduction.
    for ax in axs.flat:
        ax.tick_params(labelsize=6.7)
        ax.xaxis.label.set_size(7.4)
        ax.yaxis.label.set_size(7.4)

    # Left panels need room for descriptor/method labels; use asymmetric spacing.
    fig.subplots_adjust(
        left=0.125,
        right=0.985,
        bottom=0.10,
        top=0.975,
        wspace=0.58,
        hspace=0.48,
    )

    style.finalize_figure(fig, strip_titles=True)
    style.save_figure(
        fig,
        outdir / "Stage4A_main",
        dpi=dpi,
        save_png=True,
        save_pdf=True,
        close=True,
    )


# -----------------------------------------------------------------------------
# Optional compact figure variant: discovery message only (a-d)
# -----------------------------------------------------------------------------

def make_core_composite(tables: dict[str, pd.DataFrame], outdir: Path, style, dpi: int):
    """Four-panel compact figure focused only on the Stage4A discovery claim."""
    fig, axs = style.subplots(
        kind="s4a_composite_2x2",
        nrows=2,
        ncols=2,
        constrained_layout=False,
    )
    draw_panel_a(axs[0, 0], tables["overall"], style, add_label=True)
    draw_panel_b(axs[0, 1], tables["contrasts"], style, add_label=True)
    draw_panel_c(axs[1, 0], tables["state_budget"], style, add_label=True)
    draw_panel_d(axs[1, 1], tables["phenotype"], style, add_label=True)

    for ax in axs.flat:
        ax.tick_params(labelsize=6.8)
        ax.xaxis.label.set_size(7.5)
        ax.yaxis.label.set_size(7.5)

    fig.subplots_adjust(
        left=0.16,
        right=0.985,
        bottom=0.11,
        top=0.975,
        wspace=0.60,
        hspace=0.48,
    )
    style.finalize_figure(fig, strip_titles=True)
    style.save_figure(
        fig,
        outdir / "Stage4A_core",
        dpi=dpi,
        save_png=True,
        save_pdf=True,
        close=True,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot manuscript-ready Stage4A figures using immutable plot_style_v4 canvases"
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Project root")
    ap.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Stage4A analysis directory; default ROOT/plot/Stage4A/analysis",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output directory; default ROOT/plot/Stage4A/figure",
    )
    ap.add_argument("--dpi", type=int, default=600, help="Raster export DPI")
    ap.add_argument("--show", action="store_true", help="Show figures after generation")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Project root not found: {root}")

    style = load_project_style(root)
    style.set_paper_style()

    analysis_dir = (
        args.analysis_dir.resolve()
        if args.analysis_dir is not None
        else root / "plot" / "Stage4A" / "analysis"
    )
    outdir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "plot" / "Stage4A" / "figure"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    print("[Stage4A plot] Reading manuscript-analysis tables ...")
    tables = {
        "overall": require_csv(analysis_dir / "overall_validated_method_summary.csv"),
        "contrasts": require_csv(analysis_dir / "paired_dynamics_aware_vs_baselines.csv"),
        "state_budget": require_csv(analysis_dir / "gain_by_regime_and_budget_vs_spectral.csv"),
        "phenotype": require_csv(analysis_dir / "structural_phenotype_shifts_vs_spectral.csv"),
        "correlations": require_csv(analysis_dir / "descriptor_shift_vs_gain_correlations_spectral.csv"),
        "search_validation": require_csv(analysis_dir / "search_vs_validation_task_audit.csv"),
    }

    print("[Stage4A plot] Creating standalone panels ...")
    make_standalone_panels(tables, outdir, style, args.dpi)

    # Nature redraw final: leaf panels only. Final manuscript composition is manual.
    print("[Stage4A plot] Leaf-panel-only mode: skipping legacy composite figures.")

    # Write a minimal manifest for traceability.
    manifest = pd.DataFrame(
        [
            {"panel": "a", "figure_kind": "s4a_method", "input": "overall_validated_method_summary.csv", "message": "validated method performance"},
            {"panel": "b", "figure_kind": "s4a_gain", "input": "paired_dynamics_aware_vs_baselines.csv", "message": "matched gain over static baselines"},
            {"panel": "c", "figure_kind": "s4a_budget", "input": "gain_by_regime_and_budget_vs_spectral.csv", "message": "state- and budget-conditioned discovery landscape"},
            {"panel": "d", "figure_kind": "s4a_descriptor", "input": "structural_phenotype_shifts_vs_spectral.csv", "message": "structural departure from spectral baseline"},
            {"panel": "e", "figure_kind": "s4a_correlation", "input": "descriptor_shift_vs_gain_correlations_spectral.csv", "message": "descriptor departure associated with score gain"},
            {"panel": "f", "figure_kind": "s4a_validation", "input": "search_vs_validation_task_audit.csv", "message": "search/validation separation"},
        ]
    )
    manifest.to_csv(outdir / "Stage4A_figure_manifest.csv", index=False, encoding="utf-8-sig")
    style.figure_size_table().to_csv(outdir / "fixed_figure_sizes.csv", index=False, encoding="utf-8-sig")

    print()
    print("[Stage4A plot] DONE")
    print(f"  analysis : {analysis_dir}")
    print(f"  output   : {outdir}")
    print("  panels   : figure/panels/Stage4A_[a-f]_*.png/pdf/csv")
    print("  composite: intentionally not generated; assemble final Fig.3 manually")

    if args.show:
        # Files are already saved/closed; this flag is retained intentionally as
        # a CLI compatibility switch.  Re-run without automatic closing only if
        # interactive inspection becomes necessary.
        print("--show requested: figures were exported successfully. Open the PNG/PDF outputs for inspection.")


if __name__ == "__main__":
    main()
