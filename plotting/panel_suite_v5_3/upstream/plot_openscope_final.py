#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
OpenScope manuscript figure atlas FINAL (fixed-canvas)
====================================

Purpose
-------
Generate a dense, publication-oriented figure atlas ONLY for the frozen
OpenScope Illusion 000248 analysis.

This script:
- does NOT re-read NWB;
- does NOT refit decoders;
- does NOT redefine state / response windows / C / top-unit budget;
- does NOT add new biological endpoints;
- only visualizes already-frozen v2.3 + FinalClosure outputs;
- separates MAIN candidates from SI/supporting figures;
- requires the package-local plot_style_v4 fixed-canvas standard;
- creates no axes titles / suptitles;
- exports 600-dpi PNG + vector PDF;
- saves the exact plotted table for every generated panel;
- logs skipped optional figures rather than failing the whole atlas.

Recommended location
--------------------
Place this file directly under:
    D:\Research\Neural Science\plot

Default inputs
--------------
    D:\Research\Neural Science\plot\OpenScope_Illusion_Analysis_v2_3
    D:\Research\Neural Science\plot\OpenScope_FinalClosure_v1

Default output
--------------
    D:\Research\Neural Science\plot\OpenScope_Manuscript_FigureAtlas_Final
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path.cwd()

# ---------------------------------------------------------------------
# Unified fixed-canvas style (required)
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_style_v4 as ps
ps.set_paper_style()
HAVE_PLOT_STYLE_V4 = True
COLORS = ps.COLORS
MAIN_PALETTE = ps.MAIN_PALETTE
FS_LABEL = ps.FONT_SIZE_LABEL
FS_TICK = ps.FONT_SIZE_TICK
FS_LEGEND = ps.FONT_SIZE_LEGEND
LW_AXIS = ps.LINEWIDTH_AXIS
LW_DATA = ps.LINEWIDTH_DATA
MS = ps.MARKER_SIZE

C_HOME = COLORS["leverage"]
C_CROSS = COLORS["mismatch"]
C_BIO = COLORS["biology"]
C_SUPPORT = COLORS["support"]
C_SECONDARY = COLORS["secondary"]
C_COMPLEXITY = COLORS["complexity"]
C_GRAY = COLORS["mid_gray"]
C_DARK = COLORS["dark_gray"]
C_LIGHT = COLORS["light_gray"]
C_BLACK = COLORS["black"]
C_WHITE = COLORS["white"]


def fixed_figure(kind: str):
    """Create the only legal canvas for a semantic OpenScope plot type."""
    return ps.new_figure(kind)

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot frozen OpenScope v2.3 + FinalClosure manuscript atlas."
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--main-only", action="store_true")
    ap.add_argument("--si-only", action="store_true")
    ap.add_argument("--strict", action="store_true")
    return ap.parse_args()


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def find_named_file(root: Path, basename: str) -> Path | None:
    direct = root / "analysis" / basename
    if direct.exists():
        return direct
    direct2 = root / basename
    if direct2.exists():
        return direct2
    matches = list(root.rglob(basename)) if root.exists() else []
    return matches[0] if matches else None


def read_csv_optional(root: Path, basename: str) -> tuple[pd.DataFrame | None, Path | None]:
    p = find_named_file(root, basename)
    if p is None:
        return None, None
    try:
        return pd.read_csv(p), p
    except Exception:
        return None, p


def read_json_optional(root: Path, basename: str):
    p = find_named_file(root, basename)
    if p is None:
        return None, None
    try:
        return json.loads(p.read_text(encoding="utf-8-sig")), p
    except Exception:
        return None, p


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def subject_order(df: pd.DataFrame) -> list[str]:
    if "subject" not in df.columns:
        return []
    vals = df["subject"].dropna().astype(str).unique().tolist()
    def key(x):
        try:
            return (0, int(float(x)))
        except Exception:
            return (1, x)
    return sorted(vals, key=key)


def aggregate_session_mean(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    keys = [c for c in ["subject", "session"] if c in df.columns]
    if not keys:
        return pd.DataFrame()
    cols = [c for c in value_cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return (
        df.groupby(keys, as_index=False)[cols]
        .mean(numeric_only=True)
    )


def primary_C(df: pd.DataFrame) -> pd.DataFrame:
    if "C" not in df.columns:
        return df.copy()
    c = pd.to_numeric(df["C"], errors="coerce")
    if np.isclose(c.dropna(), 1.0).any():
        return df[np.isclose(c, 1.0)].copy()
    return df.copy()


def save_plot_data(outroot: Path, fig_id: str, df: pd.DataFrame):
    p = outroot / "plot_data" / f"{fig_id}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def open_axes(ax, matrix=False):
    # no titles, ever
    try:
        ax.set_title("")
    except Exception:
        pass
    if matrix:
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(LW_AXIS)
            s.set_color(C_BLACK)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_linewidth(LW_AXIS)
        ax.spines["bottom"].set_linewidth(LW_AXIS)
        ax.spines["left"].set_color(C_BLACK)
        ax.spines["bottom"].set_color(C_BLACK)
    ax.tick_params(direction="out", length=3, width=LW_AXIS, colors=C_BLACK)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)


def finalize_and_save(fig, outroot: Path, tier: str, fig_id: str, dpi: int):
    # strip titles from all axes and figure
    try:
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_text("")
    except Exception:
        pass
    for ax in fig.axes:
        try:
            ax.set_title("")
        except Exception:
            pass
        open_axes(ax, matrix=(len(ax.images) > 0))
    fig.patch.set_facecolor(C_WHITE)

    d = outroot / ("MAIN_candidates" if tier == "MAIN" else "SI_support")
    d.mkdir(parents=True, exist_ok=True)
    outstem = d / fig_id
    saved = ps.save_figure(
        fig, outstem, dpi=dpi, save_png=True, save_pdf=True,
        transparent=False, close=True
    )
    png = next(p for p in saved if p.suffix.lower() == ".png")
    pdf = next(p for p in saved if p.suffix.lower() == ".pdf")
    return png, pdf


# ---------------------------------------------------------------------
# Plot primitives
# ---------------------------------------------------------------------
def paired_plot(df, a, b, label_a, label_b, ylabel, ref=None,
                color_a=C_HOME, color_b=C_CROSS):
    d = df[[c for c in ["subject", a, b] if c in df.columns]].dropna().copy()
    if a not in d.columns or b not in d.columns or d.empty:
        raise ValueError(f"Missing paired columns: {a}, {b}")
    fig, ax = fixed_figure("os_paired")
    x0, x1 = 0.0, 1.0
    for _, r in d.iterrows():
        ax.plot([x0, x1], [r[a], r[b]], color=C_LIGHT, lw=0.7, zorder=1)
    ax.scatter(np.full(len(d), x0), d[a], s=18, color=color_a, edgecolor="none", zorder=3)
    ax.scatter(np.full(len(d), x1), d[b], s=18, color=color_b, edgecolor="none", zorder=3)
    ax.plot([x0, x1], [d[a].mean(), d[b].mean()], color=C_BLACK, lw=1.4, zorder=4)
    ax.scatter([x0, x1], [d[a].mean(), d[b].mean()], s=26,
               facecolor=C_WHITE, edgecolor=C_BLACK, linewidth=0.9, zorder=5)
    if ref is not None:
        ax.axhline(ref, color=C_GRAY, lw=0.8, ls="--", zorder=0)
    ax.set_xticks([x0, x1], [label_a, label_b])
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.35, 1.35)
    open_axes(ax)
    return fig, d


def mouse_effect_plot(df, col, ylabel, ref=0.0, color=C_HOME):
    d = df[[c for c in ["subject", col] if c in df.columns]].dropna().copy()
    if col not in d.columns or d.empty:
        raise ValueError(f"Missing column: {col}")
    d["subject"] = d["subject"].astype(str)
    order = subject_order(d)
    d["_order"] = pd.Categorical(d["subject"], categories=order, ordered=True)
    d = d.sort_values("_order")
    x = np.arange(len(d))
    fig, ax = fixed_figure("os_mouse")
    ax.axhline(ref, color=C_GRAY, lw=0.8, ls="--", zorder=0)
    for xi, yi in zip(x, d[col]):
        ax.plot([xi, xi], [ref, yi], color=C_LIGHT, lw=0.8, zorder=1)
    ax.scatter(x, d[col], s=22, color=color, edgecolor="none", zorder=3)
    mean = float(d[col].mean())
    ax.axhline(mean, color=C_BLACK, lw=1.2, zorder=2)
    ax.set_xticks(x, [str(s) for s in d["subject"]], rotation=60, ha="right")
    ax.set_xlabel("Mouse")
    ax.set_ylabel(ylabel)
    open_axes(ax)
    return fig, d.drop(columns=["_order"])


def mouse_value_plot(df, col, ylabel, ref=None, color=C_BIO):
    return mouse_effect_plot(df, col, ylabel, ref=(0.0 if ref is None else ref), color=color)


def group_ci_plot(stats_df, endpoint, ylabel="Effect relative to null",
                  color=C_HOME):
    d = stats_df[stats_df["endpoint"].astype(str).eq(endpoint)].copy()
    if d.empty:
        raise ValueError(f"Endpoint not found: {endpoint}")
    r = d.iloc[0]
    mean = float(r["mean_effect"])
    lo = float(r["bootstrap_ci_low"])
    hi = float(r["bootstrap_ci_high"])
    fig, ax = fixed_figure("os_group_ci")
    ax.axhline(0, color=C_GRAY, lw=0.8, ls="--")
    ax.errorbar([0], [mean], yerr=[[mean-lo], [hi-mean]], fmt="o",
                color=color, ecolor=color, elinewidth=1.2, capsize=2.5,
                markersize=5)
    ax.set_xlim(-0.7, 0.7)
    ax.set_xticks([0], ["12 mice"])
    ax.set_ylabel(ylabel)
    open_axes(ax)
    return fig, d


def summary_forest(stats_df, endpoints: list[tuple[str, str]], value_col="mean_effect",
                   lo_col="bootstrap_ci_low", hi_col="bootstrap_ci_high",
                   xlabel="Effect relative to null", color=C_HOME):
    rows = []
    for endpoint, label in endpoints:
        g = stats_df[stats_df["endpoint"].astype(str).eq(endpoint)]
        if g.empty:
            continue
        r = g.iloc[0]
        rows.append({
            "endpoint": endpoint,
            "label": label,
            "value": pd.to_numeric(r.get(value_col), errors="coerce"),
            "lo": pd.to_numeric(r.get(lo_col), errors="coerce"),
            "hi": pd.to_numeric(r.get(hi_col), errors="coerce"),
        })
    d = pd.DataFrame(rows).dropna(subset=["value"])
    if d.empty:
        raise ValueError("No endpoints available for forest")
    fig, ax = fixed_figure("os_forest")
    y = np.arange(len(d))[::-1]
    ax.axvline(0, color=C_GRAY, lw=0.8, ls="--")
    for yi, (_, r) in zip(y, d.iterrows()):
        if np.isfinite(r["lo"]) and np.isfinite(r["hi"]):
            ax.plot([r["lo"], r["hi"]], [yi, yi], color=color, lw=1.3)
        ax.scatter([r["value"]], [yi], s=24, color=color, zorder=3)
    ax.set_yticks(y, d["label"])
    ax.set_xlabel(xlabel)
    open_axes(ax)
    return fig, d


def line_by_group(df, xcol, ycol, xlabel, ylabel, groupcol="subject",
                  ref=None, highlight_x=None, color=C_HOME):
    cols = [c for c in [groupcol, xcol, ycol] if c in df.columns]
    d = df[cols].dropna().copy()
    if xcol not in d.columns or ycol not in d.columns or d.empty:
        raise ValueError(f"Missing line columns: {xcol}, {ycol}")
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.dropna(subset=[xcol, ycol])
    fig, ax = fixed_figure("os_line")
    if ref is not None:
        ax.axhline(ref, color=C_GRAY, lw=0.8, ls="--")
    if groupcol in d.columns:
        for _, g in d.groupby(groupcol):
            gg = g.groupby(xcol, as_index=False)[ycol].mean().sort_values(xcol)
            ax.plot(gg[xcol], gg[ycol], color=C_LIGHT, lw=0.7, alpha=0.9)
    mean = d.groupby(xcol, as_index=False)[ycol].mean().sort_values(xcol)
    ax.plot(mean[xcol], mean[ycol], "-o", color=color, lw=1.4, ms=4.2)
    if highlight_x is not None:
        ax.axvline(highlight_x, color=C_DARK, lw=0.8, ls=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    open_axes(ax)
    return fig, d


def repeat_distribution_by_mouse(df, col, ylabel, ref=0.0, color=C_HOME):
    d = df[[c for c in ["subject", "repeat", col] if c in df.columns]].dropna().copy()
    if col not in d.columns or d.empty:
        raise ValueError(f"Missing repeat column: {col}")
    d["subject"] = d["subject"].astype(str)
    order = subject_order(d)
    fig, ax = fixed_figure("os_repeat_mouse")
    ax.axhline(ref, color=C_GRAY, lw=0.8, ls="--")
    rng = np.random.default_rng(20260822)
    for xi, s in enumerate(order):
        g = d[d["subject"].eq(s)]
        jitter = rng.normal(0, 0.045, size=len(g))
        ax.scatter(np.full(len(g), xi)+jitter, g[col], s=9, color=C_LIGHT,
                   edgecolor="none", zorder=1)
        ax.scatter([xi], [g[col].mean()], s=23, color=color, edgecolor="none", zorder=3)
    ax.set_xticks(np.arange(len(order)), order, rotation=60, ha="right")
    ax.set_xlabel("Mouse")
    ax.set_ylabel(ylabel)
    open_axes(ax)
    return fig, d


def raw_vs_controlled_plot(controlled, raw, col, ylabel, absolute=False):
    dc = aggregate_session_mean(controlled, [col])
    dr = aggregate_session_mean(raw, [col])
    keys = [c for c in ["subject", "session"] if c in dc.columns and c in dr.columns]
    if not keys:
        raise ValueError("No session keys for raw-controlled comparison")
    m = dc.merge(dr, on=keys, suffixes=("_controlled", "_raw"))
    a = f"{col}_raw"
    b = f"{col}_controlled"
    if absolute:
        m[a] = m[a].abs()
        m[b] = m[b].abs()
    return paired_plot(m, a, b, "Raw", "Controlled", ylabel,
                       color_a=C_GRAY, color_b=C_HOME)


def multi_measure_mouse_plot(df, cols_labels, ylabel, ref=0.0, colors=None):
    cols = [c for c, _ in cols_labels]
    keep = [c for c in ["subject"] + cols if c in df.columns]
    d = df[keep].dropna().copy()
    if len(d) == 0 or any(c not in d.columns for c in cols):
        raise ValueError("Missing columns for multi-measure plot")
    labels = [lab for _, lab in cols_labels]
    x = np.arange(len(cols))
    fig, ax = fixed_figure("os_multi_measure")
    ax.axhline(ref, color=C_GRAY, lw=0.8, ls="--")
    for _, r in d.iterrows():
        ax.plot(x, [r[c] for c in cols], color=C_LIGHT, lw=0.7)
    colors = colors or [C_HOME] * len(cols)
    for i, c in enumerate(cols):
        ax.scatter(np.full(len(d), i), d[c], s=12, color=colors[i], edgecolor="none")
        ax.scatter([i], [d[c].mean()], s=25, facecolor=C_WHITE,
                   edgecolor=colors[i], linewidth=1.0, zorder=4)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    open_axes(ax)
    return fig, d


def histogram_null(null_values, observed, xlabel, color=C_GRAY, observed_color=C_HOME):
    arr = np.asarray(null_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 20:
        raise ValueError("Insufficient null draws")
    fig, ax = fixed_figure("os_histogram")
    ax.hist(arr, bins=36, density=True, color=color, alpha=0.65,
            edgecolor="none")
    ax.axvline(observed, color=observed_color, lw=1.5)
    ax.axvline(0, color=C_DARK, lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Null density")
    open_axes(ax)
    d = pd.DataFrame({"null": arr})
    d["observed"] = observed
    return fig, d


def bar_or_point_by_subject(df, col, ylabel, color=C_SUPPORT, ref=None):
    d = aggregate_session_mean(df, [col])
    if d.empty:
        raise ValueError(f"No session mean for {col}")
    if ref is None:
        ref = 0.0
    return mouse_effect_plot(d, col, ylabel, ref=ref, color=color)


def trial_counts_plot(df, cols, ylabel):
    d = aggregate_session_mean(df, cols)
    if d.empty or any(c not in d.columns for c in cols):
        raise ValueError("Trial count columns missing")
    fig, ax = fixed_figure("os_stacked_bar")
    x = np.arange(len(d))
    bottom = np.zeros(len(d))
    colors = [C_COMPLEXITY, C_HOME, C_BIO, C_SUPPORT, C_SECONDARY, C_CROSS]
    for i, c in enumerate(cols):
        vals = d[c].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, width=0.72, color=colors[i % len(colors)],
               edgecolor="none", label=c)
        bottom += vals
    ax.set_xticks(x, d["subject"].astype(str), rotation=60, ha="right")
    ax.set_xlabel("Mouse")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    open_axes(ax)
    return fig, d


def infer_null_vector(df: pd.DataFrame, endpoint: str) -> np.ndarray:
    # Wide form preferred.
    candidates = [
        f"{endpoint}_null",
        endpoint,
        endpoint.replace("decoder_auc_crossover", "decoder_auc_crossover_null"),
        endpoint.replace("delta_rho", "delta_rho_null"),
        endpoint.replace("delta_jaccard", "delta_jaccard_null"),
    ]
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(float)

    # Long form.
    endpoint_cols = [c for c in df.columns if "endpoint" in c.lower() or "metric" in c.lower()]
    value_cols = [c for c in df.columns if "null" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
    for ec in endpoint_cols:
        mask = df[ec].astype(str).str.contains(endpoint, case=False, regex=False)
        if mask.any():
            for vc in value_cols:
                vals = pd.to_numeric(df.loc[mask, vc], errors="coerce").dropna().to_numpy(float)
                if len(vals):
                    return vals
    return np.array([], dtype=float)


def extract_loo_table(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    endpoint_cols = [c for c in df.columns if "endpoint" in c.lower() or "metric" in c.lower()]
    if endpoint_cols:
        ec = endpoint_cols[0]
        d = df[df[ec].astype(str).eq(endpoint)].copy()
        if d.empty:
            d = df[df[ec].astype(str).str.contains(endpoint, case=False, regex=False)].copy()
        if d.empty:
            return pd.DataFrame()
        effect_candidates = [
            "loo_mean_effect", "mean_effect", "group_mean_effect",
            "effect", "mean_without_mouse", "leave_one_out_mean",
        ]
        effect = next((c for c in effect_candidates if c in d.columns), None)
        if effect is None:
            num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
            num = [c for c in num if "subject" not in c.lower() and "session" not in c.lower()]
            effect = num[-1] if num else None
        if effect is None:
            return pd.DataFrame()
        subj = next((c for c in d.columns if "left_out" in c.lower() or "omit" in c.lower()), None)
        if subj is None:
            subj = next((c for c in d.columns if "subject" in c.lower()), None)
        out = pd.DataFrame({
            "left_out": d[subj].astype(str) if subj else np.arange(len(d)).astype(str),
            "effect": pd.to_numeric(d[effect], errors="coerce"),
        }).dropna()
        return out

    # Wide form.
    if endpoint in df.columns:
        subj = next((c for c in df.columns if "left_out" in c.lower() or "subject" in c.lower()), None)
        return pd.DataFrame({
            "left_out": df[subj].astype(str) if subj else np.arange(len(df)).astype(str),
            "effect": pd.to_numeric(df[endpoint], errors="coerce"),
        }).dropna()
    return pd.DataFrame()


def loo_plot(df: pd.DataFrame, endpoint: str, ylabel: str, color=C_HOME):
    d = extract_loo_table(df, endpoint)
    if d.empty:
        raise ValueError(f"LOO endpoint unavailable: {endpoint}")
    fig, ax = fixed_figure("os_loo")
    ax.axhline(0, color=C_GRAY, lw=0.8, ls="--")
    x = np.arange(len(d))
    ax.scatter(x, d["effect"], s=22, color=color, edgecolor="none")
    ax.plot(x, d["effect"], color=C_LIGHT, lw=0.7)
    ax.set_xticks(x, d["left_out"], rotation=60, ha="right")
    ax.set_xlabel("Left-out mouse")
    ax.set_ylabel(ylabel)
    open_axes(ax)
    return fig, d


def extract_boundary_pairs(df: pd.DataFrame, metric: str):
    if df is None or df.empty:
        return pd.DataFrame()

    # Common direct forms.
    pairs = {
        "decoder": [
            ("primary_decoder_auc_crossover", "real_edge_decoder_auc_crossover"),
            ("decoder_auc_crossover_primary", "decoder_auc_crossover_real_edge"),
            ("primary", "real_edge"),
        ],
        "rho": [
            ("primary_delta_rho", "real_edge_delta_rho"),
            ("delta_rho_primary", "delta_rho_real_edge"),
        ],
        "topset": [
            ("primary_topset_auc_crossover", "real_edge_topset_auc_crossover"),
            ("topset_auc_crossover_primary", "topset_auc_crossover_real_edge"),
        ],
    }
    for a, b in pairs.get(metric, []):
        if a in df.columns and b in df.columns:
            cols = [c for c in ["subject", "session", a, b] if c in df.columns]
            return df[cols].dropna().copy().rename(columns={a: "primary", b: "real_edge"})

    # Long form with condition/source column.
    condition_col = next((c for c in df.columns if c.lower() in {"condition", "analysis", "edge_type", "dataset"}), None)
    value_candidates = {
        "decoder": ["decoder_auc_crossover", "R_decoder", "r_decoder"],
        "rho": ["delta_rho", "R_rho", "r_rho"],
        "topset": ["topset_auc_crossover", "R_top", "r_top"],
    }
    vc = next((c for c in value_candidates.get(metric, []) if c in df.columns), None)
    if condition_col and vc and "subject" in df.columns:
        idx = [c for c in ["subject", "session"] if c in df.columns]
        w = df.pivot_table(index=idx, columns=condition_col, values=vc, aggfunc="mean").reset_index()
        pc = next((c for c in w.columns if "primary" in str(c).lower()), None)
        rc = next((c for c in w.columns if "real" in str(c).lower()), None)
        if pc and rc:
            return w[idx + [pc, rc]].rename(columns={pc: "primary", rc: "real_edge"})
    return pd.DataFrame()


def boundary_paired_plot(df, metric, ylabel):
    d = extract_boundary_pairs(df, metric)
    if d.empty:
        raise ValueError(f"Boundary pairs unavailable: {metric}")
    return paired_plot(d, "primary", "real_edge", "Primary", "Real-edge", ylabel,
                       color_a=C_HOME, color_b=C_SUPPORT)


# ---------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    root = args.root.resolve()
    plot_root = root / "plot"
    v23_root = plot_root / "OpenScope_Illusion_Analysis_v2_3"
    closure_root = plot_root / "OpenScope_FinalClosure_v1"
    outroot = (
        args.output.resolve()
        if args.output is not None
        else plot_root / "OpenScope_Manuscript_FigureAtlas_Final"
    )
    outroot.mkdir(parents=True, exist_ok=True)
    (outroot / "MAIN_candidates").mkdir(exist_ok=True)
    (outroot / "SI_support").mkdir(exist_ok=True)
    (outroot / "plot_data").mkdir(exist_ok=True)

    # Core frozen data.
    session, p_session = read_csv_optional(v23_root, "session_primary_v23_summary.csv")
    repeats, p_repeats = read_csv_optional(v23_root, "all_primary_v23_repeat_summary.csv")
    landscapes, p_land = read_csv_optional(v23_root, "all_primary_v23_landscape_summary.csv")
    topsets, p_top = read_csv_optional(v23_root, "all_primary_v23_topset_summary.csv")
    raw_repeats, p_raw = read_csv_optional(v23_root, "all_raw_state_repeat_summary.csv")
    real_repeats, p_real = read_csv_optional(v23_root, "all_real_edge_v23_repeat_summary.csv")

    formal, p_formal = read_csv_optional(closure_root, "formal_12mouse_statistics.csv")
    endpoint_table, p_endpoint = read_csv_optional(closure_root, "formal_12mouse_endpoint_table.csv")
    high_null, p_highnull = read_csv_optional(closure_root, "high_precision_hierarchical_null.csv")
    loo_df, p_loo = read_csv_optional(closure_root, "leave_one_mouse_out.csv")
    boundary_df, p_boundary = read_csv_optional(closure_root, "primary_vs_real_edge_paired_boundary.csv")

    perm_summary, p_perm_summary = read_json_optional(closure_root, "high_precision_permutation_summary.json")
    loo_summary, p_loo_summary = read_json_optional(closure_root, "leave_one_mouse_out_summary.json")
    boundary_stats, p_boundary_stats = read_json_optional(closure_root, "primary_vs_real_edge_boundary_statistics.json")

    # Optional frozen sensitivity/QC tables.
    top_budget, p_top_budget = read_csv_optional(v23_root, "topset_budget_sensitivity_inference.csv")
    reg_sens, p_reg_sens = read_csv_optional(v23_root, "landscape_regularization_sensitivity_inference.csv")
    raw_session, p_raw_session = read_csv_optional(v23_root, "session_raw_state_sensitivity.csv")
    confound_comp, p_confound_comp = read_csv_optional(v23_root, "state_confound_control_comparison.csv")
    real_session, p_real_session = read_csv_optional(v23_root, "session_real_edge_v23_control.csv")
    shuffle30, p_shuffle30 = read_csv_optional(v23_root, "hierarchical_v23_state_shuffle_null.csv")

    # Fallback session table can be reconstructed from repeats.
    if session is None and repeats is not None:
        rr = primary_C(repeats)
        session = aggregate_session_mean(rr, [
            "decoder_auc_crossover", "decoder_auc_home", "decoder_auc_cross",
            "topset_auc_crossover", "topset_auc_home", "topset_auc_cross",
            "delta_rho", "rho_within", "rho_cross", "delta_jaccard",
            "jaccard_within", "jaccard_cross",
            "activity_residual_delta_rho",
            "activity_selectivity_residual_delta_rho",
            "half_model_home_auc_mean", "same_image_state_auc_mean",
            "same_image_state_auc_excess_mean", "state_stimulus_cramers_v",
            "state_score_time_spearman", "running_state_cohens_d",
            "pc1_explained", "n_units", "n_analysis_trials", "n_discovery", "n_final",
            "state0_fraction", "state1_fraction",
        ])
        p_session = p_repeats

    # Normalize numeric.
    for name in ["session", "repeats", "landscapes", "topsets", "raw_repeats", "real_repeats",
                 "formal", "endpoint_table", "high_null", "loo_df", "boundary_df",
                 "top_budget", "reg_sens", "raw_session", "confound_comp",
                 "real_session", "shuffle30"]:
        obj = locals().get(name)
        if isinstance(obj, pd.DataFrame):
            for c in obj.columns:
                if c not in {"subject", "session", "endpoint", "block", "positive_labels",
                             "negative_labels", "state_mode", "condition", "metric"}:
                    obj[c] = pd.to_numeric(obj[c], errors="ignore")

    catalog = []
    skipped = []

    def record(fig_id, tier, role, claim, source_paths, generated, reason=""):
        catalog.append({
            "figure_id": fig_id,
            "tier": tier,
            "suggested_role": role,
            "claim_or_support": claim,
            "source_files": " | ".join(str(p) for p in source_paths if p is not None),
            "generated": bool(generated),
            "reason": reason,
        })

    def run_job(fig_id: str, tier: str, role: str, claim: str,
                sources: list[Path | None], fn: Callable[[], tuple]):
        if args.main_only and tier != "MAIN":
            return
        if args.si_only and tier != "SI":
            return
        try:
            fig, data = fn()
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            save_plot_data(outroot, fig_id, data)
            finalize_and_save(fig, outroot, tier, fig_id, args.dpi)
            record(fig_id, tier, role, claim, sources, True, "")
            print(f"[OK] {fig_id}")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            skipped.append({
                "figure_id": fig_id,
                "tier": tier,
                "reason": msg,
                "traceback": traceback.format_exc(),
            })
            record(fig_id, tier, role, claim, sources, False, msg)
            print(f"[SKIP] {fig_id}: {msg}")
            if args.strict:
                raise

    if session is None or repeats is None:
        raise SystemExit(
            "Core OpenScope v2.3 summary tables are missing. Expected "
            "session_primary_v23_summary.csv and all_primary_v23_repeat_summary.csv."
        )

    rep1 = primary_C(repeats)
    land1 = primary_C(landscapes) if landscapes is not None else None
    real1 = primary_C(real_repeats) if real_repeats is not None else None

    # ================================================================
    # MAIN candidates
    # ================================================================
    run_job(
        "OS_M01_decoder_home_vs_cross_auc_paired", "MAIN",
        "headline population-code transfer",
        "Home-state IC/LC decoder performance exceeds cross-state transfer within mouse.",
        [p_session],
        lambda: paired_plot(
            session, "decoder_auc_home", "decoder_auc_cross",
            "Home", "Cross", "Held-out decoder AUC", ref=0.5,
            color_a=C_HOME, color_b=C_CROSS
        ),
    )

    run_job(
        "OS_M02_Rdecoder_by_mouse", "MAIN",
        "headline mouse-level effect",
        "Mouse/session-level R_decoder is predominantly positive.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "decoder_auc_crossover",
            r"$R_{\mathrm{decoder}}$", ref=0.0, color=C_HOME
        ),
    )

    if formal is not None:
        run_job(
            "OS_M03_Rdecoder_formal_effect_ci", "MAIN",
            "formal 12-mouse inference",
            "Frozen 12-mouse R_decoder effect with bootstrap confidence interval.",
            [p_formal],
            lambda: group_ci_plot(
                formal, "decoder_state_transfer", r"$R_{\mathrm{decoder}}$",
                color=C_HOME
            ),
        )

    def main_perm_decoder():
        observed = None
        if isinstance(perm_summary, dict):
            observed = perm_summary.get("decoder_auc_crossover", {}).get(
                "matched_observed_repeat0_group_mean"
            )
        vals = infer_null_vector(high_null, "decoder_auc_crossover") if high_null is not None else np.array([])
        # Fallback: concatenate all per-session high-precision null tables and
        # compute global mean by perm index if needed.
        if vals.size < 20:
            files = list((closure_root / "null_sessions").glob("*high_precision_null.csv"))
            frames = []
            for p in files:
                try:
                    d = pd.read_csv(p)
                    d["_source"] = p.name
                    frames.append(d)
                except Exception:
                    pass
            if frames:
                alln = pd.concat(frames, ignore_index=True)
                # If individual session rows include perm_index, average across mice.
                pc = next((c for c in alln.columns if "perm" in c.lower() and "index" in c.lower()), None)
                vc = next((c for c in alln.columns if "decoder_auc_crossover" in c and "null" in c), None)
                if pc and vc:
                    vals = (
                        alln.groupby(pc)[vc].mean()
                        .dropna().to_numpy(float)
                    )
                else:
                    vals = infer_null_vector(alln, "decoder_auc_crossover")
        if observed is None and formal is not None:
            observed = float(
                formal.loc[
                    formal["endpoint"].eq("decoder_state_transfer"),
                    "mean_effect"
                ].iloc[0]
            )
        fig, d = histogram_null(vals, float(observed),
                                r"Fixed-split null $R_{\mathrm{decoder}}$",
                                color=C_LIGHT, observed_color=C_HOME)
        return fig, d

    run_job(
        "OS_M04_Rdecoder_high_precision_permutation", "MAIN",
        "formal permutation confirmation",
        "Fixed-split state-label permutation confirms the decoder-transfer effect.",
        [p_highnull, p_perm_summary],
        main_perm_decoder,
    )

    run_job(
        "OS_M05_same_image_state_auc_by_mouse", "MAIN",
        "same-input biological state imprint",
        "The same external image carries a strong poststimulus signature of prestimulus collective state.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "same_image_state_auc_mean",
            "Same-image state-decoding AUC", ref=0.5, color=C_BIO
        ),
    )

    run_job(
        "OS_M06_same_image_state_excess_by_mouse", "MAIN",
        "same-input effect relative to chance",
        "Mouse-level same-image state information is consistently above chance.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "same_image_state_auc_excess_mean",
            "AUC excess above chance", ref=0.0, color=C_BIO
        ),
    )

    run_job(
        "OS_M07_coefficient_rho_within_vs_cross_paired", "MAIN",
        "coefficient-landscape organization",
        "Within-state coefficient landscapes are more similar than cross-state landscapes across repeated splits.",
        [p_session],
        lambda: paired_plot(
            session, "rho_within", "rho_cross",
            "Within", "Cross", "Coefficient-rank correlation",
            color_a=C_HOME, color_b=C_CROSS
        ),
    )

    run_job(
        "OS_M08_Rrho_by_mouse", "MAIN",
        "supportive leverage-landscape effect",
        "Repeated-split mouse-level R_rho is positive, while final permutation remains a stated boundary.",
        [p_session, p_formal, p_perm_summary],
        lambda: mouse_effect_plot(
            session, "delta_rho", r"$R_{\rho}$", ref=0.0, color=C_SUPPORT
        ),
    )

    run_job(
        "OS_M09_landscape_residualization_by_mouse", "MAIN",
        "activity/selectivity robustness",
        "Landscape reconfiguration persists after activity and activity+selectivity residualization.",
        [p_session],
        lambda: multi_measure_mouse_plot(
            session,
            [
                ("delta_rho", "Raw"),
                ("activity_residual_delta_rho", "Activity residual"),
                ("activity_selectivity_residual_delta_rho", "Activity + selectivity"),
            ],
            r"$R_{\rho}$",
            ref=0.0,
            colors=[C_SUPPORT, C_SECONDARY, C_COMPLEXITY],
        ),
    )

    run_job(
        "OS_M10_half_model_predictive_validity_by_mouse", "MAIN",
        "estimator validity",
        "Split-half coefficient models retain held-out predictive validity.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "half_model_home_auc_mean",
            "Split-half home-state AUC", ref=0.5, color=C_COMPLEXITY
        ),
    )

    run_job(
        "OS_M11_state_time_confound_by_mouse", "MAIN",
        "confound control",
        "Confound-controlled state is weakly related to recording time.",
        [p_session],
        lambda: mouse_effect_plot(
            session.assign(_abs_time=session["state_score_time_spearman"].abs()),
            "_abs_time", r"$|\rho|$ state score vs time",
            ref=0.0, color=C_SECONDARY
        ),
    )

    run_job(
        "OS_M12_running_confound_by_mouse", "MAIN",
        "confound control",
        "Residual running differences between state groups are limited.",
        [p_session],
        lambda: mouse_effect_plot(
            session.assign(_abs_run=session["running_state_cohens_d"].abs()),
            "_abs_run", r"$|d|$ running difference",
            ref=0.0, color=C_SECONDARY
        ),
    )

    run_job(
        "OS_M13_stimulus_state_cramers_v_by_mouse", "MAIN",
        "confound control",
        "Stimulus identity is only weakly associated with the inferred prestimulus state.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "state_stimulus_cramers_v",
            "State–stimulus Cramér's V", ref=0.0, color=C_SECONDARY
        ),
    )

    if loo_df is not None:
        run_job(
            "OS_M14_leave_one_mouse_out_Rdecoder", "MAIN",
            "mouse-level robustness",
            "The headline decoder-transfer effect remains positive after removing any single mouse.",
            [p_loo],
            lambda: loo_plot(
                loo_df, "decoder_auc_crossover",
                r"LOO mean $R_{\mathrm{decoder}}$", color=C_HOME
            ),
        )

    if boundary_df is not None:
        run_job(
            "OS_M15_primary_vs_real_edge_Rdecoder", "MAIN",
            "interpretation boundary",
            "Primary and real-edge state-transfer effects are comparable; the claim is general state-dependent coding, not illusion-specific.",
            [p_boundary, p_boundary_stats],
            lambda: boundary_paired_plot(
                boundary_df, "decoder", r"$R_{\mathrm{decoder}}$"
            ),
        )

    if formal is not None:
        run_job(
            "OS_M16_formal_effect_summary_forest", "MAIN",
            "compact biological closure summary",
            "Frozen formal effects across decoder transfer, coefficient landscape, same-image state imprint and predictive validity.",
            [p_formal],
            lambda: summary_forest(
                formal,
                [
                    ("decoder_state_transfer", r"$R_{\mathrm{decoder}}$"),
                    ("coefficient_landscape_reconfiguration", r"$R_{\rho}$"),
                    ("top20_unitset_transfer_supportive", r"$R_{\mathrm{top20}}$"),
                    ("activity_residual_landscape", r"Activity-resid. $R_{\rho}$"),
                ],
                xlabel="Effect relative to null",
                color=C_HOME,
            ),
        )
        run_job(
            "OS_M17_same_image_formal_effect_ci", "MAIN",
            "formal same-input biological support",
            "Frozen 12-mouse same-exact-image state-imprint effect above chance.",
            [p_formal],
            lambda: group_ci_plot(
                formal, "same_exact_image_state_imprint",
                "AUC excess above chance", color=C_BIO
            ),
        )
        run_job(
            "OS_M18_half_model_validity_formal_effect_ci", "MAIN",
            "formal estimator-validity support",
            "Frozen 12-mouse split-half predictive-validity excess above chance.",
            [p_formal],
            lambda: group_ci_plot(
                formal, "half_model_predictive_validity",
                "AUC excess above chance", color=C_COMPLEXITY
            ),
        )

    # ================================================================
    # SI / supporting
    # ================================================================
    run_job(
        "OS_S01_decoder_home_auc_by_mouse", "SI",
        "decoder validity detail",
        "Per-mouse home-state held-out AUC.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "decoder_auc_home", "Home-state decoder AUC",
            ref=0.5, color=C_HOME
        ),
    )

    run_job(
        "OS_S02_decoder_cross_auc_by_mouse", "SI",
        "decoder transfer detail",
        "Per-mouse cross-state held-out AUC.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "decoder_auc_cross", "Cross-state decoder AUC",
            ref=0.5, color=C_CROSS
        ),
    )

    run_job(
        "OS_S03_Rdecoder_repeat_distribution", "SI",
        "repeat-level robustness",
        "Repeated independent splits underlying each mouse-level R_decoder estimate.",
        [p_repeats],
        lambda: repeat_distribution_by_mouse(
            rep1, "decoder_auc_crossover",
            r"$R_{\mathrm{decoder}}$ across repeats",
            ref=0.0, color=C_HOME
        ),
    )

    run_job(
        "OS_S04_decoder_balanced_accuracy_crossover_repeat_distribution", "SI",
        "metric robustness",
        "Balanced-accuracy transfer direction across repeated splits.",
        [p_repeats],
        lambda: repeat_distribution_by_mouse(
            rep1, "decoder_balacc_crossover",
            "Balanced-accuracy crossover",
            ref=0.0, color=C_SECONDARY
        ),
    )

    run_job(
        "OS_S05_same_image_auc_repeat_distribution", "SI",
        "same-input repeat robustness",
        "Same-image state imprint remains strong across repeated splits.",
        [p_repeats],
        lambda: repeat_distribution_by_mouse(
            rep1, "same_image_state_auc_mean",
            "Same-image state AUC across repeats",
            ref=0.5, color=C_BIO
        ),
    )

    def half_model_min_plot():
        d = aggregate_session_mean(rep1, ["half_model_home_auc_min"])
        return mouse_effect_plot(
            d, "half_model_home_auc_min",
            "Minimum split-half home-state AUC",
            ref=0.5, color=C_COMPLEXITY
        )

    run_job(
        "OS_S06_half_model_min_auc_by_mouse", "SI",
        "estimator validity lower bound",
        "Worst split-half home-state decoder validity by mouse.",
        [p_repeats],
        half_model_min_plot,
    )

    run_job(
        "OS_S07_topset_home_vs_cross_auc_paired", "SI",
        "top-unit transfer support",
        "Source-state top-unit subsets are more useful in their home state than across states.",
        [p_session],
        lambda: paired_plot(
            session, "topset_auc_home", "topset_auc_cross",
            "Home", "Cross", "Top-set refit AUC", ref=0.5,
            color_a=C_HOME, color_b=C_CROSS
        ),
    )

    run_job(
        "OS_S08_topset_R_by_mouse", "SI",
        "top-unit transfer support",
        "Mouse-level top-set transfer effect.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "topset_auc_crossover",
            r"$R_{\mathrm{topset}}$", ref=0.0, color=C_SUPPORT
        ),
    )

    if topsets is not None:
        run_job(
            "OS_S09_topset_budget_sensitivity_auc", "SI",
            "budget sensitivity",
            "Top-unit transfer effect across the pre-specified 5/10/20/40% budgets.",
            [p_top],
            lambda: line_by_group(
                topsets, "top_fraction", "topset_auc_crossover",
                "Top-unit fraction", r"$R_{\mathrm{topset}}$",
                ref=0.0, highlight_x=0.20, color=C_SUPPORT
            ),
        )
        run_job(
            "OS_S10_topset_budget_sensitivity_balanced_accuracy", "SI",
            "budget sensitivity",
            "Balanced-accuracy top-set transfer across pre-specified budgets.",
            [p_top],
            lambda: line_by_group(
                topsets, "top_fraction", "topset_balacc_crossover",
                "Top-unit fraction", "Balanced-accuracy crossover",
                ref=0.0, highlight_x=0.20, color=C_SECONDARY
            ),
        )

    run_job(
        "OS_S11_jaccard_within_vs_cross_paired", "SI",
        "top-unit overlap",
        "Within-state top-unit overlap compared with cross-state overlap.",
        [p_session],
        lambda: paired_plot(
            session, "jaccard_within", "jaccard_cross",
            "Within", "Cross", "Top-set Jaccard overlap",
            color_a=C_HOME, color_b=C_CROSS
        ),
    )

    run_job(
        "OS_S12_delta_jaccard_by_mouse", "SI",
        "top-unit overlap",
        "Mouse-level within-minus-cross top-set Jaccard effect.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "delta_jaccard",
            r"$\Delta$ Jaccard", ref=0.0, color=C_SUPPORT
        ),
    )

    run_job(
        "OS_S13_activity_residual_Rrho_by_mouse", "SI",
        "activity residualization",
        "Coefficient-landscape state dependence after activity residualization.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "activity_residual_delta_rho",
            r"Activity-residual $R_{\rho}$", ref=0.0, color=C_SECONDARY
        ),
    )

    run_job(
        "OS_S14_activity_selectivity_residual_Rrho_by_mouse", "SI",
        "activity/selectivity residualization",
        "Coefficient-landscape state dependence after activity and selectivity residualization.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "activity_selectivity_residual_delta_rho",
            r"Activity+selectivity residual $R_{\rho}$",
            ref=0.0, color=C_COMPLEXITY
        ),
    )

    if landscapes is not None:
        run_job(
            "OS_S15_regularization_sensitivity_Rrho", "SI",
            "regularization sensitivity",
            "R_rho is stable across the pre-specified logistic regularization values.",
            [p_land],
            lambda: line_by_group(
                landscapes, "C", "delta_rho",
                "Logistic C", r"$R_{\rho}$",
                ref=0.0, highlight_x=1.0, color=C_SUPPORT
            ),
        )
        run_job(
            "OS_S16_regularization_sensitivity_rho_within", "SI",
            "regularization sensitivity",
            "Within-state coefficient similarity across regularization.",
            [p_land],
            lambda: line_by_group(
                landscapes, "C", "rho_within",
                "Logistic C", r"$\rho_{\mathrm{within}}$",
                highlight_x=1.0, color=C_HOME
            ),
        )
        run_job(
            "OS_S17_regularization_sensitivity_rho_cross", "SI",
            "regularization sensitivity",
            "Cross-state coefficient similarity across regularization.",
            [p_land],
            lambda: line_by_group(
                landscapes, "C", "rho_cross",
                "Logistic C", r"$\rho_{\mathrm{cross}}$",
                highlight_x=1.0, color=C_CROSS
            ),
        )
        run_job(
            "OS_S18_regularization_sensitivity_activity_residual_Rrho", "SI",
            "regularization + confound robustness",
            "Activity-residual landscape effect across regularization.",
            [p_land],
            lambda: line_by_group(
                landscapes, "C", "activity_residual_delta_rho",
                "Logistic C", r"Activity-residual $R_{\rho}$",
                ref=0.0, highlight_x=1.0, color=C_SECONDARY
            ),
        )
        run_job(
            "OS_S19_regularization_sensitivity_activity_selectivity_Rrho", "SI",
            "regularization + confound robustness",
            "Activity+selectivity-residual landscape effect across regularization.",
            [p_land],
            lambda: line_by_group(
                landscapes, "C", "activity_selectivity_residual_delta_rho",
                "Logistic C", r"Activity+selectivity residual $R_{\rho}$",
                ref=0.0, highlight_x=1.0, color=C_COMPLEXITY
            ),
        )
        run_job(
            "OS_S20_regularization_sensitivity_delta_jaccard", "SI",
            "regularization sensitivity",
            "Top-unit overlap state dependence across regularization.",
            [p_land],
            lambda: line_by_group(
                landscapes, "C", "delta_jaccard",
                "Logistic C", r"$\Delta$ Jaccard",
                ref=0.0, highlight_x=1.0, color=C_SUPPORT
            ),
        )

    run_job(
        "OS_S21_pc1_explained_by_mouse", "SI",
        "state-estimator audit",
        "Variance captured by the frozen discovery-only PC1 state axis.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "pc1_explained",
            "PC1 explained variance", ref=0.0, color=C_SECONDARY
        ),
    )

    def state_fraction_plot():
        d = session.copy()
        if "state0_fraction" not in d.columns:
            # reconstruct from repeat summary
            d = aggregate_session_mean(rep1, ["state0_fraction", "state1_fraction"])
        return mouse_effect_plot(
            d, "state0_fraction",
            "State-0 trial fraction", ref=0.5, color=C_SECONDARY
        )

    run_job(
        "OS_S22_state_fraction_balance_by_mouse", "SI",
        "state-balance audit",
        "Primary state split remains close to balanced at the mouse/session level.",
        [p_session, p_repeats],
        state_fraction_plot,
    )

    run_job(
        "OS_S23_n_units_by_mouse", "SI",
        "sample-size audit",
        "Number of analyzed units per mouse/session.",
        [p_session],
        lambda: mouse_effect_plot(
            session, "n_units", "Analyzed units",
            ref=0.0, color=C_GRAY
        ),
    )

    def n_trials_plot():
        d = session.copy()
        if "n_analysis_trials" not in d.columns:
            d = aggregate_session_mean(rep1, ["n_analysis_trials"])
        return mouse_effect_plot(
            d, "n_analysis_trials", "Analysis trials",
            ref=0.0, color=C_GRAY
        )

    run_job(
        "OS_S24_n_analysis_trials_by_mouse", "SI",
        "sample-size audit",
        "Number of analysis trials per mouse/session.",
        [p_session, p_repeats],
        n_trials_plot,
    )

    run_job(
        "OS_S25_discovery_final_trial_counts", "SI",
        "split-size audit",
        "Discovery/final split sizes used by the frozen v2.3 design.",
        [p_repeats],
        lambda: trial_counts_plot(
            rep1,
            ["n_discovery", "n_final"],
            "Trial count"
        ),
    )

    if raw_repeats is not None:
        run_job(
            "OS_S26_raw_vs_controlled_time_confound", "SI",
            "confound-removal audit",
            "Residualization sharply reduces the raw time-state association.",
            [p_raw, p_repeats],
            lambda: raw_vs_controlled_plot(
                rep1, raw_repeats, "state_score_time_spearman",
                r"$|\rho|$ state score vs time", absolute=True
            ),
        )
        run_job(
            "OS_S27_raw_vs_controlled_running_confound", "SI",
            "confound-removal audit",
            "Residualization reduces raw locomotion differences between state groups.",
            [p_raw, p_repeats],
            lambda: raw_vs_controlled_plot(
                rep1, raw_repeats, "running_state_cohens_d",
                r"$|d|$ running difference", absolute=True
            ),
        )
        run_job(
            "OS_S28_raw_vs_controlled_stimulus_association", "SI",
            "confound-removal audit",
            "State-stimulus association remains small after confound control.",
            [p_raw, p_repeats],
            lambda: raw_vs_controlled_plot(
                rep1, raw_repeats, "state_stimulus_cramers_v",
                "State–stimulus Cramér's V", absolute=True
            ),
        )

    if real_repeats is not None:
        real_session_agg = aggregate_session_mean(
            real1,
            ["decoder_auc_home", "decoder_auc_cross", "decoder_auc_crossover",
             "delta_rho", "rho_within", "rho_cross",
             "topset_auc_crossover"]
        )
        run_job(
            "OS_S29_real_edge_home_vs_cross_auc", "SI",
            "real-edge boundary",
            "Matched real-edge control also exhibits state-dependent decoder transfer.",
            [p_real],
            lambda: paired_plot(
                real_session_agg, "decoder_auc_home", "decoder_auc_cross",
                "Home", "Cross", "Real-edge decoder AUC", ref=0.5,
                color_a=C_HOME, color_b=C_CROSS
            ),
        )
        run_job(
            "OS_S30_real_edge_Rdecoder_by_mouse", "SI",
            "real-edge boundary",
            "Mouse-level R_decoder in the matched real-edge condition.",
            [p_real],
            lambda: mouse_effect_plot(
                real_session_agg, "decoder_auc_crossover",
                r"Real-edge $R_{\mathrm{decoder}}$",
                ref=0.0, color=C_SUPPORT
            ),
        )
        run_job(
            "OS_S31_real_edge_Rrho_by_mouse", "SI",
            "real-edge boundary",
            "Mouse-level coefficient-landscape effect in the matched real-edge condition.",
            [p_real],
            lambda: mouse_effect_plot(
                real_session_agg, "delta_rho",
                r"Real-edge $R_{\rho}$",
                ref=0.0, color=C_SUPPORT
            ),
        )

    if boundary_df is not None:
        run_job(
            "OS_S32_primary_vs_real_edge_Rrho", "SI",
            "specificity boundary",
            "Primary-minus-real-edge R_rho does not justify an illusion-specific claim.",
            [p_boundary],
            lambda: boundary_paired_plot(
                boundary_df, "rho", r"$R_{\rho}$"
            ),
        )
        run_job(
            "OS_S33_primary_vs_real_edge_topset", "SI",
            "specificity boundary",
            "Top-set transfer is compared directly between primary and real-edge conditions.",
            [p_boundary],
            lambda: boundary_paired_plot(
                boundary_df, "topset", r"$R_{\mathrm{topset}}$"
            ),
        )

    def perm_boundary_rho():
        observed = None
        if isinstance(perm_summary, dict):
            observed = perm_summary.get("delta_rho", {}).get(
                "matched_observed_repeat0_group_mean"
            )
        vals = infer_null_vector(high_null, "delta_rho") if high_null is not None else np.array([])
        if vals.size < 20:
            files = list((closure_root / "null_sessions").glob("*high_precision_null.csv"))
            frames = []
            for p in files:
                try:
                    frames.append(pd.read_csv(p))
                except Exception:
                    pass
            if frames:
                alln = pd.concat(frames, ignore_index=True)
                pc = next((c for c in alln.columns if "perm" in c.lower() and "index" in c.lower()), None)
                vc = next((c for c in alln.columns if "delta_rho" in c and "null" in c), None)
                if pc and vc:
                    vals = alln.groupby(pc)[vc].mean().dropna().to_numpy(float)
                else:
                    vals = infer_null_vector(alln, "delta_rho")
        if observed is None:
            observed = 0.0
        return histogram_null(vals, float(observed),
                              r"Fixed-split null $R_{\rho}$",
                              color=C_LIGHT, observed_color=C_SUPPORT)

    run_job(
        "OS_S34_Rrho_high_precision_permutation_boundary", "SI",
        "formal permutation boundary",
        "The fixed-split permutation does not confirm R_rho, despite repeated-split mouse-level support.",
        [p_highnull, p_perm_summary],
        perm_boundary_rho,
    )

    def perm_boundary_jaccard():
        observed = None
        if isinstance(perm_summary, dict):
            observed = perm_summary.get("delta_jaccard", {}).get(
                "matched_observed_repeat0_group_mean"
            )
        vals = infer_null_vector(high_null, "delta_jaccard") if high_null is not None else np.array([])
        if vals.size < 20:
            files = list((closure_root / "null_sessions").glob("*high_precision_null.csv"))
            frames = []
            for p in files:
                try:
                    frames.append(pd.read_csv(p))
                except Exception:
                    pass
            if frames:
                alln = pd.concat(frames, ignore_index=True)
                pc = next((c for c in alln.columns if "perm" in c.lower() and "index" in c.lower()), None)
                vc = next((c for c in alln.columns if "delta_jaccard" in c and "null" in c), None)
                if pc and vc:
                    vals = alln.groupby(pc)[vc].mean().dropna().to_numpy(float)
                else:
                    vals = infer_null_vector(alln, "delta_jaccard")
        if observed is None:
            observed = 0.0
        return histogram_null(vals, float(observed),
                              r"Fixed-split null $\Delta$ Jaccard",
                              color=C_LIGHT, observed_color=C_SUPPORT)

    run_job(
        "OS_S35_delta_jaccard_high_precision_permutation_boundary", "SI",
        "formal permutation boundary",
        "Fixed-split permutation boundary for top-unit overlap.",
        [p_highnull, p_perm_summary],
        perm_boundary_jaccard,
    )

    if loo_df is not None:
        run_job(
            "OS_S36_leave_one_mouse_out_Rrho", "SI",
            "mouse-level robustness",
            "Repeated-split R_rho remains positive after removing any single mouse.",
            [p_loo],
            lambda: loo_plot(
                loo_df, "delta_rho",
                r"LOO mean $R_{\rho}$", color=C_SUPPORT
            ),
        )
        run_job(
            "OS_S37_leave_one_mouse_out_same_image", "SI",
            "mouse-level robustness",
            "Same-image state imprint remains above chance after removing any mouse.",
            [p_loo],
            lambda: loo_plot(
                loo_df, "same_image_state_auc_mean",
                "LOO mean AUC excess above chance", color=C_BIO
            ),
        )
        run_job(
            "OS_S38_leave_one_mouse_out_topset", "SI",
            "mouse-level robustness",
            "Top-set transfer remains directionally stable under leave-one-mouse-out.",
            [p_loo],
            lambda: loo_plot(
                loo_df, "topset_auc_crossover",
                r"LOO mean $R_{\mathrm{topset}}$", color=C_SUPPORT
            ),
        )

    if formal is not None and "paired_standardized_effect_dz" in formal.columns:
        def dz_forest():
            endpoints = [
                ("decoder_state_transfer", r"$R_{\mathrm{decoder}}$"),
                ("coefficient_landscape_reconfiguration", r"$R_{\rho}$"),
                ("top20_unitset_transfer_supportive", r"$R_{\mathrm{top20}}$"),
                ("top20_landscape_jaccard", r"$\Delta$ Jaccard"),
                ("activity_residual_landscape", "Activity residual"),
            ]
            rows = []
            for ep, lab in endpoints:
                g = formal[formal["endpoint"].astype(str).eq(ep)]
                if g.empty:
                    continue
                rows.append({
                    "endpoint": ep,
                    "label": lab,
                    "value": float(g.iloc[0]["paired_standardized_effect_dz"])
                })
            d = pd.DataFrame(rows)
            if d.empty:
                raise ValueError("No dz rows")
            fig, ax = fixed_figure("os_forest")
            y = np.arange(len(d))[::-1]
            ax.axvline(0, color=C_GRAY, lw=0.8, ls="--")
            ax.scatter(d["value"], y, s=24, color=C_COMPLEXITY)
            ax.set_yticks(y, d["label"])
            ax.set_xlabel("Paired standardized effect $d_z$")
            open_axes(ax)
            return fig, d
        run_job(
            "OS_S39_formal_standardized_effects_dz", "SI",
            "formal effect-size audit",
            "Standardized mouse-level effect sizes for the frozen endpoint family.",
            [p_formal],
            dz_forest,
        )

    # Repeated-split stability: session SD over repeats.
    def repeat_sd_plot(col, ylabel, color):
        d = rep1.groupby(["subject", "session"], as_index=False)[col].std()
        return mouse_effect_plot(d, col, ylabel, ref=0.0, color=color)

    run_job(
        "OS_S40_Rdecoder_repeat_sd_by_mouse", "SI",
        "split stability",
        "Across-repeat variability of R_decoder for each mouse/session.",
        [p_repeats],
        lambda: repeat_sd_plot(
            "decoder_auc_crossover",
            r"SD across repeats of $R_{\mathrm{decoder}}$", C_HOME
        ),
    )

    run_job(
        "OS_S41_Rrho_repeat_sd_by_mouse", "SI",
        "split stability",
        "Across-repeat variability of R_rho for each mouse/session.",
        [p_repeats],
        lambda: repeat_sd_plot(
            "delta_rho",
            r"SD across repeats of $R_{\rho}$", C_SUPPORT
        ),
    )

    # Exact-label balanced half sizes.
    def half_trial_min_plot():
        cols = ["n_half_trials_0A", "n_half_trials_0B", "n_half_trials_1A", "n_half_trials_1B"]
        d = aggregate_session_mean(rep1, cols)
        d["min_half_trials"] = d[cols].min(axis=1)
        return mouse_effect_plot(
            d, "min_half_trials",
            "Minimum split-half trials", ref=0.0, color=C_GRAY
        )

    run_job(
        "OS_S42_split_half_trial_support_by_mouse", "SI",
        "sampling audit",
        "Minimum exact-label-balanced split-half sample size per mouse/session.",
        [p_repeats],
        half_trial_min_plot,
    )

    # State imbalance magnitude.
    def state_imbalance_plot():
        d = aggregate_session_mean(rep1, ["state0_fraction"])
        d["state_imbalance"] = (d["state0_fraction"] - 0.5).abs()
        return mouse_effect_plot(
            d, "state_imbalance",
            "|State-0 fraction − 0.5|", ref=0.0, color=C_GRAY
        )

    run_job(
        "OS_S43_state_balance_deviation_by_mouse", "SI",
        "state-balance audit",
        "Magnitude of residual state-group imbalance after the frozen median-split procedure.",
        [p_repeats],
        state_imbalance_plot,
    )

    # Optional 30-shuffle nulls.
    if shuffle30 is not None:
        def shuffle30_hist(endpoint, xlabel, observed_col, color):
            vals = infer_null_vector(shuffle30, endpoint)
            obs = float(session[observed_col].mean())
            return histogram_null(vals, obs, xlabel, color=C_LIGHT, observed_color=color)

        run_job(
            "OS_S44_original_30shuffle_decoder_null", "SI",
            "legacy permutation support",
            "Original v2.3 30-shuffle null retained as a lower-precision precursor to FinalClosure.",
            [p_shuffle30],
            lambda: shuffle30_hist(
                "decoder_auc_crossover",
                r"30-shuffle null $R_{\mathrm{decoder}}$",
                "decoder_auc_crossover", C_HOME
            ),
        )
        run_job(
            "OS_S45_original_30shuffle_Rrho_null", "SI",
            "legacy permutation support",
            "Original v2.3 30-shuffle R_rho null retained as intermediate analysis support.",
            [p_shuffle30],
            lambda: shuffle30_hist(
                "delta_rho",
                r"30-shuffle null $R_{\rho}$",
                "delta_rho", C_SUPPORT
            ),
        )

    # Optional already-frozen inference tables.
    if top_budget is not None:
        # Try plotting any recognizable group-effect column.
        xcol = next((c for c in top_budget.columns if "fraction" in c.lower() or "budget" in c.lower()), None)
        ycol = next((c for c in top_budget.columns if "mean" in c.lower() and "crossover" in c.lower()), None)
        if xcol and ycol:
            run_job(
                "OS_S46_topset_budget_inference_summary", "SI",
                "budget inference table",
                "Pre-computed budget-sensitivity inference from the frozen v2.3 analysis.",
                [p_top_budget],
                lambda: line_by_group(
                    top_budget.assign(subject="group"),
                    xcol, ycol, "Top-unit fraction",
                    "Group transfer effect", groupcol="subject",
                    ref=0.0, highlight_x=0.20, color=C_SUPPORT
                ),
            )

    if reg_sens is not None:
        xcol = next((c for c in reg_sens.columns if c.lower() == "c" or "regular" in c.lower()), None)
        ycol = next((c for c in reg_sens.columns if "mean" in c.lower() and "rho" in c.lower()), None)
        if xcol and ycol:
            run_job(
                "OS_S47_regularization_inference_summary", "SI",
                "regularization inference table",
                "Pre-computed regularization-sensitivity inference from the frozen v2.3 analysis.",
                [p_reg_sens],
                lambda: line_by_group(
                    reg_sens.assign(subject="group"),
                    xcol, ycol, "Logistic C",
                    "Group landscape effect", groupcol="subject",
                    ref=0.0, highlight_x=1.0, color=C_SUPPORT
                ),
            )

    # -----------------------------------------------------------------
    # Write atlas metadata
    # -----------------------------------------------------------------
    cat = pd.DataFrame(catalog)
    cat.to_csv(outroot / "openscope_figure_catalog.csv", index=False, encoding="utf-8-sig")
    ps.figure_size_table().to_csv(outroot / "fixed_figure_sizes.csv", index=False, encoding="utf-8-sig")

    skip = pd.DataFrame(skipped)
    if skip.empty:
        skip = pd.DataFrame(columns=["figure_id", "tier", "reason", "traceback"])
    skip.to_csv(outroot / "openscope_skipped_figures.csv", index=False, encoding="utf-8-sig")

    source_manifest = {
        "v23_root": str(v23_root),
        "closure_root": str(closure_root),
        "plot_style_v4_loaded": HAVE_PLOT_STYLE_V4,
        "generated_main": int(((cat["tier"] == "MAIN") & cat["generated"]).sum()) if not cat.empty else 0,
        "generated_si": int(((cat["tier"] == "SI") & cat["generated"]).sum()) if not cat.empty else 0,
        "skipped": int((~cat["generated"]).sum()) if not cat.empty else 0,
        "frozen_interpretation": {
            "headline": (
                "A population code learned in one collective neural state generalizes "
                "preferentially within that state and degrades when transferred across states."
            ),
            "same_image_support": (
                "Identical visual inputs produce population responses carrying a strong "
                "signature of prestimulus collective state."
            ),
            "Rrho_boundary": (
                "R_rho is supportive across repeated splits and mice but does not survive "
                "the matched fixed-split high-precision permutation test."
            ),
            "specificity_boundary": (
                "Real-edge control prevents an illusion-specific claim; the defensible "
                "conclusion is a general collective-state-dependent cortical coding principle."
            ),
            "causal_boundary": "Predictive/functional state dependence is not causal leverage.",
        },
        "no_new_endpoints": True,
        "no_model_refit": True,
        "no_nwb_read": True,
    }
    (outroot / "openscope_figure_atlas_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    readme = f"""# OpenScope Manuscript Figure Atlas v1

Generated MAIN candidates: {source_manifest['generated_main']}
Generated SI/support: {source_manifest['generated_si']}
Skipped optional panels: {source_manifest['skipped']}

Rules:
- No new endpoint.
- No refitting.
- No NWB read.
- No figure title.
- 600-dpi PNG + PDF.
- Immutable semantic canvas sizes from plot_style_v4; bbox_inches=tight is forbidden.
- Each plotted panel has a matching CSV under `plot_data/`.

Interpretation hierarchy:
1. Headline: R_decoder.
2. Strong biological support: same-exact-image state imprint.
3. Mechanistic/supportive: repeated-split R_rho and residualization.
4. Formal boundary: fixed-split R_rho permutation is null.
5. Specificity boundary: real-edge control supports a general state-dependent coding principle, not illusion specificity.
"""
    (outroot / "README_OPEN_SCOPE_FIGURES.md").write_text(readme, encoding="utf-8")

    print()
    print("=" * 88)
    print("OpenScope manuscript figure atlas complete")
    print(f"MAIN generated : {source_manifest['generated_main']}")
    print(f"SI generated   : {source_manifest['generated_si']}")
    print(f"Skipped        : {source_manifest['skipped']}")
    print(f"Output         : {outroot}")
    print("=" * 88)


if __name__ == "__main__":
    main()
