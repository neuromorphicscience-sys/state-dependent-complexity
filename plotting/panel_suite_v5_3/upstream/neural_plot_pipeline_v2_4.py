#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
NeuralScience Plot Pipeline v2.4.3

Default project root:
    D:\Research\Neural Science

Default plot root:
    D:\Research\Neural Science\plot

Core goals:
1. Inventory all Stage 1 -> current-stage outputs without changing raw results.
2. Build a reproducible stage manifest.
3. Generate high-DPI PNG + exact plot-source CSV + metadata JSON.
4. Record source/style/script SHA256 hashes.
5. Discover existing analyze/analysis scripts and optionally run one explicitly.
6. Start with Stage 1; later stages can be added to the same framework.

This script deliberately does NOT blindly execute all analyze scripts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path.cwd()
DEFAULT_PLOT = DEFAULT_ROOT / "plot"
STYLE_FILENAME = "plot_style_complexity.py"

SUPPORTED_EXT = {
    ".csv", ".tsv", ".json", ".jsonl", ".md", ".txt",
    ".npy", ".npz", ".parquet", ".pkl", ".pickle",
    ".png", ".pdf", ".py", ".log"
}

SKIP_DIR_NAMES = {
    ".git", ".idea", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", "venv", ".venv", "env"
}

# Plot outputs should not be rediscovered as raw scientific inputs.
PLOT_SUBDIRS_TO_SKIP = {
    "_inventory", "_pipeline", "_analyzer_runs"
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def normalize_stage_token(text: str) -> Optional[str]:
    """
    Detect common stage labels from a path/string:
      stage1, stage_1, stage 1
      stage1.5, stage_1_5
      stage2a, stage_2b
      stage4f
    """
    s = text.lower().replace("\\", "/")

    patterns = [
        r"stage[\s_\-]*(\d+)\.(\d+)([a-z]?)",
        r"stage[\s_\-]*(\d+)[_\-](\d+)([a-z]?)",
        r"stage[\s_\-]*(\d+)([a-z])",
        r"stage[\s_\-]*(\d+)",
    ]
    for i, pat in enumerate(patterns):
        m = re.search(pat, s, flags=re.I)
        if not m:
            continue
        if i in (0, 1):
            major, minor, suffix = m.groups()
            suffix = suffix.upper() if suffix else ""
            return f"Stage{major}.{minor}{suffix}"
        if i == 2:
            major, suffix = m.groups()
            return f"Stage{major}{suffix.upper()}"
        major = m.group(1)
        return f"Stage{major}"
    return None


def stage_sort_key(stage: str):
    m = re.match(r"Stage(\d+)(?:\.(\d+))?([A-Z]?)$", stage or "")
    if not m:
        return (999, 999, "Z")
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    suffix = m.group(3) or ""
    return (major, minor, suffix)


def iter_project_files(root: Path, plot_root: Path) -> Iterable[Path]:
    root = root.resolve()
    plot_root = plot_root.resolve()

    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)

        # prune standard noise
        dirnames[:] = [
            name for name in dirnames
            if name not in SKIP_DIR_NAMES
        ]

        # Skip generated plot subtrees except the style file itself.
        if d == plot_root:
            dirnames[:] = [
                name for name in dirnames
                if name not in PLOT_SUBDIRS_TO_SKIP and not re.match(r"^Stage", name, re.I)
            ]
        elif plot_root in d.parents:
            continue

        for name in filenames:
            p = d / name
            if p.suffix.lower() in SUPPORTED_EXT:
                yield p


def inspect_csv_header(path: Path):
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep, nrows=5)
        cols = list(map(str, df.columns))
        return cols, len(df)
    except Exception:
        return [], None


def infer_legacy_stage(path: Path, root: Path, current_stage: Optional[str]) -> Optional[str]:
    """
    Map the original unversioned baseline layout to Stage1.

    The project began with:
        configs/stage1_baseline.json
        results/runs.csv
        results/summary_by_parameter.csv
        results/candidates.csv
        scripts/summarize.py
        scripts/run_sweep.py

    Stage1B and later moved to explicit results_stageXB names.
    This mapping is intentionally narrow so unrelated root files are not
    silently assigned to Stage1.
    """
    if current_stage:
        return current_stage

    rel = safe_rel(path, root).replace("\\\\", "/").replace("\\", "/").lower()

    legacy_stage1_paths = {
        "results/runs.csv",
        "results/summary_by_parameter.csv",
        "results/candidates.csv",
        "scripts/summarize.py",
        "scripts/run_sweep.py",
        "configs/smoke.json",
    }

    if rel in legacy_stage1_paths and (root / "configs" / "stage1_baseline.json").exists():
        return "Stage1"

    return current_stage


def build_inventory(root: Path, plot_root: Path) -> pd.DataFrame:
    rows = []
    for p in iter_project_files(root, plot_root):
        try:
            st = p.stat()
        except OSError:
            continue

        stage = normalize_stage_token(str(p))
        stage = infer_legacy_stage(p, root, stage)
        cols = []
        if p.suffix.lower() in {".csv", ".tsv"}:
            cols, _ = inspect_csv_header(p)

        rows.append({
            "stage": stage or "",
            "relative_path": safe_rel(p, root),
            "filename": p.name,
            "extension": p.suffix.lower(),
            "size_bytes": st.st_size,
            "modified_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
            "columns_preview": " | ".join(cols[:40]),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["_stage_key"] = df["stage"].map(lambda x: stage_sort_key(x) if x else (999,999,"Z"))
        df = df.sort_values(["_stage_key", "relative_path"]).drop(columns="_stage_key").reset_index(drop=True)
    return df


def discover_analyzers(root: Path, plot_root: Path) -> pd.DataFrame:
    rows = []
    for p in iter_project_files(root, plot_root):
        if p.suffix.lower() != ".py":
            continue
        n = p.name.lower()
        if not (
            "analy" in n
            or "analysis" in n
            or n.startswith("report")
            or "summar" in n
        ):
            continue
        stage = normalize_stage_token(str(p))
        rows.append({
            "stage": stage or "",
            "relative_path": safe_rel(p, root),
            "filename": p.name,
            "sha256": sha256_file(p),
        })
    return pd.DataFrame(rows)


def ensure_layout(plot_root: Path, stages: Iterable[str]):
    (plot_root / "_inventory").mkdir(parents=True, exist_ok=True)
    (plot_root / "_pipeline").mkdir(parents=True, exist_ok=True)
    (plot_root / "_analyzer_runs").mkdir(parents=True, exist_ok=True)

    for stage in stages:
        sdir = plot_root / stage
        for child in ["figures", "source_data", "metadata", "logs"]:
            (sdir / child).mkdir(parents=True, exist_ok=True)


def import_style(style_path: Path):
    if not style_path.exists():
        raise FileNotFoundError(
            f"Style file not found: {style_path}\n"
            f"Expected your existing {STYLE_FILENAME} in the plot directory."
        )
    spec = importlib.util.spec_from_file_location("plot_style_complexity", str(style_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def save_figure_bundle(
    fig,
    plot_df: pd.DataFrame,
    stage_dir: Path,
    stem: str,
    source_files: list[Path],
    style_path: Path,
    script_path: Path,
    dpi: int = 600,
):
    """
    Save:
      figures/<stem>.png
      source_data/<stem>.csv
      metadata/<stem>.json

    The CSV is the exact dataframe passed to the final plotting function.
    """
    png = stage_dir / "figures" / f"{stem}.png"
    pdf = stage_dir / "figures" / f"{stem}.pdf"
    csv_path = stage_dir / "source_data" / f"{stem}.csv"
    meta = stage_dir / "metadata" / f"{stem}.json"

    png.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)

    plot_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches=None)
    fig.savefig(pdf, facecolor="white", bbox_inches=None)
    plt.close(fig)

    source_records = []
    for p in source_files:
        if p.exists() and p.is_file():
            source_records.append({
                "path": str(p),
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            })

    payload = {
        "artifact_stem": stem,
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dpi": dpi,
        "png_path": str(png),
        "pdf_path": str(pdf),
        "source_data_csv": str(csv_path),
        "source_data_sha256": sha256_file(csv_path),
        "source_files": source_records,
        "style_path": str(style_path),
        "style_sha256": sha256_file(style_path),
        "plot_script": str(script_path),
        "plot_script_sha256": sha256_file(script_path),
        "n_plot_rows": int(len(plot_df)),
        "columns": list(map(str, plot_df.columns)),
    }
    write_json(meta, payload)
    return png, csv_path, meta


MODEL_ALIASES = [
    "model", "model_name", "neuron_model", "architecture", "method", "variant"
]
TASK_ALIASES = [
    "task", "task_name", "benchmark", "dataset", "problem"
]
METRIC_ALIASES = [
    "accuracy", "acc", "test_accuracy", "mean_accuracy",
    "score", "performance", "balanced_accuracy"
]
SEED_ALIASES = [
    "seed", "random_seed", "trial_seed", "replicate", "run_seed"
]


def pick_col(columns, aliases):
    cmap = {str(c).lower(): str(c) for c in columns}
    for a in aliases:
        if a.lower() in cmap:
            return cmap[a.lower()]
    # fuzzy fallback
    for c in columns:
        cl = str(c).lower()
        for a in aliases:
            if a.lower() in cl:
                return str(c)
    return None


def stage1_candidates(inventory: pd.DataFrame, root: Path) -> list[Path]:
    if inventory.empty:
        return []

    exact = inventory[inventory["stage"] == "Stage1"].copy()
    exact = exact[exact["extension"].isin([".csv", ".tsv"])]
    paths = [root / rp for rp in exact["relative_path"].tolist()]

    # Also accept explicit stage1 filename/path that did not parse because of unusual naming,
    # but avoid stage1.5 / stage_1_5.
    for _, r in inventory.iterrows():
        rp = str(r["relative_path"])
        low = rp.lower().replace("\\", "/")
        if r["extension"] not in [".csv", ".tsv"]:
            continue
        if "stage1" in low or "stage_1" in low or "stage-1" in low:
            if any(x in low for x in ["stage1.5", "stage_1_5", "stage-1-5"]):
                continue
            p = root / rp
            if p not in paths:
                paths.append(p)

    return paths


def score_stage1_table(path: Path) -> tuple[int, dict]:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep)
    except Exception:
        return -999, {}

    model = pick_col(df.columns, MODEL_ALIASES)
    task = pick_col(df.columns, TASK_ALIASES)
    metric = pick_col(df.columns, METRIC_ALIASES)
    seed = pick_col(df.columns, SEED_ALIASES)

    score = 0
    if model: score += 5
    if task: score += 4
    if metric: score += 6
    if seed: score += 1
    if len(df) >= 5: score += 1
    if len(df) >= 20: score += 1

    return score, {
        "df": df,
        "model": model,
        "task": task,
        "metric": metric,
        "seed": seed,
    }


def _aggregate_fraction_metric(df: pd.DataFrame, fraction_col: str, metric_col: str) -> pd.DataFrame:
    """
    Pandas-version-stable aggregation.

    Important:
    - use named aggregation so no stray ``index`` column is created;
    - give each metric its own n-column so multiple metric tables can be
      outer-merged without n_x/n_y collisions.
    """
    d = df[[fraction_col, metric_col]].copy()
    d[fraction_col] = pd.to_numeric(d[fraction_col], errors="coerce")
    d[metric_col] = pd.to_numeric(d[metric_col], errors="coerce")
    d = d.dropna(subset=[fraction_col, metric_col])

    g = (
        d.groupby(fraction_col, sort=True)[metric_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    g["std"] = g["std"].fillna(0.0)
    g["sem"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
    g["ci95"] = 1.96 * g["sem"]

    g = g.rename(columns={
        fraction_col: "complexity_fraction",
        "mean": f"{metric_col}_mean",
        "std": f"{metric_col}_std",
        "n": f"{metric_col}_n",
        "sem": f"{metric_col}_sem",
        "ci95": f"{metric_col}_ci95",
    })

    return (
        g.sort_values("complexity_fraction")
         .reset_index(drop=True)
    )


def _plot_stage1_fraction_metric(
    raw: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    stem: str,
    source: Path,
    stage_dir: Path,
    style_path: Path,
    sty,
    dpi: int,
):
    agg = _aggregate_fraction_metric(raw, "hh_fraction", metric_col)
    if agg.empty:
        return None

    mean_col = f"{metric_col}_mean"
    ci_col = f"{metric_col}_ci95"

    fig, ax = sty.create_standard_figure()
    x = agg["complexity_fraction"].to_numpy(dtype=float)
    y = agg[mean_col].to_numpy(dtype=float)
    ci = agg[ci_col].to_numpy(dtype=float)

    line_color = getattr(sty, "COLOR_COMPLEXITY_HIGH", "#C7654C")
    ax.plot(
        x, y,
        color=line_color,
        linewidth=3.0,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2.0,
        markeredgecolor=line_color,
    )
    ax.fill_between(
        x, y - ci, y + ci,
        color=line_color,
        alpha=0.16,
        linewidth=0,
    )

    sty.apply_standard_axis_settings(
        ax,
        xlabel="High-complexity fraction",
        ylabel=ylabel,
        label_fontsize=32,
        tick_labelsize=28,
    )

    if len(x):
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        pad = max((xmax - xmin) * 0.04, 0.005)
        ax.set_xlim(xmin - pad, xmax + pad)

    # Explicit semantic mapping while retaining the raw source field name.
    plot_df = agg.copy()
    plot_df["source_fraction_column"] = "hh_fraction"
    plot_df["semantic_fraction_name"] = "high-complexity fraction"
    plot_df["aggregation_scope"] = (
        "global Stage1 baseline sweep; aggregated across all other sampled "
        "coupling/connection-probability/noise conditions"
    )

    return save_figure_bundle(
        fig=fig,
        plot_df=plot_df,
        stage_dir=stage_dir,
        stem=stem,
        source_files=[source],
        style_path=style_path,
        script_path=Path(__file__),
        dpi=dpi,
    )


def plot_stage1(root: Path, plot_root: Path, inventory: pd.DataFrame, dpi: int):
    """
    Stage1 is the original unversioned baseline parameter sweep.

    Canonical raw source:
        results/runs.csv

    Existing summaries:
        results/summary_by_parameter.csv
        results/candidates.csv

    We do not rerun the simulation or summarize.py when these outputs already
    exist. Stage1 figures are historical/global sweep summaries, not final
    causal claims. They are archived so later manuscript selection can decide
    whether they belong in Main, Supplementary, or history-only material.
    """
    style_path = plot_root / STYLE_FILENAME
    sty = import_style(style_path)
    stage_dir = plot_root / "Stage1"
    ensure_layout(plot_root, ["Stage1"])

    raw_path = root / "results" / "runs.csv"
    summary_path = root / "results" / "summary_by_parameter.csv"
    candidates_path = root / "results" / "candidates.csv"
    baseline_cfg = root / "configs" / "stage1_baseline.json"

    report = [
        "# Stage 1 plotting report",
        "",
        "Stage 1 identified as the original unversioned baseline sweep.",
        "",
        f"- Raw runs: `{raw_path}`",
        f"- Parameter summary: `{summary_path}`",
        f"- Candidates: `{candidates_path}`",
        f"- Baseline config: `{baseline_cfg}`",
        "",
    ]

    if not raw_path.exists():
        report += [
            "ERROR: `results/runs.csv` was not found.",
            "No plot was generated and no simulation was rerun.",
        ]
        (stage_dir / "stage1_plotting_report.md").write_text("\n".join(report), encoding="utf-8")
        print(f"[Stage1] Missing canonical raw source: {raw_path}")
        return []

    raw = pd.read_csv(raw_path)
    required = {
        "hh_fraction",
        "mean_rate_hz",
        "synchrony_proxy",
        "dominant_frequency_hz",
        "rhythm_score",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        report += [
            f"ERROR: canonical raw table is missing required fields: {missing}",
            "No scientific plot was fabricated.",
        ]
        (stage_dir / "stage1_plotting_report.md").write_text("\n".join(report), encoding="utf-8")
        print("[Stage1] Required columns missing:", missing)
        return []

    # Canonical raw Stage1 export: exact source rows plus semantic alias.
    canonical_raw = raw.copy()
    canonical_raw["complexity_fraction"] = pd.to_numeric(
        canonical_raw["hh_fraction"], errors="coerce"
    )
    canonical_raw["source_fraction_column"] = "hh_fraction"
    canonical_raw["semantic_fraction_name"] = "high-complexity fraction"
    canonical_raw.to_csv(
        stage_dir / "source_data" / "stage1_canonical_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary2 = summary.copy()
        if "hh_fraction" in summary2.columns:
            summary2["complexity_fraction"] = pd.to_numeric(
                summary2["hh_fraction"], errors="coerce"
            )
        summary2["source_fraction_column"] = "hh_fraction"
        summary2["semantic_fraction_name"] = "high-complexity fraction"
        summary2.to_csv(
            stage_dir / "source_data" / "stage1_parameter_summary_existing.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if candidates_path.exists():
        cand = pd.read_csv(candidates_path)
        cand2 = cand.copy()
        if "hh_fraction" in cand2.columns:
            cand2["complexity_fraction"] = pd.to_numeric(
                cand2["hh_fraction"], errors="coerce"
            )
        cand2["source_fraction_column"] = "hh_fraction"
        cand2["semantic_fraction_name"] = "high-complexity fraction"
        cand2.to_csv(
            stage_dir / "source_data" / "stage1_candidates_existing.csv",
            index=False,
            encoding="utf-8-sig",
        )

    specs = [
        (
            "rhythm_score",
            "Rhythm score",
            "stage1_global_complexity_fraction_rhythm_score",
        ),
        (
            "mean_rate_hz",
            "Mean firing rate (Hz)",
            "stage1_global_complexity_fraction_mean_rate",
        ),
        (
            "synchrony_proxy",
            "Synchrony proxy",
            "stage1_global_complexity_fraction_synchrony",
        ),
        (
            "dominant_frequency_hz",
            "Dominant frequency (Hz)",
            "stage1_global_complexity_fraction_frequency",
        ),
    ]

    artifacts = []
    for metric, ylabel, stem in specs:
        out = _plot_stage1_fraction_metric(
            raw=raw,
            metric_col=metric,
            ylabel=ylabel,
            stem=stem,
            source=raw_path,
            stage_dir=stage_dir,
            style_path=style_path,
            sty=sty,
            dpi=dpi,
        )
        if out:
            artifacts.append(out)

    # Metric-by-fraction master table for exact reconstruction.
    master = None
    for metric, _, _ in specs:
        g = _aggregate_fraction_metric(raw, "hh_fraction", metric)
        keep = ["complexity_fraction"] + [c for c in g.columns if c != "complexity_fraction"]
        g = g[keep]
        if master is None:
            master = g
        else:
            overlap = (set(master.columns) & set(g.columns)) - {"complexity_fraction"}
            if overlap:
                raise RuntimeError(
                    "Unexpected duplicate columns before Stage1 master merge: "
                    + ", ".join(sorted(overlap))
                )
            master = master.merge(
                g,
                on="complexity_fraction",
                how="outer",
                validate="one_to_one",
            )

    if master is not None:
        # n columns are duplicated across metrics; retain them explicitly rather than guessing equality.
        master.to_csv(
            stage_dir / "source_data" / "stage1_global_fraction_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    report += [
        f"Raw rows loaded: {len(raw):,}",
        f"Generated figure bundles: {len(artifacts)}",
        "",
        "Generated plots:",
        "- rhythm score vs high-complexity fraction",
        "- mean firing rate vs high-complexity fraction",
        "- synchrony proxy vs high-complexity fraction",
        "- dominant frequency vs high-complexity fraction",
        "",
        "Important interpretation:",
        "These are GLOBAL Stage1 sweep summaries aggregated across all other sampled "
        "network parameters. They are intended for historical/project organization and "
        "initial visualization, not as isolated causal evidence for the final manuscript.",
        "",
        "Terminology:",
        "The raw source field `hh_fraction` is preserved in source data, but figures use "
        "`High-complexity fraction` as the scientific semantic label.",
        "",
        "No simulation or analyzer was rerun because the canonical raw runs and existing "
        "summary tables are already present.",
    ]
    (stage_dir / "stage1_plotting_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"[Stage1] Canonical raw rows: {len(raw):,}")
    print(f"[Stage1] Generated {len(artifacts)} high-DPI figure bundle(s).")
    print(f"[Stage1] Output: {stage_dir}")
    return artifacts



DISPLAY_LABELS = {
    # Stage1B windows
    "frequency_transition": "Frequency transition",
    "heterogeneity_optimum": "Heterogeneity optimum",
    "sparse_complexity": "Sparse complexity",

    # Frequency states
    "high_frequency": "High frequency",
    "high_intermediate": "High-intermediate",
    "intermediate": "Intermediate",
    "low_frequency": "Low frequency",
    "weak_or_irregular": "Weak / irregular",

    # Common placement methods
    "gpu_optimized": "Dynamics-aware optimized",
    "optimized": "Dynamics-aware optimized",
    "random": "Random",
    "high_degree": "High degree",
    "degree": "High degree",
    "spectral": "Spectral",
    "feedback_hub": "Feedback hub",
    "module_bridge": "Module bridge",
    "cycle_proxy": "Cycle proxy",
}


def display_label(value) -> str:
    s = str(value)
    return DISPLAY_LABELS.get(s, s.replace("_", " ").strip().title())


def _legend_above(ax, ncol=3, fontsize=16, y=1.015):
    """
    Put the legend in the reserved top margin, not over scientific data.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=max(1, int(ncol)),
        fontsize=fontsize,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
        handletextpad=0.55,
        borderaxespad=0.0,
    )


def _placement_color(sty, placement: str, index: int = 0) -> str:
    """
    Stable method colors. Unknown methods receive a muted deterministic palette.
    """
    key = str(placement)
    if hasattr(sty, "METHOD_COLORS") and key in sty.METHOD_COLORS:
        return sty.METHOD_COLORS[key]

    fallback = [
        getattr(sty, "COLOR_OPTIMIZED", "#168B8C"),
        getattr(sty, "COLOR_SPECTRAL", "#7568A9"),
        getattr(sty, "COLOR_HIGH_DEGREE", "#D49A3A"),
        getattr(sty, "COLOR_FEEDBACK", "#8A6D62"),
        getattr(sty, "COLOR_MODULE_BRIDGE", "#5F9473"),
        getattr(sty, "COLOR_CYCLE_PROXY", "#B67598"),
        getattr(sty, "COLOR_GREY_DARK", "#666666"),
    ]
    return fallback[index % len(fallback)]


def _topology_color(sty, topology: str, index: int = 0) -> str:
    key = str(topology)
    if hasattr(sty, "TOPOLOGY_COLORS") and key in sty.TOPOLOGY_COLORS:
        return sty.TOPOLOGY_COLORS[key]
    fallback = [
        getattr(sty, "COLOR_SCALE_FREE", "#3E6FA3"),
        getattr(sty, "COLOR_SMALL_WORLD", "#3D8B7A"),
        getattr(sty, "COLOR_MODULAR", "#8A6AA5"),
        getattr(sty, "COLOR_ERDOS_RENYI", "#C1833E"),
    ]
    return fallback[index % len(fallback)]


FREQUENCY_STATE_COLORS = {
    "high_frequency": "#4F789D",
    "high_intermediate": "#3D8B7A",
    "intermediate": "#C39A46",
    "low_frequency": "#A94F63",
    "weak_or_irregular": "#8174A8",
}


def _stage1b_prepare_fraction_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    needed = ["window", "hh_fraction", metric]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Stage1B table missing columns: {missing}")

    d = df[needed].copy()
    d["hh_fraction"] = pd.to_numeric(d["hh_fraction"], errors="coerce")
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d.dropna(subset=["hh_fraction", metric])

    g = (
        d.groupby(["window", "hh_fraction"], sort=True)[metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    g["std"] = g["std"].fillna(0.0)
    g["sem"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
    g["ci95"] = 1.96 * g["sem"]
    g = g.rename(columns={
        "hh_fraction": "complexity_fraction",
        "mean": f"{metric}_mean",
        "std": f"{metric}_std",
        "n": f"{metric}_n",
        "sem": f"{metric}_sem",
        "ci95": f"{metric}_ci95",
    })
    g["source_fraction_column"] = "hh_fraction"
    g["semantic_fraction_name"] = "high-complexity fraction"
    return g.sort_values(["window", "complexity_fraction"]).reset_index(drop=True)


def _stage1b_plot_metric_by_window(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    stem: str,
    source_files: list[Path],
    stage_dir: Path,
    style_path: Path,
    sty,
    dpi: int,
):
    mean_col = f"{metric}_mean"
    ci_col = f"{metric}_ci95"

    fig, ax = sty.create_standard_figure()

    state_colors = [
        getattr(sty, "COLOR_STATE_SPARSE", "#4E79A7"),
        getattr(sty, "COLOR_STATE_MID", "#B58A3B"),
        getattr(sty, "COLOR_STATE_DENSE", "#9E3F55"),
        getattr(sty, "COLOR_OPTIMIZED", "#168B8C"),
        getattr(sty, "COLOR_SPECTRAL", "#7568A9"),
        getattr(sty, "COLOR_HIGH_DEGREE", "#D49A3A"),
    ]

    for idx, (window, g) in enumerate(summary.groupby("window", sort=True)):
        g = g.sort_values("complexity_fraction")
        x = g["complexity_fraction"].to_numpy(dtype=float)
        y = g[mean_col].to_numpy(dtype=float)
        ci = g[ci_col].to_numpy(dtype=float)
        color = state_colors[idx % len(state_colors)]

        ax.plot(
            x, y,
            linewidth=2.8,
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.8,
            color=color,
            markeredgecolor=color,
            label=display_label(window),
        )
        ax.fill_between(
            x, y-ci, y+ci,
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    sty.apply_standard_axis_settings(
        ax,
        xlabel="High-complexity fraction",
        ylabel=ylabel,
        label_fontsize=32,
        tick_labelsize=27,
    )
    if summary["window"].nunique() > 1:
        _legend_above(
            ax,
            ncol=min(3, int(summary["window"].nunique())),
            fontsize=16,
            y=1.018,
        )

    return save_figure_bundle(
        fig=fig,
        plot_df=summary,
        stage_dir=stage_dir,
        stem=stem,
        source_files=source_files,
        style_path=style_path,
        script_path=Path(__file__),
        dpi=dpi,
    )


def _stage1b_plot_frequency_composition(
    comp: pd.DataFrame,
    stage_dir: Path,
    style_path: Path,
    sty,
    source_file: Path,
    dpi: int,
):
    required = {"window", "hh_fraction", "frequency_state", "fraction"}
    missing = sorted(required - set(comp.columns))
    if missing:
        raise ValueError(f"frequency_state_composition missing columns: {missing}")

    d = comp.copy()
    d["hh_fraction"] = pd.to_numeric(d["hh_fraction"], errors="coerce")
    d["fraction"] = pd.to_numeric(d["fraction"], errors="coerce")
    d = d.dropna(subset=["hh_fraction", "fraction", "frequency_state"])
    d = d.rename(columns={"hh_fraction": "complexity_fraction"})
    d["source_fraction_column"] = "hh_fraction"
    d["semantic_fraction_name"] = "high-complexity fraction"

    outputs = []

    for window, wdf in d.groupby("window", sort=True):
        pivot = (
            wdf.pivot_table(
                index="complexity_fraction",
                columns="frequency_state",
                values="fraction",
                aggfunc="sum",
                fill_value=0.0,
            )
            .sort_index()
        )

        fig, ax = sty.create_standard_figure()
        x = pivot.index.to_numpy(dtype=float)
        ys = [pivot[c].to_numpy(dtype=float) for c in pivot.columns]

        ax.stackplot(
            x,
            *ys,
            labels=[display_label(c) for c in pivot.columns],
            colors=[
                FREQUENCY_STATE_COLORS.get(
                    str(c),
                    getattr(sty, "COLOR_GREY", "#9A9A9A")
                )
                for c in pivot.columns
            ],
            alpha=0.90,
            linewidth=0,
        )

        sty.apply_standard_axis_settings(
            ax,
            xlabel="High-complexity fraction",
            ylabel="State fraction",
            label_fontsize=32,
            tick_labelsize=27,
        )
        ax.set_ylim(0, 1)
        _legend_above(
            ax,
            ncol=min(3, max(1, len(pivot.columns))),
            fontsize=14,
            y=1.018,
        )

        safe_window = re.sub(r"[^A-Za-z0-9_-]+", "_", str(window)).strip("_")
        stem = f"stage1b_{safe_window}_frequency_state_composition"

        source_df = (
            pivot.reset_index()
            .melt(
                id_vars="complexity_fraction",
                var_name="frequency_state",
                value_name="fraction",
            )
        )
        source_df["window"] = window
        source_df["source_fraction_column"] = "hh_fraction"
        source_df["semantic_fraction_name"] = "high-complexity fraction"

        outputs.append(
            save_figure_bundle(
                fig=fig,
                plot_df=source_df,
                stage_dir=stage_dir,
                stem=stem,
                source_files=[source_file],
                style_path=style_path,
                script_path=Path(__file__),
                dpi=dpi,
            )
        )

    return outputs


def plot_stage1b(root: Path, plot_root: Path, dpi: int):
    """
    Stage1B: emergence/transition-focused analysis.

    Uses existing analyzed outputs. Does not rerun simulation or analyzer.
    """
    style_path = plot_root / STYLE_FILENAME
    sty = import_style(style_path)
    stage_dir = plot_root / "Stage1B"
    ensure_layout(plot_root, ["Stage1B"])

    base = root / "results_stage1b"
    runs_path = base / "runs_stage1b.csv"
    summary_path = base / "stage1b_parameter_summary.csv"
    candidates_path = base / "stage1b_candidates.csv"
    comp_path = base / "frequency_state_composition.csv"
    cfg_path = root / "configs" / "stage1b.json"
    analyzer_path = root / "scripts" / "analyze_stage1b.py"

    required_files = [runs_path, summary_path, candidates_path, comp_path]
    missing = [p for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage1B required analyzed outputs are missing:\n" +
            "\n".join(str(p) for p in missing)
        )

    runs = pd.read_csv(runs_path)
    summary = pd.read_csv(summary_path)
    candidates = pd.read_csv(candidates_path)
    comp = pd.read_csv(comp_path)

    # Canonical archived copies with semantic alias.
    for name, df in [
        ("stage1b_canonical_runs.csv", runs),
        ("stage1b_parameter_summary_existing.csv", summary),
        ("stage1b_candidates_existing.csv", candidates),
        ("stage1b_frequency_state_composition_existing.csv", comp),
    ]:
        d = df.copy()
        if "hh_fraction" in d.columns:
            d["complexity_fraction"] = pd.to_numeric(d["hh_fraction"], errors="coerce")
            d["source_fraction_column"] = "hh_fraction"
            d["semantic_fraction_name"] = "high-complexity fraction"
        d.to_csv(stage_dir / "source_data" / name, index=False, encoding="utf-8-sig")

    # Stage1B's most useful derived quantities.
    metric_specs = [
        ("rhythm_score_mean", "Rhythm score", "stage1b_window_rhythm_score"),
        ("emergence_gain", "Emergence gain", "stage1b_window_emergence_gain"),
        ("complexity_efficiency", "Complexity efficiency", "stage1b_window_complexity_efficiency"),
        # These two diagnostics remain selected in ED02 and therefore need the
        # same reproducible PNG/PDF bundle as every other manuscript asset.
        ("rhythm_transition_strength", "Rhythm transition strength", "stage1b_window_rhythm_transition"),
        ("frequency_transition_strength", "Frequency transition strength", "stage1b_window_frequency_transition"),
    ]

    # Transition-strength fields are retained as diagnostics/source data rather than
    # main visual outputs because their definitions can explode when a normalization
    # denominator approaches zero.
    transition_diag_cols = [
        c for c in [
            "window", "hh_fraction",
            "rhythm_transition_strength",
            "frequency_transition_strength"
        ] if c in summary.columns
    ]
    if len(transition_diag_cols) >= 3:
        diag = summary[transition_diag_cols].copy()
        diag["complexity_fraction"] = pd.to_numeric(
            diag["hh_fraction"], errors="coerce"
        )
        diag.to_csv(
            stage_dir / "source_data" / "stage1b_transition_strength_diagnostic.csv",
            index=False,
            encoding="utf-8-sig",
        )

    outputs = []
    for metric, ylabel, stem in metric_specs:
        if metric not in summary.columns:
            continue

        # summary already contains one row per parameter point; aggregate by window/fraction.
        temp = summary[["window", "hh_fraction", metric]].copy()
        temp = temp.rename(columns={metric: metric.replace("_mean", "") if metric == "rhythm_score_mean" else metric})

        actual_metric = metric.replace("_mean", "") if metric == "rhythm_score_mean" else metric
        # For rhythm_score_mean, retain values but call aggregation helper with rhythm_score.
        if metric == "rhythm_score_mean":
            temp = temp.rename(columns={"rhythm_score": "rhythm_score_stage1b"})
            actual_metric = "rhythm_score_stage1b"

        agg = _stage1b_prepare_fraction_summary(temp, actual_metric)

        outputs.append(
            _stage1b_plot_metric_by_window(
                summary=agg,
                metric=actual_metric,
                ylabel=ylabel,
                stem=stem,
                source_files=[summary_path],
                stage_dir=stage_dir,
                style_path=style_path,
                sty=sty,
                dpi=dpi,
            )
        )

    outputs.extend(
        _stage1b_plot_frequency_composition(
            comp=comp,
            stage_dir=stage_dir,
            style_path=style_path,
            sty=sty,
            source_file=comp_path,
            dpi=dpi,
        )
    )

    # Candidate audit table: don't plot flags yet; just archive concise candidate subset.
    flag_cols = [c for c in candidates.columns if c.startswith("candidate_")]
    keep = [
        c for c in [
            "window", "hh_fraction", "coupling", "connection_prob", "noise_sigma",
            "rhythm_score_mean", "lif_baseline_score", "hh_baseline_score",
            "linear_mixture_expectation", "emergence_gain",
            "complexity_efficiency", "rhythm_transition_strength",
            "frequency_transition_strength"
        ] if c in candidates.columns
    ] + flag_cols
    candidate_audit = candidates[keep].copy()
    if "hh_fraction" in candidate_audit.columns:
        candidate_audit["complexity_fraction"] = pd.to_numeric(
            candidate_audit["hh_fraction"], errors="coerce"
        )
    candidate_audit.to_csv(
        stage_dir / "source_data" / "stage1b_candidate_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = [
        "# Stage 1B plotting report",
        "",
        f"Runs loaded: {len(runs):,}",
        f"Parameter-summary rows: {len(summary):,}",
        f"Candidate rows: {len(candidates):,}",
        f"Frequency-composition rows: {len(comp):,}",
        f"Generated figure bundles: {len(outputs)}",
        "",
        "Scientific role:",
        "Stage1B refines Stage1 phenomenology by explicitly quantifying deviation from "
        "the linear mixture expectation (emergence gain), efficiency per complexity budget, "
        "and rhythm/frequency transition strength across predefined parameter windows.",
        "",
        "Interpretation boundary:",
        "Stage1B is still a broad parameter-window analysis. It identifies candidate nonlinear "
        "regimes and transitions; it does not by itself establish the later causal separation "
        "between activity gain, synchrony modulation, and frequency control.",
        "",
        "No simulation or analyzer was rerun because analyzed Stage1B tables already exist.",
        f"Config: {cfg_path}",
        f"Analyzer: {analyzer_path}",
    ]
    (stage_dir / "stage1b_plotting_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(f"[Stage1B] Runs loaded: {len(runs):,}")
    print(f"[Stage1B] Generated {len(outputs)} high-DPI figure bundle(s).")
    print(f"[Stage1B] Output: {stage_dir}")
    return outputs



def _aggregate_stage2(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str,
) -> pd.DataFrame:
    cols = group_cols + [metric]
    d = df[cols].copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    if "hh_fraction_actual" in d.columns:
        d["hh_fraction_actual"] = pd.to_numeric(
            d["hh_fraction_actual"], errors="coerce"
        )
    d = d.dropna(subset=[metric])

    g = (
        d.groupby(group_cols, sort=True, dropna=False)[metric]
        .agg(mean="mean", std="std", n="count", median="median")
        .reset_index()
    )
    g["std"] = g["std"].fillna(0.0)
    g["sem"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
    g["ci95"] = 1.96 * g["sem"]
    return g


def _plot_stage2_method_lines(
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    stem: str,
    source_file: Path,
    stage_dir: Path,
    style_path: Path,
    sty,
    dpi: int,
):
    required = {"placement", "hh_fraction_actual", metric}
    if not required.issubset(summary_df.columns):
        return None

    agg = _aggregate_stage2(
        summary_df,
        ["placement", "hh_fraction_actual"],
        metric,
    )

    fig, ax = sty.create_standard_figure()
    placements = list(agg["placement"].dropna().astype(str).unique())

    # Put random first visually in grey; then structural methods.
    placements = sorted(
        placements,
        key=lambda p: (0 if p == "random" else 1, p)
    )

    for i, placement in enumerate(placements):
        g = agg[agg["placement"].astype(str) == placement].sort_values(
            "hh_fraction_actual"
        )
        x = g["hh_fraction_actual"].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)
        ci = g["ci95"].to_numpy(dtype=float)
        color = _placement_color(sty, placement, i)

        ls = "--" if placement == "random" else "-"
        lw = 2.0 if placement == "random" else 2.7
        alpha_fill = 0.06 if placement == "random" else 0.10

        ax.plot(
            x, y,
            linestyle=ls,
            linewidth=lw,
            marker="o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.6,
            color=color,
            markeredgecolor=color,
            label=display_label(placement),
        )
        ax.fill_between(
            x, y-ci, y+ci,
            color=color,
            alpha=alpha_fill,
            linewidth=0,
        )

    sty.apply_standard_axis_settings(
        ax,
        xlabel="High-complexity fraction",
        ylabel=ylabel,
        label_fontsize=32,
        tick_labelsize=26,
    )

    if metric == "placement_gain_vs_random":
        ax.axhline(
            0.0,
            color=getattr(sty, "COLOR_GREY_DARK", "#666666"),
            linewidth=1.3,
            linestyle=":",
            zorder=0,
        )

    _legend_above(
        ax,
        ncol=min(4, max(1, len(placements))),
        fontsize=13,
        y=1.018,
    )

    plot_df = agg.rename(
        columns={"hh_fraction_actual": "complexity_fraction"}
    ).copy()
    plot_df["source_fraction_column"] = "hh_fraction_actual"
    plot_df["semantic_fraction_name"] = "high-complexity fraction"

    return save_figure_bundle(
        fig=fig,
        plot_df=plot_df,
        stage_dir=stage_dir,
        stem=stem,
        source_files=[source_file],
        style_path=style_path,
        script_path=Path(__file__),
        dpi=dpi,
    )


def _plot_stage2_heatmap(
    summary_df: pd.DataFrame,
    row_col: str,
    metric: str,
    stem: str,
    source_file: Path,
    stage_dir: Path,
    style_path: Path,
    sty,
    dpi: int,
):
    required = {row_col, "placement", metric}
    if not required.issubset(summary_df.columns):
        return None

    tab = (
        summary_df.groupby([row_col, "placement"], dropna=False)[metric]
        .mean()
        .unstack("placement")
        .sort_index()
    )

    if tab.empty:
        return None

    fig, ax = sty.create_standard_figure()
    arr = tab.to_numpy(dtype=float)

    # Diverging scale centered at zero for gain-vs-random.
    if metric == "placement_gain_vs_random":
        finite = arr[np.isfinite(arr)]
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
        vmax = max(vmax, 1e-12)
        im = ax.imshow(
            arr,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
    else:
        im = ax.imshow(
            arr,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )

    ax.set_xticks(np.arange(len(tab.columns)))
    ax.set_xticklabels(
        [display_label(c) for c in tab.columns],
        rotation=28,
        ha="right",
        fontsize=19,
    )
    ax.set_yticks(np.arange(len(tab.index)))
    ax.set_yticklabels(
        [display_label(x) for x in tab.index],
        fontsize=22,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    ax.set_xlabel("Allocation rule", fontsize=30, labelpad=15)
    ax.set_ylabel(display_label(row_col), fontsize=30, labelpad=15)
    ax.tick_params(axis="both", which="major", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label(
        "Gain vs random" if metric == "placement_gain_vs_random" else display_label(metric),
        fontsize=22,
        labelpad=10,
    )

    # Numeric annotation only if the matrix is reasonably small.
    if arr.size <= 80:
        finite = arr[np.isfinite(arr)]
        scale = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                val = arr[r, c]
                if not np.isfinite(val):
                    continue
                color = "white" if abs(val) > 0.55 * scale else "black"
                ax.text(
                    c, r, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=14,
                    color=color,
                )

    source_df = (
        tab.reset_index()
        .melt(
            id_vars=row_col,
            var_name="placement",
            value_name=f"{metric}_mean",
        )
    )

    return save_figure_bundle(
        fig=fig,
        plot_df=source_df,
        stage_dir=stage_dir,
        stem=stem,
        source_files=[source_file],
        style_path=style_path,
        script_path=Path(__file__),
        dpi=dpi,
    )


def _plot_stage2_minimum_budget(
    minimal_df: pd.DataFrame,
    source_file: Path,
    stage_dir: Path,
    style_path: Path,
    sty,
    dpi: int,
):
    required = {"placement", "minimum_fraction_for_80pct_best"}
    if not required.issubset(minimal_df.columns):
        return None

    d = minimal_df[[
        c for c in [
            "regime", "topology", "placement",
            "minimum_fraction_for_80pct_best",
            "minimum_n_hh_for_80pct_best",
            "score_at_minimum",
            "condition_best_score",
        ] if c in minimal_df.columns
    ]].copy()

    d["minimum_fraction_for_80pct_best"] = pd.to_numeric(
        d["minimum_fraction_for_80pct_best"], errors="coerce"
    )
    d = d.dropna(subset=["minimum_fraction_for_80pct_best", "placement"])

    agg = (
        d.groupby("placement")["minimum_fraction_for_80pct_best"]
        .agg(mean="mean", std="std", median="median", n="count")
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    agg["sem"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
    agg["ci95"] = 1.96 * agg["sem"]
    agg = agg.sort_values("mean").reset_index(drop=True)

    fig, ax = sty.create_standard_figure()
    y = np.arange(len(agg))

    for i, row in agg.iterrows():
        placement = str(row["placement"])
        color = _placement_color(sty, placement, i)
        ax.errorbar(
            float(row["mean"]),
            i,
            xerr=float(row["ci95"]),
            fmt="o",
            markersize=9,
            markerfacecolor="white",
            markeredgewidth=2.0,
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [display_label(x) for x in agg["placement"]],
        fontsize=22,
    )
    ax.invert_yaxis()

    sty.apply_standard_axis_settings(
        ax,
        xlabel="Minimum high-complexity fraction for 80% of best",
        ylabel="Allocation rule",
        label_fontsize=29,
        tick_labelsize=23,
    )
    # apply_standard_axis_settings may reset y tick label size but not labels.
    ax.set_yticklabels(
        [display_label(x) for x in agg["placement"]],
        fontsize=22,
    )

    return save_figure_bundle(
        fig=fig,
        plot_df=agg,
        stage_dir=stage_dir,
        stem="stage2_minimum_complexity_for_80pct_best",
        source_files=[source_file],
        style_path=style_path,
        script_path=Path(__file__),
        dpi=dpi,
    )


def _plot_stage2_best_rule_map(
    summary_df: pd.DataFrame,
    source_file: Path,
    stage_dir: Path,
    style_path: Path,
    sty,
    dpi: int,
):
    """
    For each topology x regime x complexity fraction, identify the placement
    with the highest mean rhythm score. Plot winner fractions as a compact
    descriptive allocation atlas.
    """
    required = {
        "topology", "regime", "placement",
        "hh_fraction_actual", "rhythm_score_mean"
    }
    if not required.issubset(summary_df.columns):
        return None

    d = summary_df[list(required)].copy()
    d["rhythm_score_mean"] = pd.to_numeric(
        d["rhythm_score_mean"], errors="coerce"
    )
    d["hh_fraction_actual"] = pd.to_numeric(
        d["hh_fraction_actual"], errors="coerce"
    )
    d = d.dropna()

    means = (
        d.groupby(
            ["topology", "regime", "hh_fraction_actual", "placement"],
            as_index=False,
        )["rhythm_score_mean"].mean()
    )
    idx = means.groupby(
        ["topology", "regime", "hh_fraction_actual"]
    )["rhythm_score_mean"].idxmax()
    winners = means.loc[idx].copy()

    # Summarize how often each rule wins across budget points.
    atlas = (
        winners.groupby(["topology", "regime", "placement"])
        .size()
        .rename("wins")
        .reset_index()
    )
    totals = (
        winners.groupby(["topology", "regime"])
        .size()
        .rename("total")
        .reset_index()
    )
    atlas = atlas.merge(totals, on=["topology", "regime"], how="left")
    atlas["win_fraction"] = atlas["wins"] / atlas["total"]

    # One figure: rows = topology/regime pairs, columns = placement methods.
    atlas["row_label"] = (
        atlas["topology"].astype(str).map(display_label)
        + " | "
        + atlas["regime"].astype(str).map(display_label)
    )
    tab = atlas.pivot_table(
        index="row_label",
        columns="placement",
        values="win_fraction",
        aggfunc="sum",
        fill_value=0.0,
    )

    fig, ax = sty.create_standard_figure()
    im = ax.imshow(
        tab.to_numpy(dtype=float),
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(tab.columns)))
    ax.set_xticklabels(
        [display_label(x) for x in tab.columns],
        rotation=28,
        ha="right",
        fontsize=18,
    )
    ax.set_yticks(np.arange(len(tab.index)))
    ax.set_yticklabels(tab.index.tolist(), fontsize=17)
    ax.set_xlabel("Winning allocation rule", fontsize=28, labelpad=15)
    ax.set_ylabel("Network condition", fontsize=28, labelpad=15)
    ax.tick_params(axis="both", which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("Winner fraction across budgets", fontsize=21, labelpad=10)

    return save_figure_bundle(
        fig=fig,
        plot_df=atlas,
        stage_dir=stage_dir,
        stem="stage2_allocation_rule_winner_atlas",
        source_files=[source_file],
        style_path=style_path,
        script_path=Path(__file__),
        dpi=dpi,
    )


def plot_stage2(root: Path, plot_root: Path, dpi: int):
    """
    Stage2: transition from amount-only effects to allocation effects.

    Existing analyzed outputs are used directly; simulation/analyzer are not rerun.
    """
    style_path = plot_root / STYLE_FILENAME
    sty = import_style(style_path)
    stage_dir = plot_root / "Stage2"
    ensure_layout(plot_root, ["Stage2"])

    base = root / "results_stage2"
    runs_path = base / "runs_stage2.csv"
    summary_path = base / "stage2_parameter_summary.csv"
    candidates_path = base / "stage2_candidates.csv"
    minimal_path = base / "stage2_minimal_complexity.csv"
    cfg_path = root / "configs" / "stage2.json"
    analyzer_path = root / "scripts" / "analyze_stage2.py"

    required_files = [
        runs_path, summary_path, candidates_path, minimal_path
    ]
    missing = [p for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Stage2 required outputs missing:\n"
            + "\n".join(str(p) for p in missing)
        )

    # Large runs table is archived by path/hash in figure metadata where used;
    # processed plots use the already analyzed parameter summary.
    summary = pd.read_csv(summary_path)
    candidates = pd.read_csv(candidates_path)
    minimal = pd.read_csv(minimal_path)

    # Semantic aliases without destroying raw names.
    for name, df in [
        ("stage2_parameter_summary_existing.csv", summary),
        ("stage2_candidates_existing.csv", candidates),
        ("stage2_minimal_complexity_existing.csv", minimal),
    ]:
        d = df.copy()
        if "hh_fraction_actual" in d.columns:
            d["complexity_fraction"] = pd.to_numeric(
                d["hh_fraction_actual"], errors="coerce"
            )
            d["source_fraction_column"] = "hh_fraction_actual"
            d["semantic_fraction_name"] = "high-complexity fraction"
        d.to_csv(
            stage_dir / "source_data" / name,
            index=False,
            encoding="utf-8-sig",
        )

    outputs = []

    # A. Placement advantage over random -- most direct Stage2 allocation result.
    out = _plot_stage2_method_lines(
        summary,
        metric="placement_gain_vs_random",
        ylabel="Placement gain vs random",
        stem="stage2_placement_gain_vs_random",
        source_file=summary_path,
        stage_dir=stage_dir,
        style_path=style_path,
        sty=sty,
        dpi=dpi,
    )
    if out:
        outputs.append(out)

    # B. Fraction of the condition-specific optimum captured at each budget.
    out = _plot_stage2_method_lines(
        summary,
        metric="fraction_of_condition_best",
        ylabel="Fraction of condition best",
        stem="stage2_fraction_of_condition_best",
        source_file=summary_path,
        stage_dir=stage_dir,
        style_path=style_path,
        sty=sty,
        dpi=dpi,
    )
    if out:
        outputs.append(out)

    # C. Efficiency per high-complexity unit: useful but interpreted with budget caution.
    out = _plot_stage2_method_lines(
        summary,
        metric="gain_per_hh_neuron",
        ylabel="Gain per high-complexity unit",
        stem="stage2_gain_per_complex_unit",
        source_file=summary_path,
        stage_dir=stage_dir,
        style_path=style_path,
        sty=sty,
        dpi=dpi,
    )
    if out:
        outputs.append(out)

    # D/E. How placement advantage depends on topology and dynamical regime.
    for row_col, stem in [
        ("topology", "stage2_topology_by_allocation_gain"),
        ("regime", "stage2_regime_by_allocation_gain"),
    ]:
        out = _plot_stage2_heatmap(
            summary,
            row_col=row_col,
            metric="placement_gain_vs_random",
            stem=stem,
            source_file=summary_path,
            stage_dir=stage_dir,
            style_path=style_path,
            sty=sty,
            dpi=dpi,
        )
        if out:
            outputs.append(out)

    # F. Minimal budget needed for near-optimal performance.
    out = _plot_stage2_minimum_budget(
        minimal,
        source_file=minimal_path,
        stage_dir=stage_dir,
        style_path=style_path,
        sty=sty,
        dpi=dpi,
    )
    if out:
        outputs.append(out)

    # G. Descriptive atlas of which allocation rule wins across budgets.
    out = _plot_stage2_best_rule_map(
        summary,
        source_file=summary_path,
        stage_dir=stage_dir,
        style_path=style_path,
        sty=sty,
        dpi=dpi,
    )
    if out:
        outputs.append(out)

    # Candidate audit for later paper selection.
    keep = [
        c for c in [
            "regime", "topology", "placement",
            "n_hh", "hh_fraction_actual",
            "coupling", "connection_prob", "noise_sigma",
            "rhythm_score_mean", "rhythm_score_std",
            "placement_gain_vs_random", "placement_ratio_vs_random",
            "lif_only_score", "gain_over_lif",
            "gain_per_hh_neuron", "condition_best_score",
            "fraction_of_condition_best", "candidate_score"
        ] if c in candidates.columns
    ]
    cand_audit = candidates[keep].copy()
    if "hh_fraction_actual" in cand_audit.columns:
        cand_audit["complexity_fraction"] = pd.to_numeric(
            cand_audit["hh_fraction_actual"], errors="coerce"
        )
    cand_audit.to_csv(
        stage_dir / "source_data" / "stage2_candidate_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = [
        "# Stage 2 plotting report",
        "",
        f"Parameter-summary rows: {len(summary):,}",
        f"Candidate rows: {len(candidates):,}",
        f"Minimal-complexity rows: {len(minimal):,}",
        f"Generated figure bundles: {len(outputs)}",
        "",
        "Scientific role:",
        "Stage2 is treated as the first allocation stage: it asks whether a fixed "
        "high-complexity budget performs differently depending on placement, topology, "
        "and dynamical regime.",
        "",
        "Primary Stage2 quantities:",
        "- placement_gain_vs_random",
        "- fraction_of_condition_best",
        "- minimum_fraction_for_80pct_best",
        "",
        "Secondary quantity:",
        "- gain_per_hh_neuron (budget-sensitive; not interpreted alone near zero budget)",
        "",
        "No simulation/analyzer rerun. Existing analyzed Stage2 tables are used.",
        f"Config: {cfg_path}",
        f"Analyzer: {analyzer_path}",
    ]
    (stage_dir / "stage2_plotting_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(f"[Stage2] Parameter-summary rows: {len(summary):,}")
    print(f"[Stage2] Generated {len(outputs)} high-DPI figure bundle(s).")
    print(f"[Stage2] Output: {stage_dir}")
    return outputs



# =====================================================================
# Stage 3 — comprehensive scientific figure atlas
# =====================================================================

STAGE3_ALIASES = {
    "placement": [
        "placement", "allocation", "allocation_rule", "selector",
        "method", "placement_method"
    ],
    "topology": [
        "topology", "network_topology", "graph_type", "topology_name"
    ],
    "regime": [
        "regime", "state", "window", "operating_regime", "network_regime"
    ],
    "fraction": [
        "complexity_fraction", "hh_fraction_actual", "hh_fraction",
        "high_complexity_fraction", "fraction_hh"
    ],
    "n_complex": [
        "n_hh", "hh_count", "n_complex", "n_high_complexity",
        "complex_count", "high_complexity_count"
    ],
    "network_size": [
        "N", "n_neurons", "network_size", "n_total", "num_neurons"
    ],
    "score": [
        "rhythm_score_mean", "rhythm_score", "final_score",
        "validation_score", "score_mean", "score"
    ],
    "gain_random": [
        "placement_gain_vs_random", "gain_vs_random", "delta_vs_random",
        "score_gain_vs_random", "allocation_gain_vs_random",
        "targeted_gain_vs_random"
    ],
    "ratio_random": [
        "placement_ratio_vs_random", "ratio_vs_random",
        "score_ratio_vs_random", "targeted_ratio_vs_random"
    ],
    "emergence": [
        "emergence_gain", "nonlinear_gain", "synergy_gain",
        "gain_over_linear_mixture"
    ],
    "saving_fraction": [
        "saving_fraction", "complexity_saving_fraction",
        "hh_saving_fraction", "fraction_saved", "relative_saving",
        "saving_ratio", "complexity_saving"
    ],
    "random_required": [
        "random_n_hh", "n_hh_random", "random_required_n_hh",
        "random_complexity_required", "random_budget_required"
    ],
    "targeted_required": [
        "targeted_n_hh", "n_hh_targeted", "targeted_required_n_hh",
        "structured_n_hh", "optimized_n_hh", "targeted_budget_required"
    ],
    "minimum_fraction": [
        "minimum_fraction_for_80pct_best", "minimum_fraction",
        "min_fraction", "minimum_complexity_fraction"
    ],
    "tau_m": [
        "tau_m", "tau_m_ms", "membrane_tau", "membrane_time_constant",
        "membrane_time_constant_ms"
    ],
    "frequency": [
        "dominant_frequency_hz", "lfp_dominant_frequency_hz",
        "lfp_peak_frequency_hz", "dominant_freq_hz",
        "frequency_hz"
    ],
    "synchrony": [
        "synchrony_proxy", "synchrony", "population_synchrony",
        "sync_proxy"
    ],
    "rate": [
        "mean_rate_hz", "population_rate_hz", "firing_rate_hz",
        "mean_firing_rate_hz"
    ],
    "total_spikes": [
        "total_spikes", "population_spikes", "spike_count"
    ],
    "seed": [
        "seed", "graph_seed", "simulation_seed", "noise_seed", "random_seed"
    ],
}


def _stage3_find_col(df: pd.DataFrame, semantic: str):
    cmap = {str(c).lower(): str(c) for c in df.columns}
    for candidate in STAGE3_ALIASES.get(semantic, []):
        if candidate.lower() in cmap:
            return cmap[candidate.lower()]
    # Conservative fuzzy fallback for a few unambiguous semantic fields.
    for c in df.columns:
        cl = str(c).lower()
        for candidate in STAGE3_ALIASES.get(semantic, []):
            cc = candidate.lower()
            if len(cc) >= 8 and cc in cl:
                return str(c)
    return None


def _stage3_safe_read(path: Path):
    try:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    except Exception:
        return None


def _stage3_candidate_tables(root: Path, inventory: pd.DataFrame):
    """
    Discover Stage3 tables from the inventory without assuming fixed legacy names.
    """
    paths = []
    if not inventory.empty and "stage" in inventory.columns:
        sdf = inventory[
            (inventory["stage"].astype(str).str.upper() == "STAGE3")
            & (inventory["extension"].isin([".csv", ".tsv"]))
        ].copy()
        for rp in sdf["relative_path"].tolist():
            p = root / str(rp)
            if p.exists():
                paths.append(p)

    # Recovery scan for unusual stage3 naming that the stage parser may miss.
    for pattern in [
        "results_stage3/**/*.csv",
        "results_stage3/**/*.tsv",
        "results_stage3*.csv",
        "results_stage3*.tsv",
        "stage3/**/*.csv",
        "stage3/**/*.tsv",
    ]:
        for p in root.glob(pattern):
            if p.is_file() and p not in paths:
                paths.append(p)

    # Exclude generated plot products.
    paths = [
        p for p in paths
        if "plot" not in {x.lower() for x in p.parts}
    ]
    return sorted(set(paths), key=lambda x: str(x).lower())


def _stage3_table_profile(path: Path, df: pd.DataFrame):
    sem = {}
    for key in STAGE3_ALIASES:
        sem[key] = _stage3_find_col(df, key)

    score = 0
    # Scientific table priority score.
    if sem["placement"]: score += 4
    if sem["topology"]: score += 3
    if sem["regime"]: score += 2
    if sem["fraction"] or sem["n_complex"]: score += 4
    if sem["score"]: score += 4
    if sem["gain_random"]: score += 5
    if sem["saving_fraction"] or (sem["random_required"] and sem["targeted_required"]):
        score += 6
    if sem["emergence"]: score += 5
    if sem["tau_m"] and sem["frequency"]: score += 5
    if sem["synchrony"] or sem["rate"] or sem["total_spikes"]: score += 2

    roles = []
    if sem["placement"] and (sem["score"] or sem["gain_random"]):
        roles.append("allocation_performance")
    if sem["saving_fraction"] or (sem["random_required"] and sem["targeted_required"]):
        roles.append("complexity_saving")
    if sem["emergence"]:
        roles.append("emergence")
    if sem["tau_m"] and sem["frequency"]:
        roles.append("timescale_frequency")
    if (sem["fraction"] or sem["n_complex"]) and (sem["rate"] or sem["synchrony"] or sem["total_spikes"]):
        roles.append("complexity_mechanism")

    return {
        "path": path,
        "rows": len(df),
        "columns": len(df.columns),
        "score": score,
        "roles": roles,
        "semantic": sem,
    }


def _stage3_fraction_series(df: pd.DataFrame, profile: dict):
    frac = profile["semantic"]["fraction"]
    if frac and frac in df.columns:
        return pd.to_numeric(df[frac], errors="coerce"), frac

    nc = profile["semantic"]["n_complex"]
    ns = profile["semantic"]["network_size"]
    if nc and ns and nc in df.columns and ns in df.columns:
        n1 = pd.to_numeric(df[nc], errors="coerce")
        n2 = pd.to_numeric(df[ns], errors="coerce").replace(0, np.nan)
        return n1 / n2, f"{nc}/{ns}"

    return None, None


def _stage3_add_fraction_alias(df: pd.DataFrame, profile: dict):
    d = df.copy()
    s, source = _stage3_fraction_series(d, profile)
    if s is not None:
        d["complexity_fraction"] = s
        d["source_fraction_column"] = source
        d["semantic_fraction_name"] = "high-complexity fraction"
    return d


def _stage3_readable_context(value):
    return re.sub(r"\s+", " ", display_label(value)).strip()


def _stage3_context_slug(*values):
    txt = "_".join(str(v) for v in values if v is not None and str(v) != "")
    txt = re.sub(r"[^A-Za-z0-9_-]+", "_", txt).strip("_")
    return txt[:120] or "all"


def _stage3_register(catalog, stem, family, question, source, notes=""):
    catalog.append({
        "figure_stem": stem,
        "figure_family": family,
        "scientific_question": question,
        "source_file": str(source),
        "decision": "",           # intentionally blank until scientific review
        "main_figure": "",
        "si_figure": "",
        "notes": notes,
    })


def _stage3_aggregate_lines(
    df: pd.DataFrame,
    xcol: str,
    group_col: str,
    metric_col: str,
):
    d = df[[xcol, group_col, metric_col]].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[metric_col] = pd.to_numeric(d[metric_col], errors="coerce")
    d = d.dropna(subset=[xcol, metric_col, group_col])

    g = (
        d.groupby([group_col, xcol], sort=True, dropna=False)[metric_col]
        .agg(mean="mean", std="std", n="count", median="median")
        .reset_index()
    )
    g["std"] = g["std"].fillna(0.0)
    g["sem"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
    g["ci95"] = 1.96 * g["sem"]
    return g


def _stage3_plot_lines(
    df: pd.DataFrame,
    xcol: str,
    group_col: str,
    metric_col: str,
    xlabel: str,
    ylabel: str,
    stem: str,
    source: Path,
    stage_dir: Path,
    style_path: Path,
    sty,
    dpi: int,
    zero_line=False,
):
    agg = _stage3_aggregate_lines(df, xcol, group_col, metric_col)
    if agg.empty:
        return None

    fig, ax = sty.create_standard_figure()
    groups = list(agg[group_col].astype(str).drop_duplicates())

    for i, group in enumerate(groups):
        g = agg[agg[group_col].astype(str) == group].sort_values(xcol)
        x = g[xcol].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)
        ci = g["ci95"].to_numpy(dtype=float)

        # Placement methods use method colors; topologies use topology colors.
        if group_col.lower() in {"placement", "allocation", "allocation_rule", "method", "selector"}:
            color = _placement_color(sty, group, i)
            linestyle = "--" if str(group).lower() == "random" else "-"
        elif "topology" in group_col.lower():
            color = _topology_color(sty, group, i)
            linestyle = "-"
        else:
            palette = [
                getattr(sty, "COLOR_STATE_SPARSE", "#4E79A7"),
                getattr(sty, "COLOR_STATE_MID", "#B58A3B"),
                getattr(sty, "COLOR_STATE_DENSE", "#9E3F55"),
                getattr(sty, "COLOR_OPTIMIZED", "#168B8C"),
                getattr(sty, "COLOR_SPECTRAL", "#7568A9"),
            ]
            color = palette[i % len(palette)]
            linestyle = "-"

        ax.plot(
            x, y,
            color=color,
            linewidth=2.6,
            linestyle=linestyle,
            marker="o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            label=display_label(group),
        )
        ax.fill_between(
            x, y-ci, y+ci,
            color=color,
            alpha=0.09,
            linewidth=0,
        )

    sty.apply_standard_axis_settings(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        label_fontsize=30,
        tick_labelsize=25,
    )

    if zero_line:
        ax.axhline(
            0,
            color=getattr(sty, "COLOR_GREY_DARK", "#666666"),
            linewidth=1.2,
            linestyle=":",
            zorder=0,
        )

    _legend_above(
        ax,
        ncol=min(4, max(1, len(groups))),
        fontsize=13,
        y=1.018,
    )

    plot_df = agg.copy()
    return save_figure_bundle(
        fig=fig,
        plot_df=plot_df,
        stage_dir=stage_dir,
        stem=stem,
        source_files=[source],
        style_path=style_path,
        script_path=Path(__file__),
        dpi=dpi,
    )


def _stage3_plot_saving_scatter(
    df: pd.DataFrame,
    profile: dict,
    source: Path,
    stage_dir: Path,
    style_path: Path,
    sty,
    dpi: int,
):
    rc = profile["semantic"]["random_required"]
    tc = profile["semantic"]["targeted_required"]
    topo = profile["semantic"]["topology"]

    if not rc or not tc:
        return None

    keep = [rc, tc] + ([topo] if topo else [])
    d = df[keep].copy()
    d[rc] = pd.to_numeric(d[rc], errors="coerce")
    d[tc] = pd.to_numeric(d[tc], errors="coerce")
    d = d.dropna(subset=[rc, tc])

    if d.empty:
        return None

    fig, ax = sty.create_standard_figure()

    if topo:
        groups = list(d[topo].astype(str).drop_duplicates())
        for i, name in enumerate(groups):
            g = d[d[topo].astype(str) == name]
            ax.scatter(
                g[rc], g[tc],
                s=42,
                facecolors="white",
                edgecolors=_topology_color(sty, name, i),
                linewidths=1.5,
                alpha=0.82,
                label=display_label(name),
            )
        _legend_above(
            ax,
            ncol=min(4, len(groups)),
            fontsize=14,
            y=1.018,
        )
    else:
        ax.scatter(
            d[rc], d[tc],
            s=42,
            facecolors="white",
            edgecolors=getattr(sty, "COLOR_OPTIMIZED", "#168B8C"),
            linewidths=1.5,
            alpha=0.82,
        )

    finite = np.concatenate([
        d[rc].to_numpy(dtype=float),
        d[tc].to_numpy(dtype=float),
    ])
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lo, hi = float(np.min(finite)), float(np.max(finite))
        pad = max((hi-lo)*0.04, 0.5)
        ax.plot(
            [lo-pad, hi+pad],
            [lo-pad, hi+pad],
            linestyle="--",
            linewidth=1.5,
            color=getattr(sty, "COLOR_GREY", "#9A9A9A"),
        )
        ax.set_xlim(lo-pad, hi+pad)
        ax.set_ylim(lo-pad, hi+pad)

    sty.apply_standard_axis_settings(
        ax,
        xlabel="Complex units required: random allocation",
        ylabel="Complex units required: targeted allocation",
        label_fontsize=28,
        tick_labelsize=24,
    )

    out_df = d.rename(columns={
        rc: "random_complex_units_required",
        tc: "targeted_complex_units_required",
    })
    return save_figure_bundle(
        fig=fig,
        plot_df=out_df,
        stage_dir=stage_dir,
        stem="stage3_targeted_vs_random_complexity_requirement",
        source_files=[source],
        style_path=style_path,
        script_path=Path(__file__),
        dpi=dpi,
    )


def _stage3_compute_saving(df: pd.DataFrame, profile: dict):
    sf = profile["semantic"]["saving_fraction"]
    rc = profile["semantic"]["random_required"]
    tc = profile["semantic"]["targeted_required"]

    if sf:
        vals = pd.to_numeric(df[sf], errors="coerce")
        return vals, sf

    if rc and tc:
        r = pd.to_numeric(df[rc], errors="coerce").replace(0, np.nan)
        t = pd.to_numeric(df[tc], errors="coerce")
        return (r - t) / r, f"derived:({rc}-{tc})/{rc}"

    return None, None


def _stage3_plot_saving_by_topology(
    df: pd.DataFrame,
    profile: dict,
    source: Path,
    stage_dir: Path,
    style_path: Path,
    sty,
    dpi: int,
):
    topo = profile["semantic"]["topology"]
    values, source_field = _stage3_compute_saving(df, profile)
    if values is None:
        return []

    d = pd.DataFrame({"complexity_saving_fraction": values})
    if topo:
        d["topology"] = df[topo].astype(str)
    else:
        d["topology"] = "All"
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    if d.empty:
        return []

    outputs = []

    # A. Distribution by topology.
    order = list(d.groupby("topology")["complexity_saving_fraction"].median()
                 .sort_values(ascending=False).index)

    fig, ax = sty.create_standard_figure()
    arrays = [
        d.loc[d["topology"] == name, "complexity_saving_fraction"].to_numpy()
        for name in order
    ]
    bp = ax.boxplot(
        arrays,
        positions=np.arange(len(order)),
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 2.0, "color": "#222222"},
        whiskerprops={"linewidth": 1.4},
        capprops={"linewidth": 1.4},
        boxprops={"linewidth": 1.4},
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_topology_color(sty, order[i], i))
        patch.set_alpha(0.48)

    # Jittered raw points with deterministic offsets.
    rng = np.random.default_rng(12345)
    for i, name in enumerate(order):
        vals = d.loc[d["topology"] == name, "complexity_saving_fraction"].to_numpy()
        if len(vals) > 250:
            vals = vals[np.linspace(0, len(vals)-1, 250).astype(int)]
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=12,
            color=_topology_color(sty, name, i),
            alpha=0.28,
            edgecolors="none",
        )

    ax.axhline(
        0, linestyle=":", linewidth=1.3,
        color=getattr(sty, "COLOR_GREY_DARK", "#666666")
    )
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([display_label(x) for x in order], fontsize=22)
    sty.apply_standard_axis_settings(
        ax,
        xlabel="Network topology",
        ylabel="Complexity saving fraction",
        label_fontsize=30,
        tick_labelsize=24,
    )
    ax.set_xticklabels([display_label(x) for x in order], fontsize=22)

    outputs.append(
        save_figure_bundle(
            fig=fig,
            plot_df=d.assign(source_saving_field=source_field),
            stage_dir=stage_dir,
            stem="stage3_complexity_saving_by_topology",
            source_files=[source],
            style_path=style_path,
            script_path=Path(__file__),
            dpi=dpi,
        )
    )

    # B. Positive-saving rate + median, compact point plot.
    agg = (
        d.groupby("topology")["complexity_saving_fraction"]
        .agg(
            median="median",
            mean="mean",
            n="count",
            positive_fraction=lambda s: float(np.mean(s > 0)),
        )
        .reset_index()
        .sort_values("positive_fraction", ascending=False)
    )

    fig, ax = sty.create_standard_figure()
    x = np.arange(len(agg))
    for i, row in agg.reset_index(drop=True).iterrows():
        color = _topology_color(sty, str(row["topology"]), i)
        ax.plot(
            [i, i],
            [0, float(row["positive_fraction"])],
            color=color,
            linewidth=2.0,
            alpha=0.6,
        )
        ax.scatter(
            [i],
            [float(row["positive_fraction"])],
            s=90,
            facecolors="white",
            edgecolors=color,
            linewidths=2.0,
            zorder=3,
        )
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_label(v) for v in agg["topology"]],
        fontsize=21,
    )
    sty.apply_standard_axis_settings(
        ax,
        xlabel="Network topology",
        ylabel="Fraction with positive complexity saving",
        label_fontsize=29,
        tick_labelsize=23,
    )
    ax.set_xticklabels(
        [display_label(v) for v in agg["topology"]],
        fontsize=21,
    )

    outputs.append(
        save_figure_bundle(
            fig=fig,
            plot_df=agg,
            stage_dir=stage_dir,
            stem="stage3_positive_saving_fraction_by_topology",
            source_files=[source],
            style_path=style_path,
            script_path=Path(__file__),
            dpi=dpi,
        )
    )
    return outputs


def _stage3_plot_metric_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    metric_col: str,
    stem: str,
    source: Path,
    stage_dir: Path,
    style_path: Path,
    sty,
    dpi: int,
    center_zero=True,
):
    d = df[[row_col, col_col, metric_col]].copy()
    d[metric_col] = pd.to_numeric(d[metric_col], errors="coerce")
    d = d.dropna(subset=[row_col, col_col, metric_col])
    if d.empty:
        return None

    tab = (
        d.groupby([row_col, col_col])[metric_col]
        .mean()
        .unstack(col_col)
        .sort_index()
    )
    if tab.empty:
        return None

    fig, ax = sty.create_standard_figure()
    arr = tab.to_numpy(dtype=float)

    if center_zero:
        finite = arr[np.isfinite(arr)]
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
        vmax = max(vmax, 1e-12)
        im = ax.imshow(
            arr, aspect="auto", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, interpolation="nearest"
        )
    else:
        im = ax.imshow(
            arr, aspect="auto", cmap="viridis",
            interpolation="nearest"
        )

    ax.set_xticks(np.arange(len(tab.columns)))
    ax.set_xticklabels(
        [display_label(x) for x in tab.columns],
        rotation=25, ha="right", fontsize=17,
    )
    ax.set_yticks(np.arange(len(tab.index)))
    ax.set_yticklabels(
        [display_label(x) for x in tab.index],
        fontsize=20,
    )
    ax.set_xlabel(display_label(col_col), fontsize=28, labelpad=14)
    ax.set_ylabel(display_label(row_col), fontsize=28, labelpad=14)
    ax.tick_params(axis="both", which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.ax.tick_params(labelsize=17)
    cbar.set_label(display_label(metric_col), fontsize=20, labelpad=8)

    if arr.size <= 100:
        finite = arr[np.isfinite(arr)]
        scale = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                val = arr[r, c]
                if not np.isfinite(val):
                    continue
                txt_color = "white" if abs(val) > 0.55*scale else "black"
                ax.text(
                    c, r, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=12, color=txt_color,
                )

    source_df = (
        tab.reset_index()
        .melt(id_vars=row_col, var_name=col_col, value_name=f"{metric_col}_mean")
    )
    return save_figure_bundle(
        fig=fig,
        plot_df=source_df,
        stage_dir=stage_dir,
        stem=stem,
        source_files=[source],
        style_path=style_path,
        script_path=Path(__file__),
        dpi=dpi,
    )


def _stage3_plot_timescale_frequency(
    df: pd.DataFrame,
    profile: dict,
    source: Path,
    stage_dir: Path,
    style_path: Path,
    sty,
    dpi: int,
):
    tau = profile["semantic"]["tau_m"]
    freq = profile["semantic"]["frequency"]
    if not tau or not freq:
        return None

    d = df[[tau, freq]].copy()
    d[tau] = pd.to_numeric(d[tau], errors="coerce")
    d[freq] = pd.to_numeric(d[freq], errors="coerce")
    d = d.dropna()
    d = d[d[tau] > 0]
    if len(d) < 4:
        return None

    # Collapse exact tau duplicates; this is descriptive, not a causal regression test.
    agg = (
        d.groupby(tau)[freq]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values(tau)
    )
    agg["std"] = agg["std"].fillna(0)
    agg["sem"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
    agg["ci95"] = 1.96 * agg["sem"]

    fig, ax = sty.create_standard_figure()
    color = getattr(sty, "COLOR_OPTIMIZED", "#168B8C")
    x = agg[tau].to_numpy(dtype=float)
    y = agg["mean"].to_numpy(dtype=float)
    ci = agg["ci95"].to_numpy(dtype=float)

    ax.plot(
        x, y,
        color=color, linewidth=2.8,
        marker="o", markersize=7,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.8,
    )
    ax.fill_between(x, y-ci, y+ci, color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log")

    sty.apply_standard_axis_settings(
        ax,
        xlabel="Membrane timescale",
        ylabel="Dominant LFP frequency (Hz)",
        label_fontsize=29,
        tick_labelsize=24,
    )

    out = agg.rename(columns={tau: "membrane_timescale", freq: "frequency"})
    return save_figure_bundle(
        fig=fig,
        plot_df=out,
        stage_dir=stage_dir,
        stem="stage3_membrane_timescale_frequency",
        source_files=[source],
        style_path=style_path,
        script_path=Path(__file__),
        dpi=dpi,
    )


def _stage3_plot_complexity_mechanism(
    df: pd.DataFrame,
    profile: dict,
    source: Path,
    stage_dir: Path,
    style_path: Path,
    sty,
    dpi: int,
):
    d = _stage3_add_fraction_alias(df, profile)
    if "complexity_fraction" not in d.columns:
        return []

    outputs = []
    metric_specs = []
    if profile["semantic"]["rate"]:
        metric_specs.append(
            (profile["semantic"]["rate"], "Mean firing rate (Hz)", "stage3_complexity_activity_gain")
        )
    if profile["semantic"]["synchrony"]:
        metric_specs.append(
            (profile["semantic"]["synchrony"], "Synchrony", "stage3_complexity_synchrony")
        )
    if profile["semantic"]["total_spikes"]:
        metric_specs.append(
            (profile["semantic"]["total_spikes"], "Total spikes", "stage3_complexity_total_spikes")
        )

    for metric, ylabel, stem in metric_specs:
        g = (
            d[["complexity_fraction", metric]]
            .assign(**{metric: pd.to_numeric(d[metric], errors="coerce")})
            .dropna()
            .groupby("complexity_fraction")[metric]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
            .sort_values("complexity_fraction")
        )
        if g.empty:
            continue
        g["std"] = g["std"].fillna(0)
        g["sem"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
        g["ci95"] = 1.96 * g["sem"]

        fig, ax = sty.create_standard_figure()
        color = getattr(sty, "COLOR_COMPLEXITY_HIGH", "#C7654C")
        x = g["complexity_fraction"].to_numpy(float)
        y = g["mean"].to_numpy(float)
        ci = g["ci95"].to_numpy(float)

        ax.plot(
            x, y, color=color, linewidth=2.8,
            marker="o", markersize=7,
            markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=1.8
        )
        ax.fill_between(x, y-ci, y+ci, color=color, alpha=0.12, linewidth=0)

        sty.apply_standard_axis_settings(
            ax,
            xlabel="High-complexity fraction",
            ylabel=ylabel,
            label_fontsize=30,
            tick_labelsize=25,
        )
        outputs.append(
            save_figure_bundle(
                fig=fig,
                plot_df=g,
                stage_dir=stage_dir,
                stem=stem,
                source_files=[source],
                style_path=style_path,
                script_path=Path(__file__),
                dpi=dpi,
            )
        )
    return outputs


def plot_stage3(
    root: Path,
    plot_root: Path,
    inventory: pd.DataFrame,
    dpi: int,
):
    """
    Stage3 comprehensive scientific atlas.

    Design principle:
    - generate all scientifically interpretable, non-redundant panels first;
    - do NOT decide Main vs SI in code;
    - record every panel in stage3_figure_catalog.csv with blank decision fields;
    - use existing result/analyzed tables only; never rerun simulation automatically.
    """
    style_path = plot_root / STYLE_FILENAME
    sty = import_style(style_path)
    stage_dir = plot_root / "Stage3"
    ensure_layout(plot_root, ["Stage3"])
    (stage_dir / "review").mkdir(parents=True, exist_ok=True)

    table_paths = _stage3_candidate_tables(root, inventory)
    profiles = []
    for p in table_paths:
        df = _stage3_safe_read(p)
        if df is None:
            profiles.append({
                "path": p,
                "rows": "",
                "columns": "",
                "score": -1,
                "roles": ["read_failed"],
                "semantic": {},
            })
            continue
        profiles.append(_stage3_table_profile(p, df))

    # Table audit.
    audit_rows = []
    for pr in profiles:
        audit_rows.append({
            "relative_path": safe_rel(pr["path"], root),
            "rows": pr.get("rows", ""),
            "columns": pr.get("columns", ""),
            "scientific_score": pr.get("score", ""),
            "roles": " | ".join(pr.get("roles", [])),
            "semantic_columns": json.dumps(
                pr.get("semantic", {}),
                ensure_ascii=False
            ),
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(
        stage_dir / "review" / "stage3_table_catalog.csv",
        index=False,
        encoding="utf-8-sig",
    )

    catalog = []
    generated = set()

    # Archive all recognized Stage3 tables without altering originals.
    for pr in profiles:
        p = pr["path"]
        df = _stage3_safe_read(p)
        if df is None:
            continue
        archived = _stage3_add_fraction_alias(df, pr)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem)[:100]
        archived.to_csv(
            stage_dir / "source_data" / f"stage3_archive_{safe_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # -------------------------
    # 1. Allocation/performance
    # -------------------------
    allocation_profiles = [
        pr for pr in profiles
        if "allocation_performance" in pr.get("roles", [])
    ]
    # Highest-score analyzed table first; still allow multiple distinct sources.
    allocation_profiles = sorted(
        allocation_profiles, key=lambda x: (-x["score"], str(x["path"]))
    )

    for pr in allocation_profiles:
        p = pr["path"]
        df = _stage3_safe_read(p)
        if df is None:
            continue
        df = _stage3_add_fraction_alias(df, pr)

        placement = pr["semantic"]["placement"]
        topology = pr["semantic"]["topology"]
        regime = pr["semantic"]["regime"]
        score_col = pr["semantic"]["score"]
        gain_col = pr["semantic"]["gain_random"]

        xcol = "complexity_fraction" if "complexity_fraction" in df.columns else pr["semantic"]["n_complex"]
        xlabel = "High-complexity fraction" if xcol == "complexity_fraction" else "Number of high-complexity units"
        if not xcol or not placement:
            continue

        # Per topology/regime contexts: these are the comprehensive atlas panels.
        context_cols = [c for c in [topology, regime] if c]
        if context_cols:
            contexts = df[context_cols].drop_duplicates()
            # Prevent accidental explosion from malformed high-cardinality 'regime' fields.
            if len(contexts) <= 40:
                for _, crow in contexts.iterrows():
                    mask = np.ones(len(df), dtype=bool)
                    vals = []
                    for c in context_cols:
                        mask &= (df[c].astype(str) == str(crow[c]))
                        vals.append(crow[c])
                    sub = df.loc[mask].copy()
                    slug = _stage3_context_slug(*vals)
                    human = " / ".join(_stage3_readable_context(v) for v in vals)

                    if score_col:
                        stem = f"stage3_score_vs_budget_{slug}"
                        if stem not in generated:
                            out = _stage3_plot_lines(
                                sub, xcol, placement, score_col,
                                xlabel, "Collective-dynamics score",
                                stem, p, stage_dir, style_path, sty, dpi,
                                zero_line=False,
                            )
                            if out:
                                generated.add(stem)
                                _stage3_register(
                                    catalog, stem, "allocation_performance",
                                    f"How does allocation rule change collective performance at fixed complexity budget in {human}?",
                                    p,
                                )

                    if gain_col:
                        stem = f"stage3_gain_vs_random_{slug}"
                        if stem not in generated:
                            out = _stage3_plot_lines(
                                sub, xcol, placement, gain_col,
                                xlabel, "Allocation gain vs random",
                                stem, p, stage_dir, style_path, sty, dpi,
                                zero_line=True,
                            )
                            if out:
                                generated.add(stem)
                                _stage3_register(
                                    catalog, stem, "allocation_gain",
                                    f"When does targeted allocation outperform random placement in {human}?",
                                    p,
                                )

        # Global topology × placement heatmap for gain.
        if topology and placement and gain_col:
            stem = "stage3_topology_by_allocation_gain"
            if stem not in generated:
                out = _stage3_plot_metric_heatmap(
                    df, topology, placement, gain_col, stem, p,
                    stage_dir, style_path, sty, dpi, center_zero=True
                )
                if out:
                    generated.add(stem)
                    _stage3_register(
                        catalog, stem, "allocation_gain",
                        "How topology-selective is the benefit of targeted complexity allocation?",
                        p,
                    )

        # Regime × placement heatmap.
        if regime and placement and gain_col:
            stem = "stage3_regime_by_allocation_gain"
            if stem not in generated:
                out = _stage3_plot_metric_heatmap(
                    df, regime, placement, gain_col, stem, p,
                    stage_dir, style_path, sty, dpi, center_zero=True
                )
                if out:
                    generated.add(stem)
                    _stage3_register(
                        catalog, stem, "allocation_gain",
                        "How does collective operating regime change the value of structural allocation rules?",
                        p,
                    )

        # Only use the strongest allocation table for global plots to avoid duplicate panels.
        if generated:
            break

    # -------------------------
    # 2. Complexity saving
    # -------------------------
    saving_profiles = [
        pr for pr in profiles
        if "complexity_saving" in pr.get("roles", [])
    ]
    saving_profiles = sorted(
        saving_profiles, key=lambda x: (-x["score"], str(x["path"]))
    )
    if saving_profiles:
        pr = saving_profiles[0]
        p = pr["path"]
        df = _stage3_safe_read(p)
        if df is not None:
            out = _stage3_plot_saving_scatter(
                df, pr, p, stage_dir, style_path, sty, dpi
            )
            if out:
                stem = "stage3_targeted_vs_random_complexity_requirement"
                generated.add(stem)
                _stage3_register(
                    catalog, stem, "complexity_saving",
                    "Can targeted allocation reach the same dynamical target with fewer high-complexity units than random placement?",
                    p,
                )

            saving_outs = _stage3_plot_saving_by_topology(
                df, pr, p, stage_dir, style_path, sty, dpi
            )
            stems = [
                "stage3_complexity_saving_by_topology",
                "stage3_positive_saving_fraction_by_topology",
            ]
            questions = [
                "How strongly does topology determine the amount of complexity saved by targeted allocation?",
                "In what fraction of conditions does targeted allocation actually save complexity, and how topology-dependent is that probability?",
            ]
            for stem, q, out_item in zip(stems, questions, saving_outs):
                generated.add(stem)
                _stage3_register(
                    catalog, stem, "complexity_saving", q, p
                )

    # -------------------------
    # 3. Emergence/nonlinearity
    # -------------------------
    emergence_profiles = [
        pr for pr in profiles
        if "emergence" in pr.get("roles", [])
    ]
    emergence_profiles = sorted(
        emergence_profiles, key=lambda x: (-x["score"], str(x["path"]))
    )
    if emergence_profiles:
        pr = emergence_profiles[0]
        p = pr["path"]
        df = _stage3_safe_read(p)
        if df is not None:
            df = _stage3_add_fraction_alias(df, pr)
            em = pr["semantic"]["emergence"]
            topo = pr["semantic"]["topology"]
            placement = pr["semantic"]["placement"]

            if "complexity_fraction" in df.columns and topo:
                stem = "stage3_emergence_vs_budget_by_topology"
                out = _stage3_plot_lines(
                    df, "complexity_fraction", topo, em,
                    "High-complexity fraction", "Emergence gain",
                    stem, p, stage_dir, style_path, sty, dpi,
                    zero_line=True,
                )
                if out:
                    generated.add(stem)
                    _stage3_register(
                        catalog, stem, "emergence",
                        "Does nonlinear collective benefit persist across topologies and complexity budgets?",
                        p,
                    )
            elif "complexity_fraction" in df.columns and placement:
                stem = "stage3_emergence_vs_budget_by_allocation"
                out = _stage3_plot_lines(
                    df, "complexity_fraction", placement, em,
                    "High-complexity fraction", "Emergence gain",
                    stem, p, stage_dir, style_path, sty, dpi,
                    zero_line=True,
                )
                if out:
                    generated.add(stem)
                    _stage3_register(
                        catalog, stem, "emergence",
                        "How does nonlinear emergence vary with complexity allocation?",
                        p,
                    )

            # Positive emergence fraction summary.
            d = df[[em] + ([topo] if topo else [])].copy()
            d[em] = pd.to_numeric(d[em], errors="coerce")
            d = d.dropna(subset=[em])
            if not d.empty:
                if topo:
                    agg = (
                        d.groupby(topo)[em]
                        .agg(
                            mean="mean",
                            median="median",
                            n="count",
                            positive_fraction=lambda s: float(np.mean(s > 0)),
                        )
                        .reset_index()
                        .rename(columns={topo: "topology"})
                    )
                else:
                    agg = pd.DataFrame([{
                        "topology": "All",
                        "mean": d[em].mean(),
                        "median": d[em].median(),
                        "n": len(d),
                        "positive_fraction": float(np.mean(d[em] > 0)),
                    }])

                fig, ax = sty.create_standard_figure()
                x = np.arange(len(agg))
                for i, row in agg.iterrows():
                    color = _topology_color(sty, str(row["topology"]), i)
                    ax.scatter(
                        [i], [row["positive_fraction"]],
                        s=100, facecolors="white",
                        edgecolors=color, linewidths=2.0,
                    )
                ax.set_ylim(0, 1.02)
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [display_label(v) for v in agg["topology"]],
                    fontsize=21
                )
                sty.apply_standard_axis_settings(
                    ax,
                    xlabel="Network topology",
                    ylabel="Fraction with positive emergence",
                    label_fontsize=29,
                    tick_labelsize=23,
                )
                ax.set_xticklabels(
                    [display_label(v) for v in agg["topology"]],
                    fontsize=21
                )
                stem = "stage3_positive_emergence_fraction"
                save_figure_bundle(
                    fig=fig,
                    plot_df=agg,
                    stage_dir=stage_dir,
                    stem=stem,
                    source_files=[p],
                    style_path=style_path,
                    script_path=Path(__file__),
                    dpi=dpi,
                )
                generated.add(stem)
                _stage3_register(
                    catalog, stem, "emergence",
                    "How often does heterogeneous cellular complexity produce performance beyond the additive expectation?",
                    p,
                )

    # -------------------------
    # 4. Mechanistic / timescale support
    # -------------------------
    for pr in sorted(profiles, key=lambda x: (-x["score"], str(x["path"]))):
        p = pr["path"]
        df = _stage3_safe_read(p)
        if df is None:
            continue

        if "timescale_frequency" in pr.get("roles", []):
            stem = "stage3_membrane_timescale_frequency"
            if stem not in generated:
                out = _stage3_plot_timescale_frequency(
                    df, pr, p, stage_dir, style_path, sty, dpi
                )
                if out:
                    generated.add(stem)
                    _stage3_register(
                        catalog, stem, "mechanism",
                        "Is the accessible population frequency governed primarily by intrinsic membrane timescale rather than complexity amount?",
                        p,
                        "Mechanistic support; frequency should be interpreted using the validated LFP-band measure.",
                    )

        if "complexity_mechanism" in pr.get("roles", []):
            outs = _stage3_plot_complexity_mechanism(
                df, pr, p, stage_dir, style_path, sty, dpi
            )
            for stem, question in [
                (
                    "stage3_complexity_activity_gain",
                    "How does increasing cellular complexity modulate population activity gain?"
                ),
                (
                    "stage3_complexity_synchrony",
                    "How does increasing cellular complexity reshape population synchrony?"
                ),
                (
                    "stage3_complexity_total_spikes",
                    "How does the complexity budget alter total population spiking output?"
                ),
            ]:
                if stem not in generated and outs:
                    # Some outputs may not contain all stems, but registration is harmless only
                    # if the corresponding PNG exists.
                    if (stage_dir / "figures" / f"{stem}.png").exists():
                        generated.add(stem)
                        _stage3_register(
                            catalog, stem, "mechanism", question, p
                        )

    # -------------------------
    # 5. Catalog + review sheet
    # -------------------------
    catalog_df = pd.DataFrame(catalog)
    if catalog_df.empty:
        catalog_df = pd.DataFrame(columns=[
            "figure_stem", "figure_family", "scientific_question",
            "source_file", "decision", "main_figure", "si_figure", "notes"
        ])
    catalog_df.to_csv(
        stage_dir / "review" / "stage3_figure_catalog.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = [
        "# Stage 3 comprehensive figure-atlas report",
        "",
        f"Stage3 candidate tables discovered: {len(table_paths)}",
        f"Readable/profiler tables: {sum(1 for p in profiles if p.get('score', -1) >= 0)}",
        f"Scientific figure bundles generated: {len(catalog_df)}",
        "",
        "Policy:",
        "- No Main/SI decision is made automatically.",
        "- Every generated panel has 600-dpi PNG + source-data CSV + metadata JSON.",
        "- `review/stage3_figure_catalog.csv` is the decision sheet for later manuscript triage.",
        "- Existing simulations and analyzers are not rerun automatically.",
        "",
        "Interpretation safeguards:",
        "- Complexity-saving plots are preferred when the underlying table directly provides",
        "  saving or matched random/targeted requirements; otherwise they are not fabricated.",
        "- Transition or ratio metrics with unstable near-zero denominators are not generated",
        "  unless they have a directly interpretable finite scale.",
        "- Frequency mechanism panels should use validated LFP-band frequency fields where",
        "  available; full-spectrum spike-contaminated peaks should remain diagnostic only.",
        "",
        "Next step:",
        "Review the complete Stage3 atlas visually, then fill the `decision` column with",
        "`MAIN`, `SI`, `DIAGNOSTIC`, or `DROP` according to the final scientific narrative.",
    ]
    (stage_dir / "stage3_plotting_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(f"[Stage3] Candidate tables: {len(table_paths)}")
    print(f"[Stage3] Generated figure bundles: {len(catalog_df)}")
    print(f"[Stage3] Review catalog: {stage_dir / 'review' / 'stage3_figure_catalog.csv'}")
    print(f"[Stage3] Output: {stage_dir}")
    return catalog_df


def run_analyzer(root: Path, plot_root: Path, script: Path, extra_args: list[str]):
    """
    Explicit-only analyzer execution.
    Runs from project root and captures stdout/stderr.
    Does not infer or invent arguments.
    """
    script = script.resolve()
    if not script.exists():
        raise FileNotFoundError(script)
    if script.suffix.lower() != ".py":
        raise ValueError("Analyzer must be a .py script")

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = plot_root / "_analyzer_runs" / f"{script.stem}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(script)] + extra_args
    env = os.environ.copy()
    env["NEURALSCIENCE_PLOT_OUTPUT"] = str(plot_root)

    proc = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    (run_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (run_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8", errors="replace")
    write_json(run_dir / "run_metadata.json", {
        "command": cmd,
        "cwd": str(root),
        "returncode": proc.returncode,
        "script_sha256": sha256_file(script),
        "started_via": "NeuralScience_PlotPipeline_v1",
    })

    print(f"[Analyzer] return code: {proc.returncode}")
    print(f"[Analyzer] logs: {run_dir}")
    return proc.returncode


def write_stage_manifests(inventory: pd.DataFrame, plot_root: Path):
    if inventory.empty:
        return
    stages = sorted(
        [x for x in inventory["stage"].dropna().unique().tolist() if x],
        key=stage_sort_key,
    )
    ensure_layout(plot_root, stages)
    for stage in stages:
        sdf = inventory[inventory["stage"] == stage].copy()
        sdf.to_csv(plot_root / stage / f"{stage}_manifest.csv", index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser(description="NeuralScience reproducible plotting pipeline")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--plot-root", default=str(DEFAULT_PLOT))
    ap.add_argument("--dpi", type=int, default=600)

    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inventory", help="Scan project and build project/stage manifests")
    sub.add_parser("stage1", help="Inventory + generate conservative Stage 1 plot bundles")
    sub.add_parser("stage1b", help="Generate Stage 1B emergence/transition plot bundles")
    sub.add_parser("stage2", help="Generate Stage 2 complexity-allocation plot bundles")
    sub.add_parser("stage3", help="Generate comprehensive Stage 3 scientific figure atlas")
    sub.add_parser("list-analyzers", help="Inventory and list discovered analyze/analysis scripts")

    rp = sub.add_parser("run-analyzer", help="Explicitly run one existing analyzer and capture logs")
    rp.add_argument("script", help="Analyzer .py path (absolute or relative to project root)")
    rp.add_argument("analyzer_args", nargs=argparse.REMAINDER)

    args = ap.parse_args()

    root = Path(args.root)
    plot_root = Path(args.plot_root)
    style_path = plot_root / STYLE_FILENAME

    if not root.exists():
        raise SystemExit(f"Project root not found: {root}")
    plot_root.mkdir(parents=True, exist_ok=True)

    if args.cmd != "run-analyzer" and not style_path.exists():
        print(f"[WARN] Style file not found yet: {style_path}")
        print("Inventory can still run, but plotting requires plot_style_complexity.py.")

    inventory = build_inventory(root, plot_root)
    ensure_layout(plot_root, [])
    inv_path = plot_root / "_inventory" / "project_inventory.csv"
    inventory.to_csv(inv_path, index=False, encoding="utf-8-sig")
    write_stage_manifests(inventory, plot_root)

    analyzers = discover_analyzers(root, plot_root)
    ana_path = plot_root / "_inventory" / "analyzers.csv"
    analyzers.to_csv(ana_path, index=False, encoding="utf-8-sig")

    if args.cmd == "inventory":
        print(f"[OK] Inventory: {inv_path}")
        print(f"[OK] Analyzers: {ana_path}")
        if not inventory.empty:
            counts = inventory.groupby("stage").size()
            print("\nDetected stage-tagged files:")
            for stage, n in counts.items():
                if stage:
                    print(f"  {stage:12s} {int(n):5d}")
        return

    if args.cmd == "list-analyzers":
        print(analyzers.to_string(index=False) if not analyzers.empty else "No analyzer scripts discovered.")
        print(f"\nSaved: {ana_path}")
        return

    if args.cmd == "stage1":
        if not style_path.exists():
            raise SystemExit(f"Cannot plot Stage1: missing {style_path}")
        plot_stage1(root, plot_root, inventory, args.dpi)
        print(f"[OK] Stage1 directory: {plot_root / 'Stage1'}")
        return

    if args.cmd == "stage1b":
        if not style_path.exists():
            raise SystemExit(f"Cannot plot Stage1B: missing {style_path}")
        plot_stage1b(root, plot_root, args.dpi)
        print(f"[OK] Stage1B directory: {plot_root / 'Stage1B'}")
        return

    if args.cmd == "stage2":
        if not style_path.exists():
            raise SystemExit(f"Cannot plot Stage2: missing {style_path}")
        plot_stage2(root, plot_root, args.dpi)
        print(f"[OK] Stage2 directory: {plot_root / 'Stage2'}")
        return

    if args.cmd == "stage3":
        if not style_path.exists():
            raise SystemExit(f"Cannot plot Stage3: missing {style_path}")
        plot_stage3(root, plot_root, inventory, args.dpi)
        print(f"[OK] Stage3 directory: {plot_root / 'Stage3'}")
        return

    if args.cmd == "run-analyzer":
        p = Path(args.script)
        if not p.is_absolute():
            p = root / p
        rc = run_analyzer(root, plot_root, p, args.analyzer_args)
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
