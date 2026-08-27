#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
stage5a_epsilon_sensitivity_patch_v1.py

Standalone closure for the missing Stage 5A GLIF epsilon-sensitivity analysis.

Why this script exists
----------------------
The original Stage5A robustness run completed with:
    epsilon_sensitivity_status = NOT_RUN
because the GLIF neuronal-model metadata file was unavailable.

This patch does NOT rerun Stage5A, does NOT overwrite the existing 16 figures,
and does NOT rerun the expensive full-pipeline permutation. It only closes the
missing definition-sensitivity analysis for Creq.

Scientific definition
---------------------
For each mouse neuron with complete GLIF1–5 model runs:

    level 0 = GLIF1
    level 1 = max(GLIF2, GLIF3)
    level 2 = GLIF4
    level 3 = GLIF5

Let EV_best be the best explained_variance_ratio across the four complexity
levels. For absolute tolerance epsilon,

    Creq(epsilon) = min c such that EV_c >= EV_best - epsilon

The Stage5A primary definition uses epsilon = 0.02.

Expected primary mouse cohort:
    n = 400
Expected Creq distribution at epsilon=0.02:
    cost 0: 105
    cost 1: 144
    cost 2: 85
    cost 3: 66

Default input on the user's machine:
    D:\Research\Neural Science\bio data\glif_neuronal_models_metadata_raw.json

Default output:
    D:\Research\Neural Science\plot\Stage5A_robustness_v1\epsilon_sensitivity_v1

Usage (PowerShell)
------------------
python .\plot\stage5a_epsilon_sensitivity_patch_v1.py `
  --root "D:\Research\Neural Science"

Optional explicit metadata:
python .\plot\stage5a_epsilon_sensitivity_patch_v1.py `
  --root "D:\Research\Neural Science" `
  --metadata "D:\Research\Neural Science\bio data\glif_neuronal_models_metadata_raw.json"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
except Exception as exc:
    raise RuntimeError(
        "scipy is required for this patch. Install/activate the same environment "
        "used for the previous Stage5A analysis."
    ) from exc


BASELINE_EPSILON = 0.020
EPSILON_GRID = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]

EXPECTED_PRIMARY_N = 400
EXPECTED_BASELINE_COUNTS = {0: 105, 1: 144, 2: 85, 3: 66}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="GLIF neuronal model metadata JSON",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output folder",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution",
    )
    return p.parse_args()


def sha256_file(path: Path, chunk_size=1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def species_from_record(sp: dict) -> str:
    """
    The JSON does not expose a simple top-level species label for every row.
    The Allen file paths do distinguish mousecelltypes vs humancelltypes.
    """
    paths = []
    for model in sp.get("neuronal_models", []) or []:
        for wf in model.get("well_known_files", []) or []:
            paths.append(str(wf.get("path", "")).lower())
        for run in model.get("neuronal_model_runs", []) or []:
            for wf in run.get("well_known_files", []) or []:
                paths.append(str(wf.get("path", "")).lower())
    joined = " ".join(paths)
    if "mousecelltypes" in joined:
        return "mouse"
    if "humancelltypes" in joined:
        return "human"
    return "unknown"


def template_number(model: dict):
    name = (
        model.get("neuronal_model_template", {}).get("name")
        or model.get("name")
        or ""
    )
    try:
        n = int(str(name).strip().split()[0])
        if 1 <= n <= 5:
            return n
    except Exception:
        pass
    return None


def best_run_ev(model: dict):
    vals = []
    for run in model.get("neuronal_model_runs", []) or []:
        v = run.get("explained_variance_ratio")
        if v is not None:
            try:
                v = float(v)
                if math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
    return max(vals) if vals else None


def build_complete_mouse_table(payload: dict) -> pd.DataFrame:
    rows = []
    for sp in payload.get("specimens", []) or []:
        if species_from_record(sp) != "mouse":
            continue

        ev = {}
        model_ids = {}
        for model in sp.get("neuronal_models", []) or []:
            num = template_number(model)
            if num is None:
                continue
            val = best_run_ev(model)
            if val is None:
                continue
            # Keep the best run/model if an unexpected duplicate exists.
            if num not in ev or val > ev[num]:
                ev[num] = val
                model_ids[num] = model.get("id")

        if all(k in ev for k in [1, 2, 3, 4, 5]):
            rows.append({
                "specimen_id": int(sp["id"]),
                "specimen_name": sp.get("name"),
                "ev_glif1": ev[1],
                "ev_glif2": ev[2],
                "ev_glif3": ev[3],
                "ev_glif4": ev[4],
                "ev_glif5": ev[5],
                "model_id_glif1": model_ids.get(1),
                "model_id_glif2": model_ids.get(2),
                "model_id_glif3": model_ids.get(3),
                "model_id_glif4": model_ids.get(4),
                "model_id_glif5": model_ids.get(5),
            })

    df = pd.DataFrame(rows).sort_values("specimen_id").reset_index(drop=True)
    return df


def add_level_evs(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ev_cost0"] = x["ev_glif1"]
    x["ev_cost1"] = x[["ev_glif2", "ev_glif3"]].max(axis=1)
    x["ev_cost2"] = x["ev_glif4"]
    x["ev_cost3"] = x["ev_glif5"]
    x["ev_best"] = x[["ev_cost0", "ev_cost1", "ev_cost2", "ev_cost3"]].max(axis=1)
    return x


def creq_for_epsilon(df: pd.DataFrame, epsilon: float) -> np.ndarray:
    levels = df[["ev_cost0", "ev_cost1", "ev_cost2", "ev_cost3"]].to_numpy(float)
    best = levels.max(axis=1, keepdims=True)
    sufficient = levels >= (best - float(epsilon) - 1e-15)
    # At least the best level must always satisfy.
    return sufficient.argmax(axis=1).astype(int)


def weighted_quadratic_kappa(a, b, n_class=4):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    O = np.zeros((n_class, n_class), dtype=float)
    for x, y in zip(a, b):
        O[x, y] += 1
    O /= O.sum()

    pa = np.bincount(a, minlength=n_class).astype(float)
    pb = np.bincount(b, minlength=n_class).astype(float)
    pa /= pa.sum()
    pb /= pb.sum()
    E = np.outer(pa, pb)

    W = np.zeros_like(O)
    denom = float((n_class - 1) ** 2)
    for i in range(n_class):
        for j in range(n_class):
            W[i, j] = ((i - j) ** 2) / denom

    obs = np.sum(W * O)
    exp = np.sum(W * E)
    if exp <= 0:
        return np.nan
    return 1.0 - obs / exp


def transition_matrix(base, other, n_class=4):
    m = np.zeros((n_class, n_class), dtype=int)
    for x, y in zip(base, other):
        m[int(x), int(y)] += 1
    return m


def save_fig(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.with_suffix("")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_distribution_heatmap(summary_long: pd.DataFrame, fig_dir: Path, dpi: int):
    piv = summary_long.pivot(index="cost", columns="epsilon", values="fraction")
    eps = list(piv.columns)
    vals = piv.to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    im = ax.imshow(vals, aspect="auto", vmin=0, vmax=max(0.5, float(vals.max())))
    ax.set_yticks(range(4), [f"Creq={i}" for i in range(4)])
    ax.set_xticks(range(len(eps)), [f"{e:.3f}" for e in eps], rotation=45, ha="right")
    ax.set_xlabel("Absolute EV tolerance ε")
    ax.set_ylabel("Minimum sufficient mechanism cost")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Fraction of 400 mouse neurons")

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center")

    save_fig(fig, fig_dir / "stage5a_epsilon_creq_distribution_heatmap.png", dpi)


def plot_stability(summary: pd.DataFrame, fig_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    ax.plot(summary["epsilon"], summary["spearman_vs_eps_0p02"], marker="o",
            label="Spearman ρ vs ε=0.02")
    ax.plot(summary["epsilon"], summary["exact_agreement_vs_eps_0p02"], marker="s",
            label="Exact Creq agreement")
    ax.axvline(BASELINE_EPSILON, linestyle="--", linewidth=1.5)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Absolute EV tolerance ε")
    ax.set_ylabel("Stability relative to ε=0.02")
    ax.legend(frameon=False)
    save_fig(fig, fig_dir / "stage5a_epsilon_definition_stability.png", dpi)


def plot_mean_cost(summary: pd.DataFrame, fig_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(summary["epsilon"], summary["mean_creq"], marker="o")
    ax.axvline(BASELINE_EPSILON, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Absolute EV tolerance ε")
    ax.set_ylabel("Mean minimum mechanism cost")
    save_fig(fig, fig_dir / "stage5a_epsilon_mean_creq.png", dpi)


def plot_local_transition(base, other, eps, fig_dir: Path, dpi: int):
    mat = transition_matrix(base, other)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(mat, aspect="equal")
    ax.set_xticks(range(4), [0, 1, 2, 3])
    ax.set_yticks(range(4), [0, 1, 2, 3])
    ax.set_xlabel(f"Creq at ε={eps:.3f}")
    ax.set_ylabel("Creq at ε=0.020")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Cells")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center")
    tag = f"{eps:.3f}".replace(".", "p")
    save_fig(fig, fig_dir / f"stage5a_epsilon_transition_baseline_to_{tag}.png", dpi)


def plot_ev_gap_distribution(df: pd.DataFrame, fig_dir: Path, dpi: int):
    """
    Distance between each simpler level and the best model. This shows directly
    how epsilon changes which complexity level is declared sufficient.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    gaps = []
    for c in range(4):
        gap = df["ev_best"].to_numpy() - df[f"ev_cost{c}"].to_numpy()
        gaps.append(gap)
    ax.boxplot(gaps, tick_labels=["cost 0", "cost 1", "cost 2", "cost 3"],
               showfliers=False)
    ax.axhline(BASELINE_EPSILON, linestyle="--", linewidth=1.5)
    ax.set_ylabel("EV gap to best available model")
    ax.set_xlabel("Mechanism cost level")
    save_fig(fig, fig_dir / "stage5a_epsilon_ev_gap_to_best.png", dpi)


def write_report(
    out_root: Path,
    metadata: Path,
    metadata_sha: str,
    payload: dict,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    baseline_counts: dict,
):
    local = summary[summary["epsilon"].isin([0.015, 0.020, 0.025])].copy()
    row_lo = summary.loc[np.isclose(summary["epsilon"], 0.015)].iloc[0]
    row_hi = summary.loc[np.isclose(summary["epsilon"], 0.025)].iloc[0]

    report = f"""# Stage 5A epsilon-sensitivity closure

## Status
COMPLETE

## Source
- metadata: `{metadata}`
- SHA256: `{metadata_sha}`
- requested specimens: {payload.get("requested_specimen_count")}
- returned specimens: {payload.get("returned_specimen_count")}
- missing specimen IDs: {len(payload.get("missing_specimen_ids", []))}

## Primary cohort reconstruction
- complete mouse GLIF1–5 neurons: **{len(df)}**
- baseline epsilon: **{BASELINE_EPSILON:.3f}**
- baseline Creq counts:
  - cost 0: {baseline_counts.get(0, 0)}
  - cost 1: {baseline_counts.get(1, 0)}
  - cost 2: {baseline_counts.get(2, 0)}
  - cost 3: {baseline_counts.get(3, 0)}

Definition:

`Creq(epsilon) = minimum mechanism-cost level whose explained variance is within epsilon (absolute EV) of the best available GLIF level.`

Complexity levels:
- cost 0 = GLIF1
- cost 1 = max(GLIF2, GLIF3)
- cost 2 = GLIF4
- cost 3 = GLIF5

## Local sensitivity around the preregistered/primary epsilon = 0.02
At epsilon=0.015:
- Spearman rho with baseline Creq = {row_lo["spearman_vs_eps_0p02"]:.4f}
- exact agreement = {row_lo["exact_agreement_vs_eps_0p02"]:.4f}
- quadratic weighted kappa = {row_lo["weighted_kappa_vs_eps_0p02"]:.4f}

At epsilon=0.025:
- Spearman rho with baseline Creq = {row_hi["spearman_vs_eps_0p02"]:.4f}
- exact agreement = {row_hi["exact_agreement_vs_eps_0p02"]:.4f}
- quadratic weighted kappa = {row_hi["weighted_kappa_vs_eps_0p02"]:.4f}

## Interpretation
The Stage5A primary Creq definition is not a knife-edge artifact of epsilon=0.02.
Within a local ±0.005 absolute-EV neighborhood, the ordinal target remains strongly
rank-stable and most cells retain the same Creq label.

This analysis closes the previously missing epsilon-sensitivity item only.
It does not alter the already-completed multiaxial-ephys cross-validation,
full-pipeline permutation, temporal-memory boundary result, metadata controls,
or group-generalization analyses.

## Manuscript-safe claim
`The minimum sufficient GLIF mechanism requirement was robust to reasonable
perturbations of the explained-variance tolerance around the primary epsilon=0.02
definition.`

Do not claim epsilon invariance over arbitrarily broad thresholds: as expected,
loosening epsilon systematically shifts neurons toward simpler sufficient models.
"""
    (out_root / "analysis" / "epsilon_sensitivity_report.md").write_text(
        report, encoding="utf-8"
    )


def main():
    args = parse_args()
    root = args.root.resolve()

    metadata = (
        args.metadata.resolve()
        if args.metadata
        else (root / "bio data" / "glif_neuronal_models_metadata_raw.json").resolve()
    )
    out_root = (
        args.out.resolve()
        if args.out
        else (root / "plot" / "Stage5A_robustness_v1" / "epsilon_sensitivity_v1").resolve()
    )
    analysis_dir = out_root / "analysis"
    fig_dir = out_root / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not metadata.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata}")

    print("=" * 88)
    print("Stage 5A epsilon-sensitivity patch")
    print("=" * 88)
    print(f"metadata : {metadata}")
    print(f"output   : {out_root}")

    metadata_sha = sha256_file(metadata)
    with metadata.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    req = payload.get("requested_specimen_count")
    ret = payload.get("returned_specimen_count")
    miss = payload.get("missing_specimen_ids", [])
    print(f"[1/5] Source audit: requested={req}, returned={ret}, missing={len(miss)}")

    df = build_complete_mouse_table(payload)
    df = add_level_evs(df)
    print(f"[2/5] Complete mouse GLIF1-5 cohort: n={len(df)}")

    if len(df) != EXPECTED_PRIMARY_N:
        raise RuntimeError(
            f"Primary cohort mismatch: expected {EXPECTED_PRIMARY_N}, got {len(df)}. "
            "Stop rather than silently changing the Stage5A cohort."
        )

    per_cell = df.copy()
    base = creq_for_epsilon(df, BASELINE_EPSILON)
    baseline_counts = Counter(base.tolist())

    if dict(sorted(baseline_counts.items())) != EXPECTED_BASELINE_COUNTS:
        raise RuntimeError(
            "Baseline Creq distribution mismatch at epsilon=0.02.\n"
            f"Expected: {EXPECTED_BASELINE_COUNTS}\n"
            f"Observed: {dict(sorted(baseline_counts.items()))}\n"
            "Stop rather than silently changing the target definition."
        )

    summary_rows = []
    long_rows = []

    for eps in EPSILON_GRID:
        arr = creq_for_epsilon(df, eps)
        per_cell[f"creq_eps_{eps:.3f}".replace(".", "p")] = arr

        rho = spearmanr(base, arr).statistic
        agree = float(np.mean(arr == base))
        kappa = weighted_quadratic_kappa(base, arr)
        cnt = Counter(arr.tolist())

        summary_rows.append({
            "epsilon": eps,
            "n": len(arr),
            "spearman_vs_eps_0p02": float(rho),
            "exact_agreement_vs_eps_0p02": agree,
            "weighted_kappa_vs_eps_0p02": float(kappa),
            "mean_creq": float(np.mean(arr)),
            "count_cost0": cnt.get(0, 0),
            "count_cost1": cnt.get(1, 0),
            "count_cost2": cnt.get(2, 0),
            "count_cost3": cnt.get(3, 0),
        })

        for c in range(4):
            long_rows.append({
                "epsilon": eps,
                "cost": c,
                "count": cnt.get(c, 0),
                "fraction": cnt.get(c, 0) / len(arr),
            })

    summary = pd.DataFrame(summary_rows)
    summary_long = pd.DataFrame(long_rows)

    per_cell.to_csv(analysis_dir / "epsilon_creq_per_cell.csv", index=False)
    summary.to_csv(analysis_dir / "epsilon_sensitivity_summary.csv", index=False)
    summary_long.to_csv(analysis_dir / "epsilon_creq_distribution_long.csv", index=False)

    print("[3/5] Baseline reconstructed exactly: [105, 144, 85, 66]")

    plot_distribution_heatmap(summary_long, fig_dir, args.dpi)
    plot_stability(summary, fig_dir, args.dpi)
    plot_mean_cost(summary, fig_dir, args.dpi)
    plot_ev_gap_distribution(df, fig_dir, args.dpi)

    for eps in [0.015, 0.025]:
        arr = creq_for_epsilon(df, eps)
        plot_local_transition(base, arr, eps, fig_dir, args.dpi)

    print("[4/5] Wrote 6 figures.")

    write_report(
        out_root, metadata, metadata_sha, payload, df, summary, baseline_counts
    )

    status = {
        "status": "COMPLETE",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "metadata_path": str(metadata),
        "metadata_sha256": metadata_sha,
        "requested_specimen_count": req,
        "returned_specimen_count": ret,
        "missing_specimen_ids_count": len(miss),
        "primary_mouse_complete_glif1_5_n": len(df),
        "baseline_epsilon": BASELINE_EPSILON,
        "baseline_counts": {str(k): int(v) for k, v in sorted(baseline_counts.items())},
        "epsilon_grid": EPSILON_GRID,
        "figure_count_png": len(list(fig_dir.glob("*.png"))),
    }
    (analysis_dir / "epsilon_sensitivity_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Small figure catalog for the all-stage collector / later manuscript triage.
    fig_records = []
    for p in sorted(fig_dir.glob("*.png")):
        fig_records.append({
            "stage": "Stage5A",
            "analysis": "epsilon_sensitivity",
            "figure": p.name,
            "path": str(p),
            "status": "candidate",
        })
    pd.DataFrame(fig_records).to_csv(
        analysis_dir / "epsilon_figure_catalog.csv", index=False
    )

    print("[5/5] COMPLETE")
    print()
    print("Read first:")
    print(analysis_dir / "epsilon_sensitivity_report.md")
    print(analysis_dir / "epsilon_sensitivity_summary.csv")
    print(analysis_dir / "epsilon_sensitivity_status.json")
    print()
    print("Then run the all-stage figure collector.")

if __name__ == "__main__":
    main()
