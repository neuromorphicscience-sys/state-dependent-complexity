#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5A robustness closure + full candidate figure atlas (v1)
===============================================================

Designed for the NeuralScience project. Put this script in:
    D:\\Research\\Neural Science\\plot\\stage5a_robustness_closure_v1.py

Default inputs (no manual extraction required):
    D:\\Research\\Neural Science\\bio data\\stage5_v11_review_bundle.tar.gz
    D:\\Research\\Neural Science\\bio data\\stage5_v12_results.tar.gz
    D:\\Research\\Neural Science\\bio data\\stage5_v11_hotfix.zip
    D:\\Research\\Neural Science\\bio data\\stage5_v12_biological_atlas.zip

Default outputs:
    D:\\Research\\Neural Science\\plot\\Stage5A_robustness_v1\\analysis
    D:\\Research\\Neural Science\\plot\\Stage5A_robustness_v1\\figures

What this closes
----------------
1) Reproduce Stage 5A primary OOF metrics and compare against naive baselines.
2) Re-run the primary multiaxial model with a FULL nested-CV permutation null,
   including alpha re-selection inside every permutation.
3) Test biological-group generalization:
      - leave-one-layer-out
      - leave-one-dendrite-type-out
      - VISp-only repeated nested CV
4) Quantify within-layer / within-dendrite association robustness.
5) Build an explicit cohort attrition audit from v1.1 + v1.2 archives.
6) Re-analyze the 1213-cell temporal boundary result.
7) Audit PFC projectome zero-variance features and preserve its role as
   descriptive structural organization rather than dynamical leverage.
8) OPTIONAL: C_req epsilon sensitivity (0.01/0.02/0.05/0.10) if the small
   GLIF performance atlas CSV exists locally. The script searches for it.
9) Generate a broad set of SEPARATE publication-style candidate figures.

Scientific guardrails
---------------------
- No plot title is embedded in any figure.
- Every figure is a single standalone panel (PNG 600 dpi + vector PDF).
- Search / fit statistics are not silently substituted for held-out results.
- PFC subtype effects are treated as descriptive atlas evidence, not as proof
  of dynamic leverage.
- If the GLIF atlas is absent, epsilon sensitivity is reported as NOT RUN;
  no result is fabricated.

Dependencies
------------
Python 3.10+
numpy, pandas, scipy, matplotlib

Typical run (PowerShell)
------------------------
python .\\plot\\stage5a_robustness_closure_v1.py --root "D:\\Research\\Neural Science"

Faster smoke run:
python .\\plot\\stage5a_robustness_closure_v1.py --root "D:\\Research\\Neural Science" --permutations 20 --bootstrap 300
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_ROOT = Path.cwd()

COST = {"GLIF1": 0, "GLIF2": 1, "GLIF3": 1, "GLIF4": 2, "GLIF5": 3}

MODEL_COLUMNS = {
    "Temporal profile": "pred_Creq_temporal_profile",
    "Multiaxial ephys": "pred_Creq_multiaxial_ephys",
    "Metadata only": "pred_Creq_metadata_only",
    "Ephys + metadata": "pred_Creq_ephys_plus_metadata",
    "Ephys + temporal": "pred_Creq_ephys_plus_temporal",
}

COLORS = {
    "temporal": "#8D6A9F",
    "ephys": "#355C7D",
    "metadata": "#A6A6A6",
    "combined": "#2A9D8F",
    "accent": "#B56576",
    "orange": "#BC6C25",
    "dark": "#264653",
    "gray": "#6C757D",
}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def read_csv_tar(tf: tarfile.TarFile, name: str) -> pd.DataFrame:
    f = tf.extractfile(name)
    if f is None:
        raise FileNotFoundError(name)
    return pd.read_csv(io.BytesIO(f.read()))


def read_json_tar(tf: tarfile.TarFile, name: str):
    f = tf.extractfile(name)
    if f is None:
        raise FileNotFoundError(name)
    return json.loads(f.read().decode("utf-8"))


def zip_has(path: Path, member: str) -> bool:
    with zipfile.ZipFile(path, "r") as z:
        return member in z.namelist()


def zip_json(path: Path, member: str):
    with zipfile.ZipFile(path, "r") as z:
        return json.loads(z.read(member).decode("utf-8"))


def finite_spearman(x, y) -> Tuple[float, float, int]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
        return np.nan, np.nan, int(m.sum())
    r = spearmanr(x[m], y[m])
    return float(r.statistic), float(r.pvalue), int(m.sum())


def r2_score(y, pred) -> float:
    y = np.asarray(y, float)
    p = np.asarray(pred, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 2:
        return np.nan
    den = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - np.sum((y - p) ** 2) / den) if den > 0 else np.nan


def mae(y, pred) -> float:
    y = np.asarray(y, float)
    p = np.asarray(pred, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[m] - p[m]))) if m.any() else np.nan


def quadratic_weighted_kappa(y, pred_round, n_classes: int = 4) -> float:
    y = np.asarray(y, int)
    p = np.asarray(pred_round, int)
    m = (y >= 0) & (y < n_classes) & (p >= 0) & (p < n_classes)
    y, p = y[m], p[m]
    if len(y) == 0:
        return np.nan
    O = np.zeros((n_classes, n_classes), float)
    for a, b in zip(y, p):
        O[a, b] += 1
    act = O.sum(axis=1)
    pred = O.sum(axis=0)
    E = np.outer(act, pred) / max(1.0, O.sum())
    W = np.zeros_like(O)
    den = float((n_classes - 1) ** 2)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / den
    obs = np.sum(W * O)
    exp = np.sum(W * E)
    return float(1 - obs / exp) if exp > 0 else np.nan


def ordinal_metrics(y, pred) -> Dict[str, float]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    pr = np.clip(np.rint(pred), 0, 3).astype(int)
    yi = y.astype(int)
    rho, p, n = finite_spearman(y, pred)
    return {
        "n": int(n),
        "spearman_rho": rho,
        "spearman_p": p,
        "mae": mae(y, pred),
        "r2": r2_score(y, pred),
        "rounded_accuracy": float(np.mean(pr == yi)),
        "within_one_class_accuracy": float(np.mean(np.abs(pr - yi) <= 1)),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(yi, pr, 4),
    }


def bootstrap_rho(y, pred, n_boot: int, seed: int) -> Tuple[float, float, float]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    m = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[m], pred[m]
    obs, _, _ = finite_spearman(y, pred)
    if len(y) < 5:
        return obs, np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        z = rng.choice(idx, len(idx), replace=True)
        r, _, _ = finite_spearman(y[z], pred[z])
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return obs, np.nan, np.nan
    return obs, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# -----------------------------------------------------------------------------
# Original Stage 5A ridge/CV logic, reproduced independently
# -----------------------------------------------------------------------------

def standardize_train_test(Xtr, Xte):
    Xtr = np.asarray(Xtr, float)
    Xte = np.asarray(Xte, float)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    a = np.where(np.isfinite(Xtr), Xtr, med)
    b = np.where(np.isfinite(Xte), Xte, med)
    mu = a.mean(axis=0)
    sd = a.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (a - mu) / sd, (b - mu) / sd


def ridge_fit_predict(Xtr, ytr, Xte, alpha):
    Xtr = np.asarray(Xtr, float)
    Xte = np.asarray(Xte, float)
    ytr = np.asarray(ytr, float)
    Xtr, Xte = standardize_train_test(Xtr, Xte)
    ym = ytr.mean()
    yy = ytr - ym
    A = Xtr.T @ Xtr + float(alpha) * np.eye(Xtr.shape[1])
    try:
        w = np.linalg.solve(A, Xtr.T @ yy)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ (Xtr.T @ yy)
    return Xte @ w + ym


def stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        ix = np.flatnonzero(y == cls)
        rng.shuffle(ix)
        for j, v in enumerate(ix):
            folds[j % k].append(int(v))
    return [np.asarray(sorted(x), int) for x in folds]


def repeated_nested_cv(X, y, outer_folds, repeats, alphas, seed, seedbase=0):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    pred_sum = np.zeros(n)
    pred_n = np.zeros(n)
    chosen = []
    for rep in range(int(repeats)):
        folds = stratified_folds(y, int(outer_folds), int(seed) + seedbase + rep * 97)
        for fi, test in enumerate(folds):
            train = np.setdiff1d(np.arange(n), test, assume_unique=False)
            kin = max(3, int(outer_folds) - 1)
            inner = stratified_folds(y[train], kin, int(seed) + seedbase + rep * 997 + fi)
            scores = []
            for alpha in alphas:
                sc = []
                for val_local in inner:
                    tr_local = np.setdiff1d(np.arange(len(train)), val_local)
                    pr = ridge_fit_predict(X[train][tr_local], y[train][tr_local], X[train][val_local], alpha)
                    sc.append(np.mean((pr - y[train][val_local]) ** 2))
                scores.append(np.mean(sc))
            alpha = float(alphas[int(np.argmin(scores))])
            chosen.append(alpha)
            p = ridge_fit_predict(X[train], y[train], X[test], alpha)
            pred_sum[test] += p
            pred_n[test] += 1
    pred = pred_sum / np.maximum(pred_n, 1)
    met = ordinal_metrics(y, pred)
    met["median_alpha"] = float(np.median(chosen)) if chosen else np.nan
    return pred, met


def choose_alpha_inner(X, y, alphas, seed, repeats=3, k=5):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    scores = {float(a): [] for a in alphas}
    for rep in range(repeats):
        folds = stratified_folds(y, min(k, max(3, int(np.min(np.bincount(y.astype(int)))))), seed + rep * 131)
        for val in folds:
            tr = np.setdiff1d(np.arange(len(y)), val)
            if len(tr) < 10 or len(val) == 0:
                continue
            for a in alphas:
                p = ridge_fit_predict(X[tr], y[tr], X[val], a)
                scores[float(a)].append(float(np.mean((p - y[val]) ** 2)))
    means = {a: (np.mean(v) if v else np.inf) for a, v in scores.items()}
    return min(means, key=means.get)


def leave_one_group_out_cv(X, y, groups, alphas, seed):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups, object)
    pred = np.full(len(y), np.nan)
    records = []
    unique = sorted(pd.Series(groups).dropna().astype(str).unique())
    for gi, g in enumerate(unique):
        test = np.flatnonzero(groups.astype(str) == g)
        train = np.flatnonzero(groups.astype(str) != g)
        if len(test) == 0 or len(train) < 30:
            continue
        alpha = choose_alpha_inner(X[train], y[train], alphas, seed + gi * 1009, repeats=3, k=5)
        pp = ridge_fit_predict(X[train], y[train], X[test], alpha)
        pred[test] = pp
        gm = ordinal_metrics(y[test], pp)
        records.append({"held_out_group": g, "test_n": int(len(test)), "alpha": float(alpha), **gm})
    return pred, ordinal_metrics(y[np.isfinite(pred)], pred[np.isfinite(pred)]), pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_archives(v11_tar: Path, v12_tar: Path):
    out = {}
    with tarfile.open(v12_tar, "r:gz") as tf:
        base = "stage5_biological_discovery_v12"
        out["primary"] = read_csv_tar(tf, f"{base}/20_allen_multiaxial_v12/allen_multiaxial_primary_mouse_complete.csv")
        out["assoc"] = read_csv_tar(tf, f"{base}/20_allen_multiaxial_v12/feature_Creq_associations.csv")
        out["ablation"] = read_csv_tar(tf, f"{base}/20_allen_multiaxial_v12/category_ablation.csv")
        out["summary"] = read_json_tar(tf, f"{base}/20_allen_multiaxial_v12/allen_multiaxial_summary.json")
        out["pfc"] = read_csv_tar(tf, f"{base}/10_digital_brain_v12/pfc2022_canonical_6357_features_v12.csv")
        out["pfc_effects"] = read_csv_tar(tf, f"{base}/30_structural_stats_v12/pfc2022_feature_subtype_effects.csv")
        out["pfc_centroids"] = read_csv_tar(tf, f"{base}/30_structural_stats_v12/pfc2022_subtype_centroids.csv")
        out["pfc_summary"] = read_json_tar(tf, f"{base}/30_structural_stats_v12/pfc2022_structural_stats_summary.json")
        out["verdict"] = read_json_tar(tf, f"{base}/40_final_adjudication_v12/verdict_v12.json")
    with tarfile.open(v11_tar, "r:gz") as tf:
        base = "stage5_biological_discovery_v1"
        out["extract"] = read_csv_tar(tf, f"{base}/20_allen_ephys_v11/allen_noise_extraction_summary.csv")
        out["ab"] = read_csv_tar(tf, f"{base}/30_temporal_context/AB/context_performance.csv")
        out["ba"] = read_csv_tar(tf, f"{base}/30_temporal_context/BA/context_performance.csv")
        out["ab_summary"] = read_json_tar(tf, f"{base}/30_temporal_context/AB/run_summary.json")
        out["ba_summary"] = read_json_tar(tf, f"{base}/30_temporal_context/BA/run_summary.json")
    return out


def materialize_archive_source_data(data: dict, out_source: Path) -> None:
    """Persist exact archive tables used by selected Stage5A panels."""
    out_source.mkdir(parents=True, exist_ok=True)
    exports = {
        "feature_Creq_associations.csv": data["assoc"],
        "category_ablation.csv": data["ablation"],
        "pfc2022_feature_subtype_effects.csv": data["pfc_effects"],
        "pfc2022_subtype_centroids.csv": data["pfc_centroids"],
    }
    for name, frame in exports.items():
        frame.to_csv(out_source / name, index=False, encoding="utf-8-sig")


def build_onehot(df: pd.DataFrame, fields: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    cols = []
    names = []
    for f in fields:
        levels = sorted([str(x) for x in df[f].dropna().unique()])
        for lv in levels[1:]:
            cols.append((df[f].astype(str).to_numpy() == lv).astype(float))
            names.append(f"{f}={lv}")
    return (np.column_stack(cols) if cols else np.zeros((len(df), 0))), names


def temporal_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    cols = [
        "C_temporal_consensus",
        "temporal_best_r2_mean",
        "temporal_long_short_gain_mean",
        "temporal_r2mean_10",
        "temporal_r2mean_25",
        "temporal_r2mean_50",
        "temporal_r2mean_100",
        "temporal_r2mean_200",
        "temporal_r2mean_500",
    ]
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float), cols


def temporal_full_table(ab: pd.DataFrame, ba: pd.DataFrame, eps: float = 0.02):
    contexts = sorted(set(pd.to_numeric(ab["context_ms"], errors="coerce").dropna().astype(int)))
    A = ab.pivot_table(index="specimen_id", columns="context_ms", values="r2", aggfunc="mean")
    B = ba.pivot_table(index="specimen_id", columns="context_ms", values="r2", aggfunc="mean")
    ids = sorted(set(A.index).intersection(B.index))
    rows = []
    for sid in ids:
        ra = A.loc[sid]
        rb = B.loc[sid]
        if not all(c in A.columns and c in B.columns for c in contexts):
            continue
        va = np.array([ra.get(c, np.nan) for c in contexts], float)
        vb = np.array([rb.get(c, np.nan) for c in contexts], float)
        if not np.all(np.isfinite(va)) or not np.all(np.isfinite(vb)):
            continue
        besta, bestb = np.max(va), np.max(vb)
        ca = min([c for c, v in zip(contexts, va) if v >= besta - eps])
        cb = min([c for c, v in zip(contexts, vb) if v >= bestb - eps])
        rows.append({
            "specimen_id": int(sid),
            "C_temporal_AB": int(ca),
            "C_temporal_BA": int(cb),
            "long_short_gain_AB": float(va[-1] - va[0]),
            "long_short_gain_BA": float(vb[-1] - vb[0]),
            "long_short_gain_mean": float(((va[-1] - va[0]) + (vb[-1] - vb[0])) / 2),
        })
    return pd.DataFrame(rows), contexts


# -----------------------------------------------------------------------------
# Robustness analyses
# -----------------------------------------------------------------------------

def reproduce_model_metrics(primary: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(primary["C_req_eps_0.02"], errors="coerce").to_numpy(float)
    rows = []
    for label, col in MODEL_COLUMNS.items():
        pred = pd.to_numeric(primary[col], errors="coerce").to_numpy(float)
        rows.append({"model": label, **ordinal_metrics(y, pred)})
    med = float(np.median(y))
    mean = float(np.mean(y))
    majority = int(pd.Series(y.astype(int)).value_counts().idxmax())
    rows.append({"model": "Naive median", **ordinal_metrics(y, np.full(len(y), med))})
    rows.append({"model": "Naive mean", **ordinal_metrics(y, np.full(len(y), mean))})
    rows.append({"model": "Naive majority class", **ordinal_metrics(y, np.full(len(y), majority))})
    return pd.DataFrame(rows)


def full_pipeline_permutation(X, y, cv_cfg, permutations: int, out_csv: Path):
    # Observed model is re-run through exactly the same nested pipeline used for nulls.
    pred_obs, met_obs = repeated_nested_cv(
        X, y,
        cv_cfg["outer_folds"], cv_cfg["repeats"], cv_cfg["alphas"], cv_cfg["seed"], seedbase=0,
    )
    observed = met_obs["spearman_rho"]
    rng = np.random.default_rng(int(cv_cfg["seed"]) + 77123)
    null_rows = []
    t0 = time.time()
    for i in range(permutations):
        yp = rng.permutation(y)
        _, met = repeated_nested_cv(
            X, yp,
            cv_cfg["outer_folds"], cv_cfg["repeats"], cv_cfg["alphas"], cv_cfg["seed"],
            seedbase=100000 + i * 17,
        )
        null_rows.append({"permutation": i + 1, "rho": met["spearman_rho"], "median_alpha": met["median_alpha"]})
        if (i + 1) % 10 == 0 or i + 1 == permutations:
            pd.DataFrame(null_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
            elapsed = time.time() - t0
            print(f"  full-pipeline permutation {i+1}/{permutations} elapsed={elapsed/60:.1f} min", flush=True)
    null = np.asarray([r["rho"] for r in null_rows], float)
    p = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    return pred_obs, met_obs, pd.DataFrame(null_rows), p


def run_group_robustness(primary, features, cv_cfg, out_analysis, bootstrap):
    y = pd.to_numeric(primary["C_req_eps_0.02"], errors="coerce").to_numpy(float)
    X = primary[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    rows = []

    # Standard nested rerun
    pred_std, met_std = repeated_nested_cv(X, y, cv_cfg["outer_folds"], cv_cfg["repeats"], cv_cfg["alphas"], cv_cfg["seed"])
    rows.append({"scheme": "Standard repeated nested CV", **met_std})

    # Leave one layer out
    pred_layer, met_layer, layer_details = leave_one_group_out_cv(X, y, primary["layer"].astype(str).to_numpy(), cv_cfg["alphas"], cv_cfg["seed"] + 2000)
    rows.append({"scheme": "Leave-one-layer-out", **met_layer})
    layer_details.to_csv(out_analysis / "groupcv_leave_one_layer_details.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"specimen_id": primary["specimen_id"], "y": y, "pred_leave_layer_out": pred_layer, "layer": primary["layer"]}).to_csv(out_analysis / "groupcv_leave_one_layer_predictions.csv", index=False, encoding="utf-8-sig")

    # Leave one dendrite class out
    pred_den, met_den, den_details = leave_one_group_out_cv(X, y, primary["dendrite_type"].astype(str).to_numpy(), cv_cfg["alphas"], cv_cfg["seed"] + 4000)
    rows.append({"scheme": "Leave-one-dendrite-type-out", **met_den})
    den_details.to_csv(out_analysis / "groupcv_leave_one_dendrite_details.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"specimen_id": primary["specimen_id"], "y": y, "pred_leave_dendrite_out": pred_den, "dendrite_type": primary["dendrite_type"]}).to_csv(out_analysis / "groupcv_leave_one_dendrite_predictions.csv", index=False, encoding="utf-8-sig")

    # VISp-only control: removes most regional mixture as a confound.
    vis = primary[primary["structure_parent_acronym"].astype(str) == "VISp"].copy()
    if len(vis) >= 100:
        yv = pd.to_numeric(vis["C_req_eps_0.02"], errors="coerce").to_numpy(float)
        Xv = vis[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        pred_vis, met_vis = repeated_nested_cv(Xv, yv, cv_cfg["outer_folds"], cv_cfg["repeats"], cv_cfg["alphas"], cv_cfg["seed"], seedbase=7000)
        rows.append({"scheme": "VISp-only repeated nested CV", **met_vis})
        pd.DataFrame({"specimen_id": vis["specimen_id"], "y": yv, "pred": pred_vis}).to_csv(out_analysis / "visp_only_predictions.csv", index=False, encoding="utf-8-sig")

    # Group-specific effects based on stored OOF prediction (not refit within group).
    stored = pd.to_numeric(primary["pred_Creq_multiaxial_ephys"], errors="coerce").to_numpy(float)
    group_rows = []
    for field in ["layer", "dendrite_type"]:
        for g, sub in primary.groupby(field, dropna=False):
            idx = sub.index.to_numpy()
            if len(idx) < 8:
                continue
            obs, lo, hi = bootstrap_rho(y[idx], stored[idx], bootstrap, cv_cfg["seed"] + len(group_rows) * 31)
            _, p, n = finite_spearman(y[idx], stored[idx])
            group_rows.append({"grouping": field, "group": str(g), "n": n, "rho": obs, "bootstrap95_low": lo, "bootstrap95_high": hi, "p": p})
    gdf = pd.DataFrame(group_rows)
    gdf.to_csv(out_analysis / "within_group_oof_associations.csv", index=False, encoding="utf-8-sig")

    return pd.DataFrame(rows), gdf


def c_req_from_row(row: pd.Series, eps: float):
    vals = {g: pd.to_numeric(row.get(f"{g}_EVR"), errors="coerce") for g in COST}
    if any(not np.isfinite(v) for v in vals.values()):
        return np.nan
    best = max(vals.values())
    eligible = [g for g, v in vals.items() if v >= best - eps]
    return float(min(COST[g] for g in eligible))


def find_glif_atlas(root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None and explicit.exists():
        return explicit
    candidates = [
        root / "biological_data" / "allen_cell_types" / "analysis" / "glif_performance_atlas_v1" / "tables" / "glif_performance_atlas_specimen_level.csv",
        root / "bio data" / "allen_cell_types" / "analysis" / "glif_performance_atlas_v1" / "tables" / "glif_performance_atlas_specimen_level.csv",
        root / "bio data" / "glif_performance_atlas_specimen_level.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Final exact-name search, intentionally narrow.
    try:
        hits = list(root.glob("**/glif_performance_atlas_specimen_level.csv"))
        if hits:
            return hits[0]
    except Exception:
        pass
    return None


def run_epsilon_sensitivity(primary, features, cv_cfg, glif_path: Optional[Path], out_analysis):
    status = {"status": "NOT_RUN", "reason": "GLIF performance atlas not found", "glif_path": None}
    if glif_path is None or not glif_path.exists():
        (out_analysis / "epsilon_sensitivity_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return pd.DataFrame(), status

    glif = pd.read_csv(glif_path)
    sidcol = "specimen_id" if "specimen_id" in glif.columns else None
    if sidcol is None or not all(f"{g}_EVR" in glif.columns for g in COST):
        status = {"status": "NOT_RUN", "reason": "Required specimen_id + GLIF1..5_EVR columns absent", "glif_path": str(glif_path)}
        (out_analysis / "epsilon_sensitivity_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return pd.DataFrame(), status

    g = glif.copy()
    g[sidcol] = pd.to_numeric(g[sidcol], errors="coerce")
    ids = pd.to_numeric(primary["specimen_id"], errors="coerce")
    gg = g[g[sidcol].isin(set(ids.dropna().astype(int)))].copy().set_index(sidcol)
    X = primary[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Xt, _ = temporal_matrix(primary)
    Xm, _ = build_onehot(primary, ["layer", "dendrite_type", "structure_parent_acronym"])

    rows = []
    for ei, eps in enumerate([0.01, 0.02, 0.05, 0.10]):
        y = []
        valid = []
        for i, sid in enumerate(ids):
            if not np.isfinite(sid) or int(sid) not in gg.index:
                y.append(np.nan)
                continue
            row = gg.loc[int(sid)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            y.append(c_req_from_row(row, eps))
        y = np.asarray(y, float)
        m = np.isfinite(y)
        if m.sum() < 250:
            continue
        for mi, (name, XX) in enumerate([("Multiaxial ephys", X), ("Temporal profile", Xt), ("Metadata only", Xm)]):
            _, met = repeated_nested_cv(XX[m], y[m], cv_cfg["outer_folds"], cv_cfg["repeats"], cv_cfg["alphas"], cv_cfg["seed"], seedbase=20000 + ei * 3000 + mi * 500)
            dist = Counter(y[m].astype(int))
            rows.append({"epsilon": eps, "model": name, "n": int(m.sum()), "C0": dist.get(0, 0), "C1": dist.get(1, 0), "C2": dist.get(2, 0), "C3": dist.get(3, 0), **met})
    df = pd.DataFrame(rows)
    df.to_csv(out_analysis / "epsilon_sensitivity.csv", index=False, encoding="utf-8-sig")
    status = {"status": "COMPLETE", "glif_path": str(glif_path), "rows": int(len(df))}
    (out_analysis / "epsilon_sensitivity_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return df, status


def cohort_attrition(data, primary, out_analysis):
    ext = data["extract"].copy()
    ab = data["ab"].copy()
    ba = data["ba"].copy()
    ext_ids = set(pd.to_numeric(ext["specimen_id"], errors="coerce").dropna().astype(int))
    ok_ids = set(pd.to_numeric(ext.loc[ext["status"].astype(str) == "ok", "specimen_id"], errors="coerce").dropna().astype(int))
    ab_ids = set(pd.to_numeric(ab["specimen_id"], errors="coerce").dropna().astype(int))
    ba_ids = set(pd.to_numeric(ba["specimen_id"], errors="coerce").dropna().astype(int))
    contexts_ab = ab.groupby("specimen_id")["context_ms"].nunique()
    contexts_ba = ba.groupby("specimen_id")["context_ms"].nunique()
    complete_ab = set(pd.to_numeric(contexts_ab[contexts_ab >= 6].index, errors="coerce").astype(int))
    complete_ba = set(pd.to_numeric(contexts_ba[contexts_ba >= 6].index, errors="coerce").astype(int))
    primary_ids = set(pd.to_numeric(primary["specimen_id"], errors="coerce").dropna().astype(int))
    rows = [
        {"stage": "NWB extraction inventory", "n": len(ext_ids)},
        {"stage": "Usable Noise 1 + Noise 2", "n": len(ok_ids)},
        {"stage": "AB complete (6 contexts)", "n": len(complete_ab)},
        {"stage": "BA complete (6 contexts)", "n": len(complete_ba)},
        {"stage": "AB ∩ BA complete", "n": len(complete_ab & complete_ba)},
        {"stage": "Stage 5A primary mouse + complete GLIF1–5", "n": len(primary_ids)},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_analysis / "cohort_attrition.csv", index=False, encoding="utf-8-sig")
    return df


def pfc_qc(data, out_analysis):
    pfc = data["pfc"].copy()
    numeric = pfc.select_dtypes(include=[np.number]).columns
    rows = []
    for c in numeric:
        x = pd.to_numeric(pfc[c], errors="coerce")
        rows.append({
            "feature": c,
            "n": int(x.notna().sum()),
            "n_unique": int(x.nunique(dropna=True)),
            "mean": float(x.mean()) if x.notna().any() else np.nan,
            "std": float(x.std()) if x.notna().any() else np.nan,
            "min": float(x.min()) if x.notna().any() else np.nan,
            "max": float(x.max()) if x.notna().any() else np.nan,
            "zero_variance": bool(x.nunique(dropna=True) <= 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_analysis / "pfc_feature_qc.csv", index=False, encoding="utf-8-sig")
    return df


# -----------------------------------------------------------------------------
# Figure helpers: standalone panels only, no titles
# -----------------------------------------------------------------------------

def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 10.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.16, linewidth=0.8)
    ax.set_axisbelow(True)


def save_panel(fig, path_base: Path):
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_attrition(attr, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    d = attr.iloc[::-1]
    ax.barh(np.arange(len(d)), d["n"], color=COLORS["ephys"], alpha=0.88)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["stage"])
    ax.set_xlabel("Cells / specimens")
    for i, n in enumerate(d["n"]):
        ax.text(n + max(attr["n"]) * 0.015, i, f"{int(n):,}", va="center", fontsize=9)
    clean_ax(ax, "x")
    ax.set_xlim(0, max(attr["n"]) * 1.18)
    save_panel(fig, out / "S5A_01_cohort_attrition")


def fig_model_rho(metrics, out):
    order = ["Temporal profile", "Multiaxial ephys", "Metadata only", "Ephys + metadata", "Ephys + temporal"]
    d = metrics[metrics["model"].isin(order)].set_index("model").loc[order].reset_index()
    cols = [COLORS["temporal"], COLORS["ephys"], COLORS["metadata"], COLORS["combined"], COLORS["orange"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(d))
    ax.bar(x, d["spearman_rho"], color=cols, width=0.65)
    ax.axhline(0, color="#999999", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(d["model"], rotation=26, ha="right")
    ax.set_ylabel("Cross-validated Spearman ρ")
    clean_ax(ax)
    save_panel(fig, out / "S5A_02_model_comparison_rho")


def fig_model_mae(metrics, out):
    order = ["Naive median", "Temporal profile", "Metadata only", "Multiaxial ephys", "Ephys + metadata"]
    d = metrics[metrics["model"].isin(order)].set_index("model").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(d))
    ax.bar(x, d["mae"], color=[COLORS["gray"], COLORS["temporal"], COLORS["metadata"], COLORS["ephys"], COLORS["combined"]], width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(d["model"], rotation=26, ha="right")
    ax.set_ylabel("Held-out MAE in Creq units")
    clean_ax(ax)
    save_panel(fig, out / "S5A_03_model_comparison_mae")


def fig_prediction_by_true(primary, out):
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    y = pd.to_numeric(primary["C_req_eps_0.02"], errors="coerce").to_numpy(int)
    p = pd.to_numeric(primary["pred_Creq_multiaxial_ephys"], errors="coerce").to_numpy(float)
    rng = np.random.default_rng(20260819)
    for c in range(4):
        z = p[y == c]
        jitter = rng.normal(0, 0.035, size=len(z))
        ax.scatter(np.full(len(z), c) + jitter, z, s=13, alpha=0.35, color=COLORS["ephys"], edgecolor="none")
        if len(z):
            ax.plot([c - 0.18, c + 0.18], [np.median(z), np.median(z)], color=COLORS["accent"], lw=2.2)
    ax.plot([0, 3], [0, 3], color="#AAAAAA", ls="--", lw=0.9)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Observed minimum mechanism cost Creq")
    ax.set_ylabel("OOF predicted Creq")
    clean_ax(ax)
    save_panel(fig, out / "S5A_04_prediction_by_observed_Creq")


def fig_ordinal_confusion(primary, out):
    y = pd.to_numeric(primary["C_req_eps_0.02"], errors="coerce").to_numpy(int)
    p = np.clip(np.rint(pd.to_numeric(primary["pred_Creq_multiaxial_ephys"], errors="coerce").to_numpy(float)), 0, 3).astype(int)
    cm = np.zeros((4, 4), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    fig, ax = plt.subplots(figsize=(4.7, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=("white" if cm[i, j] > cm.max() * 0.5 else "#222222"))
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xlabel("Rounded OOF prediction")
    ax.set_ylabel("Observed Creq")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Cells")
    save_panel(fig, out / "S5A_05_ordinal_confusion")


def fig_permutation(null_df, observed, pval, out):
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.hist(null_df["rho"], bins=28, color="#C7CED6", edgecolor="white", linewidth=0.6)
    ax.axvline(observed, color=COLORS["accent"], lw=2.4)
    ax.set_xlabel("Null cross-validated Spearman ρ")
    ax.set_ylabel("Permutation count")
    ax.text(0.98, 0.95, f"observed ρ = {observed:.3f}\nfull-pipeline p = {pval:.4g}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    clean_ax(ax)
    save_panel(fig, out / "S5A_06_full_pipeline_permutation")


def fig_groupcv(summary, out):
    d = summary.copy()
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    x = np.arange(len(d))
    ax.bar(x, d["spearman_rho"], color=[COLORS["ephys"], COLORS["combined"], COLORS["orange"], COLORS["accent"]][:len(d)], width=0.62)
    ax.axhline(0, color="#999999", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(d["scheme"], rotation=24, ha="right")
    ax.set_ylabel("Spearman ρ")
    clean_ax(ax)
    save_panel(fig, out / "S5A_07_group_generalization")


def fig_group_effects(gdf, grouping, out, fname):
    d = gdf[gdf["grouping"] == grouping].copy().sort_values("rho")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, max(3.6, 0.45 * len(d) + 1.4)))
    y = np.arange(len(d))
    xerr = np.vstack([d["rho"] - d["bootstrap95_low"], d["bootstrap95_high"] - d["rho"]])
    ax.errorbar(d["rho"], y, xerr=xerr, fmt="o", color=COLORS["ephys"], ecolor=COLORS["ephys"], capsize=3)
    ax.axvline(0, color="#999999", ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}  (n={n})" for g, n in zip(d["group"], d["n"])])
    ax.set_xlabel("Within-group OOF Spearman ρ")
    clean_ax(ax, "x")
    save_panel(fig, out / fname)


def fig_feature_assoc(assoc, out):
    d = assoc.copy()
    d["absrho"] = d["rho"].abs()
    d = d.sort_values("absrho", ascending=False).head(16).sort_values("rho")
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    y = np.arange(len(d))
    col = [COLORS["ephys"] if v >= 0 else COLORS["accent"] for v in d["rho"]]
    ax.hlines(y, 0, d["rho"], color=col, lw=2)
    ax.scatter(d["rho"], y, color=col, s=34, zorder=3)
    ax.axvline(0, color="#999999", ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"].str.replace("_", " "))
    ax.set_xlabel("Spearman ρ with Creq")
    clean_ax(ax, "x")
    save_panel(fig, out / "S5A_10_feature_associations")


def fig_ablation(ablation, out):
    d = ablation.sort_values("delta_rho_full_minus_without")
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    y = np.arange(len(d))
    ax.barh(y, d["delta_rho_full_minus_without"], color=COLORS["combined"], alpha=0.9)
    ax.axvline(0, color="#999999", ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(d["removed_category"].str.replace("_", " "))
    ax.set_xlabel("Δρ: full model − category ablation")
    clean_ax(ax, "x")
    save_panel(fig, out / "S5A_11_category_ablation")


def fig_temporal_heatmap(temp, contexts, out):
    c = contexts
    tab = pd.crosstab(temp["C_temporal_AB"], temp["C_temporal_BA"]).reindex(index=c, columns=c, fill_value=0)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(tab.to_numpy(), cmap="Purples")
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c)
    ax.set_xlabel("Minimum sufficient context B→A (ms)")
    ax.set_ylabel("Minimum sufficient context A→B (ms)")
    for i in range(len(c)):
        for j in range(len(c)):
            v = tab.iloc[i, j]
            if v:
                ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=8, color=("white" if v > tab.to_numpy().max() * 0.45 else "#222222"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Cells")
    save_panel(fig, out / "S5A_12_temporal_AB_BA_consistency")


def fig_long_short(temp, out):
    x = temp["long_short_gain_mean"].dropna().to_numpy(float)
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    ax.hist(x, bins=40, color=COLORS["temporal"], alpha=0.85, edgecolor="white", linewidth=0.5)
    med = float(np.median(x)) if len(x) else np.nan
    ax.axvline(0, color="#888888", ls="--", lw=0.9)
    if np.isfinite(med):
        ax.axvline(med, color=COLORS["accent"], lw=2.2)
        ax.text(0.98, 0.95, f"median = {med:.4f}\nn = {len(x):,}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.set_xlabel("Held-out R² gain: 500 ms − 10 ms")
    ax.set_ylabel("Cells")
    clean_ax(ax)
    save_panel(fig, out / "S5A_13_long_vs_short_context_gain")


def fig_epsilon_rho(epsdf, out):
    if epsdf.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    cmap = {"Multiaxial ephys": COLORS["ephys"], "Temporal profile": COLORS["temporal"], "Metadata only": COLORS["metadata"]}
    for model, g in epsdf.groupby("model"):
        g = g.sort_values("epsilon")
        ax.plot(g["epsilon"], g["spearman_rho"], marker="o", label=model, color=cmap.get(model, None))
    ax.axhline(0, color="#999999", ls="--", lw=0.9)
    ax.set_xlabel("Creq sufficiency tolerance ε")
    ax.set_ylabel("Cross-validated Spearman ρ")
    clean_ax(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3)
    save_panel(fig, out / "S5A_14_epsilon_sensitivity_rho")


def fig_epsilon_distribution(epsdf, out):
    if epsdf.empty:
        return
    d = epsdf[epsdf["model"] == "Multiaxial ephys"].sort_values("epsilon")
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    bottom = np.zeros(len(d))
    cols = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#2171B5"]
    for ci in range(4):
        vals = d[f"C{ci}"].to_numpy(float)
        ax.bar(np.arange(len(d)), vals, bottom=bottom, color=cols[ci], width=0.65, label=f"Creq={ci}")
        bottom += vals
    ax.set_xticks(np.arange(len(d))); ax.set_xticklabels([f"{x:.2f}" for x in d["epsilon"]])
    ax.set_xlabel("Creq sufficiency tolerance ε")
    ax.set_ylabel("Cells")
    clean_ax(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4)
    save_panel(fig, out / "S5A_15_epsilon_Creq_distribution")


def fig_pfc_effects(effects, out):
    d = effects.sort_values("eta2_subtype", ascending=False).head(12).sort_values("eta2_subtype")
    fig, ax = plt.subplots(figsize=(6.3, 4.9))
    y = np.arange(len(d))
    ax.barh(y, d["eta2_subtype"], color=COLORS["orange"], alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(d["feature"].str.replace("_", " "))
    ax.set_xlabel("Descriptive subtype η²")
    clean_ax(ax, "x")
    save_panel(fig, out / "S5A_16_PFC_subtype_structural_effects_descriptive")


def fig_pfc_subtype_n(centroids, out):
    d = centroids.sort_values("n", ascending=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(np.arange(len(d)), d["n"], color=COLORS["gray"], width=0.8)
    ax.set_xlabel("Projection subtype (ranked by sample size)")
    ax.set_ylabel("Canonical neurons")
    ax.set_xticks([])
    clean_ax(ax)
    save_panel(fig, out / "S5A_17_PFC_subtype_sample_sizes")


def fig_zero_variance(qc, out):
    d = qc[qc["zero_variance"] == True].copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ax.barh(np.arange(len(d)), d["std"].fillna(0), color=COLORS["accent"])
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d["feature"].str.replace("_", " "))
    ax.set_xlabel("Standard deviation across 6,357 canonical neurons")
    clean_ax(ax, "x")
    save_panel(fig, out / "S5A_18_PFC_zero_variance_audit")


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def write_report(out_analysis, metrics, perm_obs, perm_p, group_summary, temporal, eps_status, pfc_qc_df, source_manifest):
    m = metrics.set_index("model")
    e = m.loc["Multiaxial ephys"]
    t = m.loc["Temporal profile"]
    md = m.loc["Metadata only"]
    naive = m.loc["Naive median"]
    trho, tp, tn = finite_spearman(temporal["C_temporal_AB"], temporal["C_temporal_BA"])
    lsg = float(np.median(temporal["long_short_gain_mean"]))
    zvars = pfc_qc_df[pfc_qc_df["zero_variance"] == True]["feature"].tolist()

    lines = [
        "# Stage 5A robustness closure v1",
        "",
        "## Primary conclusion",
        "",
        f"- Multiaxial ephys OOF Spearman ρ = {e['spearman_rho']:.3f}; MAE = {e['mae']:.3f}; R² = {e['r2']:.3f}.",
        f"- Temporal profile OOF Spearman ρ = {t['spearman_rho']:.3f}.",
        f"- Metadata-only OOF Spearman ρ = {md['spearman_rho']:.3f}.",
        f"- Naive-median MAE = {naive['mae']:.3f}; multiaxial MAE = {e['mae']:.3f}.",
        "",
        "Interpretation: multiaxial intrinsic physiology carries reproducible rank information about minimum mechanistic complexity, but absolute ordinal prediction remains modest and should not be described as high-accuracy prediction.",
        "",
        "## Full-pipeline permutation",
        "",
        f"- Re-run observed nested-CV ρ = {perm_obs:.3f}.",
        f"- Full-pipeline permutation p = {perm_p:.6g}; alpha is re-selected inside every permuted analysis.",
        "",
        "## Biological-group generalization",
        "",
    ]
    for _, r in group_summary.iterrows():
        lines.append(f"- {r['scheme']}: ρ={r['spearman_rho']:.3f}, MAE={r['mae']:.3f}, R²={r['r2']:.3f}, n={int(r['n'])}.")
    lines += [
        "",
        "Group-CV interpretation: leave-one-layer/dendrite analyses test robustness to coarse biological identity shifts. They do not substitute for donor/Cre-line grouped CV because donor/Cre identifiers are absent from the archived primary table.",
        "",
        "## Temporal boundary result",
        "",
        f"- AB vs BA minimum-context Spearman ρ = {trho:.3f} (n={tn}, p={tp:.3g}).",
        f"- Median held-out R² gain from 10 ms to 500 ms = {lsg:.4f}.",
        "",
        "Interpretation: timescale/context length is retained as a boundary dimension, not as the primary definition of intrinsic complexity.",
        "",
        "## Creq epsilon sensitivity",
        "",
        f"- Status: {eps_status.get('status')}",
        f"- GLIF atlas: {eps_status.get('glif_path')}",
    ]
    if eps_status.get("status") != "COMPLETE":
        lines.append("- This analysis requires the small `glif_performance_atlas_specimen_level.csv`; no epsilon result is fabricated when it is absent.")
    lines += [
        "",
        "## PFC atlas QC",
        "",
        f"- Zero-variance numeric fields: {', '.join(zvars) if zvars else 'none'}.",
        "- Projection-subtype η² remains descriptive structural-atlas evidence and is not interpreted as dynamical leverage.",
        "",
        "## Manuscript-safe Stage 5A claims",
        "",
        "1. Intrinsic dynamical complexity is not reducible to a single temporal context scale.",
        "2. Multiaxial electrophysiological phenotype contains held-out information about the minimum mechanistic complexity required to reproduce neuronal dynamics.",
        "3. Static metadata alone is a substantially weaker description of this complexity axis.",
        "4. PFC projection-defined roles exhibit strong non-random projectome organization, but this does not test the dynamical-leverage claim.",
        "5. The next in-vivo stage must independently define state-resolved functional/dynamical leverage.",
        "",
        "## Input provenance",
        "",
        "```json",
        json.dumps(source_manifest, indent=2, ensure_ascii=False),
        "```",
    ]
    (out_analysis / "stage5a_robustness_report.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--v11-results", type=Path, default=None)
    ap.add_argument("--v12-results", type=Path, default=None)
    ap.add_argument("--v11-source", type=Path, default=None)
    ap.add_argument("--v12-source", type=Path, default=None)
    ap.add_argument("--glif-atlas", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--permutations", type=int, default=300)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--materialize-source-data-only", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    bio = root / "bio data"
    v11_results = args.v11_results.resolve() if args.v11_results else bio / "stage5_v11_review_bundle.tar.gz"
    v12_results = args.v12_results.resolve() if args.v12_results else bio / "stage5_v12_results.tar.gz"
    v11_source = args.v11_source.resolve() if args.v11_source else bio / "stage5_v11_hotfix.zip"
    v12_source = args.v12_source.resolve() if args.v12_source else bio / "stage5_v12_biological_atlas.zip"
    outroot = args.output.resolve() if args.output else root / "plot" / "Stage5A_robustness_v1"
    out_analysis = ensure(outroot / "analysis")
    out_fig = ensure(outroot / "figures")
    out_source = ensure(outroot / "source_data")

    for p in [v11_results, v12_results, v11_source, v12_source]:
        if not p.exists():
            raise SystemExit(f"Missing required input: {p}")

    set_pub_style()

    print("[1/9] Input provenance and archive integrity ...", flush=True)
    expected_v11 = "stage5_v11_hotfix/scripts/allen_extract_v11.py"
    expected_v12 = "stage5_v12_biological_atlas/scripts/allen_multiaxial_v12.py"
    src_cfg = zip_json(v12_source, "stage5_v12_biological_atlas/config.json")
    source_manifest = {
        "v11_results": {"path": str(v11_results), "sha256": sha256_file(v11_results)},
        "v12_results": {"path": str(v12_results), "sha256": sha256_file(v12_results)},
        "v11_source": {"path": str(v11_source), "sha256": sha256_file(v11_source), "expected_script_present": zip_has(v11_source, expected_v11)},
        "v12_source": {"path": str(v12_source), "sha256": sha256_file(v12_source), "expected_script_present": zip_has(v12_source, expected_v12)},
        "v12_source_config": src_cfg,
    }
    (out_analysis / "input_provenance.json").write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[2/9] Loading Stage 5A result archives ...", flush=True)
    data = load_archives(v11_results, v12_results)
    materialize_archive_source_data(data, out_source)
    if args.materialize_source_data_only:
        print(f"Materialized Stage5A archive source tables: {out_source}", flush=True)
        return
    primary = data["primary"].copy().reset_index(drop=True)
    features = list(data["summary"]["features"])
    cv_cfg = dict(src_cfg["cv"])
    cv_cfg["permutations"] = int(args.permutations)
    cv_cfg["bootstrap"] = int(args.bootstrap)

    print("[3/9] Reproducing OOF metrics + naive baselines ...", flush=True)
    metrics = reproduce_model_metrics(primary)
    metrics.to_csv(out_analysis / "model_metrics_and_naive_baselines.csv", index=False, encoding="utf-8-sig")

    print("[4/9] Cohort attrition + temporal boundary audit ...", flush=True)
    attr = cohort_attrition(data, primary, out_analysis)
    temp, contexts = temporal_full_table(data["ab"], data["ba"], eps=float(src_cfg.get("c_model_epsilon", 0.02)))
    temp.to_csv(out_analysis / "temporal_boundary_1213_cells.csv", index=False, encoding="utf-8-sig")
    trho, tp, tn = finite_spearman(temp["C_temporal_AB"], temp["C_temporal_BA"])
    temporal_summary = {
        "n": int(len(temp)),
        "AB_BA_spearman_rho": trho,
        "AB_BA_spearman_p": tp,
        "median_long_short_gain_mean": float(np.median(temp["long_short_gain_mean"])),
        "contexts_ms": contexts,
    }
    (out_analysis / "temporal_boundary_summary.json").write_text(json.dumps(temporal_summary, indent=2), encoding="utf-8")

    print("[5/9] FULL nested-CV permutation for multiaxial ephys ...", flush=True)
    y = pd.to_numeric(primary["C_req_eps_0.02"], errors="coerce").to_numpy(float)
    X = primary[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pred_perm_obs, perm_met, null_df, perm_p = full_pipeline_permutation(X, y, cv_cfg, int(args.permutations), out_analysis / "full_pipeline_permutation_multiaxial.csv")
    pd.DataFrame({"specimen_id": primary["specimen_id"], "C_req": y, "pred_nested_rerun": pred_perm_obs}).to_csv(out_analysis / "nested_rerun_multiaxial_predictions.csv", index=False, encoding="utf-8-sig")
    perm_summary = {"observed": perm_met, "permutations": int(args.permutations), "full_pipeline_p": perm_p}
    (out_analysis / "full_pipeline_permutation_summary.json").write_text(json.dumps(perm_summary, indent=2), encoding="utf-8")

    print("[6/9] Biological-group generalization ...", flush=True)
    group_summary, within_groups = run_group_robustness(primary, features, cv_cfg, out_analysis, int(args.bootstrap))
    group_summary.to_csv(out_analysis / "group_generalization_summary.csv", index=False, encoding="utf-8-sig")

    print("[7/9] Optional Creq epsilon sensitivity ...", flush=True)
    glif_path = find_glif_atlas(root, args.glif_atlas.resolve() if args.glif_atlas else None)
    epsdf, eps_status = run_epsilon_sensitivity(primary, features, cv_cfg, glif_path, out_analysis)

    print("[8/9] PFC structural-atlas QC ...", flush=True)
    pfcqc = pfc_qc(data, out_analysis)

    print("[9/9] Building full standalone figure atlas ...", flush=True)
    fig_attrition(attr, out_fig)
    fig_model_rho(metrics, out_fig)
    fig_model_mae(metrics, out_fig)
    fig_prediction_by_true(primary, out_fig)
    fig_ordinal_confusion(primary, out_fig)
    fig_permutation(null_df, perm_met["spearman_rho"], perm_p, out_fig)
    fig_groupcv(group_summary, out_fig)
    fig_group_effects(within_groups, "layer", out_fig, "S5A_08_within_layer_oof_effects")
    fig_group_effects(within_groups, "dendrite_type", out_fig, "S5A_09_within_dendrite_oof_effects")
    fig_feature_assoc(data["assoc"], out_fig)
    fig_ablation(data["ablation"], out_fig)
    fig_temporal_heatmap(temp, contexts, out_fig)
    fig_long_short(temp, out_fig)
    fig_epsilon_rho(epsdf, out_fig)
    fig_epsilon_distribution(epsdf, out_fig)
    fig_pfc_effects(data["pfc_effects"], out_fig)
    fig_pfc_subtype_n(data["pfc_centroids"], out_fig)
    fig_zero_variance(pfcqc, out_fig)

    # Figure catalog
    figrows = []
    for p in sorted(out_fig.glob("*.png")):
        figrows.append({"figure": p.name, "pdf": p.with_suffix(".pdf").name if p.with_suffix(".pdf").exists() else "", "role": "candidate; triage MAIN/SI after analysis"})
    pd.DataFrame(figrows).to_csv(out_analysis / "figure_catalog.csv", index=False, encoding="utf-8-sig")

    write_report(out_analysis, metrics, perm_met["spearman_rho"], perm_p, group_summary, temp, eps_status, pfcqc, source_manifest)

    final = {
        "status": "COMPLETE",
        "primary_n": int(len(primary)),
        "temporal_n": int(len(temp)),
        "full_pipeline_permutation_p": float(perm_p),
        "epsilon_sensitivity_status": eps_status.get("status"),
        "output_root": str(outroot),
        "figure_count_png": len(list(out_fig.glob("*.png"))),
    }
    (out_analysis / "stage5a_robustness_status.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Stage 5A robustness closure complete ===", flush=True)
    print(json.dumps(final, indent=2, ensure_ascii=False), flush=True)
    print("\nRead first:", flush=True)
    print(out_analysis / "stage5a_robustness_report.md")
    print(out_analysis / "group_generalization_summary.csv")
    print(out_analysis / "full_pipeline_permutation_summary.json")
    print(out_analysis / "epsilon_sensitivity_status.json")
    print(out_analysis / "figure_catalog.csv")


if __name__ == "__main__":
    main()
