#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5C — Frozen shared-feature transfer bridge v2
===================================================

Scientific purpose
------------------
Test the fixed paper hypothesis without redefining complexity inside SynPhys:

    Stage5A intrinsic physiology -> C_req(GLIF, eps=0.02)
                         |
                         | train ONLY in Stage5A
                         v
              frozen reduced shared-feature mapper
                         |
                         | apply unchanged
                         v
               SynPhys C_transfer
                         |
                         v
            local synaptic leverage

This script deliberately DOES NOT fit a new "complexity" axis in SynPhys.

Why "shared-feature" mapper?
----------------------------
The original Stage5A full multiaxial feature set and SynPhys intrinsic table do
not use identical feature definitions. A direct application of the original
full model would therefore be scientifically invalid. Instead, this script:

1. reads the original Stage5A primary mouse cohort and its real feature list
   from stage5_v12_results.tar.gz;
2. uses a PREDECLARED, semantically strict harmonization map only;
3. trains a reduced mapper using Stage5A data only;
4. validates that reduced mapper by repeated nested CV in Stage5A;
5. freezes the final mapper;
6. applies it to SynPhys without using any SynPhys leverage outcome during
   feature selection, alpha selection, scaling, or fitting;
7. tests C_transfer vs local leverage.

Primary species: mouse.
Human is exported as an extension only and is NOT used for the primary claim.

Hard guardrails
---------------
- No SynPhys leverage outcome is used to choose features.
- No feature is mapped merely because names look similar.
- "fast trough delta-V" is NOT equated with Allen absolute trough voltage.
- Missing SynPhys shared features are not median-imputed across absent cells
  for the primary cohort.
- Primary transfer requires every selected shared feature to be present.
- Model/scaler are fit on Stage5A only.
- If fewer than 4 valid shared features survive, primary bridge aborts.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

DEFAULT_ROOT = Path.cwd()
DEFAULT_PHASE1 = DEFAULT_ROOT / "allen_synap" / "stage5c_synphys_phase1"
DEFAULT_OUT = DEFAULT_ROOT / "allen_synap" / "stage5c_synphys_transfer_v2"

# ---------------------------------------------------------------------
# PREDECLARED harmonization contract.
#
# stage5a_candidates:
#   exact known/likely Allen Cell Types ephys column names.
#
# synphys:
#   audited SynPhys r2.1 intrinsic-derived column.
#
# stage5a_to_canonical and synphys_to_canonical:
#   convert both datasets to the SAME human-readable physical units.
#
# Mapping is intentionally conservative.
# ---------------------------------------------------------------------

HARMONIZATION = [
    {
        "canonical": "adaptation_index",
        "stage5a_candidates": ["adaptation"],
        "synphys": "adaptation_index",
        "stage5a_scale": 1.0,
        "synphys_scale": 1.0,
        "unit": "dimensionless",
        "status": "strict",
    },
    {
        "canonical": "fi_slope_hz_per_pa",
        "stage5a_candidates": ["f_i_curve_slope"],
        "synphys": "fi_slope",
        # Allen Cell Types API convention is typically Hz/pA;
        # SynPhys stores SI Hz/A. Convert SynPhys to Hz/pA.
        "stage5a_scale": 1.0,
        "synphys_scale": 1e-12,
        "unit": "Hz/pA",
        "status": "strict",
    },
    {
        "canonical": "input_resistance_mohm",
        "stage5a_candidates": ["input_resistance_mohm"],
        "synphys": "input_resistance",
        "stage5a_scale": 1.0,
        "synphys_scale": 1e-6,
        "unit": "MOhm",
        "status": "strict",
    },
    {
        "canonical": "sag_ratio",
        "stage5a_candidates": ["sag"],
        "synphys": "sag",
        "stage5a_scale": 1.0,
        "synphys_scale": 1.0,
        "unit": "dimensionless",
        "status": "strict",
    },
    {
        "canonical": "threshold_v_mv",
        "stage5a_candidates": ["threshold_v_long_square"],
        "synphys": "ap_threshold_v",
        "stage5a_scale": 1.0,
        "synphys_scale": 1e3,
        "unit": "mV",
        "status": "strict",
    },
    {
        "canonical": "ap_width_ms",
        "stage5a_candidates": ["width_long_square"],
        "synphys": "ap_width",
        "stage5a_scale": 1.0,
        "synphys_scale": 1e3,
        "unit": "ms",
        "status": "strict",
    },
    {
        "canonical": "upstroke_downstroke_ratio",
        "stage5a_candidates": ["upstroke_downstroke_ratio_long_square"],
        "synphys": "ap_upstroke_downstroke_ratio",
        "stage5a_scale": 1.0,
        "synphys_scale": 1.0,
        "unit": "dimensionless",
        "status": "strict",
    },
    {
        "canonical": "rheobase_pa",
        "stage5a_candidates": ["threshold_i_long_square"],
        "synphys": "rheobase",
        "stage5a_scale": 1.0,
        "synphys_scale": 1e12,
        "unit": "pA",
        "status": "strict",
    },
]

ALPHAS = np.logspace(-3, 4, 24)
OUTER_FOLDS = 5
REPEATS = 10
SEED = 20260821
MIN_SHARED_FEATURES = 4


def find_v12_tar(root: Path) -> Path:
    exact = [
        root / "stage5_v12_results.tar.gz",
        root / "stage5_v12_biological_atlas.tar.gz",
    ]
    for p in exact:
        if p.exists():
            return p

    hits = []
    for pat in ["*v12*results*.tar.gz", "*v12*.tar.gz"]:
        hits.extend(root.rglob(pat))
    hits = [p for p in hits if p.is_file()]
    if not hits:
        raise FileNotFoundError(
            f"Could not find Stage5A/Stage5 v12 tar.gz under:\n{root}\n"
            "Expected something like stage5_v12_results.tar.gz"
        )

    # Verify by member name rather than trusting filename.
    for p in sorted(hits, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with tarfile.open(p, "r:gz") as tf:
                names = tf.getnames()
                if any("allen_multiaxial_primary_mouse_complete.csv" in n for n in names):
                    return p
        except Exception:
            pass

    raise FileNotFoundError(
        "Found v12-like archives, but none contains "
        "allen_multiaxial_primary_mouse_complete.csv"
    )


def read_stage5a(v12_tar: Path):
    with tarfile.open(v12_tar, "r:gz") as tf:
        names = tf.getnames()

        def find_member(suffix):
            hits = [n for n in names if n.endswith(suffix)]
            if len(hits) != 1:
                raise RuntimeError(f"Expected exactly one {suffix}; found {len(hits)}")
            return hits[0]

        pm = find_member("allen_multiaxial_primary_mouse_complete.csv")
        sm = find_member("allen_multiaxial_summary.json")

        pf = tf.extractfile(pm)
        sf = tf.extractfile(sm)
        if pf is None or sf is None:
            raise RuntimeError("Could not extract Stage5A primary/summary.")

        primary = pd.read_csv(pf)
        summary = json.loads(sf.read().decode("utf-8"))

    return primary, summary, pm, sm


def choose_stage5a_column(row, original_features):
    candidates = row["stage5a_candidates"]
    found = [c for c in candidates if c in original_features]
    if not found:
        return None
    if len(found) > 1:
        # Predeclared candidates are aliases of same quantity; select first in contract.
        return found[0]
    return found[0]


def build_contract(stage5a_features, syn_cols):
    rows = []
    for h in HARMONIZATION:
        a = choose_stage5a_column(h, stage5a_features)
        s = h["synphys"] if h["synphys"] in syn_cols else None
        usable = a is not None and s is not None and h["status"] == "strict"
        rows.append({
            **h,
            "stage5a_selected": a,
            "synphys_present": s is not None,
            "stage5a_present": a is not None,
            "usable": usable,
        })
    return pd.DataFrame(rows)


def canonical_matrices(stage5a, syn, contract):
    usable = contract[contract["usable"]].copy()
    A = pd.DataFrame(index=stage5a.index)
    S = pd.DataFrame(index=syn.index)

    for _, r in usable.iterrows():
        name = r["canonical"]
        A[name] = pd.to_numeric(
            stage5a[r["stage5a_selected"]], errors="coerce"
        ) * float(r["stage5a_scale"])
        S[name] = pd.to_numeric(
            syn[r["synphys"]], errors="coerce"
        ) * float(r["synphys_scale"])

    return A, S, usable


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


def scaler_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    return mu, sd


def scaler_apply(X, mu, sd):
    return (X - mu) / sd


def ridge_fit(X, y, alpha):
    mu, sd = scaler_fit(X)
    Z = scaler_apply(X, mu, sd)
    ym = float(np.mean(y))
    yy = y - ym
    A = Z.T @ Z + float(alpha) * np.eye(Z.shape[1])
    try:
        w = np.linalg.solve(A, Z.T @ yy)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ (Z.T @ yy)
    return {"mu": mu, "sd": sd, "ym": ym, "w": w, "alpha": float(alpha)}


def ridge_predict(model, X):
    Z = scaler_apply(X, model["mu"], model["sd"])
    return Z @ model["w"] + model["ym"]


def choose_alpha_inner(X, y, seed, k=5):
    folds = stratified_folds(y, k, seed)
    scores = []
    for a in ALPHAS:
        mse = []
        for val in folds:
            tr = np.setdiff1d(np.arange(len(y)), val)
            m = ridge_fit(X[tr], y[tr], a)
            p = ridge_predict(m, X[val])
            mse.append(np.mean((p - y[val]) ** 2))
        scores.append(np.mean(mse))
    return float(ALPHAS[int(np.argmin(scores))])


def repeated_nested_cv(X, y):
    pred_sum = np.zeros(len(y))
    pred_n = np.zeros(len(y))
    chosen = []

    for rep in range(REPEATS):
        folds = stratified_folds(y, OUTER_FOLDS, SEED + rep * 997)
        for fi, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(y)), te)
            alpha = choose_alpha_inner(X[tr], y[tr], SEED + rep * 5003 + fi)
            chosen.append(alpha)
            m = ridge_fit(X[tr], y[tr], alpha)
            pred_sum[te] += ridge_predict(m, X[te])
            pred_n[te] += 1

    pred = pred_sum / np.maximum(pred_n, 1)
    if spearmanr is not None:
        rho, p = spearmanr(y, pred)
        rho, p = float(rho), float(p)
    else:
        rho = float(pd.Series(y).corr(pd.Series(pred), method="spearman"))
        p = np.nan

    mae = float(np.mean(np.abs(y - pred)))
    ssr = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ssr / sst if sst > 0 else np.nan
    return pred, {
        "n": int(len(y)),
        "rho": rho,
        "p": p,
        "mae": mae,
        "r2": r2,
        "median_alpha": float(np.median(chosen)),
    }


def fit_final_stage5a(X, y):
    alpha = choose_alpha_inner(X, y, SEED + 99001, k=5)
    model = ridge_fit(X, y, alpha)
    return model


def transfer_domain_qc(A, S, feature_names):
    """Flag severe extrapolation using Stage5A training mean/sd and ranges."""
    rows = []
    per_cell = pd.DataFrame(index=S.index)
    for c in feature_names:
        a = A[c].to_numpy(float)
        s = S[c].to_numpy(float)
        mu = np.mean(a)
        sd = np.std(a)
        lo, hi = np.min(a), np.max(a)
        z = (s - mu) / sd if sd > 0 else np.zeros_like(s)
        outside = (s < lo) | (s > hi)
        per_cell[f"ood_{c}"] = outside
        rows.append({
            "feature": c,
            "stage5a_n": int(np.isfinite(a).sum()),
            "synphys_n": int(np.isfinite(s).sum()),
            "stage5a_mean": float(np.nanmean(a)),
            "stage5a_sd": float(np.nanstd(a)),
            "stage5a_min": float(np.nanmin(a)),
            "stage5a_max": float(np.nanmax(a)),
            "synphys_median": float(np.nanmedian(s)),
            "synphys_outside_train_range_fraction": float(np.nanmean(outside)),
            "synphys_abs_z_gt_4_fraction": float(np.nanmean(np.abs(z) > 4)),
        })
    return pd.DataFrame(rows), per_cell


def spearman_safe(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 20:
        return {"n": int(m.sum()), "rho": None, "p": None}
    if spearmanr is None:
        return {
            "n": int(m.sum()),
            "rho": float(x[m].corr(y[m], method="spearman")),
            "p": None,
        }
    r, p = spearmanr(x[m], y[m])
    return {"n": int(m.sum()), "rho": float(r), "p": float(p)}


def nested_increment_model(df, outcome):
    """
    M1: metadata + structural leverage
    M2: M1 + frozen Stage5A-derived C_transfer

    Cluster-robust SE by experiment_id. Delta R2 is descriptive incremental fit;
    beta/p use cluster-robust covariance.
    """
    try:
        import statsmodels.api as sm
    except Exception:
        return {"status": "statsmodels_missing"}

    cols = [
        outcome, "C_transfer", "L_struct_out", "age", "cell_class",
        "cortical_layer", "cre_type", "target_region", "experiment_id",
    ]
    d = df[[c for c in cols if c in df.columns]].copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[outcome, "C_transfer", "L_struct_out", "experiment_id"])
    if len(d) < 100:
        return {"status": "too_few", "n": int(len(d))}

    X = pd.DataFrame(index=d.index)
    for c in ["age", "L_struct_out"]:
        if c in d:
            v = pd.to_numeric(d[c], errors="coerce")
            X[c] = v.fillna(v.median())

    for c in ["cell_class", "cortical_layer", "cre_type", "target_region"]:
        if c in d and d[c].notna().mean() >= 0.5:
            vc = d[c].fillna("MISSING").astype(str)
            # collapse rare levels to prevent unstable giant design
            counts = vc.value_counts()
            keep = counts[counts >= max(10, int(0.01 * len(d)))].index
            vc = vc.where(vc.isin(keep), "OTHER")
            dm = pd.get_dummies(vc, prefix=c, drop_first=True, dtype=float)
            if dm.shape[1] <= 40:
                X = pd.concat([X, dm], axis=1)

    y = pd.to_numeric(d[outcome], errors="coerce").astype(float)
    X1 = sm.add_constant(X.astype(float), has_constant="add")
    m1 = sm.OLS(y, X1).fit()

    X2 = X1.copy()
    X2["C_transfer"] = pd.to_numeric(d["C_transfer"], errors="coerce").astype(float)
    m2_plain = sm.OLS(y, X2).fit()
    m2 = m2_plain.get_robustcov_results(
        cov_type="cluster",
        groups=d["experiment_id"].astype(str).to_numpy()
    )

    # robust results returns ndarray; map names explicitly
    names = list(X2.columns)
    bi = names.index("C_transfer")
    return {
        "status": "ok",
        "n": int(len(d)),
        "n_experiments": int(d["experiment_id"].nunique()),
        "r2_M1": float(m1.rsquared),
        "r2_M2": float(m2_plain.rsquared),
        "delta_r2": float(m2_plain.rsquared - m1.rsquared),
        "beta_C": float(m2.params[bi]),
        "se_C_cluster": float(m2.bse[bi]),
        "p_C_cluster": float(m2.pvalues[bi]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--phase1", type=Path, default=DEFAULT_PHASE1)
    ap.add_argument("--v12-tar", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    root = args.root.resolve()
    phase1 = args.phase1.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    v12 = args.v12_tar.resolve() if args.v12_tar else find_v12_tar(root)
    intrinsic_path = phase1 / "stage5a_compatible_intrinsic_matrix.csv"
    leverage_path = phase1 / "cell_local_leverage_atlas.csv"

    if not intrinsic_path.exists() or not leverage_path.exists():
        raise SystemExit(
            "Phase1 outputs not found. Required:\n"
            f"{intrinsic_path}\n{leverage_path}"
        )

    print("=" * 100)
    print("Stage 5C frozen shared-feature transfer bridge v2")
    print("=" * 100)
    print("Stage5A archive:", v12)
    print("SynPhys Phase1 :", phase1)
    print("Output         :", out)
    print("=" * 100)

    stage5a, summary, primary_member, summary_member = read_stage5a(v12)
    original_features = list(summary.get("features", []))
    syn = pd.read_csv(intrinsic_path, low_memory=False)
    lev = pd.read_csv(leverage_path, low_memory=False)

    print("Stage5A primary n:", len(stage5a))
    print("Stage5A full feature count:", len(original_features))
    print("SynPhys intrinsic rows:", len(syn))

    contract = build_contract(original_features, set(syn.columns))
    contract.to_csv(out / "harmonization_contract.csv", index=False, encoding="utf-8-sig")

    A, S, usable = canonical_matrices(stage5a, syn, contract)
    feature_names = usable["canonical"].tolist()

    print("Strict shared features:", feature_names)

    if len(feature_names) < MIN_SHARED_FEATURES:
        report = {
            "status": "ABORT_INSUFFICIENT_SHARED_FEATURES",
            "n_shared": len(feature_names),
            "shared_features": feature_names,
            "stage5a_features": original_features,
        }
        (out / "transfer_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        raise SystemExit(
            f"Only {len(feature_names)} strict shared features survived; "
            f"need >= {MIN_SHARED_FEATURES}. No complexity transfer was fabricated."
        )

    # Primary Stage5A training target
    y = pd.to_numeric(stage5a["C_req_eps_0.02"], errors="coerce")

    # COMPLETE-CASE ONLY on shared features in Stage5A training.
    train_mask = A.notna().all(axis=1) & y.notna()
    At = A.loc[train_mask].to_numpy(float)
    yt = y.loc[train_mask].to_numpy(float)

    if len(yt) < 200:
        raise SystemExit(
            f"Only {len(yt)} Stage5A complete cases for shared mapper; abort."
        )

    print("Stage5A shared-feature training n:", len(yt))
    print("Running repeated nested CV...")
    oof, cv = repeated_nested_cv(At, yt)
    print("Reduced frozen-mapper CV:", cv)

    # Gate: mapper must preserve meaningful Stage5A signal before transfer.
    # This is deliberately modest because original full multiaxial rho ~0.26.
    if not np.isfinite(cv["rho"]) or cv["rho"] < 0.10:
        report = {
            "status": "ABORT_STAGE5A_SHARED_MAPPER_TOO_WEAK",
            "cv": cv,
            "shared_features": feature_names,
        }
        (out / "transfer_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        raise SystemExit(
            f"Reduced shared-feature mapper Stage5A CV rho={cv['rho']:.3f} < 0.10. "
            "No SynPhys bridge claim will be attempted."
        )

    pd.DataFrame({
        "specimen_id": stage5a.loc[train_mask, "specimen_id"].values,
        "C_req_eps_0.02": yt,
        "C_shared_oof": oof,
    }).to_csv(out / "stage5a_shared_mapper_oof.csv", index=False)

    final_model = fit_final_stage5a(At, yt)

    model_json = {
        "type": "ridge_continuous_Creq",
        "training_dataset": "Stage5A Allen Cell Types mouse complete GLIF1-5 cohort",
        "target": "C_req_eps_0.02",
        "features": feature_names,
        "alpha": final_model["alpha"],
        "mean": dict(zip(feature_names, map(float, final_model["mu"]))),
        "sd": dict(zip(feature_names, map(float, final_model["sd"]))),
        "weights": dict(zip(feature_names, map(float, final_model["w"]))),
        "intercept_target_mean": float(final_model["ym"]),
        "stage5a_cv": cv,
        "primary_member": primary_member,
        "summary_member": summary_member,
        "scientific_status": "FROZEN_REDUCED_SHARED_FEATURE_MAPPER",
    }
    (out / "frozen_shared_mapper.json").write_text(
        json.dumps(model_json, indent=2), encoding="utf-8"
    )

    # Apply only to complete-case SynPhys cells.
    syn_mask = S.notna().all(axis=1)
    syn["C_transfer"] = np.nan
    syn.loc[syn_mask, "C_transfer"] = ridge_predict(
        final_model, S.loc[syn_mask].to_numpy(float)
    )

    # Out-of-domain flags
    domain_qc, ood = transfer_domain_qc(
        A.loc[train_mask], S.loc[syn_mask], feature_names
    )
    domain_qc.to_csv(out / "transfer_domain_qc_by_feature.csv", index=False)
    syn["n_feature_ood"] = np.nan
    syn.loc[syn_mask, "n_feature_ood"] = ood.sum(axis=1).to_numpy()
    syn["transfer_domain_primary"] = (
        syn_mask & (syn["n_feature_ood"].fillna(999) <= 1)
    )

    syn.to_csv(out / "synphys_frozen_complexity_transfer.csv", index=False)

    merged = lev.merge(
        syn[["cell_id", "C_transfer", "n_feature_ood", "transfer_domain_primary"]],
        on="cell_id",
        how="left",
        suffixes=("", "_transfer"),
    )

    # Primary = mouse, complete shared features, <=1 OOD feature, structural QC.
    primary = merged[
        (merged["species"].astype(str).str.lower() == "mouse") &
        merged["transfer_domain_primary"].fillna(False)
    ].copy()

    # Scientific outputs
    outcomes = [
        "L_struct_out",
        "L_psp_out_sum_abs",
        "L_psc_out_sum_abs",
        "L_dyn50_out_sum_abs",
    ]

    assoc = {
        o: spearman_safe(primary["C_transfer"], primary[o])
        for o in outcomes if o in primary
    }

    models = {}
    for o in ["L_psp_out_sum_abs", "L_psc_out_sum_abs", "L_dyn50_out_sum_abs"]:
        if o in primary:
            d = primary[
                primary["primary_structural_qc"].fillna(False)
            ].copy()
            models[o] = nested_increment_model(d, o)

    primary.to_csv(out / "synphys_mouse_primary_transfer_bridge.csv", index=False)

    # Human extension exported but not primary inference.
    human = merged[
        (merged["species"].astype(str).str.lower() == "human") &
        merged["transfer_domain_primary"].fillna(False)
    ].copy()
    human.to_csv(out / "synphys_human_transfer_extension.csv", index=False)

    report = {
        "status": "COMPLETE",
        "scientific_design": "Stage5A-only trained frozen reduced shared-feature transfer to SynPhys",
        "stage5a_archive": str(v12),
        "original_stage5a_full_features": original_features,
        "shared_features": feature_names,
        "n_shared_features": len(feature_names),
        "stage5a_training_complete_cases": int(len(yt)),
        "stage5a_reduced_mapper_cv": cv,
        "final_alpha": final_model["alpha"],
        "synphys_all_cells": int(len(syn)),
        "synphys_complete_shared_features": int(syn_mask.sum()),
        "synphys_mouse_primary_domain_cells": int(len(primary)),
        "synphys_human_extension_domain_cells": int(len(human)),
        "mouse_primary_associations": assoc,
        "mouse_primary_nested_models": models,
        "interpretation_guardrails": [
            "C_transfer was trained only on Stage5A mouse data.",
            "No SynPhys leverage outcome influenced feature selection, scaling, alpha selection, or fitting.",
            "This is a reduced shared-feature transfer mapper, not a claim that the full original Stage5A mapper was directly portable.",
            "Primary inference is mouse only.",
            "Human results are extension only.",
            "Dynamic/weighted leverage claims require adequate pair-level QC and are conditional on Phase1 definitions.",
        ],
    }
    (out / "transfer_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "# Stage 5C — Frozen shared-feature transfer bridge v2",
        "",
        f"- Stage5A archive: `{v12}`",
        f"- Strict harmonized features: **{len(feature_names)}**",
        f"- Features: `{', '.join(feature_names)}`",
        f"- Stage5A complete-case training n: **{len(yt)}**",
        f"- Stage5A reduced-mapper nested-CV Spearman rho: **{cv['rho']:.3f}**",
        f"- Stage5A reduced-mapper MAE: **{cv['mae']:.3f} Creq units**",
        f"- SynPhys mouse primary-domain cells: **{len(primary)}**",
        "",
        "## Mouse primary associations",
        "",
    ]
    for o, r in assoc.items():
        lines.append(
            f"- `{o}`: rho={r['rho'] if r['rho'] is not None else 'NA'}, "
            f"p={r['p'] if r['p'] is not None else 'NA'}, n={r['n']}"
        )
    lines += ["", "## Mouse nested M1 -> M2 tests", ""]
    for o, r in models.items():
        lines.append(f"- `{o}`: `{json.dumps(r, ensure_ascii=False)}`")
    lines += [
        "",
        "## Interpretation rule",
        "",
        "The fixed scientific target is unchanged: test whether externally defined intrinsic "
        "complexity explains local functional/synaptic leverage beyond structural opportunity "
        "and cell identity. This script does not redefine complexity inside SynPhys.",
        "",
        "A paper-level positive bridge requires BOTH:",
        "1. the reduced shared-feature mapper retains held-out Stage5A validity; and",
        "2. `C_transfer` contributes beyond `L_struct_out` and metadata in the SynPhys mouse primary cohort.",
    ]
    (out / "00_TRANSFER_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print()
    print("=" * 100)
    print("TRANSFER COMPLETE")
    print("Stage5A CV rho:", f"{cv['rho']:.3f}")
    print("Shared features:", feature_names)
    print("Mouse primary cells:", len(primary))
    print("Output:", out)
    print("=" * 100)


if __name__ == "__main__":
    main()
