#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5C — Identity organization / variance decomposition v5
============================================================

Scientific question
-------------------
Does the observed association between externally defined intrinsic complexity
(C_transfer) and standardized synaptic dynamic role primarily reflect:

    (a) an independent within-identity complexity gradient, or
    (b) biological organization across cell identities?

This script performs a predeclared hierarchical decomposition on the v4
standardized-synapse atlas.

Primary outcome
---------------
G_norm_50hz

Secondary outcomes
------------------
G_norm_5hz, G_norm_10hz, G_norm_20hz,
tail_first_50hz, dyn_range_50hz

Model ladder
------------
M0  outcome ~ base controls
M1  outcome ~ base controls + C_transfer
M2  outcome ~ base controls + cell identity
M3  outcome ~ base controls + cell identity + C_transfer

Base controls:
    distance
    pre/post cortical layer
    pre age
    first-pulse strength A1_abs_50hz (for normalized dynamics)

Fine biological identity:
    pre/post cell_class
    pre/post cre_type

The key quantities are:
    deltaR2_C_before_identity = R2(M1) - R2(M0)
    deltaR2_C_after_identity  = R2(M3) - R2(M2)

and the attenuation:
    1 - delta_after / delta_before

This is an organization / variance-decomposition analysis, NOT a causal
mediation analysis. It must not be described as proving that cell identity
causes or mediates complexity.

Robustness
----------
1. Experiment-cluster-robust inference for beta_C in M1 and M3.
2. GroupKFold cross-validation split by experiment_id for M0-M3.
3. Within-presynaptic-Cre residual complexity analysis:
       C_within = C - mean(C | pre_cre_type)
   tested against G_norm_50hz with base controls.
4. Broad-identity decomposition using cell_class only.
5. Fine-identity decomposition using cell_class + cre_type.

No SynPhys outcome is used to define or refit C_transfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path.cwd()
DEFAULT_V4 = DEFAULT_ROOT / "stage5c_dynamic_leverage_v4"
DEFAULT_OUT = DEFAULT_ROOT / "stage5c_identity_decomposition_v5"

PRIMARY_OUTCOME = "G_norm_50hz"
SECONDARY_OUTCOMES = [
    "G_norm_5hz",
    "G_norm_10hz",
    "G_norm_20hz",
    "tail_first_50hz",
    "dyn_range_50hz",
]
SEED = 20260821
N_SPLITS = 5


def collapse_rare(s: pd.Series, min_n: int):
    s = s.fillna("MISSING").astype(str)
    vc = s.value_counts()
    keep = vc[vc >= min_n].index
    return s.where(s.isin(keep), "OTHER")


def zscore(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    sd = x.std()
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def prepare_analysis_df(df: pd.DataFrame, outcome: str):
    d = df.copy()
    d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
    d["C_transfer"] = pd.to_numeric(d["C_transfer"], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[outcome, "C_transfer", "experiment_id"])

    # Predeclare only mild outcome winsorization for stability.
    if len(d) >= 100:
        lo, hi = d[outcome].quantile([0.01, 0.99])
        d[outcome] = d[outcome].clip(lo, hi)
    return d


def base_design(d: pd.DataFrame, outcome: str):
    X = pd.DataFrame(index=d.index)

    if "distance" in d:
        x = pd.to_numeric(d["distance"], errors="coerce")
        if x.notna().sum() >= 30:
            x = x.fillna(x.median())
            X["log1p_distance_um"] = np.log1p(np.maximum(x * 1e6, 0))

    if "pre_age" in d:
        x = pd.to_numeric(d["pre_age"], errors="coerce")
        if x.notna().sum() >= 30:
            X["pre_age"] = x.fillna(x.median())

    # For normalized dynamic endpoints, explicitly account for baseline strength.
    if outcome.startswith(("G_norm_", "tail_first_", "dyn_range_")) and "A1_abs_50hz" in d:
        x = pd.to_numeric(d["A1_abs_50hz"], errors="coerce")
        x = np.log(np.maximum(x, 1e-15))
        if x.notna().sum() >= 30:
            X["log_A1_abs_50hz"] = x.fillna(x.median())

    # Layer is treated as anatomy/base context, not fine identity.
    for c in ["pre_cortical_layer", "post_cortical_layer"]:
        if c in d:
            s = collapse_rare(d[c], max(10, int(0.01 * len(d))))
            dm = pd.get_dummies(s, prefix=c, drop_first=True, dtype=float)
            if dm.shape[1] <= 20:
                X = pd.concat([X, dm], axis=1)

    return X.astype(float)


def add_identity(X: pd.DataFrame, d: pd.DataFrame, tier: str):
    out = X.copy()
    if tier == "none":
        return out
    if tier == "broad":
        cats = ["pre_cell_class", "post_cell_class"]
    elif tier == "fine":
        cats = ["pre_cell_class", "post_cell_class", "pre_cre_type", "post_cre_type"]
    else:
        raise ValueError(tier)

    for c in cats:
        if c not in d:
            continue
        s = collapse_rare(d[c], max(10, int(0.0125 * len(d))))
        dm = pd.get_dummies(s, prefix=c, drop_first=True, dtype=float)
        if dm.shape[1] <= 35:
            out = pd.concat([out, dm], axis=1)
    return out.astype(float)


def fit_model(y, X, groups):
    import statsmodels.api as sm

    Xc = sm.add_constant(X.astype(float), has_constant="add")
    plain = sm.OLS(y.astype(float), Xc).fit()
    robust = plain.get_robustcov_results(
        cov_type="cluster",
        groups=groups.astype(str).to_numpy(),
    )
    names = list(Xc.columns)
    return plain, robust, names


def extract_c(plain, robust, names):
    if "C_transfer_z" not in names:
        return {
            "beta_C_z": None,
            "se_C_z_cluster": None,
            "p_C_z_cluster": None,
        }
    i = names.index("C_transfer_z")
    return {
        "beta_C_z": float(robust.params[i]),
        "se_C_z_cluster": float(robust.bse[i]),
        "p_C_z_cluster": float(robust.pvalues[i]),
    }


def model_ladder(d: pd.DataFrame, outcome: str, identity_tier: str):
    y = d[outcome].astype(float)
    groups = d["experiment_id"]

    B = base_design(d, outcome)
    C = zscore(d["C_transfer"]).rename("C_transfer_z")
    I = add_identity(B, d, identity_tier)

    X0 = B
    X1 = B.copy()
    X1["C_transfer_z"] = C
    X2 = I
    X3 = I.copy()
    X3["C_transfer_z"] = C

    fits = {}
    for name, X in [("M0", X0), ("M1", X1), ("M2", X2), ("M3", X3)]:
        plain, robust, names = fit_model(y, X, groups)
        rec = {
            "n": int(len(d)),
            "n_experiments": int(groups.nunique()),
            "r2": float(plain.rsquared),
            "adj_r2": float(plain.rsquared_adj),
            "n_parameters": int(len(names)),
        }
        rec.update(extract_c(plain, robust, names))
        fits[name] = rec

    r0, r1, r2, r3 = [fits[m]["r2"] for m in ["M0", "M1", "M2", "M3"]]
    c_before = r1 - r0
    c_after = r3 - r2
    identity_before_c = r2 - r0
    identity_after_c = r3 - r1
    joint = r3 - r0
    shared = (r1 - r0) + (r2 - r0) - joint

    attenuation = None
    if c_before > 0:
        attenuation = 1.0 - (c_after / c_before)

    decomposition = {
        "delta_r2_C_before_identity": float(c_before),
        "delta_r2_C_after_identity": float(c_after),
        "delta_r2_identity_before_C": float(identity_before_c),
        "delta_r2_identity_after_C": float(identity_after_c),
        "joint_increment_over_base": float(joint),
        "shared_overlap_component": float(shared),
        "C_attenuation_fraction_after_identity": (
            float(attenuation) if attenuation is not None else None
        ),
        "note": "shared_overlap_component is a commonality-style descriptive quantity and may be negative under suppression/collinearity.",
    }
    return fits, decomposition


def group_cv_ladder(d: pd.DataFrame, outcome: str, identity_tier: str):
    """
    Out-of-sample R2 with GroupKFold by experiment.
    Design matrices are built globally only for categorical encoding; outcome and C are never
    used to select levels. Standard OLS coefficients are fit on train folds only.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    y = d[outcome].astype(float)
    groups = d["experiment_id"].astype(str)

    B = base_design(d, outcome)
    C = zscore(d["C_transfer"]).rename("C_transfer_z")
    I = add_identity(B, d, identity_tier)

    designs = {
        "M0": B,
        "M1": pd.concat([B, C], axis=1),
        "M2": I,
        "M3": pd.concat([I, C], axis=1),
    }

    n_groups = groups.nunique()
    n_splits = min(N_SPLITS, n_groups)
    if n_splits < 3:
        return {"status": "too_few_groups", "n_groups": int(n_groups)}

    gkf = GroupKFold(n_splits=n_splits)
    preds = {m: np.full(len(d), np.nan) for m in designs}
    yarr = y.to_numpy(float)

    for tr, te in gkf.split(np.arange(len(d)), yarr, groups):
        for m, X in designs.items():
            Xa = X.to_numpy(float)
            # Tiny ridge only for numerical stability with dummy-rich design.
            model = Ridge(alpha=1e-8, fit_intercept=True)
            model.fit(Xa[tr], yarr[tr])
            preds[m][te] = model.predict(Xa[te])

    out = {"status": "ok", "n_splits": int(n_splits)}
    for m, p in preds.items():
        valid = np.isfinite(p)
        out[f"{m}_cv_r2"] = float(r2_score(yarr[valid], p[valid]))

    out["delta_cv_r2_C_before_identity"] = (
        out["M1_cv_r2"] - out["M0_cv_r2"]
    )
    out["delta_cv_r2_C_after_identity"] = (
        out["M3_cv_r2"] - out["M2_cv_r2"]
    )
    return out


def within_identity_analysis(d: pd.DataFrame, outcome: str):
    import statsmodels.api as sm

    w = d.copy()
    if "pre_cre_type" not in w:
        return {"status": "missing_pre_cre_type"}

    # Require reasonably populated identities; otherwise preserve but do not estimate group mean
    counts = w["pre_cre_type"].fillna("MISSING").astype(str).value_counts()
    valid_types = counts[counts >= 15].index
    w["_pre_id"] = w["pre_cre_type"].fillna("MISSING").astype(str)
    w = w[w["_pre_id"].isin(valid_types)].copy()
    if len(w) < 120:
        return {"status": "too_few", "n": int(len(w))}

    c = pd.to_numeric(w["C_transfer"], errors="coerce")
    group_mean = c.groupby(w["_pre_id"]).transform("mean")
    w["C_within_cre"] = c - group_mean
    w["C_within_cre_z"] = zscore(w["C_within_cre"])

    y = w[outcome].astype(float)
    X = base_design(w, outcome)
    X["C_within_cre_z"] = w["C_within_cre_z"]
    X = sm.add_constant(X.astype(float), has_constant="add")

    plain = sm.OLS(y, X).fit()
    rob = plain.get_robustcov_results(
        cov_type="cluster",
        groups=w["experiment_id"].astype(str).to_numpy(),
    )
    names = list(X.columns)
    i = names.index("C_within_cre_z")
    return {
        "status": "ok",
        "n": int(len(w)),
        "n_pre_cre_types": int(w["_pre_id"].nunique()),
        "n_experiments": int(w["experiment_id"].nunique()),
        "beta_C_within_cre_z": float(rob.params[i]),
        "se_cluster": float(rob.bse[i]),
        "p_cluster": float(rob.pvalues[i]),
        "r2": float(plain.rsquared),
    }


def identity_summary(d: pd.DataFrame, outcome: str):
    rows = []
    for c in ["pre_cell_class", "pre_cre_type", "pre_cortical_layer"]:
        if c not in d:
            continue
        for level, g in d.groupby(c, dropna=False):
            if len(g) < 10:
                continue
            rows.append({
                "identity_variable": c,
                "identity_level": str(level),
                "n_pairs": int(len(g)),
                "n_pre_cells": int(g["pre_cell_id"].nunique()) if "pre_cell_id" in g else None,
                "C_mean": float(pd.to_numeric(g["C_transfer"], errors="coerce").mean()),
                "C_sd": float(pd.to_numeric(g["C_transfer"], errors="coerce").std()),
                "outcome_mean": float(pd.to_numeric(g[outcome], errors="coerce").mean()),
                "outcome_sd": float(pd.to_numeric(g[outcome], errors="coerce").std()),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", type=Path, default=DEFAULT_V4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    v4 = args.v4.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    atlas_path = v4 / "standardized_synapse_train_atlas.csv"
    if not atlas_path.exists():
        raise SystemExit(f"Missing v4 atlas: {atlas_path}")

    print("=" * 100)
    print("Stage 5C identity organization / variance decomposition v5")
    print("=" * 100)
    print("Input :", atlas_path)
    print("Output:", out)

    df = pd.read_csv(atlas_path, low_memory=False)

    outcomes = [PRIMARY_OUTCOME] + [
        x for x in SECONDARY_OUTCOMES if x in df.columns
    ]

    report = {
        "status": "COMPLETE",
        "primary_outcome": PRIMARY_OUTCOME,
        "interpretation": "organization/variance decomposition; not causal mediation",
        "outcomes": {},
        "guardrails": [
            "C_transfer remains frozen from Stage5A.",
            "No SynPhys outcome is used to define C_transfer.",
            "M0/M1/M2/M3 are predeclared before inspecting these model outputs.",
            "Experiment-cluster robust SEs are used for in-sample inference.",
            "GroupKFold by experiment is used for out-of-sample robustness.",
            "Cell identity attenuation is descriptive and must not be called causal mediation.",
        ],
    }

    for outcome in outcomes:
        d = prepare_analysis_df(df, outcome)
        print(f"Analyzing {outcome}: n={len(d)}")

        broad_fits, broad_dec = model_ladder(d, outcome, "broad")
        fine_fits, fine_dec = model_ladder(d, outcome, "fine")

        rec = {
            "n": int(len(d)),
            "n_experiments": int(d["experiment_id"].nunique()),
            "broad_identity": {
                "models": broad_fits,
                "decomposition": broad_dec,
                "group_cv": group_cv_ladder(d, outcome, "broad"),
            },
            "fine_identity": {
                "models": fine_fits,
                "decomposition": fine_dec,
                "group_cv": group_cv_ladder(d, outcome, "fine"),
            },
            "within_pre_cre": within_identity_analysis(d, outcome),
        }
        report["outcomes"][outcome] = rec

        if outcome == PRIMARY_OUTCOME:
            identity_summary(d, outcome).to_csv(
                out / "primary_identity_group_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )

    # Save machine-readable report
    (out / "identity_decomposition_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Compact table for manuscript / master AI
    table_rows = []
    for outcome, rec in report["outcomes"].items():
        for tier in ["broad_identity", "fine_identity"]:
            m = rec[tier]["models"]
            dec = rec[tier]["decomposition"]
            cv = rec[tier]["group_cv"]
            table_rows.append({
                "outcome": outcome,
                "identity_tier": tier,
                "n": rec["n"],
                "R2_M0": m["M0"]["r2"],
                "R2_M1_base_plus_C": m["M1"]["r2"],
                "R2_M2_base_plus_identity": m["M2"]["r2"],
                "R2_M3_identity_plus_C": m["M3"]["r2"],
                "deltaR2_C_before_identity": dec["delta_r2_C_before_identity"],
                "deltaR2_C_after_identity": dec["delta_r2_C_after_identity"],
                "C_attenuation_fraction": dec["C_attenuation_fraction_after_identity"],
                "beta_C_M1": m["M1"]["beta_C_z"],
                "p_C_M1_cluster": m["M1"]["p_C_z_cluster"],
                "beta_C_M3": m["M3"]["beta_C_z"],
                "p_C_M3_cluster": m["M3"]["p_C_z_cluster"],
                "CV_deltaR2_C_before_identity": cv.get("delta_cv_r2_C_before_identity"),
                "CV_deltaR2_C_after_identity": cv.get("delta_cv_r2_C_after_identity"),
            })

    pd.DataFrame(table_rows).to_csv(
        out / "model_ladder_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    p = report["outcomes"][PRIMARY_OUTCOME]
    fine = p["fine_identity"]
    m = fine["models"]
    dec = fine["decomposition"]
    cv = fine["group_cv"]
    within = p["within_pre_cre"]

    lines = [
        "# Stage 5C — Identity organization / variance decomposition v5",
        "",
        "## Primary outcome",
        "",
        f"`{PRIMARY_OUTCOME}`",
        "",
        "## Fine-identity model ladder",
        "",
        f"- M0 base controls: R² = **{m['M0']['r2']:.6f}**",
        f"- M1 base + C: R² = **{m['M1']['r2']:.6f}**, "
        f"β_C = **{m['M1']['beta_C_z']:.6g}**, "
        f"cluster p = **{m['M1']['p_C_z_cluster']:.6g}**",
        f"- M2 base + identity: R² = **{m['M2']['r2']:.6f}**",
        f"- M3 base + identity + C: R² = **{m['M3']['r2']:.6f}**, "
        f"β_C = **{m['M3']['beta_C_z']:.6g}**, "
        f"cluster p = **{m['M3']['p_C_z_cluster']:.6g}**",
        "",
        "## Complexity contribution before vs after identity",
        "",
        f"- ΔR²(C before identity) = **{dec['delta_r2_C_before_identity']:.6g}**",
        f"- ΔR²(C after identity) = **{dec['delta_r2_C_after_identity']:.6g}**",
        f"- attenuation fraction = **{dec['C_attenuation_fraction_after_identity']}**",
        "",
        "## Experiment-grouped cross-validation",
        "",
        f"- CV ΔR²(C before identity) = **{cv.get('delta_cv_r2_C_before_identity')}**",
        f"- CV ΔR²(C after identity) = **{cv.get('delta_cv_r2_C_after_identity')}**",
        "",
        "## Within-presynaptic-Cre residual complexity",
        "",
        f"`{json.dumps(within, ensure_ascii=False)}`",
        "",
        "## Interpretation rule",
        "",
        "If C contributes before identity but its incremental contribution collapses after "
        "fine identity is introduced, while within-Cre residual complexity is weak, the supported "
        "conclusion is that intrinsic complexity and synaptic dynamic role are largely co-organized "
        "across biological cell identities rather than forming a strong independent within-type gradient.",
        "",
        "**Do not call this causal mediation.**",
    ]

    (out / "00_IDENTITY_DECOMPOSITION_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("IDENTITY DECOMPOSITION COMPLETE")
    print("Primary fine-identity ΔR² C before identity:",
          dec["delta_r2_C_before_identity"])
    print("Primary fine-identity ΔR² C after identity:",
          dec["delta_r2_C_after_identity"])
    print("Primary attenuation:",
          dec["C_attenuation_fraction_after_identity"])
    print("Output:", out)
    print("=" * 100)


if __name__ == "__main__":
    main()
