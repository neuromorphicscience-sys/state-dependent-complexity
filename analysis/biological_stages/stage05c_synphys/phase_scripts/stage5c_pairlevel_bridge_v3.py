#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5C — Pair-level bridge v3
===============================

Purpose
-------
Keep the scientific target fixed:

    externally defined Stage5A intrinsic complexity
        -> SynPhys local functional influence
        beyond structural opportunity / cell identity

This revision fixes the main weakness of the previous cell-level sum analysis:
raw cell-level sums conflate synaptic effect size with the number of tested /
connected partners and are extremely zero-inflated.

Primary strategy
----------------
A) Structural opportunity layer (all tested directed pairs)
   connection ~ C_transfer + distance + pre/post identity
   cluster-robust SE by experiment

B) Conditional synaptic-strength layer (connected pairs only)
   log|PSP| ~ C_transfer + distance + pre/post identity
   log|PSC| ~ C_transfer + distance + pre/post identity
   cluster-robust SE by experiment

C) Dynamic/STP layer (dynamics.qc_pass == 1)
   dimensionless STP / variability endpoints ~ C_transfer
       + distance + pre/post identity
       + structural cell-level leverage L_struct_out
   cluster-robust SE by experiment

No SynPhys outcome is used to define C_transfer.
No PSP and PSC units are mixed.
No arbitrary post-hoc endpoint selection: all predeclared STP endpoints are run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path.cwd()
DEFAULT_PHASE1 = DEFAULT_ROOT / "stage5c_synphys_phase1"
DEFAULT_TRANSFER = DEFAULT_ROOT / "stage5c_synphys_transfer_v2"
DEFAULT_OUT = DEFAULT_ROOT / "stage5c_synphys_pairlevel_v3"

STP_ENDPOINTS = [
    "paired_pulse_ratio_50hz",
    "stp_initial_50hz",
    "stp_induction_50hz",
    "stp_recovery_250ms",
    "stp_recovery_single_250ms",
    "variability_resting_state",
    "variability_second_pulse_50hz",
    "variability_stp_induced_state_50hz",
]

SEED = 20260821


def winsor_series(x: pd.Series, lo=0.01, hi=0.99):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    z = x.dropna()
    if len(z) < 20:
        return x
    a, b = z.quantile([lo, hi])
    return x.clip(a, b)


def zscore(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    mu = x.mean()
    sd = x.std()
    if not np.isfinite(sd) or sd <= 0:
        return x * np.nan
    return (x - mu) / sd


def collapse_rare(s: pd.Series, min_n: int):
    s = s.fillna("MISSING").astype(str)
    vc = s.value_counts()
    keep = vc[vc >= min_n].index
    return s.where(s.isin(keep), "OTHER")


def make_design(df: pd.DataFrame, include_structural=False):
    X = pd.DataFrame(index=df.index)

    X["C_transfer_z"] = zscore(df["C_transfer"])

    if "distance" in df:
        d = pd.to_numeric(df["distance"], errors="coerce")
        d = d.fillna(d.median())
        X["log1p_distance_um"] = np.log1p(np.maximum(d * 1e6, 0))

    if include_structural and "L_struct_out" in df:
        x = pd.to_numeric(df["L_struct_out"], errors="coerce")
        X["L_struct_out"] = x.fillna(x.median())

    if "pre_age" in df:
        x = pd.to_numeric(df["pre_age"], errors="coerce")
        if x.notna().sum() > 20:
            X["pre_age"] = x.fillna(x.median())

    cat_cols = [
        "pre_cell_class",
        "post_cell_class",
        "pre_cortical_layer",
        "post_cortical_layer",
        "pre_cre_type",
        "post_cre_type",
    ]

    for c in cat_cols:
        if c not in df:
            continue
        # Require enough observations per retained level; this prevents
        # high-cardinality sparse designs.
        s = collapse_rare(df[c], max(12, int(0.01 * len(df))))
        dm = pd.get_dummies(s, prefix=c, drop_first=True, dtype=float)
        if dm.shape[1] <= 30:
            X = pd.concat([X, dm], axis=1)

    return X.astype(float)


def clustered_logit(df: pd.DataFrame):
    import statsmodels.api as sm

    d = df.copy()
    d = d.dropna(subset=["connected", "C_transfer", "experiment_id"])
    if len(d) < 500 or d["connected"].nunique() < 2:
        return {"status": "too_few", "n": int(len(d))}

    X = make_design(d, include_structural=False)
    X = sm.add_constant(X, has_constant="add")
    y = d["connected"].astype(int)

    try:
        m = sm.GLM(y, X, family=sm.families.Binomial()).fit(
            cov_type="cluster",
            cov_kwds={"groups": d["experiment_id"].astype(str).to_numpy()},
            maxiter=200,
        )
        beta = float(m.params["C_transfer_z"])
        se = float(m.bse["C_transfer_z"])
        p = float(m.pvalues["C_transfer_z"])
        return {
            "status": "ok",
            "n": int(len(d)),
            "n_experiments": int(d["experiment_id"].nunique()),
            "connection_rate": float(y.mean()),
            "beta_C_z_logodds": beta,
            "se_C_z_cluster": se,
            "p_C_z_cluster": p,
            "odds_ratio_per_1sd_C": float(np.exp(beta)),
            "aic": float(m.aic),
        }
    except Exception as e:
        return {"status": "fit_failed", "n": int(len(d)), "error": repr(e)}


def clustered_ols(df: pd.DataFrame, outcome: str, log_abs=False, include_structural=False):
    import statsmodels.api as sm

    d = df.copy()
    y = pd.to_numeric(d[outcome], errors="coerce").astype(float)

    if log_abs:
        y = np.log1p(np.abs(y) / 1e-12)
    else:
        y = winsor_series(y)

    d["_y"] = y
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["_y", "C_transfer", "experiment_id"])

    if len(d) < 100:
        return {"status": "too_few", "n": int(len(d))}

    X = make_design(d, include_structural=include_structural)
    X = sm.add_constant(X, has_constant="add")

    try:
        m0_cols = [c for c in X.columns if c != "C_transfer_z"]
        m0 = sm.OLS(d["_y"], X[m0_cols]).fit()

        m1_plain = sm.OLS(d["_y"], X).fit()
        m1 = m1_plain.get_robustcov_results(
            cov_type="cluster",
            groups=d["experiment_id"].astype(str).to_numpy(),
        )
        names = list(X.columns)
        ci = names.index("C_transfer_z")

        return {
            "status": "ok",
            "n": int(len(d)),
            "n_experiments": int(d["experiment_id"].nunique()),
            "r2_without_C": float(m0.rsquared),
            "r2_with_C": float(m1_plain.rsquared),
            "delta_r2_C": float(m1_plain.rsquared - m0.rsquared),
            "beta_C_z": float(m1.params[ci]),
            "se_C_z_cluster": float(m1.bse[ci]),
            "p_C_z_cluster": float(m1.pvalues[ci]),
        }
    except Exception as e:
        return {"status": "fit_failed", "n": int(len(d)), "error": repr(e)}


def spearman_safe(x, y):
    from scipy.stats import spearmanr
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 20:
        return {"n": int(m.sum()), "rho": None, "p": None}
    r, p = spearmanr(x[m], y[m])
    return {"n": int(m.sum()), "rho": float(r), "p": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", type=Path, default=DEFAULT_PHASE1)
    ap.add_argument("--transfer", type=Path, default=DEFAULT_TRANSFER)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    phase1 = args.phase1.resolve()
    transfer = args.transfer.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    pair_path = phase1 / "directed_pair_atlas.csv"
    lev_path = phase1 / "cell_local_leverage_atlas.csv"
    c_path = transfer / "synphys_frozen_complexity_transfer.csv"

    for p in [pair_path, lev_path, c_path]:
        if not p.exists():
            raise SystemExit(f"Missing required input: {p}")

    print("=" * 96)
    print("Stage 5C pair-level bridge v3")
    print("=" * 96)
    print("Pairs   :", pair_path)
    print("Leverage:", lev_path)
    print("C frozen:", c_path)
    print("Output  :", out)
    print("=" * 96)

    pairs = pd.read_csv(pair_path, low_memory=False)
    lev = pd.read_csv(lev_path, low_memory=False)
    comp = pd.read_csv(c_path, low_memory=False)

    # Frozen complexity only; no proxy.
    comp = comp[[
        "cell_id", "C_transfer", "transfer_domain_primary",
        "n_feature_ood", "species"
    ]].copy()

    precomp = comp.rename(columns={
        "cell_id": "pre_cell_id",
        "C_transfer": "C_transfer",
        "transfer_domain_primary": "transfer_domain_primary",
        "n_feature_ood": "n_feature_ood_pre",
        "species": "C_species",
    })

    pairs = pairs.drop(columns=[
        c for c in ["C_transfer", "transfer_domain_primary", "n_feature_ood_pre", "C_species"]
        if c in pairs.columns
    ])
    pairs = pairs.merge(precomp, on="pre_cell_id", how="left")

    # Structural leverage is an independent outcome/covariate built without C.
    lev_pre = lev[["cell_id", "L_struct_out", "n_tested_out", "n_connected_out"]].rename(
        columns={"cell_id": "pre_cell_id"}
    )
    pairs = pairs.merge(lev_pre, on="pre_cell_id", how="left")

    # Primary transfer cohort = mouse presynaptic cells within transfer domain.
    mouse = pairs[
        (pairs["pre_species"].astype(str).str.lower() == "mouse") &
        pairs["transfer_domain_primary"].fillna(False)
    ].copy()

    # Exact tested-pair primary definition from Phase1.
    tested = mouse[mouse["tested_primary"].fillna(False)].copy()
    connected = tested[tested["connected"].fillna(False)].copy()
    dyn = connected[
        pd.to_numeric(connected["dynamics_qc_pass"], errors="coerce").fillna(0).astype(int) == 1
    ].copy()

    print("Mouse transfer-domain pairs:", len(mouse))
    print("Tested pairs:", len(tested))
    print("Connected pairs:", len(connected))
    print("Dynamics-QC connected pairs:", len(dyn))

    report = {
        "status": "COMPLETE",
        "design": "pair-level hierarchical/cluster-robust bridge using frozen Stage5A-derived C_transfer",
        "counts": {
            "mouse_transfer_domain_pairs": int(len(mouse)),
            "tested_pairs": int(len(tested)),
            "connected_pairs": int(len(connected)),
            "dynamics_qc_connected_pairs": int(len(dyn)),
            "unique_pre_cells_tested": int(tested["pre_cell_id"].nunique()),
            "experiments_tested": int(tested["experiment_id"].nunique()),
        },
        "structural_connection_model": clustered_logit(tested),
        "conditional_strength_models": {},
        "dynamic_stp_models": {},
        "raw_associations_for_reference_only": {},
        "guardrails": [
            "C_transfer is frozen from Stage5A-only training.",
            "All primary models are pair-level; cell-level raw sums are not used as the primary inferential object.",
            "PSP and PSC are modeled separately.",
            "Dynamic/STP endpoints were predeclared before fitting and all are reported.",
            "Experiment-cluster robust standard errors are used.",
            "Dynamic/STP models additionally control presynaptic L_struct_out.",
        ],
    }

    # Conditional synaptic efficacy: separate physical channels.
    for outcome in ["psp_amplitude", "psc_amplitude"]:
        sub = connected[pd.to_numeric(connected[outcome], errors="coerce").notna()].copy()
        report["conditional_strength_models"][outcome] = clustered_ols(
            sub, outcome, log_abs=True, include_structural=False
        )
        report["raw_associations_for_reference_only"][outcome] = spearman_safe(
            sub["C_transfer"], np.abs(pd.to_numeric(sub[outcome], errors="coerce"))
        )

    # Dynamic / STP: all predeclared endpoints.
    for outcome in STP_ENDPOINTS:
        if outcome not in dyn.columns:
            report["dynamic_stp_models"][outcome] = {"status": "missing_column"}
            continue
        report["dynamic_stp_models"][outcome] = clustered_ols(
            dyn, outcome, log_abs=False, include_structural=True
        )
        report["raw_associations_for_reference_only"][outcome] = spearman_safe(
            dyn["C_transfer"], dyn[outcome]
        )

    # Also retain first-50Hz absolute response as a secondary dynamic-amplitude endpoint.
    if "pulse_amp_first_50hz" in dyn.columns:
        report["dynamic_stp_models"]["pulse_amp_first_50hz_abs"] = clustered_ols(
            dyn, "pulse_amp_first_50hz", log_abs=True, include_structural=True
        )

    # Save analysis-ready pair datasets.
    tested.to_csv(out / "mouse_tested_pairs_primary.csv", index=False)
    connected.to_csv(out / "mouse_connected_pairs_primary.csv", index=False)
    dyn.to_csv(out / "mouse_dynamics_qc_pairs_primary.csv", index=False)

    (out / "pairlevel_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Stage 5C — Pair-level bridge v3",
        "",
        "## Why this analysis",
        "",
        "The previous cell-level weighted/dynamic sums are intrinsically coupled to the number "
        "of tested and connected partners. This analysis therefore moves inference to the "
        "directed-pair level while retaining the frozen Stage5A-derived complexity score.",
        "",
        "## Cohort",
        "",
        f"- Tested mouse pairs: **{len(tested):,}**",
        f"- Connected mouse pairs: **{len(connected):,}**",
        f"- Dynamics-QC connected pairs: **{len(dyn):,}**",
        f"- Unique presynaptic cells: **{tested['pre_cell_id'].nunique():,}**",
        f"- Experiments: **{tested['experiment_id'].nunique():,}**",
        "",
        "## Structural connection model",
        "",
        f"`{json.dumps(report['structural_connection_model'], ensure_ascii=False)}`",
        "",
        "## Conditional synaptic-strength models",
        "",
    ]
    for k, v in report["conditional_strength_models"].items():
        lines.append(f"- `{k}`: `{json.dumps(v, ensure_ascii=False)}`")

    lines += ["", "## Dynamic / STP models", ""]
    for k, v in report["dynamic_stp_models"].items():
        lines.append(f"- `{k}`: `{json.dumps(v, ensure_ascii=False)}`")

    lines += [
        "",
        "## Interpretation rule",
        "",
        "The fixed paper goal is not that complexity must correlate positively with raw degree. "
        "The critical test is whether externally defined intrinsic complexity predicts conditional "
        "synaptic strength or temporal synaptic dynamics after accounting for structural opportunity "
        "and biological identity.",
    ]

    (out / "00_PAIRLEVEL_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print()
    print("=" * 96)
    print("PAIR-LEVEL ANALYSIS COMPLETE")
    print("Summary:", out / "00_PAIRLEVEL_SUMMARY.md")
    print("Report :", out / "pairlevel_report.json")
    print("=" * 96)


if __name__ == "__main__":
    main()
