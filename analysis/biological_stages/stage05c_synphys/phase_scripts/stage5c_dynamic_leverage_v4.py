#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5C — Standardized stochastic-synapse dynamic leverage v4
==============================================================

Goal
----
Keep the paper hypothesis fixed:

    Stage5A-defined intrinsic complexity -> local dynamic leverage

but define dynamic leverage from the Allen stochastic release model itself
rather than from raw cell-level sums or isolated STP summary endpoints.

Model implementation
--------------------
This script reproduces the EXPECTATION-mode forward dynamics of Allen
aisynphys StochasticReleaseModel using the maximum-likelihood parameters
stored in synapse_model:

    release_probability =
        (1 - depression) *
        (base_release_probability +
         (1 - base_release_probability) * facilitation)

    expected_amplitude =
        available_vesicles * release_probability * mini_amplitude

After each spike:
    if depression_amount == -1:
        vesicle depletion is used
    else:
        depression += (1 - depression) * depression_amount

    facilitation += (1 - facilitation) * facilitation_amount

Between spikes, vesicle pool / depression / facilitation recover
exponentially with their fitted tau values.

Primary standardized trains:
    5, 10, 20, 50 Hz; 12 spikes each

Per-synapse endpoints:
    E_abs_<freq>        = sum_k |A_k|                   [physical effect]
    G_norm_<freq>       = mean_k |A_k / A_1|           [temporal gain]
    tail_first_<freq>   = mean(last 3)/|A_1|            [late-train effect]
    dyn_range_<freq>    = (max|A|-min|A|)/|A_1|         [dynamic shaping]

Primary inferential endpoint:
    G_norm_50hz

Secondary / robustness:
    G_norm_5/10/20hz, tail_first, dyn_range, E_abs.

Scientific guardrails
---------------------
- C_transfer remains frozen from Stage5A only.
- No SynPhys outcome is used to tune the complexity mapping.
- All synapses receive exactly the same standardized spike trains.
- Pair-level inference is primary.
- Controls: distance + pre/post biological identity.
- Experiment-cluster-robust standard errors.
- Cell-level aggregation is exported as descriptive/secondary only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path.cwd()
DEFAULT_AUDIT = DEFAULT_ROOT / "stage5c_synapse_model_audit_v1"
DEFAULT_PHASE1 = DEFAULT_ROOT / "stage5c_synphys_phase1"
DEFAULT_OUT = DEFAULT_ROOT / "stage5c_dynamic_leverage_v4"

FREQS = [5, 10, 20, 50]
N_SPIKES = 12

PARAMS = [
    "ml_n_release_sites",
    "ml_base_release_probability",
    "ml_mini_amplitude",
    "ml_mini_amplitude_cv",
    "ml_depression_amount",
    "ml_depression_tau",
    "ml_facilitation_amount",
    "ml_facilitation_tau",
    "ml_measurement_stdev",
]


def simulate_expected_train(row, freq_hz, n_spikes=N_SPIKES):
    """Exact expectation-mode recurrence of Allen's stochastic release model."""
    n_sites = float(row["ml_n_release_sites"])
    base_pr = float(row["ml_base_release_probability"])
    mini_amp = float(row["ml_mini_amplitude"])
    dep_amt = float(row["ml_depression_amount"])
    dep_tau = float(row["ml_depression_tau"])
    fac_amt = float(row["ml_facilitation_amount"])
    fac_tau = float(row["ml_facilitation_tau"])

    if not all(np.isfinite([n_sites, base_pr, mini_amp, dep_amt, dep_tau, fac_amt, fac_tau])):
        return None
    if n_sites <= 0 or dep_tau <= 0 or fac_tau <= 0:
        return None
    if base_pr < 0 or base_pr > 1:
        return None

    dt = 1.0 / float(freq_hz)
    vesicle_pool = n_sites
    facilitation = 0.0
    depression = 0.0
    use_vesicle_depletion = np.isclose(dep_amt, -1.0)

    amps = np.empty(n_spikes, dtype=float)
    prs = np.empty(n_spikes, dtype=float)
    pools = np.empty(n_spikes, dtype=float)

    for k in range(n_spikes):
        # First event occurs at resting state; recovery occurs only after prior event.
        if k > 0:
            if use_vesicle_depletion:
                r = np.exp(-dt / dep_tau)
                vesicle_pool += (n_sites - vesicle_pool) * (1.0 - r)
            else:
                depression *= np.exp(-dt / dep_tau)

            facilitation *= np.exp(-dt / fac_tau)

        if use_vesicle_depletion:
            available = max(0.0, vesicle_pool)
        else:
            available = vesicle_pool

        pr = (1.0 - depression) * (
            base_pr + (1.0 - base_pr) * facilitation
        )
        # numerical guard; valid Allen fits should naturally remain bounded
        pr = float(np.clip(pr, 0.0, 1.0))

        expected_amp = available * pr * mini_amp

        amps[k] = expected_amp
        prs[k] = pr
        pools[k] = available

        # expectation-mode state update
        if use_vesicle_depletion:
            # In expectation mode, released vesicles = expected_amp / mini_amp
            # = available * pr.
            released = available * pr
            vesicle_pool -= released
        else:
            depression += (1.0 - depression) * dep_amt

        facilitation += (1.0 - facilitation) * fac_amt

    return amps, prs, pools


def train_features(amps):
    a = np.abs(np.asarray(amps, float))
    a1 = a[0]
    if not np.isfinite(a1) or a1 <= 1e-15:
        return None
    norm = a / a1
    return {
        "E_abs": float(np.sum(a)),
        "mean_abs": float(np.mean(a)),
        "G_norm": float(np.mean(norm)),
        "tail_first": float(np.mean(norm[-3:])),
        "second_first": float(norm[1]),
        "dyn_range": float((np.max(a) - np.min(a)) / a1),
        "A1_abs": float(a1),
    }


def collapse_rare(s, min_n):
    s = s.fillna("MISSING").astype(str)
    vc = s.value_counts()
    keep = vc[vc >= min_n].index
    return s.where(s.isin(keep), "OTHER")


def zscore(x):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    sd = x.std()
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def build_design(d, include_strength=False):
    X = pd.DataFrame(index=d.index)
    X["C_transfer_z"] = zscore(d["C_transfer"])

    if "distance" in d:
        x = pd.to_numeric(d["distance"], errors="coerce")
        if x.notna().any():
            x = x.fillna(x.median())
            X["log1p_distance_um"] = np.log1p(np.maximum(x * 1e6, 0))

    if include_strength and "A1_abs_50hz" in d:
        x = pd.to_numeric(d["A1_abs_50hz"], errors="coerce")
        x = np.log(np.maximum(x, 1e-15))
        X["log_A1_abs_50hz"] = x.fillna(x.median())

    if "pre_age" in d:
        x = pd.to_numeric(d["pre_age"], errors="coerce")
        if x.notna().sum() >= 30:
            X["pre_age"] = x.fillna(x.median())

    cats = [
        "pre_cell_class", "post_cell_class",
        "pre_cortical_layer", "post_cortical_layer",
        "pre_cre_type", "post_cre_type",
    ]
    for c in cats:
        if c not in d:
            continue
        s = collapse_rare(d[c], max(10, int(0.015 * len(d))))
        dm = pd.get_dummies(s, prefix=c, drop_first=True, dtype=float)
        if dm.shape[1] <= 30:
            X = pd.concat([X, dm], axis=1)

    return X.astype(float)


def clustered_ols(df, outcome, include_strength=False):
    import statsmodels.api as sm

    d = df.copy()
    d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[outcome, "C_transfer", "experiment_id"])

    if len(d) < 120:
        return {"status": "too_few", "n": int(len(d))}

    # Conservative 1/99% winsorization only for the outcome.
    lo, hi = d[outcome].quantile([0.01, 0.99])
    d[outcome] = d[outcome].clip(lo, hi)

    X = build_design(d, include_strength=include_strength)
    X = sm.add_constant(X, has_constant="add")
    y = d[outcome].astype(float)

    base_cols = [c for c in X.columns if c != "C_transfer_z"]
    m0 = sm.OLS(y, X[base_cols]).fit()
    m1_plain = sm.OLS(y, X).fit()
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
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--phase1", type=Path, default=DEFAULT_PHASE1)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    audit = args.audit.resolve()
    phase1 = args.phase1.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    model_path = audit / "synapse_model_mouse_transfer_overlap.csv"
    pair_path = phase1 / "directed_pair_atlas.csv"
    lev_path = phase1 / "cell_local_leverage_atlas.csv"

    for p in [model_path, pair_path, lev_path]:
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    print("=" * 100)
    print("Stage 5C standardized stochastic-synapse dynamic leverage v4")
    print("=" * 100)

    sm = pd.read_csv(model_path, low_memory=False)
    pairs = pd.read_csv(pair_path, low_memory=False)
    lev = pd.read_csv(lev_path, low_memory=False)

    # Bring audited biological identity columns from Phase1 by pair_id.
    pair_cols = [
        "pair_id", "pre_cell_id", "post_cell_id", "experiment_id", "distance",
        "pre_species", "pre_age", "pre_cell_class", "post_cell_class",
        "pre_cortical_layer", "post_cortical_layer",
        "pre_cre_type", "post_cre_type",
    ]
    pair_meta = pairs[[c for c in pair_cols if c in pairs.columns]].drop_duplicates("pair_id")

    # Avoid duplicate columns already in synapse-model join.
    keep_new = ["pair_id"] + [
        c for c in pair_meta.columns
        if c != "pair_id" and c not in sm.columns
    ]
    sm = sm.merge(pair_meta[keep_new], on="pair_id", how="left")

    # Cell-level structural opportunity for secondary aggregation/control.
    levsmall = lev[[
        c for c in ["cell_id", "L_struct_out", "n_tested_out", "n_connected_out"]
        if c in lev.columns
    ]].rename(columns={"cell_id": "pre_cell_id"})
    sm = sm.merge(levsmall, on="pre_cell_id", how="left")

    # Strict parameter QC.
    param_ok = pd.Series(True, index=sm.index)
    for c in PARAMS:
        if c not in sm.columns:
            raise SystemExit(f"Missing fitted model parameter: {c}")
        param_ok &= pd.to_numeric(sm[c], errors="coerce").notna()

    work = sm[param_ok].copy()

    # Simulate standardized trains.
    valid = []
    for idx, row in work.iterrows():
        ok = True
        for f in FREQS:
            sim = simulate_expected_train(row, f)
            if sim is None:
                ok = False
                break
            amps, prs, pools = sim
            feat = train_features(amps)
            if feat is None:
                ok = False
                break
            for k, v in feat.items():
                work.loc[idx, f"{k}_{f}hz"] = v

            # Save compact train vectors as semicolon-delimited values for audit/replay.
            work.loc[idx, f"amp_train_{f}hz"] = ";".join(f"{v:.12g}" for v in amps)
            work.loc[idx, f"pr_train_{f}hz"] = ";".join(f"{v:.12g}" for v in prs)

        valid.append((idx, ok))

    valid_idx = [i for i, ok in valid if ok]
    work = work.loc[valid_idx].copy()

    print("Model overlap rows:", len(sm))
    print("Valid standardized simulations:", len(work))
    print("Unique pre cells:", work["pre_cell_id"].nunique())
    print("Experiments:", work["experiment_id"].nunique())

    # Primary and predeclared secondary pair-level tests.
    endpoints = []
    for f in FREQS:
        endpoints += [
            f"G_norm_{f}hz",
            f"tail_first_{f}hz",
            f"dyn_range_{f}hz",
            f"E_abs_{f}hz",
        ]

    models = {}
    raw = {}
    for ep in endpoints:
        # For normalized dynamic endpoints, additionally controlling first-pulse
        # strength is a robustness analysis; for physical cumulative E_abs this
        # would partially condition away the quantity of interest, so do not.
        include_strength = ep.startswith(("G_norm_", "tail_first_", "dyn_range_"))
        models[ep] = clustered_ols(work, ep, include_strength=include_strength)
        raw[ep] = spearman_safe(work["C_transfer"], work[ep])

    # Cell-level descriptive aggregation: mean normalized dynamic phenotype
    # and opportunity-normalized cumulative effect. Not primary inference.
    agg_spec = {
        "C_transfer": "first",
        "experiment_id": "first",
        "L_struct_out": "first",
        "n_tested_out": "first",
        "n_connected_out": "first",
    }
    for f in FREQS:
        agg_spec[f"G_norm_{f}hz"] = "mean"
        agg_spec[f"tail_first_{f}hz"] = "mean"
        agg_spec[f"dyn_range_{f}hz"] = "mean"
        agg_spec[f"E_abs_{f}hz"] = "sum"
        agg_spec[f"A1_abs_{f}hz"] = "sum"

    cell = work.groupby("pre_cell_id", as_index=False).agg(agg_spec)
    cell["n_modeled_out"] = work.groupby("pre_cell_id").size().reindex(cell["pre_cell_id"]).to_numpy()
    for f in FREQS:
        denom = cell["n_tested_out"].replace(0, np.nan)
        cell[f"L_modelopportunity_E_{f}hz"] = cell[f"E_abs_{f}hz"] / denom

    work.to_csv(out / "standardized_synapse_train_atlas.csv", index=False)
    cell.to_csv(out / "standardized_cell_dynamic_leverage.csv", index=False)

    report = {
        "status": "COMPLETE",
        "definition": {
            "frequencies_hz": FREQS,
            "n_spikes": N_SPIKES,
            "primary_endpoint": "G_norm_50hz",
            "primary_level": "directed modeled synapse",
            "simulation": "Allen StochasticReleaseModel expectation-mode recurrence reproduced from public source",
        },
        "counts": {
            "input_mouse_transfer_synapse_models": int(len(sm)),
            "valid_simulated_pairs": int(len(work)),
            "unique_presynaptic_cells": int(work["pre_cell_id"].nunique()),
            "experiments": int(work["experiment_id"].nunique()),
        },
        "pair_level_models": models,
        "raw_associations_reference_only": raw,
        "guardrails": [
            "Frozen Stage5A C_transfer is unchanged.",
            "All synapses receive identical standardized spike trains.",
            "Primary inference is pair-level.",
            "G_norm endpoints separate temporal shaping from baseline synaptic scale.",
            "First-pulse amplitude is additionally controlled for normalized dynamic endpoints.",
            "Cell-level aggregation is secondary/descriptive.",
            "No pulse_response table is read.",
        ],
    }
    (out / "dynamic_leverage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    prim = models.get("G_norm_50hz", {})
    lines = [
        "# Stage 5C — Standardized stochastic-synapse dynamic leverage v4",
        "",
        f"- valid modeled mouse pairs: **{len(work):,}**",
        f"- presynaptic cells: **{work['pre_cell_id'].nunique():,}**",
        f"- experiments: **{work['experiment_id'].nunique():,}**",
        "",
        "## Primary definition",
        "",
        "`G_norm_50hz = mean_k |A_k / A_1|` for a common 12-spike 50-Hz train "
        "simulated in expectation mode from each synapse's fitted Allen stochastic-release parameters.",
        "",
        "## Primary model",
        "",
        f"`{json.dumps(prim, ensure_ascii=False)}`",
        "",
        "## All pair-level endpoints",
        "",
    ]
    for ep in endpoints:
        lines.append(f"- `{ep}`: `{json.dumps(models[ep], ensure_ascii=False)}`")

    lines += [
        "",
        "## Interpretation",
        "",
        "This is the first Stage 5C endpoint that operationalizes dynamic leverage under an identical "
        "presynaptic input rather than using raw degree, raw synaptic sums, or one isolated STP statistic.",
    ]
    (out / "00_DYNAMIC_LEVERAGE_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print()
    print("=" * 100)
    print("STANDARDIZED DYNAMIC LEVERAGE COMPLETE")
    print("Primary G_norm_50hz:", prim)
    print("Output:", out)
    print("=" * 100)


if __name__ == "__main__":
    main()
