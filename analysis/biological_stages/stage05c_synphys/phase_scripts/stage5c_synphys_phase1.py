#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5C — Allen Synaptic Physiology local circuit bridge, Phase 1
================================================================

This script uses the *audited r2.1 schema* and builds publication-ready derived
tables for the first SynPhys analysis pass.

It does NOT touch the enormous pulse_response BLOB-like/raw-response path.
It only queries:
    slice
    experiment
    cell
    intrinsic
    morphology
    cortical_cell_location
    pair
    synapse
    dynamics

Primary goals
-------------
1. Build a cell-level intrinsic-physiology atlas.
2. Build a directed pair-level connectivity/synapse/dynamics atlas.
3. Define structural local leverage with tested-opportunity correction.
4. Define conservative weighted/dynamic local leverage from real measured
   synaptic amplitudes / 50-Hz dynamics WITHOUT mixing PSP and PSC units.
5. Generate a Stage5A-compatible intrinsic feature matrix.
6. Run baseline association/control models using an explicitly labeled
   exploratory intrinsic-complexity proxy (PCA1 of robust-scaled intrinsic
   features). This proxy is NOT a replacement for the frozen Stage5A mapper.
7. Save all derived tables and a machine-readable report for the next step.

Primary structural testing rule
-------------------------------
A directed pair is "tested" for structural leverage if has_synapse IS NOT NULL.
Sensitivity flag: n_ex_test_spikes + n_in_test_spikes > 0.

Primary local leverage outputs
------------------------------
L_struct_out = connected outgoing / tested outgoing
L_struct_in  = connected incoming / tested incoming

Separate physical measurement channels (never mixed in volts vs amps):
L_psp_out_sum_abs
L_psc_out_sum_abs

Dynamic-QC channel:
L_dyn50_out_sum_abs = sum(abs(dynamics.pulse_amp_first_50hz))
over qc_pass=1 outgoing pairs.

The script also exports STP feature summaries rather than forcing an arbitrary
single STP formula.

Notes
-----
- mouse and human are reported separately.
- experiment_id is retained for hierarchical/mixed-effect analyses.
- cell identity, layer, age, species, distance, ACSF/internal are retained.
- READ ONLY SQLite connection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

DEFAULT_DB = Path.cwd() / "synphys_r2.1_full.sqlite"
DEFAULT_OUT_NAME = "stage5c_synphys_phase1"

INTRINSIC_FEATURES = [
    "rheobase",
    "fi_slope",
    "input_resistance",
    "input_resistance_ss",
    "sag",
    "tau",
    "ap_upstroke_downstroke_ratio",
    "ap_width",
    "ap_upstroke",
    "ap_downstroke",
    "ap_threshold_v",
    "ap_peak_deltav",
    "ap_fast_trough_deltav",
    "firing_rate_rheo",
    "latency_rheo",
    "firing_rate_40pa",
    "latency_40pa",
    "adaptation_index",
    "isi_cv",
    "isi_adapt_ratio",
    "upstroke_adapt_ratio",
    "downstroke_adapt_ratio",
    "width_adapt_ratio",
    "threshold_v_adapt_ratio",
]

STAGE5A_CORE_OVERLAP = [
    "rheobase",
    "fi_slope",
    "input_resistance",
    "sag",
    "tau",
    "ap_width",
    "ap_threshold_v",
    "ap_fast_trough_deltav",
    "firing_rate_rheo",
    "firing_rate_40pa",
    "adaptation_index",
]

STP_FEATURES = [
    "paired_pulse_ratio_50hz",
    "stp_initial_50hz",
    "stp_induction_50hz",
    "stp_recovery_250ms",
    "stp_recovery_single_250ms",
    "variability_resting_state",
    "variability_second_pulse_50hz",
    "variability_stp_induced_state_50hz",
]


def ro_connect(path: Path):
    uri = f"file:{quote(str(path).replace(os.sep, '/'))}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True, timeout=60)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA busy_timeout=60000")
    return con


def robust_z(df: pd.DataFrame):
    out = pd.DataFrame(index=df.index)
    params = {}
    for c in df.columns:
        x = pd.to_numeric(df[c], errors="coerce").astype(float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            q1, q3 = np.nanpercentile(x, [25, 75])
            scale = (q3 - q1) / 1.349
        if not np.isfinite(scale) or scale <= 0:
            scale = np.nanstd(x)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        out[c] = (x - med) / scale
        params[c] = {"median": float(med) if np.isfinite(med) else None,
                     "scale": float(scale) if np.isfinite(scale) else None}
    return out, params


def pca1_with_missing(X: pd.DataFrame):
    # Keep features with enough support; median-impute after robust scaling.
    support = X.notna().mean()
    keep = support[support >= 0.30].index.tolist()
    if len(keep) < 3:
        return pd.Series(np.nan, index=X.index), {}, keep

    Z, scale_params = robust_z(X[keep])
    Z = Z.clip(-8, 8)
    for c in keep:
        Z[c] = Z[c].fillna(Z[c].median())
    A = Z.to_numpy(float)
    A -= np.nanmean(A, axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    score = A @ vt[0]
    # Fix sign so higher score tends to align with adaptation / richer spike dynamics
    anchor_cols = [c for c in ["adaptation_index", "ap_fast_trough_deltav", "ap_width"] if c in keep]
    if anchor_cols:
        anchor = Z[anchor_cols].mean(axis=1).to_numpy()
        if np.corrcoef(score, anchor)[0, 1] < 0:
            score = -score
            vt[0] = -vt[0]
    explained = float((s[0] ** 2) / np.sum(s ** 2)) if np.sum(s ** 2) > 0 else np.nan
    meta = {
        "features": keep,
        "loadings": {c: float(v) for c, v in zip(keep, vt[0])},
        "explained_variance_fraction": explained,
        "scaling": scale_params,
        "warning": "Exploratory SynPhys PCA proxy only; not the frozen Stage5A complexity mapper.",
    }
    return pd.Series(score, index=X.index), meta, keep


def spearman(x, y):
    try:
        from scipy.stats import spearmanr
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10:
            return np.nan, np.nan, int(m.sum())
        r, p = spearmanr(np.asarray(x)[m], np.asarray(y)[m])
        return float(r), float(p), int(m.sum())
    except Exception:
        return np.nan, np.nan, 0


def fit_ols_increment(df, outcome, complexity, base_cols):
    """Simple OLS + one-hot controls. Exploratory baseline, not final mixed model."""
    try:
        import statsmodels.api as sm
    except Exception:
        return {"status": "statsmodels_missing"}

    use = df[[outcome, complexity] + base_cols].copy()
    use = use.replace([np.inf, -np.inf], np.nan).dropna(subset=[outcome, complexity])
    if len(use) < 50:
        return {"status": "too_few_rows", "n": int(len(use))}

    # Drop control columns that are almost all missing.
    ctrl = []
    for c in base_cols:
        if use[c].notna().mean() >= 0.50 and use[c].nunique(dropna=True) > 1:
            ctrl.append(c)

    X0 = pd.DataFrame(index=use.index)
    for c in ctrl:
        if pd.api.types.is_numeric_dtype(use[c]):
            x = pd.to_numeric(use[c], errors="coerce")
            X0[c] = x.fillna(x.median())
        else:
            d = pd.get_dummies(use[c].fillna("MISSING").astype(str), prefix=c, drop_first=True, dtype=float)
            # Avoid explosive very-high-cardinality identity columns
            if d.shape[1] <= 30:
                X0 = pd.concat([X0, d], axis=1)

    y = pd.to_numeric(use[outcome], errors="coerce").astype(float)
    c = pd.to_numeric(use[complexity], errors="coerce").astype(float)

    X0 = sm.add_constant(X0.astype(float), has_constant="add")
    try:
        m0 = sm.OLS(y, X0).fit()
        X1 = X0.copy()
        X1[complexity] = c
        m1 = sm.OLS(y, X1).fit()
        return {
            "status": "ok",
            "n": int(len(use)),
            "r2_base": float(m0.rsquared),
            "r2_plus_complexity": float(m1.rsquared),
            "delta_r2": float(m1.rsquared - m0.rsquared),
            "beta_complexity": float(m1.params.get(complexity, np.nan)),
            "p_complexity": float(m1.pvalues.get(complexity, np.nan)),
            "controls_used": ctrl,
        }
    except Exception as e:
        return {"status": "fit_failed", "error": repr(e), "n": int(len(use))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    db = args.db.resolve()
    if not db.exists():
        raise SystemExit(f"Database not found: {db}")
    out = (args.out or (db.parent / DEFAULT_OUT_NAME)).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("Stage 5C SynPhys Phase 1 — intrinsic × local leverage")
    print("DB:", db)
    print("OUT:", out)
    print("READ-ONLY")
    print("=" * 96)

    con = ro_connect(db)
    t0 = time.time()

    # ---- Cell atlas ----
    print("[1/5] Building cell intrinsic atlas...")
    cell_sql = """
    SELECT
        c.id AS cell_id,
        c.experiment_id,
        c.ext_id AS cell_ext_id,
        c.cre_type,
        c.target_layer,
        c.depth,
        c.cell_class,
        c.cell_class_nonsynaptic,
        e.slice_id,
        e.project_name,
        e.target_region,
        e.internal,
        e.acsf,
        e.target_temperature,
        e.rig_name,
        s.species,
        s.age,
        s.sex,
        s.genotype,
        i.rheobase,
        i.fi_slope,
        i.input_resistance,
        i.input_resistance_ss,
        i.sag,
        i.tau,
        i.ap_upstroke_downstroke_ratio,
        i.ap_width,
        i.ap_upstroke,
        i.ap_downstroke,
        i.ap_threshold_v,
        i.ap_peak_deltav,
        i.ap_fast_trough_deltav,
        i.firing_rate_rheo,
        i.latency_rheo,
        i.firing_rate_40pa,
        i.latency_40pa,
        i.adaptation_index,
        i.isi_cv,
        i.chirp_peak_freq,
        i.chirp_3db_freq,
        i.chirp_peak_ratio,
        i.chirp_peak_impedance,
        i.chirp_sync_freq,
        i.chirp_inductive_phase,
        i.isi_adapt_ratio,
        i.upstroke_adapt_ratio,
        i.downstroke_adapt_ratio,
        i.width_adapt_ratio,
        i.threshold_v_adapt_ratio,
        m.pyramidal,
        m.qual_morpho_type,
        m.dendrite_type,
        cl.cortical_layer,
        cl.distance_to_pia,
        cl.distance_to_wm,
        cl.fractional_depth,
        cl.fractional_layer_depth
    FROM cell c
    LEFT JOIN experiment e ON e.id = c.experiment_id
    LEFT JOIN slice s ON s.id = e.slice_id
    LEFT JOIN intrinsic i ON i.cell_id = c.id
    LEFT JOIN morphology m ON m.cell_id = c.id
    LEFT JOIN cortical_cell_location cl ON cl.cell_id = c.id
    """
    cells = pd.read_sql_query(cell_sql, con)
    cells = cells.drop_duplicates(subset=["cell_id"])

    # Exploratory complexity proxy, class-conditioned by species for scaling
    cells["C_synphys_proxy"] = np.nan
    pca_meta = {}
    for species, idx in cells.groupby("species", dropna=False).groups.items():
        sub = cells.loc[idx, INTRINSIC_FEATURES]
        score, meta, keep = pca1_with_missing(sub)
        if len(keep) >= 3:
            cells.loc[idx, "C_synphys_proxy"] = score.values
        pca_meta[str(species)] = meta

    # Explicit Stage5A-compatible matrix
    feature_matrix_cols = ["cell_id", "species", "cell_class", "cre_type", "target_layer",
                           "cortical_layer", "experiment_id"] + STAGE5A_CORE_OVERLAP
    cells[feature_matrix_cols].to_csv(out / "stage5a_compatible_intrinsic_matrix.csv", index=False)

    # ---- Directed pair atlas ----
    print("[2/5] Building directed pair / synapse / dynamics atlas...")
    pair_sql = """
    SELECT
        p.id AS pair_id,
        p.experiment_id,
        p.pre_cell_id,
        p.post_cell_id,
        p.has_synapse,
        p.has_polysynapse,
        p.has_electrical,
        p.n_ex_test_spikes,
        p.n_in_test_spikes,
        p.distance,
        p.lateral_distance,
        p.vertical_distance,
        sy.synapse_type,
        sy.latency,
        sy.psp_amplitude,
        sy.psp_rise_time,
        sy.psp_decay_tau,
        sy.psc_amplitude,
        sy.psc_rise_time,
        sy.psc_decay_tau,
        d.qc_pass AS dynamics_qc_pass,
        d.n_source_events,
        d.paired_pulse_ratio_50hz,
        d.stp_initial_50hz,
        d.stp_induction_50hz,
        d.stp_recovery_250ms,
        d.stp_recovery_single_250ms,
        d.pulse_amp_90th_percentile,
        d.noise_amp_90th_percentile,
        d.pulse_amp_first_50hz,
        d.noise_std,
        d.variability_resting_state,
        d.variability_second_pulse_50hz,
        d.variability_stp_induced_state_50hz
    FROM pair p
    LEFT JOIN synapse sy ON sy.pair_id = p.id
    LEFT JOIN dynamics d ON d.pair_id = p.id
    """
    pairs = pd.read_sql_query(pair_sql, con)
    pairs = pairs.drop_duplicates(subset=["pair_id"])

    pairs["tested_primary"] = pairs["has_synapse"].notna()
    pairs["tested_spike_sensitivity"] = (
        pairs["n_ex_test_spikes"].fillna(0) + pairs["n_in_test_spikes"].fillna(0)
    ) > 0
    pairs["connected"] = pairs["has_synapse"].fillna(0).astype(float) == 1
    pairs["psp_abs"] = pd.to_numeric(pairs["psp_amplitude"], errors="coerce").abs()
    pairs["psc_abs"] = pd.to_numeric(pairs["psc_amplitude"], errors="coerce").abs()
    pairs["dyn50_abs"] = np.where(
        pd.to_numeric(pairs["dynamics_qc_pass"], errors="coerce").fillna(0).astype(int) == 1,
        pd.to_numeric(pairs["pulse_amp_first_50hz"], errors="coerce").abs(),
        np.nan,
    )

    # Add pre/post identities without huge joins in SQLite
    cell_meta_cols = [
        "cell_id", "species", "age", "sex", "cre_type", "cell_class",
        "cell_class_nonsynaptic", "target_layer", "cortical_layer",
        "target_region", "internal", "acsf", "target_temperature",
        "experiment_id", "C_synphys_proxy"
    ]
    pre = cells[cell_meta_cols].copy().add_prefix("pre_")
    post = cells[cell_meta_cols].copy().add_prefix("post_")
    pairs = pairs.merge(pre, left_on="pre_cell_id", right_on="pre_cell_id", how="left")
    pairs = pairs.merge(post, left_on="post_cell_id", right_on="post_cell_id", how="left")

    # ---- Leverage aggregation ----
    print("[3/5] Aggregating structural / weighted / dynamic leverage...")

    def aggregate_out(g):
        tested = g["tested_primary"].sum()
        tested_sens = g["tested_spike_sensitivity"].sum()
        connected = (g["tested_primary"] & g["connected"]).sum()
        connected_sens = (g["tested_spike_sensitivity"] & g["connected"]).sum()
        dqc = pd.to_numeric(g["dynamics_qc_pass"], errors="coerce").fillna(0).astype(int) == 1

        outrec = {
            "n_possible_out_pairs": len(g),
            "n_tested_out": int(tested),
            "n_connected_out": int(connected),
            "L_struct_out": connected / tested if tested else np.nan,
            "n_tested_out_spike_sens": int(tested_sens),
            "n_connected_out_spike_sens": int(connected_sens),
            "L_struct_out_spike_sens": connected_sens / tested_sens if tested_sens else np.nan,
            "L_psp_out_sum_abs": g.loc[g["connected"], "psp_abs"].sum(min_count=1),
            "L_psc_out_sum_abs": g.loc[g["connected"], "psc_abs"].sum(min_count=1),
            "n_dyn50_out_qc": int(dqc.sum()),
            "L_dyn50_out_sum_abs": g.loc[dqc, "dyn50_abs"].sum(min_count=1),
            "L_dyn50_out_mean_abs": g.loc[dqc, "dyn50_abs"].mean(),
            "mean_out_distance": pd.to_numeric(g.loc[g["tested_primary"], "distance"], errors="coerce").mean(),
        }
        for f in STP_FEATURES:
            x = pd.to_numeric(g.loc[dqc, f], errors="coerce")
            outrec[f"out_{f}_mean"] = x.mean()
            outrec[f"out_{f}_sd"] = x.std()
        return pd.Series(outrec)

    def aggregate_in(g):
        tested = g["tested_primary"].sum()
        connected = (g["tested_primary"] & g["connected"]).sum()
        return pd.Series({
            "n_possible_in_pairs": len(g),
            "n_tested_in": int(tested),
            "n_connected_in": int(connected),
            "L_struct_in": connected / tested if tested else np.nan,
            "L_psp_in_sum_abs": g.loc[g["connected"], "psp_abs"].sum(min_count=1),
            "L_psc_in_sum_abs": g.loc[g["connected"], "psc_abs"].sum(min_count=1),
        })

    outlev = pairs.groupby("pre_cell_id", sort=False).apply(aggregate_out, include_groups=False)
    outlev.index.name = "cell_id"
    outlev = outlev.reset_index()

    inlev = pairs.groupby("post_cell_id", sort=False).apply(aggregate_in, include_groups=False)
    inlev.index.name = "cell_id"
    inlev = inlev.reset_index()

    celllev = cells.merge(outlev, on="cell_id", how="left").merge(inlev, on="cell_id", how="left")

    # QC-gated cohorts
    celllev["primary_structural_qc"] = celllev["n_tested_out"].fillna(0) >= 3
    celllev["primary_dynamic_qc"] = celllev["n_dyn50_out_qc"].fillna(0) >= 1
    celllev["intrinsic_core_complete_n"] = celllev[STAGE5A_CORE_OVERLAP].notna().sum(axis=1)
    celllev["intrinsic_core_qc"] = celllev["intrinsic_core_complete_n"] >= 6

    # ---- Baseline statistical summary ----
    print("[4/5] Running conservative baseline statistics...")
    report = {
        "status": "PHASE1_COMPLETE",
        "database": str(db),
        "elapsed_seconds": None,
        "counts": {
            "cells": int(len(cells)),
            "pairs": int(len(pairs)),
            "tested_pairs_primary": int(pairs["tested_primary"].sum()),
            "connected_pairs_primary": int((pairs["tested_primary"] & pairs["connected"]).sum()),
            "dynamics_qc_pairs": int(
                (pd.to_numeric(pairs["dynamics_qc_pass"], errors="coerce").fillna(0).astype(int) == 1).sum()
            ),
        },
        "species": {},
        "complexity_proxy": {
            "definition": "PCA1 of robust-scaled multiaxial intrinsic features, fit separately by species.",
            "status": "EXPLORATORY_ONLY_NOT_FROZEN_STAGE5A_MAPPING",
            "meta": pca_meta,
        },
        "primary_structural_definition": "has_synapse IS NOT NULL defines tested opportunities.",
        "dynamic_definition": "sum(abs(pulse_amp_first_50hz)) over dynamics.qc_pass=1 outgoing pairs; PSP/PSC units are never mixed.",
    }

    base_controls = [
        "age", "sex", "cell_class", "cre_type", "cortical_layer",
        "target_region", "internal", "acsf"
    ]

    for species in sorted(celllev["species"].dropna().unique()):
        sub = celllev[celllev["species"] == species].copy()
        srep = {"n_cells": int(len(sub)), "associations": {}, "models": {}}

        for outcome in [
            "L_struct_out",
            "L_psp_out_sum_abs",
            "L_psc_out_sum_abs",
            "L_dyn50_out_sum_abs",
        ]:
            x = pd.to_numeric(sub["C_synphys_proxy"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(sub[outcome], errors="coerce").to_numpy(float)
            r, pv, n = spearman(x, y)
            srep["associations"][outcome] = {"spearman_rho": r, "p": pv, "n": n}

        # "Beyond structure" test for dynamic leverage.
        dyn = sub[
            sub["intrinsic_core_qc"] &
            sub["primary_structural_qc"] &
            sub["primary_dynamic_qc"]
        ].copy()

        controls = base_controls + ["L_struct_out"]
        srep["models"]["L_dyn50_out_sum_abs"] = fit_ols_increment(
            dyn, "L_dyn50_out_sum_abs", "C_synphys_proxy", controls
        )
        report["species"][str(species)] = srep

    # ---- Save ----
    print("[5/5] Saving derived tables...")
    cells.to_csv(out / "cell_intrinsic_atlas.csv", index=False)
    pairs.to_csv(out / "directed_pair_atlas.csv", index=False)
    celllev.to_csv(out / "cell_local_leverage_atlas.csv", index=False)

    # Compact physical pair table for downstream paper analyses
    pair_keep = [
        "pair_id", "experiment_id", "pre_cell_id", "post_cell_id",
        "tested_primary", "tested_spike_sensitivity", "connected",
        "distance", "synapse_type", "psp_amplitude", "psc_amplitude",
        "dynamics_qc_pass", "n_source_events", "pulse_amp_first_50hz",
        "paired_pulse_ratio_50hz", "stp_initial_50hz", "stp_induction_50hz",
        "stp_recovery_250ms", "stp_recovery_single_250ms",
        "variability_resting_state",
        "pre_species", "pre_cell_class", "pre_cre_type", "pre_target_layer",
        "post_species", "post_cell_class", "post_cre_type", "post_target_layer",
    ]
    pairs[[c for c in pair_keep if c in pairs.columns]].to_csv(
        out / "pair_physical_core.csv", index=False
    )

    report["elapsed_seconds"] = time.time() - t0
    (out / "phase1_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        "# Stage 5C SynPhys Phase 1",
        "",
        f"- Cells: **{report['counts']['cells']:,}**",
        f"- Directed pairs: **{report['counts']['pairs']:,}**",
        f"- Structurally tested pairs: **{report['counts']['tested_pairs_primary']:,}**",
        f"- Connected tested pairs: **{report['counts']['connected_pairs_primary']:,}**",
        f"- Dynamics-QC pairs: **{report['counts']['dynamics_qc_pairs']:,}**",
        "",
        "## Definitions",
        "",
        "- `L_struct_out = connected_out / tested_out`, with `has_synapse IS NOT NULL` as the primary tested-opportunity rule.",
        "- PSP and PSC amplitudes are kept as separate physical channels; they are **not mixed**.",
        "- `L_dyn50_out_sum_abs = sum(abs(pulse_amp_first_50hz))` over `dynamics.qc_pass=1` outgoing pairs.",
        "- `C_synphys_proxy` is an **exploratory** multiaxial PCA proxy only. It is not the frozen Stage 5A complexity mapping.",
        "",
        "## Critical interpretation constraint",
        "",
        "No paper-level complexity→leverage claim should use `C_synphys_proxy` as the final complexity variable. "
        "The next step is to transfer the frozen Stage 5A mapping onto `stage5a_compatible_intrinsic_matrix.csv`, "
        "then rerun the same leverage tests with that externally defined complexity score.",
        "",
        "## Outputs",
        "",
        "- `cell_intrinsic_atlas.csv`",
        "- `stage5a_compatible_intrinsic_matrix.csv`",
        "- `directed_pair_atlas.csv`",
        "- `pair_physical_core.csv`",
        "- `cell_local_leverage_atlas.csv`",
        "- `phase1_report.json`",
    ]
    (out / "00_PHASE1_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    con.close()
    print()
    print("=" * 96)
    print("PHASE 1 COMPLETE")
    print("Output:", out)
    print("Summary:", out / "00_PHASE1_SUMMARY.md")
    print("Report :", out / "phase1_report.json")
    print("=" * 96)


if __name__ == "__main__":
    main()
