#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5C — Integrated Allen Synaptic Physiology complexity–leverage bridge v1
=============================================================================

Single end-to-end paper-level analysis. This replaces the previous sequential
schema-audit -> phase1 -> transfer -> pair-level -> synapse-model -> dynamic ->
identity-decomposition workflow.

Scientific target
-----------------
Test whether an externally defined intrinsic-dynamical-complexity score, learned
ONLY from Stage5A Allen Cell Types data, is associated with local circuit
influence in Allen Synaptic Physiology, and distinguish:

1) structural opportunity (connection probability),
2) conditional synaptic effect size,
3) standardized temporal synaptic transformation,
4) between-cell-identity organization,
5) within-identity complexity gradients.

Important corrections relative to the legacy Stage5C scripts
--------------------------------------------------------------
- SynPhys `fi_slope` is stored after multiplying the IPFX Hz/pA slope by 1e-12.
  To recover Hz/pA it MUST be multiplied by 1e12. The legacy v2 script used
  1e-12, a 1e24 scaling error in the recovery factor.
- Do not infer "co-organization across identities" merely because an already
  weak C effect attenuates after adding identity covariates. The integrated
  analysis explicitly separates between-identity and within-identity C.
- The pair level is the primary inferential unit. Raw presynaptic cell sums are
  descriptive only because they conflate effect size with tested/connected
  partner count.
- Standardized stochastic-synapse responses are called "dynamic synaptic
  influence/phenotype". They are not claimed to be whole-network causal leverage.

Default local paths
-------------------
DB:
D:\\Research\\Neural Science\\bio data\\allen_synap\\synphys_r2.1_full.sqlite

Stage5A archive is auto-discovered under:
D:\\Research\\Neural Science\\bio data\\

Output:
D:\\Research\\Neural Science\\plot\\Stage5C_synphys_integrated_v1

Run
---
python .\\plot\\stage5c_synphys_integrated_v1.py --root "D:\\Research\\Neural Science"

Optional quick run:
python .\\plot\\stage5c_synphys_integrated_v1.py --root "D:\\Research\\Neural Science" --quick

Outputs
-------
analysis/
tables/
figures/
source_data/

All plots:
- separate panels
- no titles
- 600-dpi PNG
- vector PDF
- source-data CSV
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sqlite3
import tarfile
import time
import warnings
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, norm
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

SEED = 20260821
RNG = np.random.default_rng(SEED)
FREQS = [5, 10, 20, 50]
N_SPIKES = 12
ALPHAS = np.logspace(-3, 4, 24)

# ---------------------------------------------------------------------
# Harmonization contract
# ---------------------------------------------------------------------
# "primary_shared": same biological construct, compatible physical units,
# but not every feature is protocol-identical across the two Allen datasets.
# "protocol_conservative": narrower set that avoids the most obvious
# rheobase-vs-hero / across-sweep summary mismatches.
#
# We deliberately DO NOT map SynPhys ap_fast_trough_deltav directly to
# Stage5A fast_trough_v_long_square because delta-V and absolute voltage,
# and hero-vs-rheobase protocols, differ.
HARMONIZATION = [
    dict(
        canonical="adaptation_index",
        stage5a_candidates=["adaptation"],
        synphys="adaptation_index",
        stage5a_scale=1.0, synphys_scale=1.0,
        unit="dimensionless", tier="primary_shared",
        note="same adaptation construct; SynPhys is an across-sweep summary"
    ),
    dict(
        canonical="fi_slope_hz_per_pa",
        stage5a_candidates=["f_i_curve_slope"],
        synphys="fi_slope",
        stage5a_scale=1.0, synphys_scale=1e12,
        unit="Hz/pA", tier="protocol_conservative",
        note="CRITICAL FIX: SynPhys DB value = IPFX Hz/pA * 1e-12"
    ),
    dict(
        canonical="input_resistance_mohm",
        stage5a_candidates=["input_resistance_mohm"],
        synphys="input_resistance",
        stage5a_scale=1.0, synphys_scale=1e-6,
        unit="MOhm", tier="protocol_conservative",
        note="SynPhys stores ohm; convert to MOhm"
    ),
    dict(
        canonical="sag_ratio",
        stage5a_candidates=["sag"],
        synphys="sag",
        stage5a_scale=1.0, synphys_scale=1.0,
        unit="dimensionless", tier="protocol_conservative",
        note="hyperpolarizing sag ratio"
    ),
    dict(
        canonical="threshold_v_mv",
        stage5a_candidates=["threshold_v_long_square"],
        synphys="ap_threshold_v",
        stage5a_scale=1.0, synphys_scale=1e3,
        unit="mV", tier="primary_shared",
        note="same threshold construct; protocol differs (Stage5A rheobase vs SynPhys hero)"
    ),
    dict(
        canonical="upstroke_downstroke_ratio",
        stage5a_candidates=["upstroke_downstroke_ratio_long_square"],
        synphys="ap_upstroke_downstroke_ratio",
        stage5a_scale=1.0, synphys_scale=1.0,
        unit="dimensionless", tier="primary_shared",
        note="same spike-shape construct; protocol differs"
    ),
    dict(
        canonical="rheobase_pa",
        stage5a_candidates=["threshold_i_long_square"],
        synphys="rheobase",
        stage5a_scale=1.0, synphys_scale=1e12,
        unit="pA", tier="protocol_conservative",
        note="SynPhys stores A; convert to pA"
    ),
]

INTRINSIC_COLS = [
    "rheobase","fi_slope","input_resistance","input_resistance_ss","sag","tau",
    "ap_upstroke_downstroke_ratio","ap_width","ap_upstroke","ap_downstroke",
    "ap_threshold_v","ap_peak_deltav","ap_fast_trough_deltav",
    "firing_rate_rheo","latency_rheo","firing_rate_40pa","latency_40pa",
    "adaptation_index","isi_cv","chirp_peak_freq","chirp_3db_freq",
    "chirp_peak_ratio","chirp_peak_impedance","chirp_sync_freq",
    "chirp_inductive_phase","isi_adapt_ratio","upstroke_adapt_ratio",
    "downstroke_adapt_ratio","width_adapt_ratio","threshold_v_adapt_ratio",
]

STP_ENDPOINTS = [
    "paired_pulse_ratio_50hz","stp_initial_50hz","stp_induction_50hz",
    "stp_recovery_250ms","stp_recovery_single_250ms",
    "variability_resting_state","variability_second_pulse_50hz",
    "variability_stp_induced_state_50hz",
]

SYNAPSE_MODEL_PARAMS = [
    "ml_n_release_sites","ml_base_release_probability","ml_mini_amplitude",
    "ml_mini_amplitude_cv","ml_depression_amount","ml_depression_tau",
    "ml_facilitation_amount","ml_facilitation_tau","ml_measurement_stdev",
]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--stage5a", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--mapper-bootstrap", type=int, default=50)
    p.add_argument("--cv-repeats", type=int, default=10)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()

def ro_connect(path: Path):
    uri = f"file:{quote(str(path).replace(os.sep, '/'))}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True, timeout=60)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA busy_timeout=60000")
    return con

def find_stage5a(root: Path) -> Path:
    preferred = [
        root / "bio data" / "stage5_v12_results.tar.gz",
        root / "bio data" / "stage5_v12_biological_atlas.tar.gz",
    ]
    for p in preferred:
        if p.exists():
            return p
    hits = list((root / "bio data").rglob("*v12*.tar.gz"))
    for p in sorted(hits, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with tarfile.open(p, "r:gz") as tf:
                names = tf.getnames()
                if any(n.endswith("allen_multiaxial_primary_mouse_complete.csv") for n in names):
                    return p
        except Exception:
            pass
    raise FileNotFoundError(
        "Could not find Stage5A v12 archive containing "
        "allen_multiaxial_primary_mouse_complete.csv"
    )

def read_stage5a(path: Path):
    with tarfile.open(path, "r:gz") as tf:
        names = tf.getnames()
        pm = [n for n in names if n.endswith("allen_multiaxial_primary_mouse_complete.csv")]
        sm = [n for n in names if n.endswith("allen_multiaxial_summary.json")]
        if len(pm) != 1 or len(sm) != 1:
            raise RuntimeError(f"Unexpected Stage5A archive members: primary={len(pm)}, summary={len(sm)}")
        f1 = tf.extractfile(pm[0]); f2 = tf.extractfile(sm[0])
        if f1 is None or f2 is None:
            raise RuntimeError("Could not extract Stage5A primary/summary")
        primary = pd.read_csv(f1)
        summary = json.loads(f2.read().decode("utf-8"))
    return primary, summary, pm[0], sm[0]

def jdump(obj, path: Path):
    def conv(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if pd.isna(x) if not isinstance(x, (dict,list,tuple,str,bool)) else False: return None
        return x
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv), encoding="utf-8")

def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    m = np.isfinite(p)
    pv = p[m]
    if len(pv) == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked)+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    out[m] = q[inv]
    return out

def zscore_series(x):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    sd = x.std()
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd

def collapse_rare(s, min_n):
    s = s.fillna("MISSING").astype(str)
    vc = s.value_counts()
    keep = vc[vc >= min_n].index
    return s.where(s.isin(keep), "OTHER")

def save_panel(fig, stem: str, figdir: Path, dpi: int):
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(figdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def set_plot_defaults():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.transparent": False,
    })

# ---------------------------------------------------------------------
# Database extraction
# ---------------------------------------------------------------------

def assert_db_schema(con):
    required = {
        "cell": ["id","experiment_id","cre_type","cell_class"],
        "experiment": ["id","slice_id"],
        "slice": ["id","species","age"],
        "intrinsic": ["cell_id","fi_slope","input_resistance","sag","rheobase"],
        "pair": ["id","experiment_id","pre_cell_id","post_cell_id","has_synapse","distance"],
        "synapse": ["pair_id","psp_amplitude","psc_amplitude"],
        "dynamics": ["pair_id","qc_pass","pulse_amp_first_50hz"],
        "synapse_model": ["pair_id","ml_n_release_sites","ml_base_release_probability"],
    }
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = [t for t in required if t not in tables]
    if missing:
        raise RuntimeError(f"Missing required DB tables: {missing}")
    for t, cols in required.items():
        have = {r[1] for r in con.execute(f'PRAGMA table_info("{t}")')}
        miss = [c for c in cols if c not in have]
        if miss:
            raise RuntimeError(f"Table {t} missing columns: {miss}")

def build_cell_atlas(con):
    cols = ",\n        ".join([f"i.{c}" for c in INTRINSIC_COLS])
    sql = f"""
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
        {cols},
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
    d = pd.read_sql_query(sql, con).drop_duplicates("cell_id")
    return d

def build_pair_atlas(con, cells):
    sql = """
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
    p = pd.read_sql_query(sql, con).drop_duplicates("pair_id")
    p["tested_primary"] = p["has_synapse"].notna()
    p["connected"] = pd.to_numeric(p["has_synapse"], errors="coerce").fillna(0).eq(1)
    p["tested_spike_sensitivity"] = (
        pd.to_numeric(p["n_ex_test_spikes"], errors="coerce").fillna(0)
        + pd.to_numeric(p["n_in_test_spikes"], errors="coerce").fillna(0)
    ) > 0
    p["psp_abs"] = pd.to_numeric(p["psp_amplitude"], errors="coerce").abs()
    p["psc_abs"] = pd.to_numeric(p["psc_amplitude"], errors="coerce").abs()

    meta = ["cell_id","species","age","sex","cre_type","cell_class","cell_class_nonsynaptic",
            "target_layer","cortical_layer","target_region","internal","acsf","target_temperature",
            "experiment_id"]
    pre = cells[meta].copy().add_prefix("pre_")
    post = cells[meta].copy().add_prefix("post_")
    p = p.merge(pre, on="pre_cell_id", how="left")
    p = p.merge(post, on="post_cell_id", how="left")
    return p

def read_synapse_models(con):
    return pd.read_sql_query("SELECT * FROM synapse_model", con).drop_duplicates("pair_id")

# ---------------------------------------------------------------------
# Complexity transfer
# ---------------------------------------------------------------------

def stage5a_feature_set(primary, summary):
    fs = set(summary.get("features", []))
    # Require actual presence as the final gate.
    return fs.intersection(primary.columns)

def choose_stage5a_col(spec, feature_set):
    for c in spec["stage5a_candidates"]:
        if c in feature_set:
            return c
    return None

def harmonized_matrices(stage5a, cells, feature_set, mode="primary_shared"):
    # mode primary_shared uses both primary_shared and protocol_conservative;
    # protocol_conservative uses only the latter.
    rows = []
    A = pd.DataFrame(index=stage5a.index)
    S = pd.DataFrame(index=cells.index)
    for h in HARMONIZATION:
        use = h["tier"] == "protocol_conservative" or mode == "primary_shared"
        if not use:
            continue
        a = choose_stage5a_col(h, feature_set)
        s = h["synphys"] if h["synphys"] in cells.columns else None
        usable = a is not None and s is not None
        rec = {**h, "stage5a_selected": a, "synphys_present": s is not None,
               "stage5a_present": a is not None, "usable": usable, "mode": mode}
        rows.append(rec)
        if usable:
            A[h["canonical"]] = pd.to_numeric(stage5a[a], errors="coerce") * float(h["stage5a_scale"])
            S[h["canonical"]] = pd.to_numeric(cells[s], errors="coerce") * float(h["synphys_scale"])
    return A, S, pd.DataFrame(rows)

def stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    folds = [[] for _ in range(k)]
    for c in np.unique(y[np.isfinite(y)]):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for j, ii in enumerate(idx):
            folds[j % k].append(int(ii))
    return [np.array(sorted(x), dtype=int) for x in folds if len(x)]

def scaler_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0
    return mu, sd

def scaler_apply(X, mu, sd):
    return (X - mu) / sd

def ridge_fit_np(X, y, alpha):
    mu, sd = scaler_fit(X)
    Z = scaler_apply(X, mu, sd)
    ym = float(np.mean(y))
    yy = y - ym
    M = Z.T @ Z + float(alpha) * np.eye(Z.shape[1])
    try:
        w = np.linalg.solve(M, Z.T @ yy)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(M) @ (Z.T @ yy)
    return {"mu":mu, "sd":sd, "ym":ym, "w":w, "alpha":float(alpha)}

def ridge_pred_np(m, X):
    return scaler_apply(X, m["mu"], m["sd"]) @ m["w"] + m["ym"]

def choose_alpha(X, y, seed, k=5):
    folds = stratified_folds(y, min(k, max(3, len(np.unique(y)))), seed)
    vals = []
    for a in ALPHAS:
        mse=[]
        for te in folds:
            tr=np.setdiff1d(np.arange(len(y)),te)
            mm=ridge_fit_np(X[tr],y[tr],a)
            pr=ridge_pred_np(mm,X[te])
            mse.append(float(np.mean((pr-y[te])**2)))
        vals.append(np.mean(mse))
    return float(ALPHAS[int(np.argmin(vals))])

def repeated_nested_cv(X, y, repeats=10):
    ps=np.zeros(len(y)); pn=np.zeros(len(y)); al=[]
    for rep in range(repeats):
        folds=stratified_folds(y,5,SEED+997*rep)
        for fi,te in enumerate(folds):
            tr=np.setdiff1d(np.arange(len(y)),te)
            a=choose_alpha(X[tr],y[tr],SEED+5003*rep+fi)
            m=ridge_fit_np(X[tr],y[tr],a)
            ps[te]+=ridge_pred_np(m,X[te]); pn[te]+=1; al.append(a)
    pred=ps/np.maximum(pn,1)
    rho,p=spearmanr(y,pred)
    return pred, {
        "n":int(len(y)),"rho":float(rho),"p":float(p),
        "mae":float(np.mean(np.abs(y-pred))),
        "r2":float(r2_score(y,pred)),
        "median_alpha":float(np.median(al))
    }

def fit_transfer(stage5a, cells, summary, mode, repeats, n_boot):
    feature_set=stage5a_feature_set(stage5a,summary)
    A,S,contract=harmonized_matrices(stage5a,cells,feature_set,mode)
    features=list(A.columns)
    if len(features)<4:
        raise RuntimeError(f"{mode}: only {len(features)} usable shared features: {features}")
    y=pd.to_numeric(stage5a["C_req_eps_0.02"],errors="coerce")
    tm=A.notna().all(axis=1)&y.notna()
    X=A.loc[tm].to_numpy(float); yy=y.loc[tm].to_numpy(float)
    if len(yy)<200:
        raise RuntimeError(f"{mode}: too few complete Stage5A cases ({len(yy)})")
    oof,cv=repeated_nested_cv(X,yy,repeats)
    final_alpha=choose_alpha(X,yy,SEED+8888)
    final=ridge_fit_np(X,yy,final_alpha)

    syn_complete=S.notna().all(axis=1)
    pred=np.full(len(cells),np.nan)
    pred[syn_complete.to_numpy()]=ridge_pred_np(final,S.loc[syn_complete].to_numpy(float))

    # Bootstrap training uncertainty, without using SynPhys outcomes.
    n_boot=int(n_boot)
    boot_pred=np.full((n_boot,len(cells)),np.nan,dtype=np.float32) if n_boot>0 else None
    rng=np.random.default_rng(SEED+123)
    if n_boot>0 and syn_complete.any():
        SX=S.loc[syn_complete].to_numpy(float)
        for b in range(n_boot):
            idx=rng.integers(0,len(yy),len(yy))
            mb=ridge_fit_np(X[idx],yy[idx],final_alpha)
            boot_pred[b,syn_complete.to_numpy()]=ridge_pred_np(mb,SX).astype(np.float32)

    # Domain-of-applicability.
    Am=A.loc[tm]
    train_min=Am.min(); train_max=Am.max()
    med=Am.median(); mad=(Am-med).abs().median()*1.4826
    mad=mad.mask((~np.isfinite(mad))|(mad<=1e-12),Am.std().replace(0,1))
    Z=(S-med)/mad
    dist=np.sqrt((Z**2).sum(axis=1))
    trdist=np.sqrt((((Am-med)/mad)**2).sum(axis=1))
    dist_thr=float(np.nanquantile(trdist,0.995))
    ood=((S.lt(train_min)) | (S.gt(train_max))).sum(axis=1)
    domain=syn_complete & (ood<=1) & (dist<=dist_thr)

    # Gross physical scale audit (would catch the historical fi_slope bug).
    qc=[]
    for f in features:
        a=Am[f].dropna()
        s=S.loc[syn_complete,f].dropna()
        rec=dict(feature=f,stage5a_n=len(a),synphys_n=len(s),
                 stage5a_mean=float(a.mean()),stage5a_sd=float(a.std()),
                 stage5a_median=float(a.median()),synphys_median=float(s.median()),
                 stage5a_min=float(a.min()),stage5a_max=float(a.max()))
        if f in {"fi_slope_hz_per_pa","input_resistance_mohm","rheobase_pa"}:
            aa=abs(rec["stage5a_median"]); ss=abs(rec["synphys_median"])
            rec["median_magnitude_ratio_syn_over_stage"] = ss/max(aa,1e-30)
            if not (1e-4 <= rec["median_magnitude_ratio_syn_over_stage"] <= 1e4):
                raise RuntimeError(
                    f"Gross unit mismatch for {f}: SynPhys/Stage5A median magnitude ratio="
                    f"{rec['median_magnitude_ratio_syn_over_stage']:.3e}"
                )
        qc.append(rec)

    out=cells[["cell_id","species","experiment_id","cre_type","cell_class",
               "target_layer","cortical_layer","target_region","age"]].copy()
    out[f"C_{mode}"]=pred
    out[f"C_{mode}_complete"]=syn_complete.to_numpy()
    out[f"C_{mode}_domain"]=domain.to_numpy()
    out[f"C_{mode}_ood_count"]=ood.to_numpy()
    out[f"C_{mode}_domain_distance"]=dist.to_numpy()
    if boot_pred is not None:
        out[f"C_{mode}_sd"]=np.nanstd(boot_pred,axis=0)
    else:
        out[f"C_{mode}_sd"]=np.nan
    return {
        "mode":mode,"features":features,"contract":contract,"stage5a_train_mask":tm,
        "stage5a_X":X,"stage5a_y":yy,"stage5a_oof":oof,"cv":cv,
        "final":final,"cell_scores":out,"feature_qc":pd.DataFrame(qc),
        "boot_pred":boot_pred,"domain_threshold":dist_thr
    }

# ---------------------------------------------------------------------
# Standardized stochastic-synapse dynamics
# ---------------------------------------------------------------------

def simulate_expected_train(row, freq_hz, n_spikes=N_SPIKES):
    n_sites=float(row["ml_n_release_sites"])
    base_pr=float(row["ml_base_release_probability"])
    mini=float(row["ml_mini_amplitude"])
    dep_amt=float(row["ml_depression_amount"])
    dep_tau=float(row["ml_depression_tau"])
    fac_amt=float(row["ml_facilitation_amount"])
    fac_tau=float(row["ml_facilitation_tau"])
    vals=[n_sites,base_pr,mini,dep_amt,dep_tau,fac_amt,fac_tau]
    if not all(np.isfinite(vals)) or n_sites<=0 or dep_tau<=0 or fac_tau<=0 or not (0<=base_pr<=1):
        return None
    dt=1/float(freq_hz)
    pool=n_sites; fac=0.; dep=0.
    depletion=np.isclose(dep_amt,-1.0)
    amps=np.empty(n_spikes,float); prs=np.empty(n_spikes,float)
    for k in range(n_spikes):
        if k>0:
            if depletion:
                r=np.exp(-dt/dep_tau); pool += (n_sites-pool)*(1-r)
            else:
                dep *= np.exp(-dt/dep_tau)
            fac *= np.exp(-dt/fac_tau)
        avail=max(0.,pool) if depletion else pool
        pr=(1-dep)*(base_pr+(1-base_pr)*fac)
        pr=float(np.clip(pr,0,1))
        amps[k]=avail*pr*mini; prs[k]=pr
        if depletion:
            pool -= avail*pr
        else:
            dep += (1-dep)*dep_amt
        fac += (1-fac)*fac_amt
    return amps,prs

def train_features(amps):
    a=np.abs(np.asarray(amps,float)); a1=a[0]
    if not np.isfinite(a1) or a1<=1e-15:
        return None
    n=a/a1
    return dict(E_abs=float(a.sum()),G_norm=float(n.mean()),
                tail_first=float(n[-3:].mean()),
                second_first=float(n[1]),
                dyn_range=float((a.max()-a.min())/a1),A1_abs=float(a1))

def build_standardized_model_atlas(smodels, pairs, cscores, score_col):
    pc=pairs.drop_duplicates("pair_id")
    cols=["pair_id","pre_cell_id","post_cell_id","experiment_id","distance",
          "pre_species","pre_age","pre_cell_class","post_cell_class",
          "pre_cortical_layer","post_cortical_layer","pre_cre_type","post_cre_type"]
    m=smodels.merge(pc[[c for c in cols if c in pc.columns]],on="pair_id",how="left")
    ccols=["cell_id",score_col,score_col+"_domain",score_col+"_sd"]
    sc=cscores[[c for c in ccols if c in cscores.columns]].rename(columns={"cell_id":"pre_cell_id"})
    m=m.merge(sc,on="pre_cell_id",how="left")
    ok=pd.Series(True,index=m.index)
    for c in SYNAPSE_MODEL_PARAMS:
        if c not in m:
            raise RuntimeError(f"synapse_model missing parameter {c}")
        ok &= pd.to_numeric(m[c],errors="coerce").notna()
    m=m[ok & m[score_col].notna()].copy()
    rows=[]
    for idx,row in m.iterrows():
        rec=row.to_dict(); good=True
        for f in FREQS:
            sim=simulate_expected_train(row,f)
            if sim is None: good=False; break
            amps,prs=sim; feat=train_features(amps)
            if feat is None: good=False; break
            for k,v in feat.items(): rec[f"{k}_{f}hz"]=v
            rec[f"amp_train_{f}hz"]=";".join(f"{v:.10g}" for v in amps)
            rec[f"pr_train_{f}hz"]=";".join(f"{v:.10g}" for v in prs)
        if good: rows.append(rec)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Statistical models
# ---------------------------------------------------------------------

def base_design(d, outcome=None, add_identity=False, add_strength=False, score_col=None):
    X=pd.DataFrame(index=d.index)
    if score_col:
        X["C_z"]=zscore_series(d[score_col])
    if "distance" in d:
        x=pd.to_numeric(d["distance"],errors="coerce")
        if x.notna().sum()>=30:
            med=x.median(); X["log1p_distance_um"]=np.log1p(np.maximum(x.fillna(med)*1e6,0))
    if "pre_age" in d:
        x=pd.to_numeric(d["pre_age"],errors="coerce")
        if x.notna().sum()>=30: X["pre_age"]=x.fillna(x.median())
    if add_strength and "A1_abs_50hz" in d:
        x=pd.to_numeric(d["A1_abs_50hz"],errors="coerce")
        if x.notna().sum()>=30: X["log_A1_abs_50hz"]=np.log(np.maximum(x,1e-15)).fillna(np.log(np.maximum(x,1e-15)).median())
    # Layer is anatomy/context in both base and identity models.
    for c in ["pre_cortical_layer","post_cortical_layer"]:
        if c in d:
            s=collapse_rare(d[c],max(10,int(.01*len(d))))
            dm=pd.get_dummies(s,prefix=c,drop_first=True,dtype=float)
            if dm.shape[1]<=20: X=pd.concat([X,dm],axis=1)
    if add_identity:
        for c in ["pre_cell_class","post_cell_class","pre_cre_type","post_cre_type"]:
            if c in d:
                s=collapse_rare(d[c],max(12,int(.0125*len(d))))
                dm=pd.get_dummies(s,prefix=c,drop_first=True,dtype=float)
                if dm.shape[1]<=40: X=pd.concat([X,dm],axis=1)
    return X.astype(float)

def clustered_ols_ladder(d, outcome, score_col, log_abs=False, add_strength=False):
    import statsmodels.api as sm
    use=d.copy()
    y=pd.to_numeric(use[outcome],errors="coerce").astype(float)
    if log_abs:
        y=np.log(np.maximum(np.abs(y),1e-15))
    else:
        z=y.dropna()
        if len(z)>=100:
            lo,hi=z.quantile([.01,.99]); y=y.clip(lo,hi)
    # Standardize the transformed outcome so continuous coefficients are comparable
    # across PSP/PSC/dynamic endpoints. R² and p-values are invariant to this scaling.
    ysd = float(y.std())
    if np.isfinite(ysd) and ysd > 1e-12:
        y = (y - y.mean()) / ysd
    use["_y"]=y
    use=use.replace([np.inf,-np.inf],np.nan).dropna(subset=["_y",score_col,"experiment_id"])
    if len(use)<100: return {"status":"too_few","n":int(len(use))}
    B=base_design(use,outcome,False,add_strength,None)
    I=base_design(use,outcome,True,add_strength,None)
    C=zscore_series(use[score_col]).rename("C_z")
    designs={"M0_base":B,"M1_base_C":pd.concat([B,C],axis=1),
             "M2_identity":I,"M3_identity_C":pd.concat([I,C],axis=1)}
    fits={}
    for name,X in designs.items():
        Xc=sm.add_constant(X,has_constant="add")
        plain=sm.OLS(use["_y"].astype(float),Xc).fit()
        robust=plain.get_robustcov_results(cov_type="cluster",groups=use["experiment_id"].astype(str).to_numpy())
        names=list(Xc.columns)
        rec={"r2":float(plain.rsquared),"adj_r2":float(plain.rsquared_adj),"n_parameters":len(names)}
        if "C_z" in names:
            i=names.index("C_z")
            rec.update(beta_C_z=float(robust.params[i]),se_C_z_cluster=float(robust.bse[i]),
                       p_C_z_cluster=float(robust.pvalues[i]))
        fits[name]=rec
    fits["delta_r2_C_before_identity"]=fits["M1_base_C"]["r2"]-fits["M0_base"]["r2"]
    fits["delta_r2_C_after_identity"]=fits["M3_identity_C"]["r2"]-fits["M2_identity"]["r2"]
    fits["n"]=int(len(use)); fits["n_experiments"]=int(use["experiment_id"].nunique())
    return {"status":"ok",**fits}

def clustered_logit_ladder(d, score_col):
    import statsmodels.api as sm
    use=d.copy().dropna(subset=["connected",score_col,"experiment_id"])
    if len(use)<500 or use["connected"].nunique()<2: return {"status":"too_few","n":len(use)}
    B=base_design(use,None,False,False,None); I=base_design(use,None,True,False,None)
    C=zscore_series(use[score_col]).rename("C_z")
    designs={"M0_base":B,"M1_base_C":pd.concat([B,C],axis=1),
             "M2_identity":I,"M3_identity_C":pd.concat([I,C],axis=1)}
    out={}
    y=use["connected"].astype(int)
    for name,X in designs.items():
        Xc=sm.add_constant(X,has_constant="add")
        try:
            fit=sm.GLM(y,Xc,family=sm.families.Binomial()).fit(
                cov_type="cluster",cov_kwds={"groups":use["experiment_id"].astype(str).to_numpy()},
                maxiter=200)
            rec={"aic":float(fit.aic),"n_parameters":Xc.shape[1]}
            if "C_z" in Xc.columns:
                rec.update(beta_C_z_logodds=float(fit.params["C_z"]),
                           se_C_z_cluster=float(fit.bse["C_z"]),
                           p_C_z_cluster=float(fit.pvalues["C_z"]),
                           odds_ratio_per_sd=float(np.exp(fit.params["C_z"])))
            out[name]=rec
        except Exception as e:
            out[name]={"error":repr(e)}
    out["n"]=int(len(use)); out["n_experiments"]=int(use["experiment_id"].nunique())
    out["connection_rate"]=float(y.mean())
    return {"status":"ok",**out}

def group_cv_continuous(d,outcome,score_col,log_abs=False,add_strength=False):
    use=d.copy()
    y=pd.to_numeric(use[outcome],errors="coerce").astype(float)
    if log_abs: y=np.log(np.maximum(np.abs(y),1e-15))
    use["_y"]=y
    use=use.replace([np.inf,-np.inf],np.nan).dropna(subset=["_y",score_col,"experiment_id"])
    if use["experiment_id"].nunique()<5: return {"status":"too_few_groups"}
    B=base_design(use,outcome,False,add_strength,None)
    I=base_design(use,outcome,True,add_strength,None)
    C=zscore_series(use[score_col]).rename("C_z")
    designs={"M0":B,"M1":pd.concat([B,C],axis=1),"M2":I,"M3":pd.concat([I,C],axis=1)}
    gkf=GroupKFold(5); groups=use["experiment_id"].astype(str).to_numpy(); yy=use["_y"].to_numpy(float)
    pred={k:np.full(len(use),np.nan) for k in designs}
    for tr,te in gkf.split(np.arange(len(use)),yy,groups):
        for k,X in designs.items():
            Xa=X.to_numpy(float)
            model=Ridge(alpha=1e-6).fit(Xa[tr],yy[tr])
            pred[k][te]=model.predict(Xa[te])
    rec={f"{k}_cv_r2":float(r2_score(yy,v)) for k,v in pred.items()}
    rec["delta_cv_r2_C_before_identity"]=rec["M1_cv_r2"]-rec["M0_cv_r2"]
    rec["delta_cv_r2_C_after_identity"]=rec["M3_cv_r2"]-rec["M2_cv_r2"]
    rec["status"]="ok"; return rec

def group_cv_connection(d,score_col):
    use=d.copy().dropna(subset=["connected",score_col,"experiment_id"])
    if use["experiment_id"].nunique()<5 or use["connected"].nunique()<2:
        return {"status":"too_few"}
    B=base_design(use,None,False,False,None)
    I=base_design(use,None,True,False,None)
    C=zscore_series(use[score_col]).rename("C_z")
    designs={"M0":B,"M1":pd.concat([B,C],axis=1),"M2":I,"M3":pd.concat([I,C],axis=1)}
    yy=use["connected"].astype(int).to_numpy(); groups=use["experiment_id"].astype(str).to_numpy()
    gkf=GroupKFold(5); probs={k:np.full(len(use),np.nan) for k in designs}
    for tr,te in gkf.split(np.arange(len(use)),yy,groups):
        for k,X in designs.items():
            try:
                m=LogisticRegression(C=1.0,max_iter=1000,class_weight=None).fit(X.to_numpy(float)[tr],yy[tr])
                probs[k][te]=m.predict_proba(X.to_numpy(float)[te])[:,1]
            except Exception:
                probs[k][te]=yy[tr].mean()
    rec={}
    for k,p in probs.items():
        rec[f"{k}_logloss"]=float(log_loss(yy,p,labels=[0,1]))
        try: rec[f"{k}_auc"]=float(roc_auc_score(yy,p))
        except: rec[f"{k}_auc"]=np.nan
    rec["delta_logloss_C_before_identity"]=rec["M0_logloss"]-rec["M1_logloss"]
    rec["delta_logloss_C_after_identity"]=rec["M2_logloss"]-rec["M3_logloss"]
    rec["status"]="ok"; return rec

def within_between_decomposition(d,outcome,score_col,identity_col="pre_cre_type",add_strength=False):
    """
    Descriptive/robust decomposition. Group means are computed at the UNIQUE
    presynaptic-cell level (not pair-weighted). The between-identity coefficient
    has few effective identity groups and is not treated as a headline p-value.
    """
    import statsmodels.api as sm
    use=d.copy().dropna(subset=[outcome,score_col,"experiment_id","pre_cell_id"])
    if identity_col not in use:
        return {"status":"missing_identity"}
    cell=use[["pre_cell_id",identity_col,score_col]].drop_duplicates("pre_cell_id").copy()
    counts=cell[identity_col].fillna("MISSING").astype(str).value_counts()
    keep=counts[counts>=8].index
    cell["_id"]=cell[identity_col].fillna("MISSING").astype(str)
    cell=cell[cell["_id"].isin(keep)].copy()
    if cell["_id"].nunique()<3: return {"status":"too_few_identity_groups"}
    means=cell.groupby("_id")[score_col].mean()
    cell["C_between"]=cell["_id"].map(means)
    cell["C_within"]=cell[score_col]-cell["C_between"]
    use=use.merge(cell[["pre_cell_id","_id","C_between","C_within"]],on="pre_cell_id",how="inner")
    y=pd.to_numeric(use[outcome],errors="coerce")
    if add_strength and "A1_abs_50hz" in use:
        pass
    B=base_design(use,outcome,False,add_strength,None)
    # Post identity is a context; pre identity dummies are NOT added because they
    # are collinear with C_between.
    for c in ["post_cell_class","post_cre_type"]:
        if c in use:
            s=collapse_rare(use[c],max(12,int(.0125*len(use))))
            dm=pd.get_dummies(s,prefix=c,drop_first=True,dtype=float)
            if dm.shape[1]<=40: B=pd.concat([B,dm],axis=1)
    B["C_between_z"]=zscore_series(use["C_between"])
    B["C_within_z"]=zscore_series(use["C_within"])
    X=sm.add_constant(B,has_constant="add")
    fit=sm.OLS(y.astype(float),X).fit().get_robustcov_results(
        cov_type="cluster",groups=use["experiment_id"].astype(str).to_numpy())
    names=list(X.columns)
    rec={"status":"ok","n_pairs":int(len(use)),"n_pre_cells":int(use["pre_cell_id"].nunique()),
         "n_identity_groups":int(cell["_id"].nunique())}
    for nm in ["C_between_z","C_within_z"]:
        i=names.index(nm)
        rec[nm]={"beta":float(fit.params[i]),"se_cluster":float(fit.bse[i]),"p_cluster":float(fit.pvalues[i])}
    return rec, cell

# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def binned_xy(x,y,n_bins=8):
    d=pd.DataFrame({"x":pd.to_numeric(x,errors="coerce"),"y":pd.to_numeric(y,errors="coerce")}).dropna()
    if len(d)<20: return pd.DataFrame()
    try:
        d["bin"]=pd.qcut(d["x"],q=min(n_bins,d["x"].nunique()),duplicates="drop")
    except Exception:
        return pd.DataFrame()
    g=d.groupby("bin",observed=True).agg(x=("x","mean"),y=("y","mean"),n=("y","size"),
                                         se=("y",lambda v:v.std()/np.sqrt(len(v)) if len(v)>1 else np.nan)).reset_index(drop=True)
    return g

def coefficient_table(models, family):
    rows=[]
    for endpoint,res in models.items():
        if not isinstance(res,dict) or res.get("status")!="ok": continue
        m=res.get("M3_identity_C",{})
        beta=m.get("beta_C_z",m.get("beta_C_z_logodds"))
        se=m.get("se_C_z_cluster")
        p=m.get("p_C_z_cluster")
        if beta is None or se is None or p is None: continue
        rows.append(dict(family=family,endpoint=endpoint,beta=beta,se=se,p=p,
                         lo=beta-1.96*se,hi=beta+1.96*se))
    df=pd.DataFrame(rows)
    if len(df): df["q_fdr"]=bh_fdr(df["p"])
    return df

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args=parse_args()
    root=args.root.resolve()
    db=(args.db or root/"bio data"/"allen_synap"/"synphys_r2.1_full.sqlite").resolve()
    stage5a=(args.stage5a.resolve() if args.stage5a else find_stage5a(root))
    out=(args.out.resolve() if args.out else root/"plot"/"Stage5C_synphys_integrated_v1")
    analysis=out/"analysis"; tables=out/"tables"; figs=out/"figures"; src=out/"source_data"
    for p in [analysis,tables,figs,src]: p.mkdir(parents=True,exist_ok=True)
    set_plot_defaults()
    if args.quick:
        repeats=2; nboot=min(5,args.mapper_bootstrap); dpi=min(180,args.dpi)
    else:
        repeats=args.cv_repeats; nboot=args.mapper_bootstrap; dpi=args.dpi

    print("="*100)
    print("Stage 5C integrated Synaptic Physiology analysis")
    print("DB      :",db)
    print("Stage5A :",stage5a)
    print("Output  :",out)
    print("="*100)
    t0=time.time()

    if not db.exists(): raise FileNotFoundError(db)
    con=ro_connect(db); assert_db_schema(con)

    print("[1/8] Extracting compact SQLite tables (no pulse_response scan)...")
    cells=build_cell_atlas(con)
    pairs=build_pair_atlas(con,cells)
    smodels=read_synapse_models(con)
    con.close()
    cells.to_csv(tables/"cell_intrinsic_atlas.csv",index=False)
    pairs.to_csv(tables/"directed_pair_atlas.csv",index=False)

    print("[2/8] Reading Stage5A and fitting corrected external complexity transfer...")
    A5,summary,pm,sm=read_stage5a(stage5a)
    primary=fit_transfer(A5,cells,summary,"primary_shared",repeats,nboot)
    conservative=fit_transfer(A5,cells,summary,"protocol_conservative",repeats,nboot)
    for r in [primary,conservative]:
        r["contract"].to_csv(analysis/f"harmonization_{r['mode']}.csv",index=False)
        r["feature_qc"].to_csv(analysis/f"transfer_feature_qc_{r['mode']}.csv",index=False)
        r["cell_scores"].to_csv(tables/f"complexity_transfer_{r['mode']}.csv",index=False)
        pd.DataFrame({"C_req":r["stage5a_y"],"OOF_pred":r["stage5a_oof"]}).to_csv(
            analysis/f"stage5a_oof_{r['mode']}.csv",index=False)

    score="C_primary_shared"
    scores=primary["cell_scores"]
    # Merge primary score into pair table.
    smerge=scores[["cell_id",score,score+"_domain",score+"_sd"]].rename(columns={"cell_id":"pre_cell_id"})
    pairs=pairs.merge(smerge,on="pre_cell_id",how="left")
    pairs.to_csv(tables/"directed_pair_with_complexity.csv",index=False)

    print("[3/8] Building standardized stochastic-synapse train atlas...")
    std=build_standardized_model_atlas(smodels,pairs,scores,score)
    std.to_csv(tables/"standardized_synapse_train_atlas.csv",index=False)

    print("[4/8] Defining primary mouse cohorts...")
    mouse_tested=pairs[
        pairs["pre_species"].astype(str).str.lower().eq("mouse") &
        pairs[score].notna() & pairs[score+"_domain"].fillna(False) &
        pairs["tested_primary"]
    ].copy()
    mouse_conn=mouse_tested[mouse_tested["connected"]].copy()
    mouse_dyn=pairs[
        pairs["pre_species"].astype(str).str.lower().eq("mouse") &
        pairs[score].notna() & pairs[score+"_domain"].fillna(False) &
        pd.to_numeric(pairs["dynamics_qc_pass"],errors="coerce").fillna(0).eq(1)
    ].copy()
    mouse_std=std[
        std["pre_species"].astype(str).str.lower().eq("mouse") &
        std[score+"_domain"].fillna(False)
    ].copy()

    mouse_tested.to_csv(tables/"mouse_tested_pairs_primary.csv",index=False)
    mouse_conn.to_csv(tables/"mouse_connected_pairs_primary.csv",index=False)
    mouse_dyn.to_csv(tables/"mouse_measured_dynamics_primary.csv",index=False)
    mouse_std.to_csv(tables/"mouse_standardized_model_pairs_primary.csv",index=False)

    print("[5/8] Running pair-level model ladders and grouped CV...")
    model_report={"connection":{},"strength":{},"measured_stp":{},"standardized":{}}
    model_report["connection"]["ladder"]=clustered_logit_ladder(mouse_tested,score)
    model_report["connection"]["group_cv"]=group_cv_connection(mouse_tested,score)

    for ep in ["psp_amplitude","psc_amplitude"]:
        d=mouse_conn[pd.to_numeric(mouse_conn[ep],errors="coerce").notna()].copy()
        model_report["strength"][ep]={
            "ladder":clustered_ols_ladder(d,ep,score,log_abs=True),
            "group_cv":group_cv_continuous(d,ep,score,log_abs=True)
        }

    for ep in STP_ENDPOINTS:
        if ep in mouse_dyn:
            model_report["measured_stp"][ep]={
                "ladder":clustered_ols_ladder(mouse_dyn,ep,score,log_abs=False),
                "group_cv":group_cv_continuous(mouse_dyn,ep,score,log_abs=False)
            }

    # Human extension: same externally trained mouse mapper, explicitly exploratory.
    human_tested=pairs[
        pairs["pre_species"].astype(str).str.lower().eq("human") &
        pairs[score].notna() & pairs[score+"_domain"].fillna(False) &
        pairs["tested_primary"]
    ].copy()
    human_conn=human_tested[human_tested["connected"]].copy()
    human_std=std[
        std["pre_species"].astype(str).str.lower().eq("human") &
        std[score+"_domain"].fillna(False)
    ].copy()
    model_report["human_extension"]={
        "counts":{"tested":int(len(human_tested)),"connected":int(len(human_conn)),
                  "standardized_models":int(len(human_std))},
        "warning":"Exploratory cross-species extension; Stage5A mapper was trained on mouse."
    }
    if len(human_tested)>=500:
        model_report["human_extension"]["connection"]={
            "ladder":clustered_logit_ladder(human_tested,score),
            "group_cv":group_cv_connection(human_tested,score)
        }
    for ep in ["psp_amplitude","psc_amplitude"]:
        hd=human_conn[pd.to_numeric(human_conn[ep],errors="coerce").notna()].copy()
        if len(hd)>=100:
            model_report["human_extension"][ep]={
                "ladder":clustered_ols_ladder(hd,ep,score,log_abs=True),
                "group_cv":group_cv_continuous(hd,ep,score,log_abs=True)
            }
    if len(human_std)>=100:
        model_report["human_extension"]["G_norm_50hz"]={
            "ladder":clustered_ols_ladder(human_std,"G_norm_50hz",score,False,True),
            "group_cv":group_cv_continuous(human_std,"G_norm_50hz",score,False,True)
        }

    std_eps=[]
    for f in FREQS:
        std_eps += [f"G_norm_{f}hz",f"tail_first_{f}hz",f"dyn_range_{f}hz",f"E_abs_{f}hz"]
    for ep in std_eps:
        if ep not in mouse_std: continue
        add_strength=ep.startswith(("G_norm_","tail_first_","dyn_range_"))
        model_report["standardized"][ep]={
            "ladder":clustered_ols_ladder(mouse_std,ep,score,log_abs=False,add_strength=add_strength),
            "group_cv":group_cv_continuous(mouse_std,ep,score,log_abs=False,add_strength=add_strength)
        }

    # FDR on standardized secondary endpoints, excluding predeclared primary from family correction.
    sec=[]
    for ep,r in model_report["standardized"].items():
        if ep=="G_norm_50hz" or r["ladder"].get("status")!="ok": continue
        m=r["ladder"]["M3_identity_C"]
        sec.append((ep,m.get("p_C_z_cluster",np.nan)))
    qs=bh_fdr([x[1] for x in sec])
    for (ep,pv),q in zip(sec,qs):
        model_report["standardized"][ep]["q_fdr_secondary"]=float(q)

    print("[6/8] Running explicit identity decomposition...")
    wb={}
    wb_cells={}
    for ident in ["pre_cell_class","pre_cre_type"]:
        res=within_between_decomposition(mouse_std,"G_norm_50hz",score,ident,add_strength=True)
        if isinstance(res,tuple):
            wb[ident]=res[0]; wb_cells[ident]=res[1]
            res[1].to_csv(analysis/f"within_between_cells_{ident}.csv",index=False)
        else:
            wb[ident]=res

    # Transfer uncertainty sensitivity: rerun primary G_norm coefficient over bootstrap C draws.
    print("[7/8] Propagating Stage5A mapper uncertainty and building figures...")
    boot_effects=[]
    bp=primary["boot_pred"]
    if bp is not None and len(mouse_std):
        # index in original cells for each pre_cell_id
        cell_pos=pd.Series(np.arange(len(cells)),index=cells["cell_id"]).to_dict()
        B=min(bp.shape[0],50 if not args.quick else 5)
        for b in range(B):
            tmp=mouse_std.copy()
            tmp["_Cb"]=tmp["pre_cell_id"].map(lambda cid: bp[b,cell_pos.get(cid,-1)] if cid in cell_pos else np.nan)
            rr=clustered_ols_ladder(tmp,"G_norm_50hz","_Cb",False,True)
            if rr.get("status")=="ok":
                mm=rr["M3_identity_C"]
                boot_effects.append(dict(draw=b,beta=mm["beta_C_z"],p=mm["p_C_z_cluster"],
                                         delta_r2=rr["delta_r2_C_after_identity"]))
    boot_effects=pd.DataFrame(boot_effects)
    boot_effects.to_csv(analysis/"transfer_mapper_uncertainty_effects.csv",index=False)

    # ---------- Figures ----------
    # F01 Stage5A reduced mapper CV
    for mode,r in [("primary_shared",primary),("protocol_conservative",conservative)]:
        sd=pd.DataFrame({"observed":r["stage5a_y"],"predicted":r["stage5a_oof"]})
        sd.to_csv(src/f"F01_{mode}_mapper_oof.csv",index=False)
        fig,ax=plt.subplots(figsize=(4.6,4.0))
        rng=np.random.default_rng(SEED)
        ax.scatter(sd["observed"]+rng.normal(0,.035,len(sd)),sd["predicted"],s=12,alpha=.45)
        g=sd.groupby("observed")["predicted"].agg(["mean","sem"]).reset_index()
        ax.errorbar(g["observed"],g["mean"],yerr=g["sem"],fmt="o",capsize=3)
        ax.set_xlabel("Stage5A minimum mechanism cost")
        ax.set_ylabel("Cross-validated transferred-complexity prediction")
        save_panel(fig,f"F01_{mode}_mapper_oof",figs,dpi)

    # F02 feature distribution QC
    q=primary["feature_qc"].copy()
    q.to_csv(src/"F02_transfer_feature_medians.csv",index=False)
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    y=np.arange(len(q))
    ax.scatter(q["stage5a_median"],y,label="Stage5A")
    ax.scatter(q["synphys_median"],y,label="SynPhys")
    ax.set_yticks(y,q["feature"])
    ax.set_xlabel("Canonical feature value (feature-specific units)")
    ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc="upper left")
    save_panel(fig,"F02_transfer_feature_medians",figs,dpi)

    # F03 complexity by broad cell class
    cplot=scores[scores[score+"_domain"].fillna(False) & scores["species"].astype(str).str.lower().eq("mouse")].copy()
    cplot.to_csv(src/"F03_complexity_by_cell_class.csv",index=False)
    cats=[x for x,n in cplot["cell_class"].fillna("MISSING").value_counts().items() if n>=30]
    fig,ax=plt.subplots(figsize=(6.0,4.2))
    data=[cplot.loc[cplot["cell_class"].fillna("MISSING").eq(c),score].dropna().to_numpy() for c in cats]
    if data:
        ax.boxplot(data,tick_labels=cats,showfliers=False)
        ax.tick_params(axis="x",rotation=30)
    ax.set_ylabel("Transferred intrinsic-complexity score")
    save_panel(fig,"F03_complexity_by_cell_class",figs,dpi)

    # F04 raw connection binned
    b=binned_xy(mouse_tested[score],mouse_tested["connected"].astype(float),8)
    b.to_csv(src/"F04_connection_probability_binned.csv",index=False)
    fig,ax=plt.subplots(figsize=(4.8,4.0))
    if len(b): ax.errorbar(b["x"],b["y"],yerr=b["se"],fmt="o-",capsize=3)
    ax.set_xlabel("Transferred intrinsic-complexity score")
    ax.set_ylabel("Observed connection probability")
    save_panel(fig,"F04_connection_probability_binned",figs,dpi)

    # F05 main coefficient forest
    coeff=[]
    # Continuous endpoints are standardized, so these coefficients are directly
    # comparable. Connection log-odds are kept separate in F04/report.
    for ep,label in [("psp_amplitude","PSP magnitude"),("psc_amplitude","PSC magnitude")]:
        rr=model_report["strength"].get(ep,{}).get("ladder",{})
        if rr.get("status")=="ok":
            m=rr["M3_identity_C"]; coeff.append(dict(endpoint=label,beta=m["beta_C_z"],se=m["se_C_z_cluster"],p=m["p_C_z_cluster"],family="strength"))
    rr=model_report["standardized"].get("G_norm_50hz",{}).get("ladder",{})
    if rr.get("status")=="ok":
        m=rr["M3_identity_C"]; coeff.append(dict(endpoint="Standardized 50-Hz temporal gain",beta=m["beta_C_z"],se=m["se_C_z_cluster"],p=m["p_C_z_cluster"],family="dynamic"))
    cf=pd.DataFrame(coeff)
    if len(cf):
        cf["lo"]=cf["beta"]-1.96*cf["se"]; cf["hi"]=cf["beta"]+1.96*cf["se"]
    cf.to_csv(src/"F05_primary_effect_forest.csv",index=False)
    fig,ax=plt.subplots(figsize=(6.0,4.2))
    if len(cf):
        yy=np.arange(len(cf))
        ax.errorbar(cf["beta"],yy,xerr=1.96*cf["se"],fmt="o",capsize=3)
        ax.axvline(0,ls="--",lw=1)
        ax.set_yticks(yy,cf["endpoint"])
    ax.set_xlabel("Standardized effect per 1 SD transferred complexity (95% CI)")
    save_panel(fig,"F05_primary_effect_forest",figs,dpi)

    # F06 standardized frequency coefficients
    rows=[]
    for f in FREQS:
        ep=f"G_norm_{f}hz"; rr=model_report["standardized"].get(ep,{}).get("ladder",{})
        if rr.get("status")=="ok":
            m=rr["M3_identity_C"]; rows.append(dict(freq=f,beta=m["beta_C_z"],se=m["se_C_z_cluster"],p=m["p_C_z_cluster"]))
    fr=pd.DataFrame(rows)
    if len(fr): fr["q_fdr"]=bh_fdr(fr["p"])
    fr.to_csv(src/"F06_dynamic_gain_by_frequency.csv",index=False)
    fig,ax=plt.subplots(figsize=(5.0,4.0))
    if len(fr): ax.errorbar(fr["freq"],fr["beta"],yerr=1.96*fr["se"],fmt="o-",capsize=3)
    ax.axhline(0,ls="--",lw=1); ax.set_xscale("log"); ax.set_xticks(FREQS,FREQS)
    ax.set_xlabel("Standardized train frequency (Hz)"); ax.set_ylabel("Complexity effect on normalized temporal gain")
    save_panel(fig,"F06_dynamic_gain_by_frequency",figs,dpi)

    # F07 Gnorm binned
    b=binned_xy(mouse_std[score],mouse_std["G_norm_50hz"],8)
    b.to_csv(src/"F07_Gnorm50_binned.csv",index=False)
    fig,ax=plt.subplots(figsize=(4.8,4.0))
    if len(b): ax.errorbar(b["x"],b["y"],yerr=b["se"],fmt="o-",capsize=3)
    ax.set_xlabel("Transferred intrinsic-complexity score"); ax.set_ylabel("Standardized 50-Hz temporal gain")
    save_panel(fig,"F07_Gnorm50_binned",figs,dpi)

    # F08 normalized trains by complexity quartile
    trrows=[]
    if len(mouse_std):
        qcat=pd.qcut(mouse_std[score],4,labels=["Q1","Q2","Q3","Q4"],duplicates="drop")
        for qv,g in mouse_std.assign(C_quartile=qcat).groupby("C_quartile",observed=True):
            vec=[]
            for s in g["amp_train_50hz"].dropna():
                a=np.array([float(x) for x in str(s).split(";")])
                if len(a)==N_SPIKES and abs(a[0])>1e-15: vec.append(np.abs(a/a[0]))
            if vec:
                M=np.vstack(vec)
                for k in range(N_SPIKES):
                    trrows.append(dict(quartile=str(qv),spike=k+1,mean=float(M[:,k].mean()),se=float(M[:,k].std(ddof=1)/np.sqrt(len(M))),n=len(M)))
    trdf=pd.DataFrame(trrows); trdf.to_csv(src/"F08_standardized_trains_by_C_quartile.csv",index=False)
    fig,ax=plt.subplots(figsize=(5.4,4.0))
    for qv,g in trdf.groupby("quartile") if len(trdf) else []:
        ax.plot(g["spike"],g["mean"],marker="o",label=qv)
    ax.set_xlabel("Spike number"); ax.set_ylabel("|Aₖ/A₁|")
    if len(trdf): ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc="upper left")
    save_panel(fig,"F08_standardized_trains_by_C_quartile",figs,dpi)

    # F09 before/after identity incremental R2
    rows=[]
    families=[("PSP",model_report["strength"].get("psp_amplitude",{}).get("ladder",{})),
              ("PSC",model_report["strength"].get("psc_amplitude",{}).get("ladder",{})),
              ("Gnorm50",model_report["standardized"].get("G_norm_50hz",{}).get("ladder",{}))]
    for name,r in families:
        if r.get("status")=="ok":
            rows += [dict(endpoint=name,context="Before identity",delta_r2=r["delta_r2_C_before_identity"]),
                     dict(endpoint=name,context="After identity",delta_r2=r["delta_r2_C_after_identity"])]
    rd=pd.DataFrame(rows); rd.to_csv(src/"F09_incremental_r2_before_after_identity.csv",index=False)
    fig,ax=plt.subplots(figsize=(5.6,4.0))
    if len(rd):
        x=np.arange(len(set(rd["endpoint"]))); names=list(dict.fromkeys(rd["endpoint"]))
        for j,ctx in enumerate(["Before identity","After identity"]):
            g=rd[rd["context"].eq(ctx)].set_index("endpoint").reindex(names)
            ax.bar(x+(j-.5)*.32,g["delta_r2"],width=.32,label=ctx)
        ax.set_xticks(x,names); ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc="upper left")
    ax.set_ylabel("Incremental in-sample R² from complexity")
    save_panel(fig,"F09_incremental_r2_before_after_identity",figs,dpi)

    # F10 within-between decomposition
    wbrows=[]
    for ident,r in wb.items():
        if isinstance(r,dict) and r.get("status")=="ok":
            for term in ["C_between_z","C_within_z"]:
                z=r[term]; wbrows.append(dict(identity=ident,term=term,beta=z["beta"],se=z["se_cluster"],p=z["p_cluster"]))
    wbd=pd.DataFrame(wbrows); wbd.to_csv(src/"F10_within_between_decomposition.csv",index=False)
    fig,ax=plt.subplots(figsize=(6.0,4.0))
    if len(wbd):
        labels=[f"{r.identity}: {r.term.replace('_z','')}" for r in wbd.itertuples()]
        yv=np.arange(len(wbd))
        ax.errorbar(wbd["beta"],yv,xerr=1.96*wbd["se"],fmt="o",capsize=3)
        ax.axvline(0,ls="--",lw=1); ax.set_yticks(yv,labels)
    ax.set_xlabel("Effect on standardized 50-Hz temporal gain (95% CI)")
    save_panel(fig,"F10_within_between_decomposition",figs,dpi)

    # F11 grouped CV incremental value
    rows=[]
    for name,r in families:
        gr=None
        if name=="PSP": gr=model_report["strength"].get("psp_amplitude",{}).get("group_cv",{})
        elif name=="PSC": gr=model_report["strength"].get("psc_amplitude",{}).get("group_cv",{})
        else: gr=model_report["standardized"].get("G_norm_50hz",{}).get("group_cv",{})
        if gr and gr.get("status")=="ok":
            rows.append(dict(endpoint=name,before=gr["delta_cv_r2_C_before_identity"],after=gr["delta_cv_r2_C_after_identity"]))
    cvd=pd.DataFrame(rows); cvd.to_csv(src/"F11_group_cv_increment.csv",index=False)
    fig,ax=plt.subplots(figsize=(5.3,4.0))
    if len(cvd):
        x=np.arange(len(cvd))
        ax.bar(x-.16,cvd["before"],.32,label="Before identity")
        ax.bar(x+.16,cvd["after"],.32,label="After identity")
        ax.set_xticks(x,cvd["endpoint"]); ax.legend(frameon=False,bbox_to_anchor=(1.02,1),loc="upper left")
    ax.axhline(0,ls="--",lw=1); ax.set_ylabel("Experiment-grouped CV ΔR²")
    save_panel(fig,"F11_group_cv_increment",figs,dpi)

    # F12 mapper uncertainty beta distribution
    boot_effects.to_csv(src/"F12_mapper_uncertainty_beta.csv",index=False)
    fig,ax=plt.subplots(figsize=(5.0,3.8))
    if len(boot_effects): ax.hist(boot_effects["beta"],bins=15)
    ax.axvline(0,ls="--",lw=1); ax.set_xlabel("Complexity effect across Stage5A mapper bootstrap draws"); ax.set_ylabel("Count")
    save_panel(fig,"F12_mapper_uncertainty_beta",figs,dpi)

    # F13 measured STP coefficient forest
    cc=[]
    for ep,r in model_report["measured_stp"].items():
        rr=r["ladder"]
        if rr.get("status")=="ok":
            m=rr["M3_identity_C"]; cc.append(dict(endpoint=ep,beta=m["beta_C_z"],se=m["se_C_z_cluster"],p=m["p_C_z_cluster"]))
    ccd=pd.DataFrame(cc)
    if len(ccd): ccd["q_fdr"]=bh_fdr(ccd["p"])
    ccd.to_csv(src/"F13_measured_stp_forest.csv",index=False)
    fig,ax=plt.subplots(figsize=(7.0,4.8))
    if len(ccd):
        yv=np.arange(len(ccd)); ax.errorbar(ccd["beta"],yv,xerr=1.96*ccd["se"],fmt="o",capsize=3)
        ax.axvline(0,ls="--",lw=1); ax.set_yticks(yv,ccd["endpoint"])
    ax.set_xlabel("Complexity effect (95% CI)")
    save_panel(fig,"F13_measured_stp_forest",figs,dpi)

    # F14 primary-vs-conservative transferred scores
    s1=primary["cell_scores"]; s2=conservative["cell_scores"]
    cmp=s1[["cell_id","C_primary_shared"]].merge(s2[["cell_id","C_protocol_conservative"]],on="cell_id").dropna()
    cmp.to_csv(src/"F14_transfer_definition_comparison.csv",index=False)
    fig,ax=plt.subplots(figsize=(4.4,4.0))
    if len(cmp): ax.scatter(cmp["C_primary_shared"],cmp["C_protocol_conservative"],s=8,alpha=.35)
    ax.set_xlabel("Primary shared-feature transfer"); ax.set_ylabel("Protocol-conservative transfer")
    save_panel(fig,"F14_transfer_definition_comparison",figs,dpi)

    # F15 cohort flow
    flow=pd.DataFrame([
        ("All cells",len(cells)),("All directed pairs",len(pairs)),
        ("Structurally tested pairs",int(pairs["tested_primary"].sum())),
        ("Mouse tested + C domain",len(mouse_tested)),
        ("Mouse connected + C domain",len(mouse_conn)),
        ("Mouse measured-dynamics + C domain",len(mouse_dyn)),
        ("Mouse modeled synapses + C domain",len(mouse_std))
    ],columns=["stage","n"])
    flow.to_csv(src/"F15_cohort_flow.csv",index=False)
    fig,ax=plt.subplots(figsize=(7.0,4.2))
    ax.bar(np.arange(len(flow)),flow["n"])
    ax.set_xticks(np.arange(len(flow)),flow["stage"],rotation=35,ha="right"); ax.set_ylabel("Observations")
    save_panel(fig,"F15_cohort_flow",figs,dpi)

    # F16 mouse vs human standardized dynamic effect (human exploratory)
    sp_rows=[]
    mr=model_report["standardized"].get("G_norm_50hz",{}).get("ladder",{})
    if mr.get("status")=="ok":
        m=mr["M3_identity_C"]; sp_rows.append(dict(species="mouse",beta=m["beta_C_z"],se=m["se_C_z_cluster"],p=m["p_C_z_cluster"],n=mr.get("n")))
    hr=model_report.get("human_extension",{}).get("G_norm_50hz",{}).get("ladder",{})
    if hr.get("status")=="ok":
        m=hr["M3_identity_C"]; sp_rows.append(dict(species="human exploratory",beta=m["beta_C_z"],se=m["se_C_z_cluster"],p=m["p_C_z_cluster"],n=hr.get("n")))
    spdf=pd.DataFrame(sp_rows); spdf.to_csv(src/"F16_species_dynamic_effect.csv",index=False)
    fig,ax=plt.subplots(figsize=(4.8,3.8))
    if len(spdf):
        yv=np.arange(len(spdf)); ax.errorbar(spdf["beta"],yv,xerr=1.96*spdf["se"],fmt="o",capsize=3)
        ax.axvline(0,ls="--",lw=1); ax.set_yticks(yv,spdf["species"])
    ax.set_xlabel("Standardized complexity effect on 50-Hz temporal gain")
    save_panel(fig,"F16_species_dynamic_effect",figs,dpi)

    # ---------- Reports ----------
    transfer_report={}
    for r in [primary,conservative]:
        transfer_report[r["mode"]]={
            "features":r["features"],"stage5a_cv":r["cv"],
            "n_synphys_complete":int(r["cell_scores"][f"C_{r['mode']}_complete"].sum()),
            "n_synphys_domain":int(r["cell_scores"][f"C_{r['mode']}_domain"].sum()),
            "domain_threshold":r["domain_threshold"]
        }

    # Adjudication
    primdyn=model_report["standardized"].get("G_norm_50hz",{}).get("ladder",{})
    cvdyn=model_report["standardized"].get("G_norm_50hz",{}).get("group_cv",{})
    if primdyn.get("status")=="ok":
        mm=primdyn["M3_identity_C"]
        within=wb.get("pre_cre_type",{}).get("C_within_z",{}) if isinstance(wb.get("pre_cre_type"),dict) else {}
        direct_positive=(mm.get("p_C_z_cluster",1)<.05 and primdyn.get("delta_r2_C_after_identity",0)>0 and cvdyn.get("delta_cv_r2_C_after_identity",0)>0)
        if direct_positive:
            verdict="POSITIVE_DIRECT_BRIDGE"
            manuscript="Externally defined intrinsic complexity predicts standardized local synaptic dynamic influence beyond anatomy and cell identity, with positive out-of-experiment predictive value."
        elif mm.get("p_C_z_cluster",1)<.05:
            verdict="IN_SAMPLE_ONLY"
            manuscript="A conditional association is detectable in-sample, but out-of-experiment predictive support is insufficient; treat as suggestive rather than a closed biological bridge."
        else:
            verdict="NO_INDEPENDENT_WITHIN_IDENTITY_BRIDGE"
            manuscript="The integrated SynPhys analysis does not support a reproducible independent complexity gradient in standardized synaptic dynamics after biological identity/context controls; this is a boundary condition, not evidence for identity mediation."
    else:
        verdict="PRIMARY_MODEL_FAILED"; manuscript="Primary model could not be fitted."

    legacy_audit = {
        "critical_fi_slope_bug": {
            "legacy_scale": 1e-12,
            "correct_recovery_scale": 1e12,
            "explanation": "SynPhys pipeline stores fi_fit_slope(Hz/pA) * 1e-12. Recover Hz/pA by *1e12.",
            "consequence": "All legacy C_transfer-dependent v2-v5 inferential results must be rerun."
        },
        "lost_shared_feature": {
            "feature": "ap_upstroke_downstroke_ratio",
            "explanation": "Present in the SynPhys DB and Stage5A feature set, but omitted from the legacy Phase1 Stage5A-compatible export, so legacy v2 incorrectly marked the SynPhys feature absent."
        },
        "identity_interpretation": {
            "problem": "Attenuation of an already weak/non-generalizing C effect after adding identity does not establish co-organization or mediation.",
            "replacement": "Report before/after identity models plus explicit within-between decomposition; reserve causal/organizational language for supported results."
        }
    }

    report={
        "status":"COMPLETE","created":time.strftime("%Y-%m-%d %H:%M:%S"),
        "db":str(db),"stage5a":str(stage5a),
        "counts":{"cells":len(cells),"pairs":len(pairs),"tested_pairs":int(pairs["tested_primary"].sum()),
                  "connected_pairs":int((pairs["tested_primary"]&pairs["connected"]).sum()),
                  "synapse_model_rows":len(smodels),"mouse_tested_primary":len(mouse_tested),
                  "mouse_connected_primary":len(mouse_conn),"mouse_measured_dynamics_primary":len(mouse_dyn),
                  "mouse_standardized_models_primary":len(mouse_std)},
        "transfer":transfer_report,
        "models":model_report,
        "within_between":wb,
        "mapper_uncertainty_draws":int(len(boot_effects)),
        "legacy_audit":legacy_audit,
        "claim_adjudication":{"verdict":verdict,"manuscript_safe_wording":manuscript},
        "elapsed_seconds":time.time()-t0,
        "guardrails":[
            "Stage5A complexity mapper is trained without SynPhys outcomes.",
            "Primary inference is mouse directed-pair level.",
            "PSP and PSC physical channels are modeled separately.",
            "Synapse-model trains use identical standardized presynaptic inputs.",
            "G_norm is a standardized synaptic temporal phenotype, not whole-network causal leverage.",
            "Human transfer, if later analyzed, is extension only.",
            "Between-identity effects have few effective identity groups and are descriptive unless independently replicated."
        ]
    }
    jdump(report,analysis/"stage5c_integrated_report.json")
    jdump(legacy_audit,analysis/"legacy_analysis_audit.json")

    # Model summary table.
    rows=[]
    def addrow(fam,ep,r):
        if r.get("status")!="ok": return
        m=r.get("M3_identity_C",{})
        rows.append(dict(family=fam,endpoint=ep,n=r.get("n"),n_experiments=r.get("n_experiments"),
                         beta=m.get("beta_C_z",m.get("beta_C_z_logodds")),
                         se=m.get("se_C_z_cluster"),p=m.get("p_C_z_cluster"),
                         delta_r2_before=r.get("delta_r2_C_before_identity"),
                         delta_r2_after=r.get("delta_r2_C_after_identity")))
    addrow("structural","connection",model_report["connection"]["ladder"])
    for ep,z in model_report["strength"].items(): addrow("strength",ep,z["ladder"])
    for ep,z in model_report["measured_stp"].items(): addrow("measured_stp",ep,z["ladder"])
    for ep,z in model_report["standardized"].items(): addrow("standardized",ep,z["ladder"])
    ms=pd.DataFrame(rows)
    if len(ms): ms["q_fdr_global_descriptive"]=bh_fdr(ms["p"])
    ms.to_csv(analysis/"model_summary.csv",index=False)

    # Human-readable report.
    md=[
        "# Stage 5C integrated Synaptic Physiology analysis","",
        "## Central adjudication","",
        f"**{verdict}**","",manuscript,"",
        "## Critical audit of the legacy analysis","",
        "1. **fi-slope unit conversion was wrong in legacy transfer v2.** The legacy code used `SynPhys fi_slope × 1e-12`; the correct recovery to Hz/pA is `× 1e12`. All legacy v2–v5 C-transfer-dependent results are therefore superseded by this integrated rerun.",
        "2. **A shared spike-shape feature was accidentally dropped.** `ap_upstroke_downstroke_ratio` exists in SynPhys and its Stage5A counterpart exists, but the intermediate Phase1 export omitted it.",
        "3. **Identity attenuation was overinterpreted.** A reduction of an already weak, out-of-sample-negative C increment after adding identity does not demonstrate biological co-organization or mediation. This script uses explicit within/between decomposition.",
        "",
        "## Transfer validity","",
        f"- Primary shared features: `{', '.join(primary['features'])}`",
        f"- Stage5A nested-CV rho: **{primary['cv']['rho']:.3f}**; MAE: **{primary['cv']['mae']:.3f}**; R²: **{primary['cv']['r2']:.3f}**",
        f"- Mouse/SynPhys domain cells (all species in score table): **{int(primary['cell_scores']['C_primary_shared_domain'].sum()):,}**",
        f"- Protocol-conservative Stage5A nested-CV rho: **{conservative['cv']['rho']:.3f}**",
        "",
        "## Primary cohorts","",
        f"- all cells: **{len(cells):,}**",
        f"- all directed pairs: **{len(pairs):,}**",
        f"- mouse tested pairs in transferred-complexity domain: **{len(mouse_tested):,}**",
        f"- mouse connected pairs: **{len(mouse_conn):,}**",
        f"- mouse measured-dynamics pairs: **{len(mouse_dyn):,}**",
        f"- mouse standardized stochastic-synapse models: **{len(mouse_std):,}**",
        "",
        "## Interpretation hierarchy","",
        "- Structural connection, conditional PSP/PSC strength, measured STP, and standardized train dynamics are reported as distinct layers.",
        "- The paper-level complexity–leverage bridge requires a reproducible conditional C effect **and** positive experiment-grouped out-of-sample increment.",
        "- If only raw/marginal associations are present but conditional/grouped-CV effects fail, the supported conclusion is a biological boundary condition rather than a direct bridge.",
        "",
        "## Read first","",
        "- `analysis/model_summary.csv`",
        "- `analysis/legacy_analysis_audit.json`",
        "- `analysis/stage5c_integrated_report.json`",
        "- `figures/F05_primary_effect_forest.png`",
        "- `figures/F10_within_between_decomposition.png`",
        "- `figures/F11_group_cv_increment.png`",
    ]
    (analysis/"00_STAGE5C_INTEGRATED_SUMMARY.md").write_text("\n".join(md)+"\n",encoding="utf-8")

    # Figure catalog
    frecs=[]
    for p in sorted(figs.glob("*.png")):
        frecs.append(dict(figure=p.stem,png=str(p),pdf=str(p.with_suffix(".pdf")),
                          source_csv=str(src/(p.stem+".csv")) if (src/(p.stem+".csv")).exists() else "see source_data/"))
    pd.DataFrame(frecs).to_csv(analysis/"figure_catalog.csv",index=False)

    print()
    print("="*100)
    print("COMPLETE")
    print("Verdict:",verdict)
    print("Output :",out)
    print("Read first:")
    print(analysis/"00_STAGE5C_INTEGRATED_SUMMARY.md")
    print(analysis/"model_summary.csv")
    print(analysis/"legacy_analysis_audit.json")
    print("="*100)

if __name__=="__main__":
    main()
