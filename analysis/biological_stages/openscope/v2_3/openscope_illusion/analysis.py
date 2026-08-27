#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenScope Illusion 000248 — final scientific closure analysis v2.3.

Why v2.3 exists
---------------
The v2.2 quick run established two method-QC facts before the formal run:

1. The original zero-to-training-mean single-unit ablation ranking was too
   unstable across independent trial splits to serve as the primary leverage
   estimator.
2. The original focal-area -> other-area continuous Ridge bridge had negative
   held-out R^2 and therefore did not have sufficient predictive validity for a
   headline leverage claim.

v2.3 freezes a reliability-normalized, decoder-based closure before the formal
12-mouse analysis.  It preserves the same biological question but replaces the
unstable estimator with regularized standardized decoder coefficients and adds
explicit within-state-versus-cross-state reproducibility controls.

Primary v2.3 endpoints
----------------------
A. State-specific full-decoder transfer
   Train an IC-vs-LC decoder in source state s on discovery trials and test it
   on untouched final trials in target state t.

       R_decoder = 0.5[(AUC_00 + AUC_11) - (AUC_01 + AUC_10)]

B. Coefficient-selected unit-set transfer
   Rank units by |standardized L2-logistic coefficient| in each discovery-state
   decoder.  Select the source-state top 20% set (5/10/20/40% sensitivity), then
   re-fit/test a decoder only on that selected set inside each target-state
   final-trial cohort.

       R_topset = 0.5[(AUC_00 + AUC_11) - (AUC_01 + AUC_10)]

C. Sampling-normalized coefficient-landscape reconfiguration
   Split each state's discovery trials into two disjoint, exact-label-balanced
   halves A/B.  Fit four independent decoders: 0A, 0B, 1A, 1B.

       rho_within = mean[rho(0A,0B), rho(1A,1B)]
       rho_cross  = mean[all four cross-state pairings]
       R_rho      = rho_within - rho_cross

   The same calculation is repeated with top-20% Jaccard overlap.  This asks
   whether between-state change exceeds ordinary estimator/sampling variability.

Primary state definition
------------------------
The state is derived from prestimulus (-300..0 ms) population activity only.
Before PCA, each neuron's baseline activity is residualized with a model fit on
*discovery trials only*:

    1 + t + t^2 + t^3 + log1p(pre-running-speed)

where time and running are scaled from discovery trials.  The fixed regression
is applied to all trials, then units are discovery-z-scored, trial-wise global
population mean is removed, and PC1 is fit on discovery trials.  PCA sign is
oriented deterministically by its largest-magnitude loading; the discovery
median is the state threshold.

Secondary same-input endpoint
-----------------------------
For IC1 and IC2 separately, decode the prestimulus state from the poststimulus
population response after removing each trial's global population response.
This is not a leverage estimator, but it directly tests whether the same exact
external image evokes state-dependent population patterns.

All leverage quantities are predictive/functional, not causal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED = 20260821
MAIN_LABELS = ["IC1", "IC2", "LC1", "LC2"]
REAL_EDGE_LABELS = ["IRE1", "IRE2", "TRE1", "TRE2"]
TARGET_AREAS = ["V1", "LM", "RL", "AL", "PM", "AM"]
PRIMARY_C = 1.0
C_SENSITIVITY = (0.25, 1.0, 4.0)
TOP_FRACTIONS = (0.05, 0.10, 0.20, 0.40)
PRIMARY_TOP_FRACTION = 0.20


def _finite_spearman(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    aa = a[ok]
    bb = b[ok]
    if np.nanstd(aa) < 1e-12 or np.nanstd(bb) < 1e-12:
        return np.nan
    return float(stats.spearmanr(aa, bb).statistic)


def _jaccard_top(a, b, frac=0.20) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    idx = np.flatnonzero(ok)
    if len(idx) < 5:
        return np.nan
    k = max(1, int(round(frac * len(idx))))
    aa = idx[np.argsort(-a[idx])[:k]]
    bb = idx[np.argsort(-b[idx])[:k]]
    A, B = set(map(int, aa)), set(map(int, bb))
    return float(len(A & B) / max(1, len(A | B)))


def stratified_discovery_split(labels, frac=0.60, seed=SEED):
    """Exact-label-stratified discovery/final split."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, str)
    disc, final = [], []
    for lab in pd.unique(labels):
        idx = np.flatnonzero(labels == lab)
        rng.shuffle(idx)
        if len(idx) < 2:
            continue
        n = int(round(frac * len(idx)))
        n = min(max(n, 1), len(idx) - 1)
        disc.extend(idx[:n])
        final.extend(idx[n:])
    return np.asarray(sorted(disc), int), np.asarray(sorted(final), int)


def balance_exact_indices(indices, exact_labels, required_labels, seed):
    """Return equal counts of every required exact stimulus label."""
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices, int)
    exact_labels = np.asarray(exact_labels, str)
    groups = []
    for lab in required_labels:
        g = indices[exact_labels[indices] == lab]
        if len(g) == 0:
            return np.array([], int)
        groups.append(g.copy())
    n = min(len(g) for g in groups)
    if n < 2:
        return np.array([], int)
    out = np.concatenate([rng.choice(g, n, replace=False) for g in groups])
    rng.shuffle(out)
    return out.astype(int)


def disjoint_balanced_halves(indices, exact_labels, required_labels, seed):
    """
    Build two disjoint exact-label-balanced halves from `indices`.

    Every half contains the same number of IC1/IC2/LC1/LC2 (or control) trials.
    """
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices, int)
    exact_labels = np.asarray(exact_labels, str)
    by_lab = []
    for lab in required_labels:
        g = indices[exact_labels[indices] == lab].copy()
        rng.shuffle(g)
        by_lab.append(g)
    n_half = min(len(g) // 2 for g in by_lab) if by_lab else 0
    if n_half < 3:
        return np.array([], int), np.array([], int)
    A = np.concatenate([g[:n_half] for g in by_lab])
    B = np.concatenate([g[n_half:2*n_half] for g in by_lab])
    rng.shuffle(A)
    rng.shuffle(B)
    return A.astype(int), B.astype(int)


def _scale_from_discovery(x, discovery_idx):
    x = np.asarray(x, float)
    d = np.asarray(discovery_idx, int)
    finite = np.isfinite(x[d])
    if finite.sum() == 0:
        return np.zeros_like(x), 0.0, 1.0, False
    med = float(np.nanmedian(x[d][finite]))
    xx = np.where(np.isfinite(x), x, med)
    mu = float(np.mean(xx[d]))
    sd = float(np.std(xx[d]))
    if sd < 1e-12:
        return np.zeros_like(xx), mu, 1.0, False
    return (xx - mu) / sd, mu, sd, True


def _confound_design(time_s, running, discovery_idx):
    """
    Discovery-scaled cubic time basis + log1p running.
    """
    t, tmu, tsd, tok = _scale_from_discovery(time_s, discovery_idx)
    cols = [np.ones(len(t)), t, t*t, t*t*t]
    names = ["intercept", "time_z", "time_z2", "time_z3"]
    run_meta = {"running_included": False}
    if running is not None:
        r = np.asarray(running, float)
        # Running speed should be nonnegative, but guard against tiny negatives.
        r = np.log1p(np.clip(r, 0, None))
        rz, rmu, rsd, rok = _scale_from_discovery(r, discovery_idx)
        if rok:
            cols.append(rz)
            names.append("log1p_running_z")
            run_meta = {
                "running_included": True,
                "running_log_mean_discovery": rmu,
                "running_log_sd_discovery": rsd,
            }
    X = np.column_stack(cols)
    return X, {
        "columns": names,
        "time_mean_discovery": tmu,
        "time_sd_discovery": tsd,
        **run_meta,
    }


def fit_prestim_state(
    B,
    discovery_local,
    time_s,
    running=None,
    *,
    mode="confound_controlled",
):
    """
    Fit a binary prestimulus population state from baseline spike counts.

    Parameters are learned only on discovery trials and then frozen.
    """
    B = np.asarray(B, float)
    d = np.asarray(discovery_local, int)
    if len(d) < 20:
        raise RuntimeError("Too few discovery trials for state fitting")

    if mode not in {"confound_controlled", "raw"}:
        raise ValueError(mode)

    Ball = B.copy()
    conf_meta = {"mode": mode, "confound_residualized": False}
    if mode == "confound_controlled":
        X, meta = _confound_design(time_s, running, d)
        beta = np.linalg.lstsq(X[d], Ball[d], rcond=None)[0]
        Ball = Ball - X @ beta
        conf_meta = {
            "mode": mode,
            "confound_residualized": True,
            **meta,
        }

    # Discovery-fit per-neuron scaling.
    mu = np.nanmean(Ball[d], axis=0)
    sd = np.nanstd(Ball[d], axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    all_filled = np.where(np.isfinite(Ball), Ball, mu)
    Z = (all_filled - mu) / sd

    # Remove trial-wise global population gain.
    Z = Z - Z.mean(axis=1, keepdims=True)
    good = np.nanstd(Z[d], axis=0) > 1e-8
    if good.sum() < 5:
        raise RuntimeError("Too few variable units for state PCA")

    pca = PCA(n_components=1, random_state=SEED)
    pca.fit(Z[d][:, good])
    score = pca.transform(Z[:, good]).ravel()

    # Deterministic PCA orientation: largest-magnitude loading is positive.
    loading = pca.components_[0].copy()
    anchor = int(np.argmax(np.abs(loading)))
    if loading[anchor] < 0:
        score = -score
        loading = -loading

    threshold = float(np.median(score[d]))
    state = (score >= threshold).astype(int)
    return {
        "score": score,
        "state": state,
        "threshold": threshold,
        "pc1_explained": float(pca.explained_variance_ratio_[0]),
        "n_pca_units": int(good.sum()),
        "loading_anchor_local_index": anchor,
        **conf_meta,
    }


def _cramers_v(state, labels):
    tab = pd.crosstab(pd.Series(state), pd.Series(labels))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan, np.nan
    chi, p, _, _ = stats.chi2_contingency(tab)
    n = tab.to_numpy().sum()
    k = min(tab.shape) - 1
    v = np.sqrt(chi / (n*k)) if n > 0 and k > 0 else np.nan
    return float(v), float(p)


def state_audit(trials, idx, state, score):
    t = trials.reset_index(drop=True)
    idx = np.asarray(idx, int)
    exact = t.loc[idx, "canonical_stimulus"].astype(str).to_numpy()
    st = np.asarray(state, int)[idx]
    sc = np.asarray(score, float)[idx]
    v, p = _cramers_v(st, exact)
    time = pd.to_numeric(t.loc[idx, "start_time"], errors="coerce").to_numpy(float)
    rho_time = _finite_spearman(sc, time)
    out = {
        "state_stimulus_cramers_v": v,
        "state_stimulus_chi2_p": p,
        "state_score_time_spearman": rho_time,
        "state0_fraction": float(np.mean(st == 0)),
        "state1_fraction": float(np.mean(st == 1)),
    }
    if "pre_running_speed" in t.columns:
        run = pd.to_numeric(t.loc[idx, "pre_running_speed"], errors="coerce").to_numpy(float)
        ok = np.isfinite(run)
        if ok.sum() >= 20 and len(np.unique(st[ok])) == 2:
            a = run[ok & (st == 0)]
            b = run[ok & (st == 1)]
            pooled = np.sqrt(0.5*(np.var(a, ddof=1) + np.var(b, ddof=1))) if len(a)>1 and len(b)>1 else np.nan
            d = (np.mean(b)-np.mean(a))/pooled if np.isfinite(pooled) and pooled>1e-12 else np.nan
            try:
                mwu = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            except Exception:
                mwu = np.nan
            out.update({
                "running_state0_mean": float(np.mean(a)),
                "running_state1_mean": float(np.mean(b)),
                "running_state_difference": float(np.mean(b)-np.mean(a)),
                "running_state_cohens_d": float(d) if np.isfinite(d) else np.nan,
                "running_state_mwu_p": mwu,
            })
    return out


@dataclass
class LinearDecoder:
    scaler: StandardScaler
    clf: LogisticRegression
    coef_signed: np.ndarray
    coef_abs: np.ndarray


def fit_decoder(X, y, *, C=PRIMARY_C, seed=SEED):
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    if X.ndim != 2 or X.shape[1] < 2 or len(np.unique(y)) != 2:
        return None
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)
    clf = LogisticRegression(
        C=float(C), solver="liblinear", max_iter=4000,
        random_state=seed,
    )
    clf.fit(Xt, y)
    w = clf.coef_.ravel().astype(float)
    return LinearDecoder(scaler, clf, w, np.abs(w))


def evaluate_decoder(model, X, y):
    if model is None:
        return {"auc": np.nan, "balanced_accuracy": np.nan, "n": 0}
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    if len(y) < 4 or len(np.unique(y)) != 2:
        return {"auc": np.nan, "balanced_accuracy": np.nan, "n": int(len(y))}
    Xz = model.scaler.transform(X)
    dec = model.clf.decision_function(Xz)
    pred = (dec >= 0).astype(int)
    return {
        "auc": float(roc_auc_score(y, dec)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n": int(len(y)),
    }


def selected_decoder_cv(X, y, selected, *, seed=SEED):
    """
    Target-state held-out utility of a source-selected feature set.

    We re-fit the target-state decoder within cross-validation, keeping only the
    selected units.  Thus the metric tests feature-set utility rather than
    source-state weight calibration.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    selected = np.asarray(selected, int)
    if len(selected) < 2 or len(y) < 12 or len(np.unique(y)) != 2:
        return {"auc": np.nan, "balanced_accuracy": np.nan, "n": int(len(y))}
    nmin = int(np.min(np.bincount(y)))
    ns = min(5, nmin)
    if ns < 2:
        return {"auc": np.nan, "balanced_accuracy": np.nan, "n": int(len(y))}
    skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed)
    aucs, bals = [], []
    for tr, te in skf.split(X, y):
        m = fit_decoder(X[tr][:, selected], y[tr], C=PRIMARY_C, seed=seed+len(aucs))
        ev = evaluate_decoder(m, X[te][:, selected], y[te])
        aucs.append(ev["auc"])
        bals.append(ev["balanced_accuracy"])
    return {
        "auc": float(np.nanmean(aucs)),
        "balanced_accuracy": float(np.nanmean(bals)),
        "n": int(len(y)),
    }


def _residualize_vector_by_unit_covariates(vector, units, covariates, include_selectivity=False):
    """Residualize a unit-level coefficient vector against static/activity covariates."""
    vector = np.asarray(vector, float)
    d = units[["unit_uid", "area"]].copy().merge(covariates, on="unit_uid", how="left")
    num = ["mean_rate", "baseline_rate"]
    if include_selectivity:
        num.append("ic_lc_selectivity")
    Xn = d[num].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Xa = pd.get_dummies(d["area"].fillna("NA"), prefix="area", drop_first=True, dtype=float)
    X = pd.concat([Xn, Xa], axis=1).to_numpy(float)
    X = np.column_stack([np.ones(len(X)), X])
    ok = np.isfinite(vector)
    if ok.sum() < X.shape[1] + 5:
        return np.full_like(vector, np.nan)
    beta = np.linalg.lstsq(X[ok], vector[ok], rcond=None)[0]
    return vector - X @ beta


def _landscape_similarity(vectors: Dict[str, np.ndarray], top_frac=PRIMARY_TOP_FRACTION):
    """Within-state vs cross-state coefficient-landscape similarity."""
    v00 = vectors["0A"]
    v01 = vectors["0B"]
    v10 = vectors["1A"]
    v11 = vectors["1B"]
    within_rhos = [_finite_spearman(v00, v01), _finite_spearman(v10, v11)]
    cross_rhos = [
        _finite_spearman(v00, v10), _finite_spearman(v00, v11),
        _finite_spearman(v01, v10), _finite_spearman(v01, v11),
    ]
    within_j = [_jaccard_top(v00, v01, top_frac), _jaccard_top(v10, v11, top_frac)]
    cross_j = [
        _jaccard_top(v00, v10, top_frac), _jaccard_top(v00, v11, top_frac),
        _jaccard_top(v01, v10, top_frac), _jaccard_top(v01, v11, top_frac),
    ]
    rho_w = float(np.nanmean(within_rhos))
    rho_c = float(np.nanmean(cross_rhos))
    j_w = float(np.nanmean(within_j))
    j_c = float(np.nanmean(cross_j))
    return {
        "rho_within": rho_w,
        "rho_cross": rho_c,
        "delta_rho": rho_w-rho_c,
        "jaccard_within": j_w,
        "jaccard_cross": j_c,
        "delta_jaccard": j_w-j_c,
        "rho_0A_0B": within_rhos[0],
        "rho_1A_1B": within_rhos[1],
        "jaccard_0A_0B": within_j[0],
        "jaccard_1A_1B": within_j[1],
    }


def activity_covariates(trials, units, mats, block="ICwcfg1", indices=None):
    t = trials.reset_index(drop=True)
    allowed = np.ones(len(t), dtype=bool)
    if indices is not None:
        allowed[:] = False
        allowed[np.asarray(indices, int)] = True
    blockmask = t["stimulus_config"].eq(block).to_numpy() & allowed
    ic = (t["stimulus_config"].eq(block) & t["stimulus_group"].eq("IC")).to_numpy() & allowed
    lc = (t["stimulus_config"].eq(block) & t["stimulus_group"].eq("LC")).to_numpy() & allowed
    full = np.asarray(mats["full"], float)
    base = np.asarray(mats["baseline"], float)
    rows = []
    for j, u in units.reset_index(drop=True).iterrows():
        mi = float(np.mean(full[ic, j])/.40) if ic.any() else np.nan
        ml = float(np.mean(full[lc, j])/.40) if lc.any() else np.nan
        rows.append({
            "unit_uid": u["unit_uid"],
            "mean_rate": float(np.mean(full[blockmask, j])/.40),
            "baseline_rate": float(np.mean(base[blockmask, j])/.30),
            "ic_lc_selectivity": float((mi-ml)/(abs(mi)+abs(ml)+1e-6)),
        })
    return pd.DataFrame(rows)


def _build_analysis_subset(trials, block, positive_labels, negative_labels):
    t = trials.reset_index(drop=True)
    req = list(positive_labels)+list(negative_labels)
    mask = t["stimulus_config"].eq(block) & t["canonical_stimulus"].isin(req)
    idx = np.flatnonzero(mask.to_numpy())
    exact = t["canonical_stimulus"].astype(str).to_numpy()
    y = np.isin(exact, list(positive_labels)).astype(int)
    return t, req, idx, exact, y


def _train_source_decoders(full, y, exact, discovery, states, req, *, C, seed):
    models = {}
    balanced_idx = {}
    for s in [0, 1]:
        d = discovery[states[discovery] == s]
        d = balance_exact_indices(d, exact, req, seed+100+s)
        if len(d) < max(20, 4*len(req)):
            return None, None
        m = fit_decoder(full[d], y[d], C=C, seed=seed+200+s)
        if m is None:
            return None, None
        models[s] = m
        balanced_idx[s] = d
    return models, balanced_idx


def decoder_transfer(full, y, exact, discovery, final, states, req, *, C=PRIMARY_C, seed=SEED):
    models, source_idx = _train_source_decoders(full, y, exact, discovery, states, req, C=C, seed=seed)
    if models is None:
        return None
    rows = []
    for target in [0, 1]:
        f = final[states[final] == target]
        f = balance_exact_indices(f, exact, req, seed+1000+target)
        for source in [0, 1]:
            ev = evaluate_decoder(models[source], full[f], y[f])
            rows.append({
                "source_state": source,
                "target_state": target,
                "auc": ev["auc"],
                "balanced_accuracy": ev["balanced_accuracy"],
                "n_final_trials": ev["n"],
                "C": float(C),
            })
    tab = pd.DataFrame(rows)
    Mauc = {(int(r.source_state), int(r.target_state)): r.auc for r in tab.itertuples()}
    Mbal = {(int(r.source_state), int(r.target_state)): r.balanced_accuracy for r in tab.itertuples()}
    rauc = .5*((Mauc[(0,0)]+Mauc[(1,1)])-(Mauc[(0,1)]+Mauc[(1,0)]))
    rbal = .5*((Mbal[(0,0)]+Mbal[(1,1)])-(Mbal[(0,1)]+Mbal[(1,0)]))
    home_auc = .5*(Mauc[(0,0)]+Mauc[(1,1)])
    cross_auc = .5*(Mauc[(0,1)]+Mauc[(1,0)])
    return {
        "table": tab,
        "models": models,
        "source_indices": source_idx,
        "summary": {
            "decoder_auc_home": float(home_auc),
            "decoder_auc_cross": float(cross_auc),
            "decoder_auc_crossover": float(rauc),
            "decoder_balacc_crossover": float(rbal),
        },
    }


def coefficient_selected_transfer(
    full, y, exact, final, states, req, source_models,
    *, top_fracs=TOP_FRACTIONS, seed=SEED,
):
    rows = []
    nunit = full.shape[1]
    for frac in top_fracs:
        k = max(5, min(nunit, int(round(frac*nunit))))
        selected = {
            s: np.argsort(-source_models[s].coef_abs)[:k]
            for s in [0,1]
        }
        for target in [0,1]:
            f = final[states[final] == target]
            f = balance_exact_indices(f, exact, req, seed+2000+target+int(frac*1000))
            for source in [0,1]:
                ev = selected_decoder_cv(
                    full[f], y[f], selected[source],
                    seed=seed+3000+target+int(frac*1000)
                )
                rows.append({
                    "source_state": source,
                    "target_state": target,
                    "top_fraction": float(frac),
                    "k": int(k),
                    "auc": ev["auc"],
                    "balanced_accuracy": ev["balanced_accuracy"],
                    "n_final_trials": ev["n"],
                })
    tab = pd.DataFrame(rows)
    summaries=[]
    for frac,g in tab.groupby("top_fraction"):
        Ma={(int(r.source_state),int(r.target_state)):r.auc for r in g.itertuples()}
        Mb={(int(r.source_state),int(r.target_state)):r.balanced_accuracy for r in g.itertuples()}
        summaries.append({
            "top_fraction":float(frac),
            "topset_auc_home":float(.5*(Ma[(0,0)]+Ma[(1,1)])),
            "topset_auc_cross":float(.5*(Ma[(0,1)]+Ma[(1,0)])),
            "topset_auc_crossover":float(.5*((Ma[(0,0)]+Ma[(1,1)])-(Ma[(0,1)]+Ma[(1,0)]))),
            "topset_balacc_crossover":float(.5*((Mb[(0,0)]+Mb[(1,1)])-(Mb[(0,1)]+Mb[(1,0)]))),
        })
    return tab, pd.DataFrame(summaries)


def coefficient_landscape(
    full, y, exact, discovery, states, req, units, covariates,
    *, C=PRIMARY_C, seed=SEED, top_frac=PRIMARY_TOP_FRACTION,
):
    """Fit 0A/0B/1A/1B independent coefficient landscapes."""
    models={}
    half_indices={}
    rows=[]
    for s in [0,1]:
        d = discovery[states[discovery] == s]
        A,B = disjoint_balanced_halves(d, exact, req, seed+500+s)
        if len(A)<12 or len(B)<12:
            return None
        for label,idx in [(f"{s}A",A),(f"{s}B",B)]:
            label_seed={"0A":11,"0B":12,"1A":21,"1B":22}[label]
            m = fit_decoder(full[idx], y[idx], C=C, seed=seed+600+label_seed)
            if m is None:
                return None
            models[label]=m
            half_indices[label]=idx
            for j,u in units.reset_index(drop=True).iterrows():
                rows.append({
                    "unit_uid":u["unit_uid"],
                    "area":u.get("area",None),
                    "half_model":label,
                    "state":int(label[0]),
                    "half":label[1],
                    "coef_signed":float(m.coef_signed[j]),
                    "coef_abs":float(m.coef_abs[j]),
                    "C":float(C),
                })

    vectors={k:m.coef_abs for k,m in models.items()}
    raw=_landscape_similarity(vectors, top_frac=top_frac)

    activity_vectors={
        k:_residualize_vector_by_unit_covariates(v,units,covariates,include_selectivity=False)
        for k,v in vectors.items()
    }
    activity=_landscape_similarity(activity_vectors, top_frac=top_frac)

    full_resid_vectors={
        k:_residualize_vector_by_unit_covariates(v,units,covariates,include_selectivity=True)
        for k,v in vectors.items()
    }
    fullres=_landscape_similarity(full_resid_vectors, top_frac=top_frac)

    # Cross-half predictive validity: A -> B and B -> A within each state.
    home_aucs=[]
    for s in [0,1]:
        A=half_indices[f"{s}A"]
        B=half_indices[f"{s}B"]
        home_aucs.append(evaluate_decoder(models[f"{s}A"],full[B],y[B])["auc"])
        home_aucs.append(evaluate_decoder(models[f"{s}B"],full[A],y[A])["auc"])

    summary={
        "C":float(C),
        **raw,
        "activity_residual_delta_rho":activity["delta_rho"],
        "activity_residual_rho_within":activity["rho_within"],
        "activity_residual_rho_cross":activity["rho_cross"],
        "activity_selectivity_residual_delta_rho":fullres["delta_rho"],
        "half_model_home_auc_mean":float(np.nanmean(home_aucs)),
        "half_model_home_auc_min":float(np.nanmin(home_aucs)),
        "n_half_trials_0A":int(len(half_indices["0A"])),
        "n_half_trials_0B":int(len(half_indices["0B"])),
        "n_half_trials_1A":int(len(half_indices["1A"])),
        "n_half_trials_1B":int(len(half_indices["1B"])),
    }
    return {
        "summary":summary,
        "unit_coefficients":pd.DataFrame(rows),
        "models":models,
        "half_indices":half_indices,
    }


def exact_image_state_imprint(trials, mats, discovery, final, states, *, seed=SEED):
    """
    Decode confound-controlled prestim state from poststim population pattern for
    IC1 and IC2 separately, after subtracting each trial's global response.
    """
    t=trials.reset_index(drop=True)
    exact=t["canonical_stimulus"].astype(str).to_numpy()
    X=np.asarray(mats["full"],float)
    X=X-X.mean(axis=1,keepdims=True)
    rows=[]
    for i,stim in enumerate(["IC1","IC2"]):
        d=discovery[exact[discovery]==stim]
        f=final[exact[final]==stim]
        # Balance state labels within each exact image.
        rng=np.random.default_rng(seed+8000+i)
        def bal_state(idx):
            a=idx[states[idx]==0]; b=idx[states[idx]==1]
            n=min(len(a),len(b))
            if n<8: return np.array([],int)
            out=np.r_[rng.choice(a,n,replace=False),rng.choice(b,n,replace=False)]
            rng.shuffle(out); return out.astype(int)
        d=bal_state(d); f=bal_state(f)
        if len(d)<20 or len(f)<12:
            continue
        m=fit_decoder(X[d],states[d],C=PRIMARY_C,seed=seed+8100+i)
        ev=evaluate_decoder(m,X[f],states[f])
        rows.append({
            "exact_stimulus":stim,
            "auc":ev["auc"],
            "balanced_accuracy":ev["balanced_accuracy"],
            "auc_excess_over_chance":ev["auc"]-.5 if np.isfinite(ev["auc"]) else np.nan,
            "n_discovery":int(len(d)),
            "n_final":int(len(f)),
        })
    tab=pd.DataFrame(rows)
    if len(tab):
        summary={
            "same_image_state_auc_mean":float(tab["auc"].mean()),
            "same_image_state_auc_excess_mean":float(tab["auc_excess_over_chance"].mean()),
        }
    else:
        summary={"same_image_state_auc_mean":np.nan,"same_image_state_auc_excess_mean":np.nan}
    return tab,summary


def formal_context_repeat(
    trials, units, mats, covariates,
    *,
    block="ICwcfg1",
    positive_labels=("IC1","IC2"),
    negative_labels=("LC1","LC2"),
    state_mode="confound_controlled",
    repeat_seed=SEED,
    discovery_fraction=.60,
    primary_top_fraction=PRIMARY_TOP_FRACTION,
    c_sensitivity=C_SENSITIVITY,
    top_fracs=TOP_FRACTIONS,
    compute_same_image_imprint=True,
    fixed_states=None,
    fixed_split=None,
):
    t,req,idx,exact,y=_build_analysis_subset(trials,block,positive_labels,negative_labels)
    if len(idx)<80:
        return None

    if fixed_split is None:
        dloc,floc=stratified_discovery_split(exact[idx],discovery_fraction,repeat_seed)
        discovery=idx[dloc]; final=idx[floc]
    else:
        discovery=np.asarray(fixed_split[0],int)
        final=np.asarray(fixed_split[1],int)

    if fixed_states is None:
        local_discovery=np.searchsorted(idx,discovery)
        time=pd.to_numeric(t.loc[idx,"start_time"],errors="coerce").to_numpy(float)
        running=None
        if "pre_running_speed" in t.columns:
            running=pd.to_numeric(t.loc[idx,"pre_running_speed"],errors="coerce").to_numpy(float)
        sf=fit_prestim_state(
            mats["baseline"][idx],local_discovery,time,running,
            mode=state_mode,
        )
        states=np.full(len(t),-1,int)
        scores=np.full(len(t),np.nan,float)
        states[idx]=sf["state"]
        scores[idx]=sf["score"]
        state_meta={
            "state_mode":state_mode,
            "pc1_explained":sf["pc1_explained"],
            "n_pca_units":sf["n_pca_units"],
            "confound_residualized":sf["confound_residualized"],
            "running_included_in_state_model":bool(sf.get("running_included",False)),
        }
        audit=state_audit(t,idx,states,scores)
    else:
        states=np.asarray(fixed_states,int)
        scores=np.full(len(t),np.nan,float)
        state_meta={"state_mode":"fixed_permutation","pc1_explained":np.nan,
                    "n_pca_units":mats["full"].shape[1],"confound_residualized":True,
                    "running_included_in_state_model":True}
        audit={}

    # Exact-label counts by state, discovery/final.
    counts={}
    for partname,part in [("discovery",discovery),("final",final)]:
        for s in [0,1]:
            for lab in req:
                counts[f"n_{partname}_s{s}_{lab}"]=int(np.sum((states[part]==s)&(exact[part]==lab)))

    full=np.asarray(mats["full"],float)
    dec=decoder_transfer(full,y,exact,discovery,final,states,req,C=PRIMARY_C,seed=repeat_seed)
    if dec is None:
        return None
    topset_tab,topset_sum=coefficient_selected_transfer(
        full,y,exact,final,states,req,dec["models"],
        top_fracs=top_fracs,seed=repeat_seed
    )

    # Unit-level nuisance covariates are also discovery-only, avoiding final-trial
    # leakage in the residualized landscape sensitivity analyses.
    local_covariates = activity_covariates(t, units, mats, block=block, indices=discovery)

    landscape_summaries=[]
    landscape_units=[]
    primary_landscape=None
    for C in c_sensitivity:
        lr=coefficient_landscape(
            full,y,exact,discovery,states,req,units,local_covariates,
            C=C,seed=repeat_seed+7000,top_frac=primary_top_fraction
        )
        if lr is None:
            continue
        landscape_summaries.append(lr["summary"])
        uu=lr["unit_coefficients"].copy()
        landscape_units.append(uu)
        if abs(C-PRIMARY_C)<1e-12:
            primary_landscape=lr["summary"]
    if primary_landscape is None:
        return None

    imprint_tab,imprint_summary=exact_image_state_imprint(
        t,mats,discovery,final,states,seed=repeat_seed
    ) if (compute_same_image_imprint and block=="ICwcfg1" and set(positive_labels)=={"IC1","IC2"}) else (pd.DataFrame(),{})

    ptop=topset_sum[np.isclose(topset_sum["top_fraction"],primary_top_fraction)]
    ptoprow=ptop.iloc[0].to_dict() if len(ptop) else {}

    summary={
        "block":block,
        "positive_labels":";".join(positive_labels),
        "negative_labels":";".join(negative_labels),
        **state_meta,
        **audit,
        **counts,
        **dec["summary"],
        **{k:v for k,v in ptoprow.items() if k!="top_fraction"},
        **primary_landscape,
        **imprint_summary,
        "n_units":int(full.shape[1]),
        "n_analysis_trials":int(len(idx)),
        "n_discovery":int(len(discovery)),
        "n_final":int(len(final)),
    }

    state_df=pd.DataFrame({
        "trial_uid":t["trial_uid"],
        "prestim_state":states,
        "prestim_score":scores,
    })
    return {
        "summary":summary,
        "decoder_transfer":dec["table"],
        "topset_transfer":topset_tab,
        "topset_summary":topset_sum,
        "landscape_summary":pd.DataFrame(landscape_summaries),
        "landscape_units":pd.concat(landscape_units,ignore_index=True) if landscape_units else pd.DataFrame(),
        "same_image_imprint":imprint_tab,
        "state_scores":state_df,
        "split":{"discovery":discovery,"final":final,"analysis":idx},
    }


def shuffle_states_within_exact(trials, base_result, *, seed):
    """Shuffle state labels within exact stimulus and separately within split."""
    rng=np.random.default_rng(seed)
    t=trials.reset_index(drop=True)
    exact=t["canonical_stimulus"].astype(str).to_numpy()
    states=base_result["state_scores"].set_index("trial_uid")["prestim_state"]
    global_state=t["trial_uid"].map(states).fillna(-1).to_numpy(int)
    discovery=np.asarray(base_result["split"]["discovery"],int)
    final=np.asarray(base_result["split"]["final"],int)
    out=global_state.copy()
    for part in [discovery,final]:
        for lab in pd.unique(exact[part]):
            ii=part[exact[part]==lab]
            if len(ii)>1:
                out[ii]=rng.permutation(out[ii])
    return out


def state_shuffle_null(
    trials,units,mats,covariates,base_result,
    *, n_perm=30, seed=SEED,
):
    """Fixed-split null for the three v2.3 key endpoints."""
    rows=[]
    split=(base_result["split"]["discovery"],base_result["split"]["final"])
    for p in range(n_perm):
        shuffled=shuffle_states_within_exact(trials,base_result,seed=seed+10000+p)
        rr=formal_context_repeat(
            trials,units,mats,covariates,
            block="ICwcfg1",
            positive_labels=("IC1","IC2"),negative_labels=("LC1","LC2"),
            state_mode="confound_controlled",
            repeat_seed=seed,
            c_sensitivity=(PRIMARY_C,),
            top_fracs=(PRIMARY_TOP_FRACTION,),
            compute_same_image_imprint=False,
            fixed_states=shuffled,
            fixed_split=split,
        )
        if rr is None:
            continue
        s=rr["summary"]
        rows.append({
            "perm_index":p,
            "decoder_auc_crossover_null":s["decoder_auc_crossover"],
            "topset_auc_crossover_null":s["topset_auc_crossover"],
            "delta_rho_null":s["delta_rho"],
            "delta_jaccard_null":s["delta_jaccard"],
        })
    return pd.DataFrame(rows)
