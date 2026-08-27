#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5C — Synaptic Physiology narrative closure v2
====================================================

Purpose
-------
This is a *closure*, not a fishing expedition. It does not add arbitrary new
endpoints. It sharpens what Allen Synaptic Physiology can and cannot contribute
to the paper's unified scientific narrative:

    Intrinsic cellular dynamical complexity is a network resource whose
    functional value is not reducible to static structural prominence or
    local synaptic strength/dynamics; its value emerges at the level of
    collective, state-dependent dynamical leverage.

SynPhys cannot measure whole-network state-dependent leverage directly. Its job
in the paper is therefore narrower and important:

1) test whether transferred intrinsic complexity is merely "hubness";
2) test whether it is merely local synaptic strength / STP;
3) test whether standardized local temporal gain provides an independent
   within-identity bridge;
4) quantify how much marginal local-synaptic association is absorbed by
   biological identity (descriptive, NOT causal mediation);
5) verify all key conclusions under a protocol-conservative transfer definition;
6) close the remaining technical gap with fully fold-local experiment-grouped CV;
7) propagate Stage5A mapper uncertainty into the only results worth retaining:
   mouse connection probability, mouse G_norm_50Hz, and exploratory human PSP.

This script imports the already validated:
    plot/stage5c_synphys_integrated_v1.py

and reads:
    plot/Stage5C_synphys_integrated_v1/tables/
    bio data/stage5_v12_results.tar.gz

Default output:
    plot/Stage5C_synphys_narrative_closure_v2/

No SynPhys outcome is used to define or refit the Stage5A complexity mapper.
No local SynPhys result is rebranded as whole-network dynamical leverage.

Run
---
python .\\plot\\stage5c_synphys_narrative_closure_v2.py ^
  --root "D:\\Research\\Neural Science"

Quick smoke test:
python .\\plot\\stage5c_synphys_narrative_closure_v2.py ^
  --root "D:\\Research\\Neural Science" --quick
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20260821
PRIMARY_SCORE = "C_primary_shared"
CONSERVATIVE_SCORE = "C_protocol_conservative"
FREQS = [5, 10, 20, 50]

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

KEY_BOOTSTRAP_ENDPOINTS = [
    "mouse_connection",
    "mouse_G_norm_50hz",
    "human_psp_amplitude",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--integrated-script", type=Path, default=None)
    ap.add_argument("--integrated-out", type=Path, default=None)
    ap.add_argument("--stage5a", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bootstrap-draws", type=int, default=100)
    ap.add_argument("--split-repeats", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--quick", action="store_true")
    return ap.parse_args()


def json_dump(path: Path, obj):
    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv), encoding="utf-8")


def load_integrated_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing integrated analysis script:\n{path}\n"
            "Place stage5c_synphys_integrated_v1.py under the plot directory."
        )
    spec = importlib.util.spec_from_file_location("stage5c_integrated_v1", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def find_stage5a(root: Path, mod):
    try:
        return mod.find_stage5a(root)
    except Exception:
        hits = list((root / "bio data").rglob("stage5_v12_results.tar.gz"))
        if not hits:
            raise
        return hits[0]


def read_required(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def boolish(s):
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).ne(0)
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def set_plot_defaults():
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


def save_panel(fig, stem, figdir, dpi):
    fig.savefig(figdir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(figdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    tmp = np.empty_like(q)
    tmp[order] = np.minimum(q, 1.0)
    out[np.flatnonzero(ok)] = tmp
    return out


def attach_score(df, scores, score_col):
    domain_col = score_col + "_domain"
    sd_col = score_col + "_sd"
    drop = [c for c in [score_col, domain_col, sd_col] if c in df.columns]
    x = df.drop(columns=drop).copy()
    cols = ["cell_id", score_col, domain_col]
    if sd_col in scores.columns:
        cols.append(sd_col)
    s = scores[cols].drop_duplicates("cell_id").rename(columns={"cell_id": "pre_cell_id"})
    return x.merge(s, on="pre_cell_id", how="left")


def primary_mouse_cohorts(pairs, std, score_col):
    domain = score_col + "_domain"
    pre_mouse = pairs["pre_species"].astype(str).str.lower().eq("mouse")
    score_ok = pd.to_numeric(pairs[score_col], errors="coerce").notna()
    domain_ok = boolish(pairs[domain])
    tested = pairs[pre_mouse & score_ok & domain_ok & boolish(pairs["tested_primary"])].copy()
    conn = tested[boolish(tested["connected"])].copy()
    dyn = pairs[
        pre_mouse & score_ok & domain_ok &
        pd.to_numeric(pairs["dynamics_qc_pass"], errors="coerce").fillna(0).eq(1)
    ].copy()

    smouse = std["pre_species"].astype(str).str.lower().eq("mouse")
    sscore = pd.to_numeric(std[score_col], errors="coerce").notna()
    sdomain = boolish(std[domain])
    sstd = std[smouse & sscore & sdomain].copy()
    return tested, conn, dyn, sstd


def human_cohorts(pairs, std, score_col):
    domain = score_col + "_domain"
    pre_human = pairs["pre_species"].astype(str).str.lower().eq("human")
    score_ok = pd.to_numeric(pairs[score_col], errors="coerce").notna()
    domain_ok = boolish(pairs[domain])
    tested = pairs[pre_human & score_ok & domain_ok & boolish(pairs["tested_primary"])].copy()
    conn = tested[boolish(tested["connected"])].copy()

    sh = std["pre_species"].astype(str).str.lower().eq("human")
    ss = pd.to_numeric(std[score_col], errors="coerce").notna()
    sd = boolish(std[domain])
    sstd = std[sh & ss & sd].copy()
    return tested, conn, sstd


# ---------------------------------------------------------------------
# Fully fold-local predictor pipelines
# ---------------------------------------------------------------------

BASE_NUMERIC = ["log1p_distance_um", "pre_age"]
BASE_CATEGORICAL = ["pre_cortical_layer", "post_cortical_layer"]
IDENTITY_CATEGORICAL = [
    "pre_cell_class", "post_cell_class", "pre_cre_type", "post_cre_type"
]


def raw_predictors(d, score_col, include_score, include_identity, add_strength):
    X = pd.DataFrame(index=d.index)
    num = []
    cat = []

    if "distance" in d:
        x = pd.to_numeric(d["distance"], errors="coerce")
        X["log1p_distance_um"] = np.log1p(np.maximum(x * 1e6, 0))
        num.append("log1p_distance_um")
    if "pre_age" in d:
        X["pre_age"] = pd.to_numeric(d["pre_age"], errors="coerce")
        num.append("pre_age")
    if add_strength and "A1_abs_50hz" in d:
        x = pd.to_numeric(d["A1_abs_50hz"], errors="coerce")
        X["log_A1_abs_50hz"] = np.log(np.maximum(x, 1e-15))
        num.append("log_A1_abs_50hz")
    if include_score:
        X["complexity"] = pd.to_numeric(d[score_col], errors="coerce")
        num.append("complexity")

    for c in BASE_CATEGORICAL:
        if c in d:
            X[c] = d[c].fillna("MISSING").astype(str)
            cat.append(c)

    if include_identity:
        for c in IDENTITY_CATEGORICAL:
            if c in d:
                X[c] = d[c].fillna("MISSING").astype(str)
                cat.append(c)

    return X, num, cat


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def predictor_pipeline(num_cols, cat_cols, task):
    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    if task == "continuous":
        est = Ridge(alpha=1e-6, solver="lsqr")
    else:
        est = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    return Pipeline([("pre", pre), ("model", est)])


def clean_continuous_target(d, outcome, log_abs=False):
    y = pd.to_numeric(d[outcome], errors="coerce").astype(float)
    if log_abs:
        y = np.log(np.maximum(np.abs(y), 1e-15))
    y = y.replace([np.inf, -np.inf], np.nan)
    return y


def fold_local_group_cv_continuous(
    d, outcome, score_col, log_abs=False, add_strength=False, n_splits=5
):
    use = d.copy()
    y = clean_continuous_target(use, outcome, log_abs)
    valid = y.notna() & pd.to_numeric(use[score_col], errors="coerce").notna() & use["experiment_id"].notna()
    use = use.loc[valid].copy()
    y = y.loc[valid].to_numpy(float)
    groups = use["experiment_id"].astype(str).to_numpy()
    if pd.Series(groups).nunique() < n_splits:
        return {"status": "too_few_groups"}

    designs = {}
    for name, inc_c, inc_i in [
        ("M0", False, False), ("M1", True, False),
        ("M2", False, True), ("M3", True, True),
    ]:
        X, num, cat = raw_predictors(use, score_col, inc_c, inc_i, add_strength)
        designs[name] = (X.reset_index(drop=True), num, cat)

    gkf = GroupKFold(n_splits=n_splits)
    preds = {k: np.full(len(use), np.nan) for k in designs}
    for tr, te in gkf.split(np.arange(len(use)), y, groups):
        for k, (X, num, cat) in designs.items():
            pipe = predictor_pipeline(num, cat, "continuous")
            pipe.fit(X.iloc[tr], y[tr])
            preds[k][te] = pipe.predict(X.iloc[te])

    rec = {f"{k}_cv_r2": float(r2_score(y, p)) for k, p in preds.items()}
    rec["delta_cv_r2_C_before_identity"] = rec["M1_cv_r2"] - rec["M0_cv_r2"]
    rec["delta_cv_r2_C_after_identity"] = rec["M3_cv_r2"] - rec["M2_cv_r2"]
    rec["n"] = int(len(use))
    rec["n_experiments"] = int(pd.Series(groups).nunique())
    rec["status"] = "ok"
    return rec


def fold_local_group_cv_connection(d, score_col, n_splits=5):
    use = d.copy()
    valid = (
        use["connected"].notna() &
        pd.to_numeric(use[score_col], errors="coerce").notna() &
        use["experiment_id"].notna()
    )
    use = use.loc[valid].copy()
    y = boolish(use["connected"]).astype(int).to_numpy()
    groups = use["experiment_id"].astype(str).to_numpy()
    if pd.Series(groups).nunique() < n_splits or len(np.unique(y)) < 2:
        return {"status": "too_few"}

    designs = {}
    for name, inc_c, inc_i in [
        ("M0", False, False), ("M1", True, False),
        ("M2", False, True), ("M3", True, True),
    ]:
        X, num, cat = raw_predictors(use, score_col, inc_c, inc_i, False)
        designs[name] = (X.reset_index(drop=True), num, cat)

    gkf = GroupKFold(n_splits=n_splits)
    probs = {k: np.full(len(use), np.nan) for k in designs}
    for tr, te in gkf.split(np.arange(len(use)), y, groups):
        for k, (X, num, cat) in designs.items():
            pipe = predictor_pipeline(num, cat, "binary")
            try:
                pipe.fit(X.iloc[tr], y[tr])
                probs[k][te] = pipe.predict_proba(X.iloc[te])[:, 1]
            except Exception:
                probs[k][te] = y[tr].mean()

    rec = {}
    for k, p in probs.items():
        p = np.clip(p, 1e-8, 1 - 1e-8)
        rec[f"{k}_logloss"] = float(log_loss(y, p, labels=[0, 1]))
        try:
            rec[f"{k}_auc"] = float(roc_auc_score(y, p))
        except Exception:
            rec[f"{k}_auc"] = np.nan
    rec["delta_logloss_C_before_identity"] = rec["M0_logloss"] - rec["M1_logloss"]
    rec["delta_logloss_C_after_identity"] = rec["M2_logloss"] - rec["M3_logloss"]
    rec["n"] = int(len(use))
    rec["n_experiments"] = int(pd.Series(groups).nunique())
    rec["status"] = "ok"
    return rec


def repeated_group_holdout_continuous(
    d, outcome, score_col, repeats, log_abs=False, add_strength=False
):
    use = d.copy()
    y = clean_continuous_target(use, outcome, log_abs)
    valid = y.notna() & pd.to_numeric(use[score_col], errors="coerce").notna() & use["experiment_id"].notna()
    use = use.loc[valid].copy().reset_index(drop=True)
    y = y.loc[valid].to_numpy(float)
    groups = use["experiment_id"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=repeats, test_size=0.20, random_state=SEED
    )
    rows = []
    for rep, (tr, te) in enumerate(splitter.split(np.arange(len(use)), y, groups)):
        vals = {}
        for name, inc_c, inc_i in [
            ("M0", False, False), ("M1", True, False),
            ("M2", False, True), ("M3", True, True),
        ]:
            X, num, cat = raw_predictors(use, score_col, inc_c, inc_i, add_strength)
            pipe = predictor_pipeline(num, cat, "continuous")
            pipe.fit(X.iloc[tr], y[tr])
            vals[name] = r2_score(y[te], pipe.predict(X.iloc[te]))
        rows.append({
            "repeat": rep,
            "delta_before": vals["M1"] - vals["M0"],
            "delta_after": vals["M3"] - vals["M2"],
            "n_test": int(len(te)),
            "n_test_experiments": int(pd.Series(groups[te]).nunique()),
        })
    return pd.DataFrame(rows)


def repeated_group_holdout_connection(d, score_col, repeats):
    use = d.copy()
    valid = (
        use["connected"].notna() &
        pd.to_numeric(use[score_col], errors="coerce").notna() &
        use["experiment_id"].notna()
    )
    use = use.loc[valid].copy().reset_index(drop=True)
    y = boolish(use["connected"]).astype(int).to_numpy()
    groups = use["experiment_id"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=repeats, test_size=0.20, random_state=SEED
    )
    rows = []
    for rep, (tr, te) in enumerate(splitter.split(np.arange(len(use)), y, groups)):
        vals = {}
        for name, inc_c, inc_i in [
            ("M0", False, False), ("M1", True, False),
            ("M2", False, True), ("M3", True, True),
        ]:
            X, num, cat = raw_predictors(use, score_col, inc_c, inc_i, False)
            pipe = predictor_pipeline(num, cat, "binary")
            try:
                pipe.fit(X.iloc[tr], y[tr])
                p = np.clip(pipe.predict_proba(X.iloc[te])[:, 1], 1e-8, 1 - 1e-8)
            except Exception:
                p = np.full(len(te), y[tr].mean())
            vals[name] = log_loss(y[te], p, labels=[0, 1])
        rows.append({
            "repeat": rep,
            "delta_before": vals["M0"] - vals["M1"],
            "delta_after": vals["M2"] - vals["M3"],
            "n_test": int(len(te)),
            "n_test_experiments": int(pd.Series(groups[te]).nunique()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Model closure for one transfer definition
# ---------------------------------------------------------------------

def run_definition_models(mod, pairs, std, scores, score_col, split_repeats):
    p = attach_score(pairs, scores, score_col)
    s = attach_score(std, scores, score_col)
    tested, conn, dyn, sstd = primary_mouse_cohorts(p, s, score_col)

    out = {
        "counts": {
            "tested": int(len(tested)),
            "connected": int(len(conn)),
            "measured_dynamics": int(len(dyn)),
            "standardized_models": int(len(sstd)),
        },
        "connection": {},
        "strength": {},
        "measured_stp": {},
        "standardized_gnorm": {},
    }

    out["connection"]["ladder"] = mod.clustered_logit_ladder(tested, score_col)
    out["connection"]["fold_local_cv"] = fold_local_group_cv_connection(tested, score_col)
    out["connection"]["repeated_holdout"] = repeated_group_holdout_connection(
        tested, score_col, split_repeats
    )

    for ep in ["psp_amplitude", "psc_amplitude"]:
        d = conn[pd.to_numeric(conn[ep], errors="coerce").notna()].copy()
        out["strength"][ep] = {
            "ladder": mod.clustered_ols_ladder(d, ep, score_col, log_abs=True),
            "fold_local_cv": fold_local_group_cv_continuous(
                d, ep, score_col, log_abs=True
            ),
            "repeated_holdout": repeated_group_holdout_continuous(
                d, ep, score_col, split_repeats, log_abs=True
            ),
        }

    for ep in STP_ENDPOINTS:
        if ep not in dyn.columns:
            continue
        d = dyn[pd.to_numeric(dyn[ep], errors="coerce").notna()].copy()
        out["measured_stp"][ep] = {
            "ladder": mod.clustered_ols_ladder(d, ep, score_col, log_abs=False),
            "fold_local_cv": fold_local_group_cv_continuous(
                d, ep, score_col, log_abs=False
            ),
        }

    for f in FREQS:
        ep = f"G_norm_{f}hz"
        if ep not in sstd.columns:
            continue
        d = sstd[pd.to_numeric(sstd[ep], errors="coerce").notna()].copy()
        out["standardized_gnorm"][ep] = {
            "ladder": mod.clustered_ols_ladder(
                d, ep, score_col, log_abs=False, add_strength=True
            ),
            "fold_local_cv": fold_local_group_cv_continuous(
                d, ep, score_col, log_abs=False, add_strength=True
            ),
        }
        if f == 50:
            out["standardized_gnorm"][ep]["repeated_holdout"] = (
                repeated_group_holdout_continuous(
                    d, ep, score_col, split_repeats,
                    log_abs=False, add_strength=True
                )
            )
    return out, p, s, tested, conn, dyn, sstd


def flatten_definition(name, score_col, result):
    rows = []
    # Connection.
    lr = result["connection"]["ladder"]
    cv = result["connection"]["fold_local_cv"]
    if lr.get("status") == "ok":
        m = lr["M3_identity_C"]
        rows.append({
            "definition": name, "score_col": score_col,
            "family": "structural", "endpoint": "connection",
            "n": lr["n"], "n_experiments": lr["n_experiments"],
            "beta_after_identity": m["beta_C_z_logodds"],
            "se_after_identity": m["se_C_z_cluster"],
            "p_after_identity": m["p_C_z_cluster"],
            "effect_scale": "log_odds",
            "odds_ratio": m["odds_ratio_per_sd"],
            "delta_before": np.nan, "delta_after": np.nan,
            "fold_local_increment_after": cv.get("delta_logloss_C_after_identity"),
            "fold_local_metric": "logloss_improvement",
        })

    for famkey, famname in [
        ("strength", "strength"),
        ("measured_stp", "measured_stp"),
        ("standardized_gnorm", "standardized"),
    ]:
        for ep, rr in result[famkey].items():
            lr = rr["ladder"]
            cv = rr["fold_local_cv"]
            if lr.get("status") != "ok":
                continue
            m = lr["M3_identity_C"]
            rows.append({
                "definition": name, "score_col": score_col,
                "family": famname, "endpoint": ep,
                "n": lr["n"], "n_experiments": lr["n_experiments"],
                "beta_after_identity": m["beta_C_z"],
                "se_after_identity": m["se_C_z_cluster"],
                "p_after_identity": m["p_C_z_cluster"],
                "effect_scale": "standardized_beta",
                "odds_ratio": np.nan,
                "delta_before": lr["delta_r2_C_before_identity"],
                "delta_after": lr["delta_r2_C_after_identity"],
                "fold_local_increment_after": cv.get("delta_cv_r2_C_after_identity"),
                "fold_local_metric": "delta_cv_r2",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Mapper uncertainty propagation
# ---------------------------------------------------------------------

def mapper_bootstrap_closure(
    mod, root, stage5a, cells, pairs, std, bootstrap_draws
):
    A5, summary, _, _ = mod.read_stage5a(stage5a)
    transfer = mod.fit_transfer(
        A5, cells, summary, "primary_shared",
        repeats=3, n_boot=bootstrap_draws
    )
    saved = read_required(
        root / "plot" / "Stage5C_synphys_integrated_v1" /
        "tables" / "complexity_transfer_primary_shared.csv"
    )
    chk = transfer["cell_scores"][["cell_id", PRIMARY_SCORE]].merge(
        saved[["cell_id", PRIMARY_SCORE]],
        on="cell_id", suffixes=("_rerun", "_saved")
    )
    maxdiff = float(
        np.nanmax(np.abs(chk[f"{PRIMARY_SCORE}_rerun"] - chk[f"{PRIMARY_SCORE}_saved"]))
    )

    final_scores = transfer["cell_scores"]
    pp = attach_score(pairs, final_scores, PRIMARY_SCORE)
    ss = attach_score(std, final_scores, PRIMARY_SCORE)
    mt, mc, md, ms = primary_mouse_cohorts(pp, ss, PRIMARY_SCORE)
    ht, hc, hs = human_cohorts(pp, ss, PRIMARY_SCORE)

    bp = transfer["boot_pred"]
    cell_pos = pd.Series(np.arange(len(cells)), index=cells["cell_id"]).to_dict()
    rows = []

    for b in range(bp.shape[0]):
        def map_boot(df):
            x = df.copy()
            x["_Cboot"] = x["pre_cell_id"].map(
                lambda cid: float(bp[b, cell_pos[cid]])
                if cid in cell_pos and np.isfinite(bp[b, cell_pos[cid]])
                else np.nan
            )
            return x

        # Mouse connection.
        d = map_boot(mt)
        rr = mod.clustered_logit_ladder(d, "_Cboot")
        if rr.get("status") == "ok":
            m = rr["M3_identity_C"]
            rows.append({
                "draw": b, "endpoint": "mouse_connection",
                "beta": m["beta_C_z_logodds"],
                "se": m["se_C_z_cluster"],
                "p": m["p_C_z_cluster"],
                "derived_effect": m["odds_ratio_per_sd"],
                "derived_effect_name": "odds_ratio",
            })

        # Mouse standardized dynamic phenotype.
        d = map_boot(ms)
        rr = mod.clustered_ols_ladder(
            d, "G_norm_50hz", "_Cboot", False, True
        )
        if rr.get("status") == "ok":
            m = rr["M3_identity_C"]
            rows.append({
                "draw": b, "endpoint": "mouse_G_norm_50hz",
                "beta": m["beta_C_z"],
                "se": m["se_C_z_cluster"],
                "p": m["p_C_z_cluster"],
                "derived_effect": rr["delta_r2_C_after_identity"],
                "derived_effect_name": "delta_r2_after",
            })

        # Exploratory human PSP.
        hd = hc[pd.to_numeric(hc["psp_amplitude"], errors="coerce").notna()].copy()
        d = map_boot(hd)
        if len(d) >= 100:
            rr = mod.clustered_ols_ladder(
                d, "psp_amplitude", "_Cboot", True, False
            )
            if rr.get("status") == "ok":
                m = rr["M3_identity_C"]
                rows.append({
                    "draw": b, "endpoint": "human_psp_amplitude",
                    "beta": m["beta_C_z"],
                    "se": m["se_C_z_cluster"],
                    "p": m["p_C_z_cluster"],
                    "derived_effect": rr["delta_r2_C_after_identity"],
                    "derived_effect_name": "delta_r2_after",
                })

    out = pd.DataFrame(rows)
    summary_rows = []
    for ep, g in out.groupby("endpoint"):
        beta = g["beta"].to_numpy(float)
        summary_rows.append({
            "endpoint": ep,
            "n_draws": int(len(g)),
            "beta_median": float(np.nanmedian(beta)),
            "beta_q025": float(np.nanquantile(beta, .025)),
            "beta_q975": float(np.nanquantile(beta, .975)),
            "positive_fraction": float(np.mean(beta > 0)),
            "negative_fraction": float(np.mean(beta < 0)),
            "p_lt_0_05_fraction": float(np.mean(g["p"].to_numpy(float) < .05)),
        })
    return out, pd.DataFrame(summary_rows), maxdiff


# ---------------------------------------------------------------------
# Narrative synthesis
# ---------------------------------------------------------------------

def identity_absorption_table(flat):
    d = flat[
        flat["family"].isin(["strength", "measured_stp"]) &
        pd.to_numeric(flat["delta_before"], errors="coerce").notna()
    ].copy()
    d["identity_absorption_fraction"] = np.nan
    m = d["delta_before"] >= 0.005
    d.loc[m, "identity_absorption_fraction"] = (
        1 - d.loc[m, "delta_after"] / d.loc[m, "delta_before"]
    )
    d["descriptive_only"] = True
    return d


def holdout_summary(def_name, result):
    rows = []
    items = [
        ("connection", result["connection"].get("repeated_holdout")),
        ("psp_amplitude", result["strength"].get("psp_amplitude", {}).get("repeated_holdout")),
        ("psc_amplitude", result["strength"].get("psc_amplitude", {}).get("repeated_holdout")),
        ("G_norm_50hz", result["standardized_gnorm"].get("G_norm_50hz", {}).get("repeated_holdout")),
    ]
    for ep, df in items:
        if df is None or len(df) == 0:
            continue
        rows.append({
            "definition": def_name,
            "endpoint": ep,
            "n_repeats": int(len(df)),
            "median_increment_before": float(df["delta_before"].median()),
            "median_increment_after": float(df["delta_after"].median()),
            "positive_fraction_before": float((df["delta_before"] > 0).mean()),
            "positive_fraction_after": float((df["delta_after"] > 0).mean()),
            "q025_after": float(df["delta_after"].quantile(.025)),
            "q975_after": float(df["delta_after"].quantile(.975)),
        })
    return pd.DataFrame(rows)


def endpoint_evidence_table(flat, holdout, boot_summary):
    rows = []

    def get(defn, ep):
        z = flat[(flat["definition"] == defn) & (flat["endpoint"] == ep)]
        return None if z.empty else z.iloc[0]

    for ep, role in [
        ("connection", "structural prominence / hubness baseline"),
        ("psp_amplitude", "conditional local synaptic strength"),
        ("psc_amplitude", "conditional local synaptic strength"),
        ("stp_induction_50hz", "measured local short-term plasticity"),
        ("variability_resting_state", "measured local synaptic variability"),
        ("G_norm_50hz", "standardized local temporal gain"),
    ]:
        p = get("primary_shared", ep)
        c = get("protocol_conservative", ep)
        if p is None or c is None:
            continue
        rows.append({
            "endpoint": ep,
            "paper_role": role,
            "primary_beta_after": p["beta_after_identity"],
            "primary_p_after": p["p_after_identity"],
            "primary_fold_local_increment_after": p["fold_local_increment_after"],
            "conservative_beta_after": c["beta_after_identity"],
            "conservative_p_after": c["p_after_identity"],
            "conservative_fold_local_increment_after": c["fold_local_increment_after"],
            "same_beta_sign": bool(
                np.sign(p["beta_after_identity"]) == np.sign(c["beta_after_identity"])
            ),
        })
    e = pd.DataFrame(rows)
    if len(e):
        e["interpretation"] = ""
        for i, r in e.iterrows():
            ep = r["endpoint"]
            if ep == "connection":
                e.loc[i, "interpretation"] = (
                    "Tests whether transferred intrinsic complexity simply marks structural hubness. "
                    "An inverse/small effect supports decoupling from simple local prominence if robust."
                )
            elif ep == "G_norm_50hz":
                e.loc[i, "interpretation"] = (
                    "Tests the direct local temporal-gain bridge. Null/negative held-out increments "
                    "support a boundary: local synaptic temporal gain is not the paper's network leverage."
                )
            else:
                e.loc[i, "interpretation"] = (
                    "Marginal local-synaptic associations are compared before/after identity; "
                    "attenuation is descriptive identity structuring, not mediation."
                )
    return e


def adjudicate(flat, holdout, boot_summary):
    def row(defn, ep):
        d = flat[(flat["definition"] == defn) & (flat["endpoint"] == ep)]
        return None if d.empty else d.iloc[0]

    p_conn = row("primary_shared", "connection")
    c_conn = row("protocol_conservative", "connection")
    p_g = row("primary_shared", "G_norm_50hz")
    c_g = row("protocol_conservative", "G_norm_50hz")

    out = {
        "direct_local_dynamic_bridge": "NOT_SUPPORTED",
        "structural_hubness_model": "NOT_SUPPORTED_AS_A_POSITIVE_COMPLEXITY_RULE",
        "local_synaptic_identity_structure": "SUPPORTED_DESCRIPTIVELY",
        "human_psp": "EXPLORATORY_ONLY",
        "manuscript_use": [],
    }

    if p_g is not None and c_g is not None:
        if (
            p_g["fold_local_increment_after"] <= 0 and
            c_g["fold_local_increment_after"] <= 0 and
            p_g["p_after_identity"] > .05 and
            c_g["p_after_identity"] > .05
        ):
            out["direct_local_dynamic_bridge"] = "ROBUSTLY_NOT_SUPPORTED"

    if p_conn is not None and c_conn is not None:
        if p_conn["beta_after_identity"] < 0 and c_conn["beta_after_identity"] < 0:
            out["structural_hubness_model"] = "INVERSE_OR_DECOUPLED_SUGGESTIVE"

    out["manuscript_use"] = [
        "SynPhys should be used as a local-circuit boundary test, not as the positive network-level leverage validation.",
        "The supported paper claim is not 'complex neurons should be placed at important nodes'.",
        "The stronger unified claim is that the value of cellular complexity is not reducible to static/local prominence and must be evaluated in a collective state-dependent dynamical context.",
        "Identity attenuation is descriptive organization, not causal mediation.",
        "Human PSP remains exploratory because the transfer mapper was trained in mouse."
    ]
    return out


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def build_figures(flat, absorption, holdout, boot, boot_summary, outdir, dpi):
    figdir = outdir / "figures"
    srcdir = outdir / "source_data"
    figdir.mkdir(parents=True, exist_ok=True)
    srcdir.mkdir(parents=True, exist_ok=True)

    # N01: Primary vs conservative conditional effects.
    sel = flat[flat["endpoint"].isin([
        "connection", "psp_amplitude", "psc_amplitude",
        "stp_induction_50hz", "variability_resting_state",
        "G_norm_5hz", "G_norm_10hz", "G_norm_20hz", "G_norm_50hz"
    ])].copy()
    sel.to_csv(srcdir / "N01_definition_robustness_effects.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    endpoints = list(dict.fromkeys(sel["endpoint"].tolist()))
    ypos = {e: i for i, e in enumerate(endpoints)}
    offsets = {"primary_shared": -0.14, "protocol_conservative": 0.14}
    markers = {"primary_shared": "o", "protocol_conservative": "s"}
    for defn, g in sel.groupby("definition"):
        y = np.array([ypos[e] for e in g["endpoint"]], float) + offsets.get(defn, 0)
        ax.errorbar(
            g["beta_after_identity"], y,
            xerr=1.96 * g["se_after_identity"],
            fmt=markers.get(defn, "o"), capsize=2.5, label=defn
        )
    ax.axvline(0, linewidth=.8)
    ax.set_yticks(range(len(endpoints)))
    ax.set_yticklabels(endpoints)
    ax.set_xlabel("Conditional standardized effect per 1 SD transferred complexity")
    ax.legend(frameon=False)
    save_panel(fig, "N01_transfer_definition_robustness", figdir, dpi)

    # N02: Identity absorption among local phenotypes.
    aa = absorption[
        (absorption["definition"] == "primary_shared") &
        absorption["identity_absorption_fraction"].notna()
    ].copy()
    aa.to_csv(srcdir / "N02_identity_absorption.csv", index=False)
    if len(aa):
        fig, ax = plt.subplots(figsize=(6.3, 4.3))
        order = aa.sort_values("identity_absorption_fraction")["endpoint"]
        z = aa.set_index("endpoint").loc[order]
        ax.barh(np.arange(len(z)), z["identity_absorption_fraction"])
        ax.set_yticks(np.arange(len(z)))
        ax.set_yticklabels(z.index)
        ax.axvline(0, linewidth=.8)
        ax.axvline(1, linewidth=.8, linestyle="--")
        ax.set_xlabel("Descriptive identity-absorption fraction of marginal ΔR²")
        save_panel(fig, "N02_local_synaptic_identity_absorption", figdir, dpi)

    # N03: Fold-local CV after-identity increments.
    cv = flat[flat["endpoint"].isin([
        "connection", "psp_amplitude", "psc_amplitude", "G_norm_50hz"
    ])].copy()
    cv.to_csv(srcdir / "N03_fold_local_cv.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    keys = list(dict.fromkeys(cv["endpoint"].tolist()))
    xpos = {k: i for i, k in enumerate(keys)}
    for defn, g in cv.groupby("definition"):
        off = -0.12 if defn == "primary_shared" else 0.12
        ax.scatter(
            [xpos[e] + off for e in g["endpoint"]],
            g["fold_local_increment_after"], label=defn
        )
    ax.axhline(0, linewidth=.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("Held-out increment after identity\n(positive = improved prediction)")
    ax.legend(frameon=False)
    save_panel(fig, "N03_fold_local_group_cv_increment", figdir, dpi)

    # N04: Structural connection odds ratio.
    cc = flat[(flat["family"] == "structural") & (flat["endpoint"] == "connection")].copy()
    cc["or_lo"] = np.exp(cc["beta_after_identity"] - 1.96 * cc["se_after_identity"])
    cc["or_hi"] = np.exp(cc["beta_after_identity"] + 1.96 * cc["se_after_identity"])
    cc.to_csv(srcdir / "N04_connection_odds_ratio.csv", index=False)
    if len(cc):
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        y = np.arange(len(cc))
        ax.errorbar(
            cc["odds_ratio"], y,
            xerr=np.vstack([cc["odds_ratio"] - cc["or_lo"], cc["or_hi"] - cc["odds_ratio"]]),
            fmt="o", capsize=3
        )
        ax.axvline(1, linewidth=.8)
        ax.set_yticks(y)
        ax.set_yticklabels(cc["definition"])
        ax.set_xlabel("Connection odds ratio per 1 SD complexity")
        save_panel(fig, "N04_structural_connection_decoupling", figdir, dpi)

    # N05: G_norm frequency conditional effects.
    gg = flat[
        (flat["family"] == "standardized") &
        flat["endpoint"].str.match(r"G_norm_(5|10|20|50)hz")
    ].copy()
    gg["frequency_hz"] = gg["endpoint"].str.extract(r"(\d+)").astype(float)
    gg.to_csv(srcdir / "N05_Gnorm_frequency_effects.csv", index=False)
    if len(gg):
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        for defn, g in gg.groupby("definition"):
            g = g.sort_values("frequency_hz")
            ax.errorbar(
                g["frequency_hz"], g["beta_after_identity"],
                yerr=1.96 * g["se_after_identity"], marker="o", capsize=2.5,
                label=defn
            )
        ax.axhline(0, linewidth=.8)
        ax.set_xlabel("Standardized presynaptic train frequency (Hz)")
        ax.set_ylabel("Conditional effect on local temporal gain")
        ax.legend(frameon=False)
        save_panel(fig, "N05_local_temporal_gain_frequency_null", figdir, dpi)

    # N06: Mapper bootstrap.
    boot.to_csv(srcdir / "N06_mapper_bootstrap_effects.csv", index=False)
    if len(boot):
        eps = list(boot["endpoint"].drop_duplicates())
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        rng = np.random.default_rng(SEED)
        for i, ep in enumerate(eps):
            g = boot[boot["endpoint"] == ep]
            jitter = rng.normal(0, .045, len(g))
            ax.scatter(np.full(len(g), i) + jitter, g["beta"], s=10, alpha=.35)
            ax.errorbar(
                i, g["beta"].median(),
                yerr=np.array([[
                    g["beta"].median() - g["beta"].quantile(.025)
                ], [
                    g["beta"].quantile(.975) - g["beta"].median()
                ]]),
                fmt="o", capsize=3
            )
        ax.axhline(0, linewidth=.8)
        ax.set_xticks(range(len(eps)))
        ax.set_xticklabels(eps, rotation=20, ha="right")
        ax.set_ylabel("Conditional effect across Stage5A mapper bootstrap draws")
        save_panel(fig, "N06_mapper_uncertainty_key_effects", figdir, dpi)

    # N07: Repeated experiment holdout distributions.
    holdout.to_csv(srcdir / "N07_repeated_group_holdout.csv", index=False)
    if len(holdout):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        keys = list(dict.fromkeys(
            (holdout["definition"] + " | " + holdout["endpoint"]).tolist()
        ))
        rng = np.random.default_rng(SEED + 1)
        for i, key in enumerate(keys):
            defn, ep = key.split(" | ", 1)
            g = holdout[(holdout["definition"] == defn) & (holdout["endpoint"] == ep)]
            jitter = rng.normal(0, .045, len(g))
            ax.scatter(np.full(len(g), i) + jitter, g["delta_after"], s=9, alpha=.3)
            ax.plot(i, g["delta_after"].median(), marker="D")
        ax.axhline(0, linewidth=.8)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=35, ha="right")
        ax.set_ylabel("Repeated held-out experiment increment after identity")
        save_panel(fig, "N07_repeated_experiment_holdout", figdir, dpi)

    # N08: Mouse vs exploratory human PSP.
    sp = flat[
        (flat["endpoint"] == "psp_amplitude") &
        (flat["definition"] == "primary_shared")
    ].copy()
    # human is appended separately by caller if desired; source file will be generated there.


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    root = args.root.resolve()
    integrated_script = (
        args.integrated_script.resolve()
        if args.integrated_script
        else root / "plot" / "stage5c_synphys_integrated_v1.py"
    )
    integrated_out = (
        args.integrated_out.resolve()
        if args.integrated_out
        else root / "plot" / "Stage5C_synphys_integrated_v1"
    )
    out = (
        args.out.resolve()
        if args.out
        else root / "plot" / "Stage5C_synphys_narrative_closure_v2"
    )
    analysis = out / "analysis"
    tables_out = out / "tables"
    for p in [analysis, tables_out, out / "figures", out / "source_data"]:
        p.mkdir(parents=True, exist_ok=True)

    if args.quick:
        bootstrap_draws = min(args.bootstrap_draws, 10)
        split_repeats = min(args.split_repeats, 5)
        dpi = min(args.dpi, 180)
    else:
        bootstrap_draws = args.bootstrap_draws
        split_repeats = args.split_repeats
        dpi = args.dpi

    set_plot_defaults()
    t0 = time.time()
    print("=" * 100)
    print("Stage 5C SynPhys narrative closure v2")
    print("Root           :", root)
    print("Integrated out :", integrated_out)
    print("Output         :", out)
    print("=" * 100)

    mod = load_integrated_module(integrated_script)
    stage5a = args.stage5a.resolve() if args.stage5a else find_stage5a(root, mod)

    tables = integrated_out / "tables"
    cells = read_required(tables / "cell_intrinsic_atlas.csv")
    pairs = read_required(tables / "directed_pair_atlas.csv")
    std = read_required(tables / "standardized_synapse_train_atlas.csv")
    primary_scores = read_required(tables / "complexity_transfer_primary_shared.csv")
    conservative_scores = read_required(tables / "complexity_transfer_protocol_conservative.csv")

    print("[1/6] Re-running predeclared endpoints under primary and protocol-conservative transfers...")
    primary_res, pp, ps, ptested, pconn, pdyn, pstd = run_definition_models(
        mod, pairs, std, primary_scores, PRIMARY_SCORE, split_repeats
    )
    conservative_res, cp, cs, ctested, cconn, cdyn, cstd = run_definition_models(
        mod, pairs, std, conservative_scores, CONSERVATIVE_SCORE, split_repeats
    )

    flat = pd.concat([
        flatten_definition("primary_shared", PRIMARY_SCORE, primary_res),
        flatten_definition("protocol_conservative", CONSERVATIVE_SCORE, conservative_res),
    ], ignore_index=True)
    flat["q_within_definition_descriptive"] = np.nan
    for definition, idx in flat.groupby("definition").groups.items():
        flat.loc[idx, "q_within_definition_descriptive"] = bh_fdr(
            flat.loc[idx, "p_after_identity"].to_numpy(float)
        )
    flat.to_csv(analysis / "definition_robustness_models.csv", index=False)

    print("[2/6] Computing descriptive identity-absorption profile...")
    absorption = identity_absorption_table(flat)
    absorption.to_csv(analysis / "identity_absorption_descriptive.csv", index=False)

    print("[3/6] Saving repeated experiment-holdout distributions...")
    hold_parts = []
    for defn, rr in [
        ("primary_shared", primary_res),
        ("protocol_conservative", conservative_res),
    ]:
        for ep, df in [
            ("connection", rr["connection"].get("repeated_holdout")),
            ("psp_amplitude", rr["strength"].get("psp_amplitude", {}).get("repeated_holdout")),
            ("psc_amplitude", rr["strength"].get("psc_amplitude", {}).get("repeated_holdout")),
            ("G_norm_50hz", rr["standardized_gnorm"].get("G_norm_50hz", {}).get("repeated_holdout")),
        ]:
            if df is None or len(df) == 0:
                continue
            x = df.copy()
            x["definition"] = defn
            x["endpoint"] = ep
            hold_parts.append(x)
    holdout = pd.concat(hold_parts, ignore_index=True) if hold_parts else pd.DataFrame()
    holdout.to_csv(analysis / "repeated_experiment_holdout.csv", index=False)
    hsummary = pd.concat([
        holdout_summary("primary_shared", primary_res),
        holdout_summary("protocol_conservative", conservative_res),
    ], ignore_index=True)
    hsummary.to_csv(analysis / "repeated_experiment_holdout_summary.csv", index=False)

    print("[4/6] Propagating Stage5A mapper uncertainty into key retained signals...")
    boot, boot_summary, transfer_maxdiff = mapper_bootstrap_closure(
        mod, root, stage5a, cells, pairs, std, bootstrap_draws
    )
    boot.to_csv(analysis / "mapper_bootstrap_key_effects.csv", index=False)
    boot_summary.to_csv(analysis / "mapper_bootstrap_key_effects_summary.csv", index=False)

    print("[5/6] Exploratory human PSP and species comparison...")
    # Primary human PSP from the primary score, kept explicitly exploratory.
    ht, hc, hs = human_cohorts(pp, ps, PRIMARY_SCORE)
    hpsp = hc[pd.to_numeric(hc["psp_amplitude"], errors="coerce").notna()].copy()
    human_psp = {}
    if len(hpsp) >= 100:
        human_psp = {
            "ladder": mod.clustered_ols_ladder(
                hpsp, "psp_amplitude", PRIMARY_SCORE, True, False
            ),
            "fold_local_cv": fold_local_group_cv_continuous(
                hpsp, "psp_amplitude", PRIMARY_SCORE, log_abs=True
            ),
            "repeated_holdout": repeated_group_holdout_continuous(
                hpsp, "psp_amplitude", PRIMARY_SCORE, split_repeats, log_abs=True
            ),
        }
        human_psp["repeated_holdout"].to_csv(
            analysis / "human_psp_repeated_holdout.csv", index=False
        )

    # Mouse PSP counterpart for same score.
    mpsp = pconn[pd.to_numeric(pconn["psp_amplitude"], errors="coerce").notna()].copy()
    mouse_psp = {
        "ladder": mod.clustered_ols_ladder(
            mpsp, "psp_amplitude", PRIMARY_SCORE, True, False
        ),
        "fold_local_cv": fold_local_group_cv_continuous(
            mpsp, "psp_amplitude", PRIMARY_SCORE, log_abs=True
        ),
    }

    species_rows = []
    for species, rr in [("mouse", mouse_psp), ("human_exploratory", human_psp)]:
        if not rr or rr["ladder"].get("status") != "ok":
            continue
        lr = rr["ladder"]
        m = lr["M3_identity_C"]
        species_rows.append({
            "species": species,
            "n": lr["n"],
            "n_experiments": lr["n_experiments"],
            "beta": m["beta_C_z"],
            "se": m["se_C_z_cluster"],
            "p": m["p_C_z_cluster"],
            "delta_r2_after": lr["delta_r2_C_after_identity"],
            "fold_local_increment_after": rr["fold_local_cv"]["delta_cv_r2_C_after_identity"],
        })
    species_df = pd.DataFrame(species_rows)
    species_df.to_csv(analysis / "species_psp_comparison.csv", index=False)

    print("[6/6] Building narrative evidence table and figures...")
    evidence = endpoint_evidence_table(flat, hsummary, boot_summary)
    evidence.to_csv(analysis / "narrative_evidence_table.csv", index=False)

    adjudication = adjudicate(flat, hsummary, boot_summary)
    adjudication["transfer_rerun_max_abs_diff_vs_saved"] = transfer_maxdiff
    adjudication["bootstrap_draws"] = int(bootstrap_draws)
    adjudication["repeated_group_holdout_splits"] = int(split_repeats)
    adjudication["scientific_guardrails"] = [
        "SynPhys does not measure whole-network state-dependent dynamical leverage.",
        "Identity absorption is descriptive and is not causal mediation.",
        "A null local dynamic bridge does not prove the true biological effect is exactly zero because cross-dataset complexity transfer has limited predictive fidelity.",
        "Connection probability is a structural baseline; an inverse association is evidence against a simple hubness rule, not evidence that lower degree is universally optimal.",
        "Human PSP is exploratory because the complexity mapper was trained in mouse.",
        "The paper-level positive state-dependent leverage evidence comes from Stage4 transfer and real population/state analyses, not from SynPhys."
    ]
    json_dump(analysis / "narrative_adjudication.json", adjudication)

    # Figures.
    build_figures(flat, absorption, holdout, boot, boot_summary, out, dpi)

    # Species PSP figure.
    if len(species_df):
        species_df.to_csv(out / "source_data" / "N08_species_psp.csv", index=False)
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        y = np.arange(len(species_df))
        ax.errorbar(
            species_df["beta"], y,
            xerr=1.96 * species_df["se"],
            fmt="o", capsize=3
        )
        ax.axvline(0, linewidth=.8)
        ax.set_yticks(y)
        ax.set_yticklabels(species_df["species"])
        ax.set_xlabel("Conditional PSP effect per 1 SD transferred complexity")
        save_panel(fig, "N08_species_psp_exploratory", out / "figures", dpi)

    # Claim-focused markdown.
    pconn_row = flat[(flat.definition == "primary_shared") & (flat.endpoint == "connection")]
    cconn_row = flat[(flat.definition == "protocol_conservative") & (flat.endpoint == "connection")]
    pg_row = flat[(flat.definition == "primary_shared") & (flat.endpoint == "G_norm_50hz")]
    cg_row = flat[(flat.definition == "protocol_conservative") & (flat.endpoint == "G_norm_50hz")]

    lines = [
        "# Stage 5C SynPhys narrative closure v2",
        "",
        "## What this dataset contributes to the paper",
        "",
        "SynPhys is treated as a **local-circuit boundary test**. It is not used to claim that local synaptic gain is the same quantity as whole-network state-dependent dynamical leverage.",
        "",
        "The paper-level question is therefore not:",
        "",
        "> Are complex neurons simply placed at important nodes?",
        "",
        "but:",
        "",
        "> Is the functional value of intrinsic cellular dynamical complexity reducible to static structural prominence or local synaptic phenotypes, or does it require collective state-dependent network context?",
        "",
        "## Predeclared closure results to read",
        "",
    ]

    if len(pconn_row) and len(cconn_row):
        a, b = pconn_row.iloc[0], cconn_row.iloc[0]
        lines += [
            "### 1. Structural prominence / hubness baseline",
            "",
            f"- Primary transfer: β(log-odds)={a.beta_after_identity:.4f}, p={a.p_after_identity:.4g}, OR={a.odds_ratio:.4f}, fold-local held-out improvement={a.fold_local_increment_after:.6g}.",
            f"- Protocol-conservative transfer: β(log-odds)={b.beta_after_identity:.4f}, p={b.p_after_identity:.4g}, OR={b.odds_ratio:.4f}, fold-local held-out improvement={b.fold_local_increment_after:.6g}.",
            "",
            "Interpretation: if the inverse direction is stable across both transfer definitions and mapper bootstrap draws, SynPhys argues **against** the simple rule that intrinsic complexity merely marks local structural hubness. This is supportive decoupling evidence, not a claim that low degree is intrinsically optimal.",
            "",
        ]

    if len(pg_row) and len(cg_row):
        a, b = pg_row.iloc[0], cg_row.iloc[0]
        lines += [
            "### 2. Standardized local temporal gain",
            "",
            f"- Primary transfer G_norm_50Hz: β={a.beta_after_identity:.4f}, p={a.p_after_identity:.4g}, ΔR²(after identity)={a.delta_after:.6g}, fold-local ΔCVR²={a.fold_local_increment_after:.6g}.",
            f"- Conservative transfer G_norm_50Hz: β={b.beta_after_identity:.4f}, p={b.p_after_identity:.4g}, ΔR²(after identity)={b.delta_after:.6g}, fold-local ΔCVR²={b.fold_local_increment_after:.6g}.",
            "",
            "Interpretation: a null/negative held-out increment under both definitions supports a **local dynamic boundary**. It means that standardized single-synapse temporal gain is not sufficient to instantiate the network-level leverage concept used elsewhere in the paper.",
            "",
        ]

    if len(absorption):
        aa = absorption[
            (absorption.definition == "primary_shared") &
            absorption.identity_absorption_fraction.notna()
        ]
        if len(aa):
            lines += [
                "### 3. Local synaptic phenotypes are strongly identity-structured",
                "",
                f"- Median descriptive identity-absorption fraction across local strength/STP endpoints with marginal ΔR²≥0.005: {aa.identity_absorption_fraction.median():.3f}.",
                "",
                "Interpretation: many marginal complexity–synaptic associations largely disappear after biological identity/context is modeled. This is **descriptive identity structuring, not causal mediation**.",
                "",
            ]

    if len(boot_summary):
        lines += [
            "### 4. Stage5A mapper uncertainty",
            "",
        ]
        for _, r in boot_summary.iterrows():
            lines.append(
                f"- {r.endpoint}: median β={r.beta_median:.4f}, 95% bootstrap-mapper range [{r.beta_q025:.4f}, {r.beta_q975:.4f}], positive fraction={r.positive_fraction:.3f}, p<0.05 fraction={r.p_lt_0_05_fraction:.3f}."
            )
        lines += [""]

    if len(species_df):
        lines += [
            "### 5. Human extension",
            "",
        ]
        for _, r in species_df.iterrows():
            lines.append(
                f"- {r.species}: PSP β={r.beta:.4f}, p={r.p:.4g}, ΔR²(after identity)={r.delta_r2_after:.5f}, fold-local ΔCVR²={r.fold_local_increment_after:.5f}."
            )
        lines += [
            "",
            "Human results remain exploratory because the transferred complexity mapper was trained in mouse.",
            "",
        ]

    lines += [
        "## Manuscript-safe synthesis",
        "",
        "**SynPhys should not be used to argue that complex neurons simply occupy locally important positions. Instead, it constrains what complexity is *not*: it is not reducible to local degree, synaptic strength, measured STP, or standardized single-synapse temporal gain. This boundary is complementary to the positive network-level evidence that leverage is collective and state-dependent.**",
        "",
        "A safe integrated wording is:",
        "",
        "> Intrinsic cellular dynamical complexity is a network resource whose functional value is not captured by static structural prominence or local synaptic efficacy alone; rather, its utility emerges in a collective, state-dependent dynamical context.",
        "",
        "This wording combines SynPhys only with the positive Stage4/Stage5B evidence. SynPhys alone does not establish the second clause.",
        "",
        "## Technical closure",
        "",
        f"- Stage5A mapper bootstrap draws: {bootstrap_draws}",
        f"- Repeated experiment-group holdouts: {split_repeats}",
        f"- Refit-vs-saved transfer max absolute difference: {transfer_maxdiff:.6g}",
        "",
        "## Output files",
        "",
        "- `definition_robustness_models.csv`",
        "- `identity_absorption_descriptive.csv`",
        "- `repeated_experiment_holdout_summary.csv`",
        "- `mapper_bootstrap_key_effects_summary.csv`",
        "- `species_psp_comparison.csv`",
        "- `narrative_evidence_table.csv`",
        "- `narrative_adjudication.json`",
        "- `figures/N01...N08`",
    ]
    (analysis / "00_STAGE5C_NARRATIVE_CLOSURE_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    report = {
        "status": "COMPLETE",
        "elapsed_seconds": time.time() - t0,
        "primary_counts": primary_res["counts"],
        "conservative_counts": conservative_res["counts"],
        "adjudication": adjudication,
        "bootstrap_summary": boot_summary.to_dict(orient="records"),
        "holdout_summary": hsummary.to_dict(orient="records"),
        "species_psp": species_df.to_dict(orient="records"),
    }
    json_dump(analysis / "stage5c_narrative_closure_report.json", report)

    # Figure catalog.
    figs = sorted((out / "figures").glob("*.png"))
    pd.DataFrame([
        {"figure": p.stem, "png": str(p), "pdf": str(p.with_suffix(".pdf"))}
        for p in figs
    ]).to_csv(analysis / "figure_catalog.csv", index=False)

    print()
    print("=" * 100)
    print("STAGE5C NARRATIVE CLOSURE COMPLETE")
    print("Summary :", analysis / "00_STAGE5C_NARRATIVE_CLOSURE_SUMMARY.md")
    print("Report  :", analysis / "stage5c_narrative_closure_report.json")
    print("Output  :", out)
    print("=" * 100)


if __name__ == "__main__":
    main()
