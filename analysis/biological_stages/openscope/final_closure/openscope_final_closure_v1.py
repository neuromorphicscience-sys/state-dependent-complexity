#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenScope Illusion 000248 — Final Closure v1
============================================

This is the last analysis-only closure after the frozen v2.3 formal run.

It does TWO things only:

A. High-precision fixed-split permutation null
   - 2000 permutations/session by default.
   - Headline endpoints only:
       R_decoder
       R_rho
     plus delta-Jaccard at essentially no extra fitting cost.
   - Reuses the exact frozen v2.3 estimator, seed schedule, state-shuffle rule,
     repeat-0 split/state definition, and C=1.
   - Existing v2.3 30-permutation session nulls are validated against the
     lightweight engine, then reused as permutations 0..29.
   - Resumable checkpoints per session.
   - Independent sessions can run in parallel.

B. Formal 12-mouse biological closure
   - 12-repeat mouse/session effect sizes.
   - bootstrap CIs.
   - exact sign-flip tests.
   - Wilcoxon signed-rank.
   - exact binomial sign tests.
   - standardized paired effect size dz.
   - leave-one-mouse-out stability.
   - matched primary-vs-real-edge boundary.
   - confound and estimator-validity audit.
   - high-precision hierarchical permutation inference.

NO new endpoint is introduced.
NO publication figure is generated.

Important permutation-matching correction
-----------------------------------------
The v2.3 null shuffles state labels while holding the repeat-0 discovery/final
split fixed. Therefore the permutation p-value in this closure is compared to
the group mean of the observed repeat-0 endpoint, not to the 12-repeat average.
The 12-repeat mouse/session average remains the formal effect-size estimate.
This makes the null and observed statistic exactly matched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from openscope_illusion import analysis as ana

VERSION = "OpenScope_FinalClosure_v1"
SEED = 20260821
DEFAULT_ROOT = Path.cwd()
DEFAULT_PERMUTATIONS = 2000
DEFAULT_WORKERS = 4
CHECKPOINT_EVERY = 25

HEADLINE_METRICS = {
    "decoder_auc_crossover": "decoder_auc_crossover_null",
    "delta_rho": "delta_rho_null",
    "delta_jaccard": "delta_jaccard_null",
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--v23-root", type=Path, default=None)
    ap.add_argument("--v23-analysis", type=Path, default=None,
                    help="Optional analysis/ directory override.")
    ap.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--force-null", action="store_true")
    ap.add_argument("--formal-only", action="store_true",
                    help="Build 12-mouse closure from existing outputs, without new nulls.")
    ap.add_argument("--self-test", action="store_true")
    return ap.parse_args()

def jdump(path: Path, obj):
    def conv(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=conv),
                    encoding="utf-8")

def load_npz_numeric(path: Path):
    required = ["baseline", "early", "late", "full"]
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in required if k not in z.files]
        if missing:
            raise RuntimeError(f"{path}: missing arrays {missing}")
        return {k: z[k] for k in required}

def resolve_v23(root: Path, override: Path | None):
    p = (override or (root / "plot" / "OpenScope_Illusion_Analysis_v2_3")).resolve()
    if not p.exists():
        raise FileNotFoundError(f"v2.3 root not found: {p}")
    return p

def resolve_analysis(v23_root: Path, override: Path | None):
    p = (override or (v23_root / "analysis")).resolve()
    required = [
        "session_primary_v23_summary.csv",
        "all_primary_v23_repeat_summary.csv",
        "session_real_edge_v23_control.csv",
        "primary_adjudication_v2_3.json",
    ]
    missing = [x for x in required if not (p / x).exists()]
    if missing:
        raise FileNotFoundError(f"v2.3 analysis missing: {missing} under {p}")
    return p

def cache_candidates(root: Path, subject: str, session: str, v23_root: Path):
    name = f"sub-{subject}_ses-{session}"
    return [
        v23_root / "cache" / name,
        root / "plot" / "OpenScope_Illusion_Analysis_v2_2" / "cache" / name,
        root / "plot" / "OpenScope_Illusion_Analysis_v2_1" / "cache" / name,
        root / "plot" / "OpenScope_Illusion_Analysis_v2" / "cache" / name,
    ]

def is_complete_cache(p: Path):
    return all((p / f).exists() for f in [
        "trials.csv", "units_primary.csv", "spike_counts.npz", "done.json"
    ])

def resolve_cache(root: Path, subject: str, session: str, v23_root: Path):
    for p in cache_candidates(root, subject, session, v23_root):
        if is_complete_cache(p):
            try:
                d = json.loads((p / "done.json").read_text(encoding="utf-8"))
                if d.get("status") == "COMPLETE":
                    return p.resolve()
            except Exception:
                pass
    raise FileNotFoundError(f"No complete cache for sub-{subject} ses-{session}")

def get_session_pairs(v23_analysis: Path):
    d = pd.read_csv(v23_analysis / "session_primary_v23_summary.csv")
    if len(d) != 12:
        raise RuntimeError(f"Expected 12 formal sessions, found {len(d)}")
    return [(str(int(r.subject)), str(int(r.session))) for r in d.itertuples()]

def build_repeat0_base(cache: Path):
    trials = pd.read_csv(cache / "trials.csv", low_memory=False)
    units = pd.read_csv(cache / "units_primary.csv", low_memory=False)
    mats = load_npz_numeric(cache / "spike_counts.npz")
    if mats["full"].shape[1] != len(units):
        raise RuntimeError(f"unit/matrix mismatch: {len(units)} vs {mats['full'].shape[1]}")
    cov = ana.activity_covariates(trials, units, mats, block="ICwcfg1")
    base = ana.formal_context_repeat(
        trials, units, mats, cov,
        block="ICwcfg1",
        positive_labels=("IC1", "IC2"),
        negative_labels=("LC1", "LC2"),
        state_mode="confound_controlled",
        repeat_seed=SEED,
        c_sensitivity=(ana.PRIMARY_C,),
        top_fracs=(ana.PRIMARY_TOP_FRACTION,),
        compute_same_image_imprint=False,
    )
    if base is None:
        raise RuntimeError("repeat-0 base reconstruction returned None")
    return trials, units, mats, base

def lightweight_perm_metrics(trials, units, mats, base, perm_index: int):
    """
    Exactly reproduces the v2.3 fixed-split null for the headline metrics
    without recomputing top-set CV, residualizations, or same-image analyses.
    """
    shuffled = ana.shuffle_states_within_exact(
        trials, base, seed=SEED + 10000 + int(perm_index)
    )

    t, req, idx, exact, y = ana._build_analysis_subset(
        trials, "ICwcfg1", ("IC1", "IC2"), ("LC1", "LC2")
    )
    discovery = np.asarray(base["split"]["discovery"], int)
    final = np.asarray(base["split"]["final"], int)
    full = np.asarray(mats["full"], float)

    dec = ana.decoder_transfer(
        full, y, exact, discovery, final, shuffled, req,
        C=ana.PRIMARY_C, seed=SEED,
    )
    if dec is None:
        raise RuntimeError(f"Permutation {perm_index}: decoder_transfer returned None")

    # Reproduce coefficient_landscape raw R_rho with the same seed schedule:
    # formal_context_repeat passes seed=repeat_seed+7000.
    landscape_seed = SEED + 7000
    vectors = {}
    for state in [0, 1]:
        d = discovery[shuffled[discovery] == state]
        A, B = ana.disjoint_balanced_halves(
            d, exact, req, landscape_seed + 500 + state
        )
        if len(A) < 12 or len(B) < 12:
            raise RuntimeError(
                f"Permutation {perm_index}: insufficient half trials state={state}"
            )
        for label, half_idx in [(f"{state}A", A), (f"{state}B", B)]:
            label_seed = {"0A":11, "0B":12, "1A":21, "1B":22}[label]
            model = ana.fit_decoder(
                full[half_idx], y[half_idx],
                C=ana.PRIMARY_C,
                seed=landscape_seed + 600 + label_seed,
            )
            if model is None:
                raise RuntimeError(
                    f"Permutation {perm_index}: fit_decoder failed for {label}"
                )
            vectors[label] = model.coef_abs

    sim = ana._landscape_similarity(
        vectors, top_frac=ana.PRIMARY_TOP_FRACTION
    )
    return {
        "perm_index": int(perm_index),
        "decoder_auc_crossover_null": float(dec["summary"]["decoder_auc_crossover"]),
        "delta_rho_null": float(sim["delta_rho"]),
        "delta_jaccard_null": float(sim["delta_jaccard"]),
    }

def existing_v23_null(v23_root: Path, subject: str, session: str):
    p = (v23_root / "session_results" /
         f"sub-{subject}_ses-{session}" /
         "primary_v23_state_shuffle_null.csv")
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    cols = ["perm_index", "decoder_auc_crossover_null",
            "delta_rho_null", "delta_jaccard_null"]
    if not set(cols).issubset(d.columns):
        return pd.DataFrame()
    return d[cols].copy()

def validate_lightweight_engine(trials, units, mats, base, old_null: pd.DataFrame):
    if len(old_null) == 0:
        return {"status":"NO_EXISTING_NULL_TO_VALIDATE"}
    # Validate three deterministic seeds if available.
    checks = []
    for pidx in [0, 1, 2]:
        row = old_null[old_null["perm_index"].eq(pidx)]
        if len(row) == 0:
            continue
        new = lightweight_perm_metrics(trials, units, mats, base, pidx)
        old = row.iloc[0]
        diffs = {
            k: abs(float(new[k]) - float(old[k]))
            for k in ["decoder_auc_crossover_null",
                      "delta_rho_null", "delta_jaccard_null"]
        }
        checks.append({"perm_index":pidx, **diffs})
        if max(diffs.values()) > 1e-10:
            raise RuntimeError(
                f"Lightweight null does not reproduce v2.3 at perm {pidx}: {diffs}"
            )
    return {"status":"EXACT_MATCH", "checks":checks}

def append_rows(path: Path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        df.to_csv(path, index=False, mode="a", header=False)
    else:
        df.to_csv(path, index=False)

def run_null_session(job):
    root = Path(job["root"])
    v23_root = Path(job["v23_root"])
    out = Path(job["out"])
    subject = job["subject"]
    session = job["session"]
    target = int(job["permutations"])
    checkpoint = int(job["checkpoint_every"])
    force = bool(job["force"])

    outdir = out / "null_sessions"
    outdir.mkdir(parents=True, exist_ok=True)
    outcsv = outdir / f"sub-{subject}_ses-{session}_high_precision_null.csv"
    audit_path = outdir / f"sub-{subject}_ses-{session}_null_audit.json"

    if force and outcsv.exists():
        outcsv.unlink()
    if force and audit_path.exists():
        audit_path.unlink()

    cache = resolve_cache(root, subject, session, v23_root)
    trials, units, mats, base = build_repeat0_base(cache)
    old = existing_v23_null(v23_root, subject, session)

    audit = validate_lightweight_engine(trials, units, mats, base, old)

    if outcsv.exists():
        cur = pd.read_csv(outcsv)
    else:
        cur = pd.DataFrame(columns=[
            "perm_index","decoder_auc_crossover_null",
            "delta_rho_null","delta_jaccard_null","source"
        ])

    # Seed the output with already-computed v2.3 permutations.
    have = set(pd.to_numeric(cur.get("perm_index", pd.Series(dtype=float)),
                             errors="coerce").dropna().astype(int).tolist())
    seed_rows = []
    if len(old):
        for r in old.itertuples():
            pidx = int(r.perm_index)
            if pidx >= target or pidx in have:
                continue
            seed_rows.append({
                "perm_index":pidx,
                "decoder_auc_crossover_null":float(r.decoder_auc_crossover_null),
                "delta_rho_null":float(r.delta_rho_null),
                "delta_jaccard_null":float(r.delta_jaccard_null),
                "source":"v2.3_existing",
            })
            have.add(pidx)
    append_rows(outcsv, seed_rows)

    pending = [p for p in range(target) if p not in have]
    batch = []
    t0 = time.time()
    for j, pidx in enumerate(pending, start=1):
        row = lightweight_perm_metrics(trials, units, mats, base, pidx)
        row["source"] = "final_closure"
        batch.append(row)
        if len(batch) >= checkpoint:
            append_rows(outcsv, batch)
            batch = []
        if j % max(100, checkpoint) == 0 or j == len(pending):
            elapsed = time.time() - t0
            print(
                f"[NULL sub-{subject}] {len(have)+j}/{target} "
                f"new={j}/{len(pending)} elapsed={elapsed/60:.1f} min",
                flush=True,
            )
    append_rows(outcsv, batch)

    final = pd.read_csv(outcsv).drop_duplicates("perm_index", keep="last")
    final = final.sort_values("perm_index")
    final = final[final["perm_index"].between(0, target-1)]
    final.to_csv(outcsv, index=False)

    missing = sorted(set(range(target)) - set(final["perm_index"].astype(int)))
    status = {
        "status":"COMPLETE" if not missing else "INCOMPLETE",
        "subject":subject,"session":session,
        "target_permutations":target,
        "completed_permutations":int(len(final)),
        "missing":missing[:50],
        "cache":str(cache),
        "validation":audit,
        "output":str(outcsv),
    }
    jdump(audit_path, status)
    if missing:
        raise RuntimeError(f"sub-{subject}: missing permutations {missing[:10]}")
    return status

def exact_signflip_p(effect, alternative="greater"):
    x = np.asarray(effect, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0: return np.nan
    obs = float(np.mean(x))
    # n=12 for this project -> exact 4096 sign configurations.
    vals = np.empty(1 << n, float)
    for mask in range(1 << n):
        signs = np.ones(n, float)
        for i in range(n):
            if (mask >> i) & 1:
                signs[i] = -1.0
        vals[mask] = np.mean(x * signs)
    if alternative == "greater":
        return float(np.mean(vals >= obs - 1e-15))
    return float(np.mean(np.abs(vals) >= abs(obs) - 1e-15))

def bootstrap_mean_ci(effect, nboot=50000, seed=SEED):
    x = np.asarray(effect, float)
    x = x[np.isfinite(x)]
    if len(x) == 0: return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(nboot, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.quantile(means, .025)),
            float(np.quantile(means, .975))]

def exact_binomial_sign_p(effect, alternative="greater"):
    x = np.asarray(effect, float)
    x = x[np.isfinite(x)]
    pos = int(np.sum(x > 0))
    neg = int(np.sum(x < 0))
    n = pos + neg
    if n == 0: return np.nan
    return float(stats.binomtest(pos, n, .5, alternative=alternative).pvalue)

def formal_inference(values, null=0.0, seed=SEED):
    raw = np.asarray(values, float)
    raw = raw[np.isfinite(raw)]
    effect = raw - float(null)
    n = len(effect)
    if n == 0:
        return {}
    try:
        w1 = float(stats.wilcoxon(effect, alternative="greater",
                                  zero_method="wilcox").pvalue)
        w2 = float(stats.wilcoxon(effect, alternative="two-sided",
                                  zero_method="wilcox").pvalue)
    except Exception:
        w1 = w2 = np.nan
    sd = float(np.std(effect, ddof=1)) if n > 1 else np.nan
    dz = float(np.mean(effect) / sd) if np.isfinite(sd) and sd > 1e-12 else np.nan
    return {
        "n_mouse_sessions":int(n),
        "null":float(null),
        "mean_raw":float(np.mean(raw)),
        "mean_effect":float(np.mean(effect)),
        "median_effect":float(np.median(effect)),
        "sd_effect":sd,
        "sem_effect":float(sd / np.sqrt(n)) if np.isfinite(sd) else np.nan,
        "bootstrap_95ci_mean_effect":bootstrap_mean_ci(effect, seed=seed),
        "min_effect":float(np.min(effect)),
        "max_effect":float(np.max(effect)),
        "positive_fraction":float(np.mean(effect > 0)),
        "n_positive":int(np.sum(effect > 0)),
        "n_negative":int(np.sum(effect < 0)),
        "exact_signflip_one_sided_p":exact_signflip_p(effect, "greater"),
        "exact_signflip_two_sided_p":exact_signflip_p(effect, "two-sided"),
        "wilcoxon_one_sided_p":w1,
        "wilcoxon_two_sided_p":w2,
        "exact_binomial_sign_one_sided_p":exact_binomial_sign_p(effect, "greater"),
        "paired_standardized_effect_dz":dz,
    }

def clopper_pearson(k, n, alpha=.05):
    if n <= 0: return [np.nan, np.nan]
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha/2, k, n-k+1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1-alpha/2, k+1, n-k))
    return [lo, hi]

def aggregate_high_precision_null(out: Path, pairs, repeat_summary: pd.DataFrame,
                                  target: int):
    tabs = []
    for subject, session in pairs:
        p = out / "null_sessions" / f"sub-{subject}_ses-{session}_high_precision_null.csv"
        if not p.exists():
            raise FileNotFoundError(f"High-precision null missing: {p}")
        d = pd.read_csv(p)
        d["subject"] = subject
        d["session"] = session
        d = d[d["perm_index"].between(0, target-1)]
        if d["perm_index"].nunique() != target:
            raise RuntimeError(
                f"sub-{subject}: expected {target} nulls, got {d['perm_index'].nunique()}"
            )
        tabs.append(d)
    N = pd.concat(tabs, ignore_index=True)
    A = out / "analysis"
    N.to_csv(A / "all_high_precision_session_null.csv", index=False)

    r0 = repeat_summary[pd.to_numeric(repeat_summary["repeat"], errors="coerce").eq(0)]
    if r0[["subject","session"]].drop_duplicates().shape[0] != 12:
        raise RuntimeError("repeat-0 observed table does not contain 12 sessions")

    summaries = {}
    global_rows = []
    for obs_col, null_col in HEADLINE_METRICS.items():
        obs0 = float(
            r0.groupby(["subject","session"], as_index=False)[obs_col]
              .mean()[obs_col].mean()
        )
        piv = N.pivot_table(
            index="perm_index", columns=["subject","session"],
            values=null_col, aggfunc="mean"
        )
        # Require every global draw to include all 12 mice.
        vals = piv.dropna(axis=0, how="any").mean(axis=1).to_numpy(float)
        B = len(vals)
        if B != target:
            raise RuntimeError(f"{obs_col}: expected {target} global draws, got {B}")
        k = int(np.sum(vals >= obs0))
        p_corr = float((k + 1) / (B + 1))
        p_raw = float(k / B)
        mcse = float(np.sqrt(max(p_raw*(1-p_raw), 1e-18) / B))
        null_sd = float(np.std(vals, ddof=1))
        z = float((obs0 - np.mean(vals))/null_sd) if null_sd > 1e-15 else np.nan
        summaries[obs_col] = {
            "matched_observed_repeat0_group_mean":obs0,
            "n_global_null_draws":int(B),
            "n_null_ge_observed":k,
            "empirical_one_sided_p_plus1":p_corr,
            "raw_exceedance_fraction":p_raw,
            "raw_exceedance_95ci_clopper_pearson":clopper_pearson(k, B),
            "monte_carlo_se_raw_p":mcse,
            "null_mean":float(np.mean(vals)),
            "null_sd":null_sd,
            "observed_vs_null_z":z,
            "null_q025":float(np.quantile(vals,.025)),
            "null_q975":float(np.quantile(vals,.975)),
        }
        for pidx, v in enumerate(vals):
            global_rows.append({
                "metric":obs_col,
                "perm_index":pidx,
                "global_null_mean":float(v),
                "matched_observed_repeat0_group_mean":obs0,
            })
    pd.DataFrame(global_rows).to_csv(
        A / "high_precision_hierarchical_null.csv", index=False
    )
    jdump(A / "high_precision_permutation_summary.json", summaries)
    return summaries

def leave_one_mouse_out(session: pd.DataFrame):
    endpoint_specs = {
        "decoder_auc_crossover":0.0,
        "delta_rho":0.0,
        "topset_auc_crossover":0.0,
        "same_image_state_auc_mean":0.5,
        "activity_residual_delta_rho":0.0,
        "activity_selectivity_residual_delta_rho":0.0,
    }
    rows = []
    for omit in session.itertuples():
        mask = ~(
            session["subject"].astype(str).eq(str(omit.subject)) &
            session["session"].astype(str).eq(str(omit.session))
        )
        keep = session[mask]
        for metric, null in endpoint_specs.items():
            eff = pd.to_numeric(keep[metric], errors="coerce") - null
            rows.append({
                "omitted_subject":str(omit.subject),
                "omitted_session":str(omit.session),
                "metric":metric,
                "n_remaining":int(eff.notna().sum()),
                "loo_mean_effect":float(eff.mean()),
                "loo_median_effect":float(eff.median()),
                "loo_positive_fraction":float((eff > 0).mean()),
            })
    return pd.DataFrame(rows)

def primary_real_edge_boundary(session: pd.DataFrame, ctrl: pd.DataFrame):
    d = session.merge(ctrl, on=["subject","session"], how="inner")
    specs = [
        ("decoder_auc_crossover","real_edge_decoder_auc_crossover","decoder_primary_minus_real_edge"),
        ("topset_auc_crossover","real_edge_topset_auc_crossover","topset_primary_minus_real_edge"),
        ("delta_rho","real_edge_delta_rho","rho_primary_minus_real_edge"),
    ]
    rows = []
    stats_out = {}
    for a,b,name in specs:
        d[name] = pd.to_numeric(d[a], errors="coerce") - pd.to_numeric(d[b], errors="coerce")
        inf = formal_inference(d[name].to_numpy(float), null=0.0, seed=SEED+101)
        stats_out[name] = inf
    keep = ["subject","session"] + [x[2] for x in specs]
    return d[keep], stats_out

def build_formal_closure(v23_analysis: Path, out: Path, high_null: dict | None):
    A = out / "analysis"
    A.mkdir(parents=True, exist_ok=True)

    session = pd.read_csv(v23_analysis / "session_primary_v23_summary.csv")
    repeats = pd.read_csv(v23_analysis / "all_primary_v23_repeat_summary.csv")
    ctrl = pd.read_csv(v23_analysis / "session_real_edge_v23_control.csv")
    reg = pd.read_csv(v23_analysis / "landscape_regularization_sensitivity_inference.csv")
    budget = pd.read_csv(v23_analysis / "topset_budget_sensitivity_inference.csv")
    conf = pd.read_csv(v23_analysis / "state_confound_control_comparison.csv")

    # Formal 12-mouse endpoint table.
    endpoint_table = session.copy()
    endpoint_table["decoder_home_minus_cross"] = (
        endpoint_table["decoder_auc_home"] - endpoint_table["decoder_auc_cross"]
    )
    endpoint_table["same_image_state_auc_excess"] = (
        endpoint_table["same_image_state_auc_mean"] - .5
    )
    endpoint_table["half_model_auc_excess"] = (
        endpoint_table["half_model_home_auc_mean"] - .5
    )
    endpoint_table.to_csv(A / "formal_12mouse_endpoint_table.csv", index=False)

    specs = {
        "decoder_state_transfer":{"column":"decoder_auc_crossover","null":0.0},
        "coefficient_landscape_reconfiguration":{"column":"delta_rho","null":0.0},
        "top20_unitset_transfer_supportive":{"column":"topset_auc_crossover","null":0.0},
        "top20_landscape_jaccard":{"column":"delta_jaccard","null":0.0},
        "activity_residual_landscape":{"column":"activity_residual_delta_rho","null":0.0},
        "activity_selectivity_residual_landscape":{"column":"activity_selectivity_residual_delta_rho","null":0.0},
        "same_exact_image_state_imprint":{"column":"same_image_state_auc_mean","null":0.5},
        "half_model_predictive_validity":{"column":"half_model_home_auc_mean","null":0.5},
    }
    stats_rows = []
    formal = {}
    for i,(name,spec) in enumerate(specs.items()):
        inf = formal_inference(
            pd.to_numeric(session[spec["column"]], errors="coerce").to_numpy(float),
            null=spec["null"], seed=SEED+1000+i
        )
        formal[name] = inf
        stats_rows.append({
            "endpoint":name,
            "column":spec["column"],
            **{k:v for k,v in inf.items() if not isinstance(v,list)},
            "bootstrap_ci_low":inf.get("bootstrap_95ci_mean_effect",[np.nan,np.nan])[0],
            "bootstrap_ci_high":inf.get("bootstrap_95ci_mean_effect",[np.nan,np.nan])[1],
        })
    pd.DataFrame(stats_rows).to_csv(A / "formal_12mouse_statistics.csv", index=False)

    loo = leave_one_mouse_out(session)
    loo.to_csv(A / "leave_one_mouse_out.csv", index=False)
    loo_summary = {}
    for metric,g in loo.groupby("metric"):
        vals = pd.to_numeric(g["loo_mean_effect"], errors="coerce").dropna().to_numpy(float)
        loo_summary[metric] = {
            "n_loo":int(len(vals)),
            "min_loo_mean_effect":float(np.min(vals)),
            "max_loo_mean_effect":float(np.max(vals)),
            "all_loo_means_positive":bool(np.all(vals > 0)),
            "positive_loo_fraction":float(np.mean(vals > 0)),
        }
    jdump(A / "leave_one_mouse_out_summary.json", loo_summary)

    boundary_table, boundary_stats = primary_real_edge_boundary(session, ctrl)
    boundary_table.to_csv(A / "primary_vs_real_edge_paired_boundary.csv", index=False)
    jdump(A / "primary_vs_real_edge_boundary_statistics.json", boundary_stats)

    confound = {
        "n_mouse_sessions":int(len(session)),
        "mean_abs_state_score_time_rho":float(
            pd.to_numeric(session["state_score_time_spearman"],errors="coerce").abs().mean()
        ),
        "max_abs_state_score_time_rho":float(
            pd.to_numeric(session["state_score_time_spearman"],errors="coerce").abs().max()
        ),
        "mean_abs_running_cohens_d":float(
            pd.to_numeric(session["running_state_cohens_d"],errors="coerce").abs().mean()
        ),
        "max_abs_running_cohens_d":float(
            pd.to_numeric(session["running_state_cohens_d"],errors="coerce").abs().max()
        ),
        "mean_state_stimulus_cramers_v":float(
            pd.to_numeric(session["state_stimulus_cramers_v"],errors="coerce").mean()
        ),
        "mean_time_absrho_reduction_vs_raw":float(
            pd.to_numeric(conf.get("time_absrho_reduction"),errors="coerce").mean()
        ) if "time_absrho_reduction" in conf else np.nan,
        "mean_running_absd_reduction_vs_raw":float(
            pd.to_numeric(conf.get("running_absd_reduction"),errors="coerce").mean()
        ) if "running_absd_reduction" in conf else np.nan,
    }

    # Formal 12-repeat mean vs matched repeat-0 permutation statistic are both recorded.
    matched_repeat0 = {}
    r0 = repeats[pd.to_numeric(repeats["repeat"],errors="coerce").eq(0)]
    for col in ["decoder_auc_crossover","delta_rho","delta_jaccard"]:
        matched_repeat0[col] = float(
            r0.groupby(["subject","session"],as_index=False)[col].mean()[col].mean()
        )

    # Objective closure label.
    dec = formal["decoder_state_transfer"]
    rho = formal["coefficient_landscape_reconfiguration"]
    same = formal["same_exact_image_state_imprint"]
    valid = formal["half_model_predictive_validity"]

    dec_perm = (high_null or {}).get("decoder_auc_crossover",{}).get(
        "empirical_one_sided_p_plus1", np.nan
    )
    rho_perm = (high_null or {}).get("delta_rho",{}).get(
        "empirical_one_sided_p_plus1", np.nan
    )
    dec_ci = dec.get("bootstrap_95ci_mean_effect",[np.nan,np.nan])
    rho_ci = rho.get("bootstrap_95ci_mean_effect",[np.nan,np.nan])

    if (
        np.isfinite(dec_perm) and dec_perm < .05 and
        np.isfinite(rho_perm) and rho_perm < .05 and
        dec_ci[0] > 0 and rho_ci[0] > 0 and
        same.get("bootstrap_95ci_mean_effect",[np.nan,np.nan])[0] > 0 and
        valid.get("mean_raw",0) > .55
    ):
        status = "STRONG_GENERAL_STATE_DEPENDENT_RECONFIGURATION"
    elif (
        np.isfinite(dec_perm) and dec_perm < .05 and
        dec_ci[0] > 0 and rho_ci[0] > 0 and
        rho.get("exact_signflip_one_sided_p",1) < .05
    ):
        status = "STRONG_DECODER_WITH_LANDSCAPE_PERMUTATION_BOUNDARY"
    elif dec_ci[0] > 0 and rho_ci[0] > 0:
        status = "SUPPORTED_WITH_PERMUTATION_BOUNDARY"
    else:
        status = "BOUNDARY_OR_MIXED"

    closure = {
        "status":status,
        "version":VERSION,
        "formal_12mouse":formal,
        "high_precision_permutation":high_null or {},
        "matched_repeat0_observed_for_permutation":matched_repeat0,
        "leave_one_mouse_out":loo_summary,
        "primary_vs_real_edge_boundary":boundary_stats,
        "confound_audit":confound,
        "regularization_sensitivity":reg.to_dict("records"),
        "budget_sensitivity":budget.to_dict("records"),
        "interpretation_guardrails":[
            "Mouse/session is the independent biological replicate (n=12).",
            "The 12-repeat session mean is the formal effect-size estimate.",
            "High-precision permutation p-values are matched to the repeat-0 fixed-split observed statistic because the null keeps that split fixed.",
            "R_decoder and R_rho are headline endpoints; top-20% unit-set transfer is supportive.",
            "Same-image state imprint establishes state-conditioned population representation under identical IC image, not by itself causal leverage.",
            "Real-edge effects bound specificity: a similar effect there supports a general state-dependent cortical coding principle rather than illusion-specificity.",
            "All leverage statements are predictive/functional, not causal.",
            "No new stimulus window, budget, state definition, or endpoint was selected after viewing v2.3 results.",
        ],
    }
    jdump(A / "FINAL_BIOLOGICAL_CLOSURE.json", closure)

    # Compact manuscript-ready report.
    def fmt(x, digits=5):
        try:
            if not np.isfinite(float(x)): return "NA"
            return f"{float(x):.{digits}g}"
        except Exception:
            return "NA"

    hp_dec = (high_null or {}).get("decoder_auc_crossover",{})
    hp_rho = (high_null or {}).get("delta_rho",{})
    report = [
        "# OpenScope Illusion 000248 — Final Biological Closure",
        "",
        f"**Closure status:** `{status}`",
        "",
        "## Frozen biological claim",
        "",
        "Prestimulus collective neural state reorganizes cortical population coding and the reproducible coefficient-defined functional-leverage landscape.",
        "",
        "## 12-mouse formal effects",
        "",
        f"- R_decoder (12-repeat session mean effect): {fmt(dec.get('mean_effect'))}; "
        f"bootstrap 95% CI [{fmt(dec_ci[0])}, {fmt(dec_ci[1])}]; "
        f"exact sign-flip one-sided p={fmt(dec.get('exact_signflip_one_sided_p'))}; "
        f"{dec.get('n_positive','NA')}/{dec.get('n_mouse_sessions','NA')} mice positive.",
        f"- R_rho = rho_within - rho_cross: {fmt(rho.get('mean_effect'))}; "
        f"bootstrap 95% CI [{fmt(rho_ci[0])}, {fmt(rho_ci[1])}]; "
        f"exact sign-flip one-sided p={fmt(rho.get('exact_signflip_one_sided_p'))}; "
        f"{rho.get('n_positive','NA')}/{rho.get('n_mouse_sessions','NA')} mice positive.",
        f"- Same-exact-image state imprint AUC: {fmt(same.get('mean_raw'))}; "
        f"excess above 0.5={fmt(same.get('mean_effect'))}.",
        f"- Half-model predictive-validity AUC: {fmt(valid.get('mean_raw'))}.",
        "",
        "## High-precision matched permutation",
        "",
        f"- R_decoder matched repeat-0 observed={fmt(hp_dec.get('matched_observed_repeat0_group_mean'))}; "
        f"B={hp_dec.get('n_global_null_draws','NA')}; "
        f"p={fmt(hp_dec.get('empirical_one_sided_p_plus1'))}.",
        f"- R_rho matched repeat-0 observed={fmt(hp_rho.get('matched_observed_repeat0_group_mean'))}; "
        f"B={hp_rho.get('n_global_null_draws','NA')}; "
        f"p={fmt(hp_rho.get('empirical_one_sided_p_plus1'))}.",
        "",
        "Permutation inference is deliberately matched to repeat 0 because the fixed-split null preserves the repeat-0 split. The 12-repeat mean above remains the effect-size estimate.",
        "",
        "## Leave-one-mouse-out stability",
        "",
        f"- R_decoder: all LOO means positive = {loo_summary.get('decoder_auc_crossover',{}).get('all_loo_means_positive','NA')}; "
        f"range [{fmt(loo_summary.get('decoder_auc_crossover',{}).get('min_loo_mean_effect'))}, "
        f"{fmt(loo_summary.get('decoder_auc_crossover',{}).get('max_loo_mean_effect'))}].",
        f"- R_rho: all LOO means positive = {loo_summary.get('delta_rho',{}).get('all_loo_means_positive','NA')}; "
        f"range [{fmt(loo_summary.get('delta_rho',{}).get('min_loo_mean_effect'))}, "
        f"{fmt(loo_summary.get('delta_rho',{}).get('max_loo_mean_effect'))}].",
        "",
        "## Specificity boundary",
        "",
        "The matched real-edge analysis is retained as a boundary condition. If primary-minus-real-edge is not positive, the defensible claim is a general collective-state-dependent cortical coding/leverage principle, not illusion-specific reconfiguration.",
        "",
        "## Confound audit",
        "",
        f"- mean |state-score vs recording-time rho| = {fmt(confound['mean_abs_state_score_time_rho'])}",
        f"- mean |running Cohen's d| = {fmt(confound['mean_abs_running_cohens_d'])}",
        f"- mean state-stimulus Cramer's V = {fmt(confound['mean_state_stimulus_cramers_v'])}",
        "",
        "## Final interpretation boundary",
        "",
        "This closure establishes predictive/functional state dependence, not causal influence. No new publication figures are generated here.",
    ]
    (A / "00_OPENSCOPE_FINAL_CLOSURE_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return closure

def self_test():
    """
    Synthetic test:
    1) build a v2.3 repeat-0 result;
    2) verify the lightweight permutation engine exactly matches the full
       v2.3 state_shuffle_null for several seeds.
    """
    rng = np.random.default_rng(123)
    n_each = 120
    labels = np.array(["IC1"]*n_each + ["IC2"]*n_each +
                      ["LC1"]*n_each + ["LC2"]*n_each)
    n = len(labels)
    nu = 100
    tm = np.linspace(0, 2000, n)
    running = np.exp(rng.normal(1.0, .4, n)) - 1
    true_state = rng.integers(0, 2, n)

    B = rng.normal(0, 1, (n, nu)).astype(np.float32)
    B[:, :15] += 1.0*(true_state[:,None]-.5)
    B[:, 15:30] += 0.5*((tm-tm.mean())/tm.std())[:,None]
    B[:, 30:45] += 0.3*np.log1p(running)[:,None]

    y = np.isin(labels, ["IC1","IC2"]).astype(float)
    F = rng.normal(0, 1, (n, nu)).astype(np.float32)
    F[:, 60:75] += 0.6*y[:,None]
    F[:, :20] += 1.2*y[:,None]*(true_state[:,None]==0)
    F[:, 20:40] += 1.2*y[:,None]*(true_state[:,None]==1)

    trials = pd.DataFrame({
        "trial_uid":np.arange(n),
        "stimulus_config":"ICwcfg1",
        "canonical_stimulus":labels,
        "stimulus_group":np.where(np.isin(labels,["IC1","IC2"]),"IC","LC"),
        "start_time":tm,
        "pre_running_speed":running,
    })
    areas = np.resize(np.array(["V1","LM","RL","AL","PM","AM"]), nu)
    units = pd.DataFrame({"unit_uid":[f"u{i}" for i in range(nu)],"area":areas})
    mats = {"baseline":B,"early":F,"late":F,"full":F}
    cov = ana.activity_covariates(trials, units, mats)
    base = ana.formal_context_repeat(
        trials, units, mats, cov,
        repeat_seed=SEED,
        c_sensitivity=(ana.PRIMARY_C,),
        top_fracs=(ana.PRIMARY_TOP_FRACTION,),
        compute_same_image_imprint=False,
    )
    if base is None:
        raise RuntimeError("SELF_TEST base failed")
    full = ana.state_shuffle_null(
        trials, units, mats, cov, base, n_perm=3, seed=SEED
    )
    for pidx in range(3):
        light = lightweight_perm_metrics(trials, units, mats, base, pidx)
        old = full[full["perm_index"].eq(pidx)].iloc[0]
        for col in ["decoder_auc_crossover_null","delta_rho_null","delta_jaccard_null"]:
            diff = abs(float(light[col]) - float(old[col]))
            if diff > 1e-10:
                raise RuntimeError(f"SELF_TEST mismatch p={pidx} {col}: {diff}")
    print("SELF_TEST lightweight null exactly reproduces frozen v2.3")
    print("SELF_TEST OK")

def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return

    if args.permutations < 100:
        raise ValueError("--permutations must be >=100 for the final closure")

    root = args.root.resolve()
    v23_root = resolve_v23(root, args.v23_root)
    v23_analysis = resolve_analysis(v23_root, args.v23_analysis)
    out = (args.out or (root / "plot" / "OpenScope_FinalClosure_v1")).resolve()
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "null_sessions").mkdir(parents=True, exist_ok=True)

    pairs = get_session_pairs(v23_analysis)

    # Freeze input provenance.
    source_manifest = {
        "version":VERSION,
        "root":str(root),
        "v23_root":str(v23_root),
        "v23_analysis":str(v23_analysis),
        "output":str(out),
        "n_sessions":len(pairs),
        "sessions":[{"subject":s,"session":se} for s,se in pairs],
        "target_permutations":int(args.permutations),
        "workers":int(args.workers),
        "formal_only":bool(args.formal_only),
        "permutation_matching_note":"Fixed-split null is compared to repeat-0 observed group mean; 12-repeat session mean is used for effect size.",
    }
    jdump(out / "analysis" / "run_manifest.json", source_manifest)

    high_null = None
    if not args.formal_only:
        print("="*108)
        print("OpenScope Final Closure v1 — HIGH-PRECISION NULL")
        print("Sessions:",len(pairs),"Permutations/session:",args.permutations,
              "Workers:",args.workers)
        print("v2.3 source:",v23_root)
        print("Output:",out)
        print("="*108)

        jobs = [{
            "root":str(root),
            "v23_root":str(v23_root),
            "out":str(out),
            "subject":s,
            "session":se,
            "permutations":int(args.permutations),
            "checkpoint_every":int(args.checkpoint_every),
            "force":bool(args.force_null),
        } for s,se in pairs]

        if args.workers <= 1:
            statuses = [run_null_session(j) for j in jobs]
        else:
            statuses = []
            with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs))) as ex:
                futs = {ex.submit(run_null_session,j):(j["subject"],j["session"]) for j in jobs}
                for fut in as_completed(futs):
                    key = futs[fut]
                    try:
                        st = fut.result()
                        statuses.append(st)
                        print(f"[DONE NULL] sub-{key[0]} ses-{key[1]} "
                              f"{st['completed_permutations']}/{st['target_permutations']}",
                              flush=True)
                    except Exception as e:
                        print(f"[FAILED NULL] sub-{key[0]} ses-{key[1]}: {e!r}",flush=True)
                        raise
        jdump(out / "analysis" / "null_session_status.json", statuses)

        repeat_summary = pd.read_csv(v23_analysis / "all_primary_v23_repeat_summary.csv")
        high_null = aggregate_high_precision_null(
            out, pairs, repeat_summary, int(args.permutations)
        )
    else:
        hp = out / "analysis" / "high_precision_permutation_summary.json"
        if hp.exists():
            high_null = json.loads(hp.read_text(encoding="utf-8"))

    closure = build_formal_closure(v23_analysis, out, high_null)

    print("="*108)
    print("FINAL CLOSURE COMPLETE")
    print("Status:",closure["status"])
    print("Read first:",out/"analysis"/"00_OPENSCOPE_FINAL_CLOSURE_REPORT.md")
    print("Machine-readable:",out/"analysis"/"FINAL_BIOLOGICAL_CLOSURE.json")
    print("No publication figures were generated.")
    print("="*108)

if __name__ == "__main__":
    main()
