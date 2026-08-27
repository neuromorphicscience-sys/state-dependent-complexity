#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenScope Illusion 000248 — Formal Integrated Analysis v2.3 (NO FIGURES)

This is the one-shot formal run.  There is intentionally no --quick mode.

Default source:
D:/Research/Neural Science/bio data/openscope_illusion_000248

Default output:
D:/Research/Neural Science/plot/OpenScope_Illusion_Analysis_v2_3

Cache reuse
-----------
v2.3 automatically reuses a COMPLETE extraction cache from v2.2 or v2.1 when
available, so the 186-GB NWB dataset does not need to be re-extracted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

from openscope_illusion.io import load_inventory,build_session_manifest,extract_session
from openscope_illusion.analysis import (
    SEED,PRIMARY_TOP_FRACTION,PRIMARY_C,C_SENSITIVITY,TOP_FRACTIONS,
    activity_covariates,formal_context_repeat,state_shuffle_null,
)

DEFAULT_ROOT = Path.cwd()
VERSION="2.3"


def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    ap.add_argument("--data-root",type=Path,default=None)
    ap.add_argument("--out",type=Path,default=None)
    ap.add_argument("--repeats",type=int,default=12)
    ap.add_argument("--control-repeats",type=int,default=6)
    ap.add_argument("--raw-state-repeats",type=int,default=4)
    ap.add_argument("--null-permutations",type=int,default=30)
    ap.add_argument("--force-analysis",action="store_true")
    ap.add_argument("--force-extract",action="store_true")
    ap.add_argument("--self-test",action="store_true")
    return ap.parse_args()


def jdump(path,obj):
    def conv(x):
        if isinstance(x,np.integer): return int(x)
        if isinstance(x,np.floating): return float(x)
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=conv),encoding="utf-8")


def load_npz_numeric(path):
    required=["baseline","early","late","full"]
    with np.load(path,allow_pickle=False) as z:
        missing=[k for k in required if k not in z.files]
        if missing:
            raise RuntimeError(f"spike_counts.npz missing {missing}")
        return {k:z[k] for k in required}


def parameter_signature(args):
    payload={
        "version":VERSION,
        "repeats":args.repeats,
        "control_repeats":args.control_repeats,
        "raw_state_repeats":args.raw_state_repeats,
        "null_permutations":args.null_permutations,
        "primary_top_fraction":PRIMARY_TOP_FRACTION,
        "primary_C":PRIMARY_C,
        "C_sensitivity":list(C_SENSITIVITY),
        "top_fractions":list(TOP_FRACTIONS),
        "state_model":"baseline residual ~ cubic time + log1p running; global-mean removal; discovery PCA1",
    }
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]


def read_done(path):
    if not path.exists(): return None
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None


def is_complete_cache(path):
    d=read_done(path/"done.json")
    return bool(d and d.get("status")=="COMPLETE" and
                (path/"trials.csv").exists() and
                (path/"units_primary.csv").exists() and
                (path/"spike_counts.npz").exists())


def resolve_cache(root,out,subject,session):
    name=f"sub-{subject}_ses-{session}"
    candidates=[
        out/"cache"/name,
        root/"plot"/"OpenScope_Illusion_Analysis_v2_2"/"cache"/name,
        root/"plot"/"OpenScope_Illusion_Analysis_v2_1"/"cache"/name,
        root/"plot"/"OpenScope_Illusion_Analysis_v2"/"cache"/name,
    ]
    for p in candidates:
        if is_complete_cache(p):
            return p
    return out/"cache"/name


def append_csv(df,path):
    if df is None or len(df)==0:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        df.to_csv(path,index=False,mode="a",header=False)
    else:
        df.to_csv(path,index=False)


def existing_repeats(path):
    if not path.exists(): return set()
    try:
        d=pd.read_csv(path,usecols=["repeat"])
        return set(pd.to_numeric(d["repeat"],errors="coerce").dropna().astype(int).tolist())
    except Exception:
        return set()


def stage_log(result,stage,status,detail=""):
    path=result/"stage_progress.csv"
    row=pd.DataFrame([{
        "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage":stage,"status":status,"detail":detail,
    }])
    append_csv(row,path)


def save_repeat_result(rr,result,prefix,repeat):
    s=pd.DataFrame([{**rr["summary"],"repeat":repeat}])

    # Write detailed tables first and the repeat summary last.  The summary is
    # the checkpoint marker used on resume, so a partially written repeat will
    # never be mistaken for a completed one.
    for key,name in [
        ("decoder_transfer","decoder_transfer"),
        ("topset_transfer","topset_transfer"),
        ("topset_summary","topset_summary"),
        ("landscape_summary","landscape_summary"),
        ("landscape_units","landscape_unit_coefficients"),
        ("same_image_imprint","same_image_state_imprint"),
        ("state_scores","state_scores"),
    ]:
        d=rr.get(key)
        if d is not None and len(d):
            x=d.copy(); x["repeat"]=repeat
            append_csv(x,result/f"{prefix}_{name}.csv")
    append_csv(s,result/f"{prefix}_repeat_summary.csv")


def run_session(cache,result,args,sig,force=False):
    result.mkdir(parents=True,exist_ok=True)
    done=result/"done.json"
    old=read_done(done)
    if old and old.get("status")=="COMPLETE" and old.get("parameter_signature")==sig and not force:
        return old

    try:
        trials=pd.read_csv(cache/"trials.csv",low_memory=False)
        units=pd.read_csv(cache/"units_primary.csv",low_memory=False)
        mats=load_npz_numeric(cache/"spike_counts.npz")
        if len(units)!=mats["full"].shape[1]:
            raise RuntimeError(f"unit/matrix mismatch: {len(units)} vs {mats['full'].shape[1]}")
        cov=activity_covariates(trials,units,mats,block="ICwcfg1")
        cov.to_csv(result/"unit_activity_covariates.csv",index=False)

        if force:
            for pattern in ["primary_v23_*.csv", "raw_state_*.csv", "real_edge_v23_*.csv"]:
                for p in result.glob(pattern):
                    p.unlink()
            if (result/"stage_progress.csv").exists():
                (result/"stage_progress.csv").unlink()

        # ---------------- Primary confound-controlled repeats ----------------
        primary_summary=result/"primary_v23_repeat_summary.csv"
        completed=existing_repeats(primary_summary)
        stage_log(result,"primary","START",f"existing repeats={sorted(completed)}")
        for r in range(args.repeats):
            if r in completed:
                print(f"      primary repeat {r+1}/{args.repeats}: reuse")
                continue
            seed=SEED+1000*r
            print(f"      primary repeat {r+1}/{args.repeats}")
            rr=formal_context_repeat(
                trials,units,mats,cov,
                block="ICwcfg1",
                positive_labels=("IC1","IC2"),negative_labels=("LC1","LC2"),
                state_mode="confound_controlled",
                repeat_seed=seed,
                c_sensitivity=C_SENSITIVITY,
            )
            if rr is None:
                raise RuntimeError(f"primary repeat {r} returned None")
            save_repeat_result(rr,result,"primary_v23",r)
        stage_log(result,"primary","COMPLETE")

        P=pd.read_csv(primary_summary)
        # Exact formal-run completeness check: prevents a Quick/checkpoint mix-up.
        reps=sorted(pd.to_numeric(P["repeat"],errors="coerce").dropna().astype(int).unique().tolist())
        if reps!=list(range(args.repeats)):
            raise RuntimeError(f"Formal primary repeat set incomplete: {reps}")

        # ---------------- Raw-state comparator ----------------
        raw_summary=result/"raw_state_repeat_summary.csv"
        if force and raw_summary.exists(): raw_summary.unlink()
        raw_done=existing_repeats(raw_summary)
        stage_log(result,"raw_state_sensitivity","START")
        for r in range(args.raw_state_repeats):
            if r in raw_done: continue
            rr=formal_context_repeat(
                trials,units,mats,cov,
                block="ICwcfg1",
                positive_labels=("IC1","IC2"),negative_labels=("LC1","LC2"),
                state_mode="raw",
                repeat_seed=SEED+40000+1000*r,
                c_sensitivity=(PRIMARY_C,),
            )
            if rr is not None:
                save_repeat_result(rr,result,"raw_state",r)
        stage_log(result,"raw_state_sensitivity","COMPLETE")

        # ---------------- Matched real-edge control ----------------
        control_summary=result/"real_edge_v23_repeat_summary.csv"
        if force and control_summary.exists(): control_summary.unlink()
        ctrl_done=existing_repeats(control_summary)
        stage_log(result,"real_edge_control","START")
        for r in range(args.control_repeats):
            if r in ctrl_done: continue
            rr=formal_context_repeat(
                trials,units,mats,cov,
                block="ICwcfg1",
                positive_labels=("IRE1","IRE2"),negative_labels=("TRE1","TRE2"),
                state_mode="confound_controlled",
                repeat_seed=SEED+60000+1000*r,
                c_sensitivity=(PRIMARY_C,),
            )
            if rr is not None:
                save_repeat_result(rr,result,"real_edge_v23",r)
        stage_log(result,"real_edge_control","COMPLETE")

        # ---------------- Fixed-split state-shuffle null ----------------
        null_path=result/"primary_v23_state_shuffle_null.csv"
        stage_log(result,"state_shuffle_null","START")
        if args.null_permutations>0 and (force or not null_path.exists()):
            # Rebuild repeat 0 deterministically to preserve split/state exactly.
            base=formal_context_repeat(
                trials,units,mats,cov,
                block="ICwcfg1",
                positive_labels=("IC1","IC2"),negative_labels=("LC1","LC2"),
                state_mode="confound_controlled",
                repeat_seed=SEED,
                c_sensitivity=(PRIMARY_C,),
            )
            null=state_shuffle_null(
                trials,units,mats,cov,base,
                n_perm=args.null_permutations,seed=SEED,
            )
            null.to_csv(null_path,index=False)
        stage_log(result,"state_shuffle_null","COMPLETE")

        P=pd.read_csv(primary_summary)
        status={
            "status":"COMPLETE",
            "analysis_version":VERSION,
            "parameter_signature":sig,
            "cache_source":str(cache),
            "n_trials":int(len(trials)),
            "n_units":int(len(units)),
            "primary_repeats_completed":int(P["repeat"].nunique()),
            "decoder_auc_crossover_mean":float(P["decoder_auc_crossover"].mean()),
            "topset_auc_crossover_mean":float(P["topset_auc_crossover"].mean()),
            "delta_rho_mean":float(P["delta_rho"].mean()),
            "delta_jaccard_mean":float(P["delta_jaccard"].mean()),
            "same_image_state_auc_mean":float(P["same_image_state_auc_mean"].mean()),
            "state_score_time_absrho_mean":float(P["state_score_time_spearman"].abs().mean()),
        }
    except Exception as e:
        status={
            "status":"FAILED",
            "analysis_version":VERSION,
            "parameter_signature":sig,
            "cache_source":str(cache),
            "error":repr(e),
            "traceback":traceback.format_exc(),
        }
        print("      ERROR:",repr(e))
    jdump(done,status)
    return status


def collect(manifest,out,filename):
    parts=[]
    for r in manifest.itertuples():
        p=out/"session_results"/f"sub-{r.subject}_ses-{r.session}"/filename
        d=out/"session_results"/f"sub-{r.subject}_ses-{r.session}"/"done.json"
        if not p.exists() or not d.exists(): continue
        st=read_done(d)
        if not st or st.get("status")!="COMPLETE": continue
        x=pd.read_csv(p,low_memory=False)
        x["subject"]=str(r.subject); x["session"]=str(r.session)
        parts.append(x)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()


def exact_signflip(x,alternative="greater"):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0: return np.nan
    obs=float(np.mean(x)); n=len(x)
    if n<=20:
        vals=[]
        for mask in range(1<<n):
            signs=np.ones(n)
            for i in range(n):
                if (mask>>i)&1: signs[i]=-1
            vals.append(np.mean(x*signs))
        vals=np.asarray(vals,float)
    else:
        rng=np.random.default_rng(SEED)
        vals=np.asarray([np.mean(x*rng.choice([-1.,1.],n)) for _ in range(200000)])
    if alternative=="greater":
        return float(np.mean(vals>=obs-1e-15))
    return float(np.mean(np.abs(vals)>=abs(obs)-1e-15))


def bootstrap_ci(x,nboot=20000):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0: return [np.nan,np.nan]
    rng=np.random.default_rng(SEED)
    b=np.asarray([np.mean(rng.choice(x,len(x),replace=True)) for _ in range(nboot)])
    return [float(np.quantile(b,.025)),float(np.quantile(b,.975))]


def infer_session(df,col,null=0.0,alternative="greater"):
    if df is None or len(df)==0 or col not in df.columns: return {}
    x=pd.to_numeric(df[col],errors="coerce").dropna().to_numpy(float)-float(null)
    if len(x)==0: return {}
    try:
        pg=float(stats.wilcoxon(x,alternative=alternative,zero_method="wilcox").pvalue)
        p2=float(stats.wilcoxon(x,alternative="two-sided",zero_method="wilcox").pvalue)
    except Exception:
        pg=p2=np.nan
    ci=bootstrap_ci(x+null)
    return {
        "n_sessions":int(len(x)),
        "mean":float(np.mean(x+null)),
        "median":float(np.median(x+null)),
        "positive_fraction_vs_null":float(np.mean(x>0)),
        "null":float(null),
        "bootstrap_95ci_mean":ci,
        "exact_signflip_one_sided_p":exact_signflip(x,"greater" if alternative=="greater" else "two-sided"),
        "exact_signflip_two_sided_p":exact_signflip(x,"two-sided"),
        "wilcoxon_one_sided_p":pg,
        "wilcoxon_two_sided_p":p2,
    }


def aggregate(manifest,out,args,sig):
    A=out/"analysis"; A.mkdir(parents=True,exist_ok=True)
    P=collect(manifest,out,"primary_v23_repeat_summary.csv")
    D=collect(manifest,out,"primary_v23_decoder_transfer.csv")
    T=collect(manifest,out,"primary_v23_topset_transfer.csv")
    TS=collect(manifest,out,"primary_v23_topset_summary.csv")
    L=collect(manifest,out,"primary_v23_landscape_summary.csv")
    I=collect(manifest,out,"primary_v23_same_image_state_imprint.csv")
    RAW=collect(manifest,out,"raw_state_repeat_summary.csv")
    CTRL=collect(manifest,out,"real_edge_v23_repeat_summary.csv")
    N=collect(manifest,out,"primary_v23_state_shuffle_null.csv")

    for name,d in [
        ("all_primary_v23_repeat_summary.csv",P),
        ("all_primary_v23_decoder_transfer.csv",D),
        ("all_primary_v23_topset_transfer.csv",T),
        ("all_primary_v23_topset_summary.csv",TS),
        ("all_primary_v23_landscape_summary.csv",L),
        ("all_primary_v23_same_image_state_imprint.csv",I),
        ("all_raw_state_repeat_summary.csv",RAW),
        ("all_real_edge_v23_repeat_summary.csv",CTRL),
        ("all_primary_v23_state_shuffle_null.csv",N),
    ]:
        d.to_csv(A/name,index=False)

    if len(P):
        session=P.groupby(["subject","session"],as_index=False).agg(
            decoder_auc_crossover=("decoder_auc_crossover","mean"),
            decoder_auc_home=("decoder_auc_home","mean"),
            decoder_auc_cross=("decoder_auc_cross","mean"),
            topset_auc_crossover=("topset_auc_crossover","mean"),
            topset_auc_home=("topset_auc_home","mean"),
            topset_auc_cross=("topset_auc_cross","mean"),
            delta_rho=("delta_rho","mean"),
            rho_within=("rho_within","mean"),
            rho_cross=("rho_cross","mean"),
            delta_jaccard=("delta_jaccard","mean"),
            jaccard_within=("jaccard_within","mean"),
            jaccard_cross=("jaccard_cross","mean"),
            activity_residual_delta_rho=("activity_residual_delta_rho","mean"),
            activity_selectivity_residual_delta_rho=("activity_selectivity_residual_delta_rho","mean"),
            half_model_home_auc_mean=("half_model_home_auc_mean","mean"),
            same_image_state_auc_mean=("same_image_state_auc_mean","mean"),
            same_image_state_auc_excess_mean=("same_image_state_auc_excess_mean","mean"),
            state_stimulus_cramers_v=("state_stimulus_cramers_v","mean"),
            state_score_time_spearman=("state_score_time_spearman","mean"),
            running_state_cohens_d=("running_state_cohens_d","mean"),
            pc1_explained=("pc1_explained","mean"),
            n_units=("n_units","median"),
        )
    else:
        session=pd.DataFrame()
    session.to_csv(A/"session_primary_v23_summary.csv",index=False)

    # Budget sensitivity, averaging repeats within mouse/session first.
    if len(TS):
        budget_rows=[]
        for frac,g in TS.groupby("top_fraction"):
            sg=g.groupby(["subject","session"],as_index=False)["topset_auc_crossover"].mean()
            budget_rows.append({"top_fraction":float(frac),**infer_session(sg,"topset_auc_crossover")})
        budget=pd.DataFrame(budget_rows)
    else: budget=pd.DataFrame()
    budget.to_csv(A/"topset_budget_sensitivity_inference.csv",index=False)

    # Regularization sensitivity for R_rho.
    if len(L):
        reg_rows=[]
        for C,g in L.groupby("C"):
            sg=g.groupby(["subject","session"],as_index=False)["delta_rho"].mean()
            reg_rows.append({"C":float(C),**infer_session(sg,"delta_rho")})
        reg=pd.DataFrame(reg_rows)
    else: reg=pd.DataFrame()
    reg.to_csv(A/"landscape_regularization_sensitivity_inference.csv",index=False)

    if len(RAW):
        raw_session=RAW.groupby(["subject","session"],as_index=False).agg(
            raw_decoder_auc_crossover=("decoder_auc_crossover","mean"),
            raw_topset_auc_crossover=("topset_auc_crossover","mean"),
            raw_delta_rho=("delta_rho","mean"),
            raw_abs_time_rho=("state_score_time_spearman",lambda x:float(np.mean(np.abs(x)))),
            raw_running_abs_d=("running_state_cohens_d",lambda x:float(np.mean(np.abs(x)))),
        )
    else: raw_session=pd.DataFrame()
    raw_session.to_csv(A/"session_raw_state_sensitivity.csv",index=False)

    if len(session) and len(raw_session):
        confound_compare=session[["subject","session","state_score_time_spearman","running_state_cohens_d"]].merge(
            raw_session,on=["subject","session"],how="inner"
        )
        confound_compare["controlled_abs_time_rho"]=confound_compare["state_score_time_spearman"].abs()
        confound_compare["controlled_abs_running_d"]=confound_compare["running_state_cohens_d"].abs()
        confound_compare["time_absrho_reduction"]=confound_compare["raw_abs_time_rho"]-confound_compare["controlled_abs_time_rho"]
        confound_compare["running_absd_reduction"]=confound_compare["raw_running_abs_d"]-confound_compare["controlled_abs_running_d"]
    else:
        confound_compare=pd.DataFrame()
    confound_compare.to_csv(A/"state_confound_control_comparison.csv",index=False)

    if len(CTRL):
        ctrl_session=CTRL.groupby(["subject","session"],as_index=False).agg(
            real_edge_decoder_auc_crossover=("decoder_auc_crossover","mean"),
            real_edge_topset_auc_crossover=("topset_auc_crossover","mean"),
            real_edge_delta_rho=("delta_rho","mean"),
        )
    else: ctrl_session=pd.DataFrame()
    ctrl_session.to_csv(A/"session_real_edge_v23_control.csv",index=False)

    # Hierarchical fixed-split permutation null: average same permutation across sessions.
    null_summary={}
    if len(N) and len(session):
        null_metrics={
            "decoder_auc_crossover":"decoder_auc_crossover_null",
            "topset_auc_crossover":"topset_auc_crossover_null",
            "delta_rho":"delta_rho_null",
            "delta_jaccard":"delta_jaccard_null",
        }
        global_rows=[]
        for metric,ncol in null_metrics.items():
            piv=N.pivot_table(index="perm_index",columns=["subject","session"],values=ncol,aggfunc="mean")
            vals=piv.mean(axis=1).dropna().to_numpy(float)
            obs=float(session[metric].mean())
            p=float((1+np.sum(vals>=obs))/(len(vals)+1)) if len(vals) else np.nan
            null_summary[metric]={
                "observed_session_mean":obs,
                "n_global_null_draws":int(len(vals)),
                "null_mean":float(np.mean(vals)) if len(vals) else np.nan,
                "empirical_one_sided_p":p,
            }
            for i,v in enumerate(vals):
                global_rows.append({"metric":metric,"perm_index":i,"global_null_mean":v})
        pd.DataFrame(global_rows).to_csv(A/"hierarchical_v23_state_shuffle_null.csv",index=False)

    primary={
        "decoder_state_transfer":infer_session(session,"decoder_auc_crossover"),
        "coefficient_selected_top20_transfer":infer_session(session,"topset_auc_crossover"),
        "sampling_normalized_landscape_reconfiguration":infer_session(session,"delta_rho"),
        "top20_landscape_jaccard_reconfiguration":infer_session(session,"delta_jaccard"),
        "activity_residual_landscape_reconfiguration":infer_session(session,"activity_residual_delta_rho"),
        "same_exact_image_state_imprint":infer_session(session,"same_image_state_auc_mean",null=.5),
        "half_model_predictive_validity_auc":infer_session(session,"half_model_home_auc_mean",null=.5),
        "raw_state_sensitivity":{
            "decoder":infer_session(raw_session,"raw_decoder_auc_crossover"),
            "topset":infer_session(raw_session,"raw_topset_auc_crossover"),
            "landscape":infer_session(raw_session,"raw_delta_rho"),
        },
        "real_edge_control":{
            "decoder":infer_session(ctrl_session,"real_edge_decoder_auc_crossover"),
            "topset":infer_session(ctrl_session,"real_edge_topset_auc_crossover"),
            "landscape":infer_session(ctrl_session,"real_edge_delta_rho"),
        },
        "budget_sensitivity":budget.to_dict("records"),
        "regularization_sensitivity":reg.to_dict("records"),
        "state_shuffle_null":null_summary,
    }

    # Objective adjudication. No p-value fishing: these criteria are descriptive gates.
    strong=False; moderate=False
    try:
        d=primary["decoder_state_transfer"]
        t=primary["coefficient_selected_top20_transfer"]
        r=primary["sampling_normalized_landscape_reconfiguration"]
        valid=primary["half_model_predictive_validity_auc"]
        same=primary["same_exact_image_state_imprint"]
        strong=(
            d.get("mean",0)>0 and t.get("mean",0)>0 and r.get("mean",0)>0 and
            r.get("positive_fraction_vs_null",0)>=0.75 and
            valid.get("mean",0)>.55 and same.get("mean",0)>.5
        )
        moderate=(r.get("mean",0)>0 and valid.get("mean",0)>.55)
    except Exception:
        pass
    label="STRONG" if strong else ("MODERATE" if moderate else "BOUNDARY_OR_NEGATIVE")

    adjudication={
        "status":label,
        "formal_version":VERSION,
        "parameter_signature":sig,
        "primary":primary,
        "state_confound_audit":{
            "mean_abs_state_score_time_rho":float(session["state_score_time_spearman"].abs().mean()) if len(session) else np.nan,
            "mean_abs_running_cohens_d":float(session["running_state_cohens_d"].abs().mean()) if len(session) else np.nan,
            "mean_stimulus_cramers_v":float(session["state_stimulus_cramers_v"].mean()) if len(session) else np.nan,
            "mean_time_absrho_reduction_vs_raw":float(confound_compare["time_absrho_reduction"].mean()) if len(confound_compare) else np.nan,
            "mean_running_absd_reduction_vs_raw":float(confound_compare["running_absd_reduction"].mean()) if len(confound_compare) else np.nan,
        },
        "decision_rule":{
            "STRONG":"Positive full-decoder transfer, positive coefficient-selected top20 transfer, positive sampling-normalized R_rho across mice, valid split-half decoders, and same-exact-image state imprint above chance after confound-controlled state definition.",
            "MODERATE":"Reliable positive sampling-normalized landscape reconfiguration with valid decoders, but transfer and/or same-image evidence is incomplete.",
            "BOUNDARY_OR_NEGATIVE":"Formal held-out transfer/landscape endpoints fail to show state-specific structure beyond sampling variability.",
        },
        "methodological_boundary":[
            "The v2.2 zero-out ranking is archived but not primary because prior split reliability was inadequate.",
            "The v2.2 cross-area Ridge bridge is archived but not primary because held-out R^2 was negative.",
            "All v2.3 state parameters and feature rankings are discovery-fit; final trials are untouched until evaluation.",
            "Mouse/session is the only independent biological replicate.",
            "Predictive decoder coefficients are a functional-leverage proxy, not causal influence.",
        ],
    }
    jdump(A/"primary_adjudication_v2_3.json",adjudication)

    report=[
        "# OpenScope Illusion 000248 — Formal Integrated Analysis v2.3",
        "",
        f"**Adjudication: {label}**",
        "",
        "## Primary state-specific decoder transfer",
        "```json",json.dumps(primary["decoder_state_transfer"],ensure_ascii=False,indent=2),"```",
        "",
        "## Coefficient-selected top-20% unit-set transfer",
        "```json",json.dumps(primary["coefficient_selected_top20_transfer"],ensure_ascii=False,indent=2),"```",
        "",
        "## Sampling-normalized coefficient-landscape reconfiguration R_rho",
        "```json",json.dumps(primary["sampling_normalized_landscape_reconfiguration"],ensure_ascii=False,indent=2),"```",
        "",
        "## Same exact-image poststimulus state imprint",
        "```json",json.dumps(primary["same_exact_image_state_imprint"],ensure_ascii=False,indent=2),"```",
        "",
        "## Split-half decoder validity",
        "```json",json.dumps(primary["half_model_predictive_validity_auc"],ensure_ascii=False,indent=2),"```",
        "",
        "## Confound audit",
        "```json",json.dumps(adjudication["state_confound_audit"],ensure_ascii=False,indent=2),"```",
        "",
        "## Interpretation boundary",
        "v2.3 tests state-dependent predictive/functional leverage beyond estimator sampling variability. It does not establish causal influence.",
        "",
        "No publication figures are generated in this formal pass.",
    ]
    (A/"00_OPENSCOPE_V2_3_FINAL_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    return adjudication


def self_test():
    """Synthetic end-to-end test with state-specific informative feature sets."""
    from openscope_illusion.analysis import activity_covariates,formal_context_repeat
    rng=np.random.default_rng(123)
    n_each=220
    labels=np.array(["IC1"]*n_each+["IC2"]*n_each+["LC1"]*n_each+["LC2"]*n_each)
    n=len(labels); nu=180
    time=np.linspace(0,4000,n)
    running=np.exp(rng.normal(1.0,.5,n))-1
    true_state=rng.integers(0,2,n)

    # Baseline state plus strong time/running nuisance.
    B=rng.normal(0,1,(n,nu)).astype(np.float32)
    B[:,0:20]+=1.2*(true_state[:,None]-.5)
    B[:,20:40]+=0.6*((time-time.mean())/time.std())[:,None]
    B[:,40:60]+=0.4*np.log1p(running)[:,None]

    y=np.isin(labels,["IC1","IC2"]).astype(float)
    F=rng.normal(0,1,(n,nu)).astype(np.float32)
    # shared decoder backbone
    F[:,100:120]+=0.7*y[:,None]
    # state-specific informative units
    F[:,0:30]+=1.6*y[:,None]*(true_state[:,None]==0)
    F[:,30:60]+=1.6*y[:,None]*(true_state[:,None]==1)
    # make state persist into response even for identical image
    F[:,70:90]+=0.6*(true_state[:,None]-.5)

    trials=pd.DataFrame({
        "trial_uid":np.arange(n),
        "stimulus_config":"ICwcfg1",
        "canonical_stimulus":labels,
        "stimulus_group":np.where(np.isin(labels,["IC1","IC2"]),"IC","LC"),
        "start_time":time,
        "pre_running_speed":running,
    })
    areas=np.resize(np.array(["V1","LM","RL","AL","PM","AM"]),nu)
    units=pd.DataFrame({"unit_uid":[f"u{i}" for i in range(nu)],"area":areas})
    mats={"baseline":B,"early":F,"late":F,"full":F}
    cov=activity_covariates(trials,units,mats)
    rr=formal_context_repeat(trials,units,mats,cov,repeat_seed=777)
    if rr is None: raise RuntimeError("SELF_TEST returned None")
    s=rr["summary"]
    print("SELF_TEST decoder_auc_crossover",s["decoder_auc_crossover"])
    print("SELF_TEST topset_auc_crossover",s["topset_auc_crossover"])
    print("SELF_TEST delta_rho",s["delta_rho"])
    print("SELF_TEST half_model_home_auc_mean",s["half_model_home_auc_mean"])
    assert np.isfinite(s["delta_rho"])
    assert s["half_model_home_auc_mean"]>.5
    print("SELF_TEST OK")


def main():
    args=parse_args()
    if args.self_test:
        self_test(); return

    root=args.root.resolve()
    data=(args.data_root or root/"bio data"/"openscope_illusion_000248").resolve()
    out=(args.out or root/"plot"/"OpenScope_Illusion_Analysis_v2_3").resolve()
    sig=parameter_signature(args)
    for p in [out,out/"cache",out/"session_results",out/"analysis"]:
        p.mkdir(parents=True,exist_ok=True)

    print("="*104)
    print("OpenScope Illusion 000248 — FORMAL Integrated Analysis v2.3 (NO FIGURES)")
    print("Data:",data)
    print("Out :",out)
    print("Primary repeats:",args.repeats,
          "Real-edge controls:",args.control_repeats,
          "Raw-state sensitivity:",args.raw_state_repeats,
          "Null/session:",args.null_permutations)
    print("Parameter signature:",sig)
    print("="*104)

    inv=load_inventory(data)
    inv.to_csv(out/"analysis"/"local_asset_inventory.csv",index=False)
    manifest=build_session_manifest(inv)
    manifest.to_csv(out/"analysis"/"session_manifest.csv",index=False)

    # Cache resolution / extraction. The OGEN-based v2.1/v2.2 cache is valid.
    cache_rows=[]
    cache_map={}
    for i,row in manifest.iterrows():
        sub=str(row["subject"]); ses=str(row["session"])
        cache=resolve_cache(root,out,sub,ses)
        if is_complete_cache(cache) and not args.force_extract:
            status="REUSED_COMPLETE_CACHE"
        else:
            print(f"[EXTRACT {i+1}/{len(manifest)}] sub-{sub} ses-{ses}")
            st=extract_session(row,out/"cache",force=args.force_extract)
            cache=out/"cache"/f"sub-{sub}_ses-{ses}"
            status=st.get("status")
            if status!="COMPLETE":
                print("   ERROR:",st.get("error"))
        cache_map[(sub,ses)]=cache
        cache_rows.append({"subject":sub,"session":ses,"cache_source":str(cache),"status":status})
    pd.DataFrame(cache_rows).to_csv(out/"analysis"/"cache_resolution.csv",index=False)

    statuses=[]
    for i,row in manifest.iterrows():
        sub=str(row["subject"]); ses=str(row["session"])
        cache=cache_map[(sub,ses)]
        if not is_complete_cache(cache):
            statuses.append({"subject":sub,"session":ses,"status":"SKIPPED_NO_COMPLETE_CACHE"})
            continue
        print(f"[ANALYZE {i+1}/{len(manifest)}] sub-{sub} ses-{ses}")
        result=out/"session_results"/f"sub-{sub}_ses-{ses}"
        st=run_session(cache,result,args,sig,force=args.force_analysis)
        statuses.append({"subject":sub,"session":ses,**st})
        print("   ",st.get("status"),
              "Rdec",st.get("decoder_auc_crossover_mean"),
              "Rtop",st.get("topset_auc_crossover_mean"),
              "Rrho",st.get("delta_rho_mean"))
    pd.DataFrame(statuses).to_csv(out/"analysis"/"analysis_status.csv",index=False)

    complete=sum(1 for s in statuses if s.get("status")=="COMPLETE")
    if complete==0:
        raise RuntimeError("No session completed v2.3 analysis; inspect analysis_status.csv")

    adj=aggregate(manifest,out,args,sig)
    manifest_out={
        "status":"COMPLETE",
        "formal_version":VERSION,
        "parameter_signature":sig,
        "created":time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root":str(data),"output_root":str(out),
        "n_sessions_complete":complete,
        "parameters":{
            "primary_block":"ICwcfg1",
            "primary_labels":["IC1","IC2","LC1","LC2"],
            "prestim_window_s":[-.30,0],
            "full_response_window_s":[0,.40],
            "discovery_fraction":.60,
            "state_confound_model":"1+t+t^2+t^3+log1p(running), discovery-fit",
            "global_population_gain_removed_before_state_PCA":True,
            "primary_decoder_C":PRIMARY_C,
            "regularization_sensitivity":list(C_SENSITIVITY),
            "primary_top_fraction":PRIMARY_TOP_FRACTION,
            "top_fraction_sensitivity":list(TOP_FRACTIONS),
            "primary_repeats":args.repeats,
            "control_repeats":args.control_repeats,
            "raw_state_repeats":args.raw_state_repeats,
            "null_permutations_per_session":args.null_permutations,
        },
        "adjudication":adj,
    }
    jdump(out/"analysis"/"run_manifest_v2_3.json",manifest_out)

    print("="*104)
    print("FORMAL v2.3 COMPLETE")
    print("Read first:",out/"analysis"/"00_OPENSCOPE_V2_3_FINAL_REPORT.md")
    print("Adjudication:",out/"analysis"/"primary_adjudication_v2_3.json")
    print("No publication figures were generated.")
    print("="*104)


if __name__=="__main__":
    main()
