from pathlib import Path
import sys,json,traceback
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage5b.common import *
from stage5b.nwb_utils import *

CFG=read_yaml(Path(__file__).resolve().parents[1]/"config/default.yaml")
ROOT=Path(CFG["steinmetz_root"]); OUT=ensure_dir(Path(CFG["output_root"])/"steinmetz")
CK=ensure_dir(OUT/"checkpoints")

def vec(df,names):
    c=candidate_col(df,names)
    return pd.to_numeric(df[c],errors="coerce").to_numpy() if c is not None else None,c

def state_masks_from_trials(times,trials,nwb):
    masks={}
    vis,cvis=vec(trials,["visual_stimulus_time","visualStim_times","visual_stimulus_times"])
    resp,cresp=vec(trials,["response_time","response_times"])
    fb,cfb=vec(trials,["feedback_time","feedback_times"])
    if vis is not None:
        masks["baseline"]=build_event_mask(times,vis,CFG["steinmetz_windows"]["baseline"])
        masks["visual"]=build_event_mask(times,vis,CFG["steinmetz_windows"]["visual"])
    if resp is not None:
        masks["decision"]=build_event_mask(times,resp,CFG["steinmetz_windows"]["decision"])
        masks["movement"]=build_event_mask(times,resp,CFG["steinmetz_windows"]["movement"])
    if fb is not None:
        masks["outcome"]=build_event_mask(times,fb,CFG["steinmetz_windows"]["outcome"])
    # Spontaneous intervals if present.
    try:
        sp=nwb.intervals.get("spontaneous")
        sdf=sp.to_dataframe()
        m=np.zeros(len(times),bool)
        for _,r in sdf.iterrows():
            m|=(times>=float(r["start_time"]))&(times<float(r["stop_time"]))
        masks["spontaneous"]=m
    except Exception:
        pass
    return {k:v for k,v in masks.items() if v.sum()>=CFG["min_state_bins"]}, \
           {"visual":cvis,"response":cresp,"feedback":cfb}

def trial_predictive_features(spikes,trials,state):
    # fixed event-aligned windows; rows=trials, cols=units
    if state in ("baseline","visual"):
        ev,_=vec(trials,["visual_stimulus_time","visualStim_times","visual_stimulus_times"])
        win=CFG["steinmetz_windows"][state]
    elif state in ("decision","movement"):
        ev,_=vec(trials,["response_time","response_times"]); win=CFG["steinmetz_windows"][state]
    else:
        ev,_=vec(trials,["feedback_time","feedback_times"]); win=CFG["steinmetz_windows"]["outcome"]
    if ev is None:return None
    X=np.zeros((len(ev),len(spikes)),np.float32)
    lo,hi=win
    for r,t in enumerate(ev):
        if not np.isfinite(t): X[r,:]=np.nan; continue
        for j,s in enumerate(spikes):
            X[r,j]=np.searchsorted(s,t+hi)-np.searchsorted(s,t+lo)
    return X

all_dyn=[]; all_pred=[]; diagnostics={}
for p in find_nwb(ROOT):
    sid=p.stem
    ck=CK/f"{sid}.done"
    dynfile=CK/f"{sid}_dyn.csv"
    predfile=CK/f"{sid}_pred.csv"
    if ck.exists() and dynfile.exists():
        print("SKIP",sid)
        all_dyn.append(pd.read_csv(dynfile))
        if predfile.exists(): all_pred.append(pd.read_csv(predfile))
        continue
    print("PROCESS",sid,flush=True)
    try:
        with open_nwb(p) as nwb:
            spikes,udf=unit_spike_times(nwb)
            trials=normalise_trial_table(table_df(getattr(nwb,"trials",None)))
            if len(spikes)<3 or len(trials)<5: raise ValueError("insufficient units/trials")
            # Determine safe session interval from trial times + spikes.
            vals=[]
            for c in trials.columns:
                if "time" in str(c).lower() or str(c).lower() in ("start_time","stop_time"):
                    a=pd.to_numeric(trials[c],errors="coerce").to_numpy()
                    vals.extend(a[np.isfinite(a)].tolist())
            if vals:
                t0=max(0.0,float(np.nanmin(vals))-2); t1=float(np.nanmax(vals))+2
            else:
                non=[s for s in spikes if len(s)]
                t0=min(s[0] for s in non);t1=max(s[-1] for s in non)
            X,times=bin_spikes(spikes,t0,t1,CFG["bin_seconds_steinmetz"])
            masks,mapping=state_masks_from_trials(times,trials,nwb)
            dyn,diag=dynamics_leverage(X,masks,CFG["latent_components"],
                CFG["dynamics_lag_bins"],CFG["ridge_alpha"],CFG["min_state_bins"],
                CFG["random_seed"])
            diag["trial_mapping"]=mapping; diagnostics[sid]=diag
            if len(dyn):
                dyn["session_id"]=sid
                dyn["neuron_id"]=udf.index.to_numpy()[dyn["neuron_index"].to_numpy()] if len(udf)==len(spikes) else dyn["neuron_index"]
                # best-effort static metadata
                for col in ["brain_region","location","quality","probe","cluster_id"]:
                    if col in udf.columns:
                        vals2=udf[col].to_numpy()
                        dyn[col]=[vals2[i] if i<len(vals2) else None for i in dyn["neuron_index"]]
                dyn.to_csv(dynfile,index=False);all_dyn.append(dyn)
            # Predictive leverage for available labels.
            pr=[]
            targets={
                "choice":["response_choice","choice"],
                "outcome":["feedback_type","feedback"],
                "engagement":["included","is_included"]
            }
            for state in ("baseline","visual","decision","movement","outcome"):
                Xt=trial_predictive_features(spikes,trials,state)
                if Xt is None: continue
                for target,names in targets.items():
                    y,c=vec(trials,names)
                    if y is None: continue
                    lev,info=predictive_ablation_logloss(Xt,y,random_state=CFG["random_seed"],
                                                         min_samples=CFG["min_trials_predictive"])
                    if lev is None: continue
                    for j,v in enumerate(lev):
                        pr.append({"session_id":sid,"neuron_index":j,
                                   "neuron_id":udf.index[j] if j<len(udf) else j,
                                   "state":state,"target":target,
                                   "L_pred":float(v),"full_logloss":info["full_logloss"]})
            pdf=pd.DataFrame(pr)
            if len(pdf): pdf.to_csv(predfile,index=False);all_pred.append(pdf)
            ck.write_text("ok",encoding="utf-8")
    except Exception as e:
        diagnostics[sid]={"status":"failed","error":repr(e),"traceback":traceback.format_exc()}
        print("FAILED",sid,repr(e))
if all_dyn:
    D=pd.concat(all_dyn,ignore_index=True)
    D.to_csv(OUT/"leverage_long.csv",index=False)
    F=summarize_flexibility(D,("session_id","neuron_id"),"L_dyn_rank")
    # controls
    ctrl=D.groupby(["session_id","neuron_id"],as_index=False).agg(mean_activity=("mean_activity","mean"),
                                                                  mean_raw_leverage=("L_dyn","mean"))
    F=F.merge(ctrl,on=["session_id","neuron_id"],how="left")
    F.to_csv(OUT/"role_flexibility.csv",index=False)
if all_pred:
    pd.concat(all_pred,ignore_index=True).to_csv(OUT/"predictive_leverage_long.csv",index=False)
write_json(OUT/"diagnostics.json",diagnostics)
print("Steinmetz leverage baseline complete")
