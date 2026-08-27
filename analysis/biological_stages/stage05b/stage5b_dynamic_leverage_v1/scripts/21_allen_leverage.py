from pathlib import Path
import sys,json,traceback,re
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage5b.common import *
from stage5b.nwb_utils import *

CFG=read_yaml(Path(__file__).resolve().parents[1]/"config/default.yaml")
ROOT=Path(CFG["allen_root"]);OUT=ensure_dir(Path(CFG["output_root"])/"allen_vbo");CK=ensure_dir(OUT/"checkpoints")
inv_path=ROOT/"download_status"/"cohort_inventory.json"
j=json.loads(inv_path.read_text(encoding="utf-8"))
cohort={int(x["experiment_id"]):x for x in j["items"]}
meta_path=ROOT/"metadata"/"ophys_experiment_table.csv"
meta=pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
# Resolve experiment id column.
meta_id=None
for c in meta.columns:
    if str(c).lower()=="ophys_experiment_id":meta_id=c;break
if meta_id is None and len(meta):
    for c in meta.columns:
        z=pd.to_numeric(meta[c],errors="coerce")
        if (z>1e8).mean()>0.8: meta_id=c;break

def experiment_meta(eid):
    if meta_id is None:return {}
    z=meta[pd.to_numeric(meta[meta_id],errors="coerce")==eid]
    if len(z)==0:return {}
    r=z.iloc[0]
    keys=["ophys_container_id","cre_line","targeted_structure","imaging_depth","experience_level","passive","session_type"]
    return {k:r.get(k) for k in keys if k in z.columns}

def activity_substates(nwb,times):
    # Best-effort running/pupil median split. Failure is nonfatal.
    masks={}
    for label,kw in [("running",["running","speed"]),("pupil",["pupil"])]:
        objs=find_timeseries(nwb,kw)
        if not objs: continue
        obj=objs[0]
        try:
            data=np.asarray(obj.data[:],float).squeeze()
            if obj.timestamps is not None: ts=np.asarray(obj.timestamps[:],float)
            else: ts=float(obj.starting_time)+np.arange(len(data))/float(obj.rate)
            if data.ndim>1:data=np.nanmean(data,axis=tuple(range(1,data.ndim)))
            v=np.interp(times,ts,data,left=np.nan,right=np.nan)
            med=np.nanmedian(v)
            masks[f"{label}_low"]=np.isfinite(v)&(v<=med)
            masks[f"{label}_high"]=np.isfinite(v)&(v>med)
        except Exception:
            pass
    return masks

allrows=[]; diagnostics={}
for p in find_nwb(ROOT/"raw"):
    eid=int(p.stem.split("_")[-1]); item=cohort.get(eid,{})
    ck=CK/f"{eid}.done"; csvp=CK/f"{eid}_lev.csv"
    if ck.exists() and csvp.exists():
        allrows.append(pd.read_csv(csvp));print("SKIP",eid);continue
    print("PROCESS",eid,item.get("state"),flush=True)
    try:
        with open_nwb(p) as nwb:
            cands=find_roi_response_series(nwb)
            if not cands: raise ValueError("no RoiResponseSeries")
            _,series_name,series=cands[0]
            X,ts,cids,roi_tab=roi_series_matrix(series)
            # Keep only finite, meaningful cell ids.
            cid=pd.to_numeric(pd.Series(cids),errors="coerce").to_numpy()
            good=np.isfinite(cid)
            X=X[:,good];cid=cid[good].astype(np.int64)
            if X.shape[1]<3: raise ValueError("too few matched cell IDs")
            Xb,tb,cnt=bin_continuous(X,ts,CFG["bin_seconds_allen"])
            # Primary state = the experiment condition. Secondary substates arousal/running.
            primary=str(item.get("state","session"))
            masks={primary:np.ones(len(tb),bool)}
            masks.update(activity_substates(nwb,tb))
            dyn,diag=dynamics_leverage(Xb,masks,CFG["latent_components"],
                    CFG["dynamics_lag_bins"],CFG["ridge_alpha"],CFG["min_state_bins"],
                    CFG["random_seed"])
            diagnostics[str(eid)]={"roi_series":series_name,**diag}
            if not len(dyn): raise ValueError("no dynamical leverage")
            dyn["cell_specimen_id"]=[cid[i] for i in dyn["neuron_index"]]
            dyn["experiment_id"]=eid
            dyn["container_id"]=item.get("container_id")
            dyn["cre_line"]=item.get("cre_line")
            dyn["experiment_state"]=primary
            for k,v in experiment_meta(eid).items():dyn[k]=v
            # Rank primary leverage within experiment for cross-session comparability.
            pm=dyn["state"].astype(str)==primary
            dyn.loc[pm,"L_primary_rank"]=rank01(dyn.loc[pm,"L_dyn"])
            dyn.to_csv(csvp,index=False);allrows.append(dyn)
            ck.write_text("ok",encoding="utf-8")
    except Exception as e:
        diagnostics[str(eid)]={"status":"failed","error":repr(e),"traceback":traceback.format_exc()}
        print("FAILED",eid,repr(e))
if allrows:
    D=pd.concat(allrows,ignore_index=True)
    D.to_csv(OUT/"leverage_long.csv",index=False)
    P=D[D["state"].astype(str)==D["experiment_state"].astype(str)].copy()
    # guarantee rank exists for old checkpoints
    if "L_primary_rank" not in P.columns:P["L_primary_rank"]=P.groupby("experiment_id")["L_dyn"].transform(rank01)
    else:
        miss=P["L_primary_rank"].isna()
        if miss.any():
            P.loc[miss,"L_primary_rank"]=P.groupby("experiment_id")["L_dyn"].transform(rank01)[miss]
    # Same-cell, same-container across experiment states.
    counts=P.groupby(["container_id","cell_specimen_id"])["experiment_state"].nunique().rename("n_states").reset_index()
    same=counts[counts["n_states"]>=2].copy()
    same.to_csv(OUT/"same_cell_map.csv",index=False)
    Q=P.merge(same[["container_id","cell_specimen_id"]],on=["container_id","cell_specimen_id"],how="inner")
    rows=[]
    for key,g in Q.groupby(["container_id","cell_specimen_id"],dropna=False):
        vals=g["L_primary_rank"].dropna().to_numpy(float)
        states=sorted(g["experiment_state"].dropna().astype(str).unique())
        r={"container_id":key[0],"cell_specimen_id":key[1],
           "n_states":len(vals),"states":";".join(states),
           "mean_leverage":float(np.mean(vals)) if len(vals) else np.nan,
           "role_flexibility":float(np.var(vals)) if len(vals)>=2 else np.nan,
           "leverage_range":float(np.ptp(vals)) if len(vals)>=2 else np.nan,
           "mean_activity":float(g["mean_activity"].mean())}
        for c in ["cre_line","targeted_structure","imaging_depth"]:
            if c in g.columns:
                v=g[c].dropna()
                r[c]=v.iloc[0] if len(v) else None
        # state-specific ranks as columns when unique
        for s,gg in g.groupby("experiment_state"):
            if len(gg):r[f"L_rank__{s}"]=float(gg["L_primary_rank"].iloc[0])
        rows.append(r)
    F=pd.DataFrame(rows)
    F.to_csv(OUT/"role_flexibility.csv",index=False)
    # Basic control statistics.
    rho,pv,n=spearman_safe(F["role_flexibility"],F["mean_activity"])
    write_json(OUT/"control_statistics.json",
               {"flexibility_vs_mean_activity":{"spearman_rho":rho,"p":pv,"n":n},
                "matched_cells_2plus_states":int(len(F)),
                "matched_cells_3plus_states":int((F["n_states"]>=3).sum())})
write_json(OUT/"diagnostics.json",diagnostics)
print("Allen VBO leverage baseline complete")
