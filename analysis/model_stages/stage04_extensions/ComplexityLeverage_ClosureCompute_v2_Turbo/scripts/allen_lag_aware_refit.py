from __future__ import annotations
import argparse, json, math, os, re, sys, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from common import *

PRIMARY_STATES=["familiar_active","novel_active","passive"]


def infer_state(meta_row):
    passive=meta_row.get("passive",False)
    try:
        if bool(passive) and str(passive).lower() not in ["false","0","nan","none"]: return "passive"
    except Exception: pass
    s=" ".join(str(meta_row.get(k,"")) for k in ["experience_level","session_type"]).lower()
    if "passive" in s: return "passive"
    if "novel" in s: return "novel_active"
    if "familiar" in s: return "familiar_active"
    return "unknown"


def load_cohort(allen_root: Path):
    inv=allen_root/"download_status"/"cohort_inventory.json"
    if inv.exists():
        j=json.loads(inv.read_text(encoding="utf-8")); items=j.get("items",j if isinstance(j,list) else [])
        rows=[]
        for x in items:
            rows.append({"experiment_id":int(x["experiment_id"]),"container_id":x.get("container_id"),"state":x.get("state"),"cre_line":x.get("cre_line")})
        d=pd.DataFrame(rows).drop_duplicates("experiment_id")
        if len(d): return d
    meta_path=allen_root/"metadata"/"ophys_experiment_table.csv"
    if not meta_path.exists(): raise FileNotFoundError(f"Need {inv} or {meta_path}")
    m=pd.read_csv(meta_path)
    idc=next((c for c in m.columns if str(c).lower()=="ophys_experiment_id"),None)
    if idc is None: raise RuntimeError("Cannot find ophys_experiment_id in metadata")
    rows=[]
    for _,r in m.iterrows():
        try: eid=int(r[idc])
        except Exception: continue
        st=infer_state(r)
        if st not in PRIMARY_STATES: continue
        rows.append({"experiment_id":eid,"container_id":r.get("ophys_container_id"),"state":st,"cre_line":r.get("cre_line")})
    return pd.DataFrame(rows).drop_duplicates("experiment_id")


def find_raw_files(allen_root: Path, cohort):
    alln=list((allen_root/"raw").rglob("*.nwb"))
    by={}
    for p in alln:
        m=re.search(r"(\d{8,12})(?=\.nwb$)",p.name)
        if not m: continue
        eid=int(m.group(1)); by.setdefault(eid,[]).append(p)
    rows=[]
    for _,r in cohort.iterrows():
        eid=int(r.experiment_id); c=by.get(eid,[])
        if not c: continue
        # duplicate aliases: shortest path then largest file, deterministic
        c=sorted(c,key=lambda p:(len(str(p)), -p.stat().st_size, str(p)))
        rows.append({**r.to_dict(),"path":str(c[0]),"alias_count":len(c),"bytes":c[0].stat().st_size})
    return pd.DataFrame(rows)


def roi_candidates(nwb):
    from pynwb.ophys import RoiResponseSeries
    out=[]
    for obj in nwb.objects.values():
        if isinstance(obj,RoiResponseSeries):
            nm=str(getattr(obj,"name","")).lower(); sc=(100 if "event" in nm else 0)+(50 if "dff" in nm else 0)
            out.append((sc,nm,obj))
    return sorted(out,key=lambda z:(-z[0],z[1]))


def roi_matrix(series):
    data=np.asarray(series.data[:],dtype=np.float32)
    if series.timestamps is not None: ts=np.asarray(series.timestamps[:],float)
    else:
        rate=float(series.rate); start=float(series.starting_time); ntime=max(data.shape); ts=start+np.arange(ntime)/rate
    try:
        ridx=np.asarray(series.rois.data[:],int); tab=series.rois.table.to_dataframe(); sub=tab.iloc[ridx].copy()
    except Exception: sub=pd.DataFrame(index=np.arange(min(data.shape)))
    if data.ndim!=2: raise ValueError(f"RoiResponseSeries must be 2D, got {data.shape}")
    if data.shape[0]==len(ts): X=data
    elif data.shape[1]==len(ts): X=data.T
    else:
        X=data if data.shape[0]>=data.shape[1] else data.T;ts=ts[:X.shape[0]]
    ids=None
    for c in sub.columns:
        if str(c).lower()=="cell_specimen_id": ids=pd.to_numeric(sub[c],errors="coerce").to_numpy();break
    if ids is None:
        ids=pd.to_numeric(pd.Series(sub.index),errors="coerce").to_numpy()
    return X,ts,ids


def bin_continuous(X,ts,bin_s):
    X=np.asarray(X,np.float32);ts=np.asarray(ts,float);m=np.isfinite(ts);X=X[m];ts=ts[m];o=np.argsort(ts);X=X[o];ts=ts[o]
    edges=np.arange(float(ts[0]),float(ts[-1])+bin_s,bin_s)
    if len(edges)<10: raise ValueError("too few bins")
    bid=np.searchsorted(edges,ts,side="right")-1;good=(bid>=0)&(bid<len(edges)-1);bid=bid[good];X=X[good]
    out=np.zeros((len(edges)-1,X.shape[1]),np.float32);cnt=np.bincount(bid,minlength=len(edges)-1).astype(np.float32)
    for j in range(X.shape[1]): np.add.at(out[:,j],bid,X[:,j])
    return out,(edges[:-1]+edges[1:])/2,cnt


def r2vw(y,p):
    from sklearn.metrics import r2_score
    return float(r2_score(y,p,multioutput="variance_weighted"))


def fit_select_lag(X,bin_s,lag_seconds,alphas,n_components=20,seed=0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    X=np.asarray(X,np.float32)
    good=np.isfinite(X).all(0)&(np.nanstd(X,axis=0)>1e-8)
    if good.sum()<3 or X.shape[0]<200: return None,{"status":"insufficient_matrix","bins":int(X.shape[0]),"neurons":int(good.sum())}
    Xg=X[:,good]; T=len(Xg); tr_end=int(T*.60); va_end=int(T*.80)
    if tr_end<100 or T-va_end<30: return None,{"status":"too_short_for_blocked_nested_split","bins":T}
    mu=Xg[:tr_end].mean(0);sd=Xg[:tr_end].std(0);sd[sd<1e-8]=1;Xz=(Xg-mu)/sd
    k=max(2,min(int(n_components),Xz.shape[1]-1,tr_end-2))
    pca=PCA(n_components=k,svd_solver="randomized",random_state=seed);pca.fit(Xz[:tr_end]);Z=pca.transform(Xz)
    grid=[]
    lagbins=sorted(set(max(1,int(round(s/bin_s))) for s in lag_seconds))
    for lag in lagbins:
        tr=np.arange(0,max(0,tr_end-lag));va=np.arange(tr_end,max(tr_end,va_end-lag))
        if len(tr)<80 or len(va)<20: continue
        for alpha in alphas:
            model=Ridge(alpha=float(alpha),fit_intercept=True);model.fit(Z[tr],Z[tr+lag]);pred=model.predict(Z[va]);rv=r2vw(Z[va+lag],pred);rp=r2vw(Z[va+lag],Z[va])
            grid.append({"lag_bins":lag,"lag_seconds":lag*bin_s,"alpha":float(alpha),"val_r2":rv,"val_persistence_r2":rp,"val_delta_vs_persistence":rv-rp})
    if not grid: return None,{"status":"no_valid_candidates"}
    gd=pd.DataFrame(grid).sort_values(["val_r2","lag_bins","alpha"],ascending=[False,True,True]);best=gd.iloc[0]
    lag=int(best.lag_bins);alpha=float(best.alpha)
    # Strict untouched final 20% test. Refit scaler/PCA/model on first 80% only after hyperparameter selection.
    mu=Xg[:va_end].mean(0);sd=Xg[:va_end].std(0);sd[sd<1e-8]=1;Xz=(Xg-mu)/sd
    k=max(2,min(int(n_components),Xz.shape[1]-1,va_end-2));pca=PCA(n_components=k,svd_solver="randomized",random_state=seed+17);pca.fit(Xz[:va_end]);Z=pca.transform(Xz);W=pca.components_.T
    trv=np.arange(0,max(0,va_end-lag));te=np.arange(va_end,max(va_end,T-lag))
    if len(te)<20: return None,{"status":"too_few_test_pairs","test_pairs":len(te)}
    model=Ridge(alpha=alpha,fit_intercept=True);model.fit(Z[trv],Z[trv+lag]);pred=model.predict(Z[te]);test_r2=r2vw(Z[te+lag],pred);persist=r2vw(Z[te+lag],Z[te])
    B=model.coef_.T;direction=np.linalg.norm(W@B,axis=1);mean_abs=np.mean(np.abs(Xz[:va_end]),axis=0);lev=mean_abs*direction;mean_activity=Xg[:va_end].mean(0);loading=np.linalg.norm(W,axis=1)
    return {"good":good,"lev":lev,"mean_activity":mean_activity,"loading":loading}, {
        "status":"ok","chosen_lag_bins":lag,"chosen_lag_seconds":lag*bin_s,"chosen_alpha":alpha,
        "validation_r2":float(best.val_r2),"validation_persistence_r2":float(best.val_persistence_r2),
        "test_r2":test_r2,"test_persistence_r2":persist,"test_delta_vs_persistence":test_r2-persist,
        "n_components":k,"pca_explained_variance":float(pca.explained_variance_ratio_.sum()),"trainval_pairs":len(trv),"test_pairs":len(te),
        "candidate_grid":grid}


def process_one(item,cfg):
    from pynwb import NWBHDF5IO
    p=Path(item["path"]);eid=int(item["experiment_id"])
    try:
        io=NWBHDF5IO(str(p),"r",load_namespaces=True);nwb=io.read()
        try:
            cand=roi_candidates(nwb)
            if not cand: raise ValueError("no RoiResponseSeries")
            _,nm,ser=cand[0];X,ts,cids=roi_matrix(ser);cid=pd.to_numeric(pd.Series(cids),errors="coerce").to_numpy();goodid=np.isfinite(cid);X=X[:,goodid];cid=cid[goodid].astype(np.int64)
            if X.shape[1]<3: raise ValueError("too few matched cell IDs")
            Xb,tb,cnt=bin_continuous(X,ts,cfg["bin_seconds"])
            fit,diag=fit_select_lag(Xb,cfg["bin_seconds"],cfg["lag_seconds"],cfg["alphas"],cfg["latent_components"],cfg["random_seed"]+eid%100000)
            if fit is None: raise RuntimeError(str(diag))
            orig=np.flatnonzero(fit["good"]);rows=[]
            for j,oi in enumerate(orig):
                rows.append({"experiment_id":eid,"container_id":item.get("container_id"),"experiment_state":item.get("state"),"cre_line":item.get("cre_line"),
                             "neuron_index":int(oi),"cell_specimen_id":int(cid[oi]),"L_dyn":float(fit["lev"][j]),"mean_activity":float(fit["mean_activity"][j]),
                             "pca_loading_norm":float(fit["loading"][j]),"latent_r2":float(diag["test_r2"]),"persistence_r2":float(diag["test_persistence_r2"]),
                             "delta_r2_vs_persistence":float(diag["test_delta_vs_persistence"]),"chosen_lag_seconds":float(diag["chosen_lag_seconds"]),"chosen_alpha":float(diag["chosen_alpha"])})
            d=pd.DataFrame(rows);d["L_primary_rank"]=rank01(d.L_dyn)
            return {"ok":True,"eid":eid,"rows":d,"diag":{"experiment_id":eid,"state":item.get("state"),"container_id":item.get("container_id"),"roi_series":nm,**diag}}
        finally: io.close()
    except Exception as e:
        return {"ok":False,"eid":eid,"error":repr(e),"traceback":traceback.format_exc()}


def residualize(d):
    parts=[]
    for eid,g in d.groupby("experiment_id",sort=False):
        h=g.copy();y=pd.to_numeric(h.L_primary_rank,errors="coerce").to_numpy(float);X=np.column_stack([np.ones(len(h)),rank01(h.mean_activity),rank01(h.pca_loading_norm)]);m=np.isfinite(y)&np.isfinite(X).all(1);res=np.full(len(h),np.nan)
        if m.sum()>=8:
            b,*_=np.linalg.lstsq(X[m],y[m],rcond=None);res[m]=y[m]-X[m]@b;h["L_resid_rank"]=rank01(res)
        else:h["L_resid_rank"]=np.nan
        parts.append(h)
    return pd.concat(parts,ignore_index=True)


def state_pairs(d,value_col,gate=None):
    x=d.copy()
    if gate is not None: x=x.query(gate)
    rows=[]
    from itertools import combinations
    for cid,g in x.groupby("container_id",dropna=False):
        for a,b in combinations(PRIMARY_STATES,2):
            ga=g[g.experiment_state.astype(str)==a][["cell_specimen_id",value_col]].dropna().drop_duplicates("cell_specimen_id")
            gb=g[g.experiment_state.astype(str)==b][["cell_specimen_id",value_col]].dropna().drop_duplicates("cell_specimen_id")
            m=ga.merge(gb,on="cell_specimen_id",suffixes=("_a","_b"));r,p,n=spearman_safe(m[f"{value_col}_a"],m[f"{value_col}_b"])
            if n>=4: rows.append({"container_id":cid,"state_a":a,"state_b":b,"rho":r,"p":p,"n_cells":n})
    return pd.DataFrame(rows)


def pair_summary(p):
    if p.empty:return pd.DataFrame()
    return p.groupby(["state_a","state_b"],as_index=False).agg(n_containers=("rho","count"),median_rho=("rho","median"),mean_rho=("rho","mean"))


def plot_heat(sumdf,outpath,label="Median within-container Spearman ρ"):
    setup_mpl();import matplotlib.pyplot as plt
    M=np.eye(3)
    for i,a in enumerate(PRIMARY_STATES):
        for j,b in enumerate(PRIMARY_STATES):
            if i==j:continue
            q=sumdf[((sumdf.state_a==a)&(sumdf.state_b==b))|((sumdf.state_a==b)&(sumdf.state_b==a))]
            M[i,j]=q.median_rho.iloc[0] if len(q) else np.nan
    fig,ax=plt.subplots(figsize=(5.2,4.5));im=ax.imshow(M,cmap="RdBu_r",vmin=-1,vmax=1);labs=[s.replace("_"," ") for s in PRIMARY_STATES];ax.set_xticks(range(3));ax.set_xticklabels(labs,rotation=30,ha="right");ax.set_yticks(range(3));ax.set_yticklabels(labs)
    for i in range(3):
        for j in range(3):
            if np.isfinite(M[i,j]):ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="white" if abs(M[i,j])>.55 else COL["dark"])
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03);cb.set_label(label);save_panel(fig,outpath)


def main():
    ap=argparse.ArgumentParser(description="Leakage-controlled lag-aware latent dynamics refit for Allen Visual Behavior 2P.")
    ap.add_argument("--project-root",default="/data/coding/NeuralScience");ap.add_argument("--allen-root",default=None);ap.add_argument("--output-root",default=None)
    ap.add_argument("--old-leverage",default=None);ap.add_argument("--workers",type=int,default=4);ap.add_argument("--bin-seconds",type=float,default=.25)
    ap.add_argument("--lags",default="0.25,0.5,1.0,2.0");ap.add_argument("--alphas",default="0.1,1,10");ap.add_argument("--latent-components",type=int,default=20);ap.add_argument("--random-seed",type=int,default=20260820)
    ap.add_argument("--limit",type=int,default=0,help="Debug only: process first N experiments")
    args=ap.parse_args();root=Path(args.project_root).resolve();allen=Path(args.allen_root).resolve() if args.allen_root else root/"biological_data"/"stage5b_dynamic"/"allen_visual_behavior_2p_official_s3"
    out=ensure_dir(Path(args.output_root) if args.output_root else root/"results_complexity_leverage_closure_v1"/"allen_lag_aware_refit");ensure_dir(out/"analysis");ensure_dir(out/"figures");ck=ensure_dir(out/"checkpoints")
    cohort=load_cohort(allen);files=find_raw_files(allen,cohort)
    if args.limit>0:files=files.head(args.limit)
    files.to_csv(out/"analysis"/"experiment_manifest.csv",index=False)
    cfg={"bin_seconds":args.bin_seconds,"lag_seconds":[float(x) for x in args.lags.split(",")],"alphas":[float(x) for x in args.alphas.split(",")],"latent_components":args.latent_components,"random_seed":args.random_seed}
    write_json(out/"analysis"/"model_selection_config.json",cfg)
    allrows=[];diags=[];todo=[]
    for _,r in files.iterrows():
        eid=int(r.experiment_id);cp=ck/f"{eid}_leverage.csv";jp=ck/f"{eid}_diag.json"
        if cp.exists() and jp.exists():allrows.append(pd.read_csv(cp));diags.append(json.loads(jp.read_text()));continue
        todo.append(r.to_dict())
    print(f"Allen lag-aware refit: experiments={len(files)}, cached={len(allrows)}, todo={len(todo)}",flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=max(1,args.workers)) as ex:
            fut={ex.submit(process_one,x,cfg):x for x in todo}
            done=0
            for f in as_completed(fut):
                res=f.result();done+=1;eid=res["eid"]
                if res["ok"]:
                    d=res["rows"];d.to_csv(ck/f"{eid}_leverage.csv",index=False);write_json(ck/f"{eid}_diag.json",res["diag"]);allrows.append(d);diags.append(res["diag"]);print(f"[{done}/{len(todo)}] OK {eid} lag={res['diag']['chosen_lag_seconds']} R2={res['diag']['test_r2']:.3f}",flush=True)
                else:
                    write_json(ck/f"{eid}_failed.json",res);print(f"[{done}/{len(todo)}] FAILED {eid}: {res['error']}",flush=True)
    if not allrows: raise RuntimeError("No successful Allen experiments")
    d=pd.concat(allrows,ignore_index=True);d=residualize(d);d.to_csv(out/"analysis"/"allen_lagaware_leverage_long.csv",index=False)
    dg=pd.DataFrame(diags);dg.to_csv(out/"analysis"/"allen_lagaware_experiment_diagnostics.csv",index=False)
    raw=state_pairs(d,"L_primary_rank");res=state_pairs(d,"L_resid_rank");raw.to_csv(out/"analysis"/"state_pairs_raw_by_container.csv",index=False);res.to_csv(out/"analysis"/"state_pairs_residual_by_container.csv",index=False)
    rs=pair_summary(raw);xs=pair_summary(res);rs.to_csv(out/"analysis"/"state_pairs_raw_summary.csv",index=False);xs.to_csv(out/"analysis"/"state_pairs_residual_summary.csv",index=False)
    gates=[]
    for label,query in [("r2_gt_0","latent_r2>0"),("r2_gt_005","latent_r2>0.05"),("beats_persistence","delta_r2_vs_persistence>0"),("r2_gt0_and_beats_persistence","latent_r2>0 and delta_r2_vs_persistence>0")]:
        q=state_pairs(d,"L_primary_rank",query);q["gate"]=label;gates.append(q)
    gate=pd.concat(gates,ignore_index=True) if gates else pd.DataFrame();gate.to_csv(out/"analysis"/"quality_gated_state_pairs.csv",index=False)
    gsum=gate.groupby(["gate","state_a","state_b"],as_index=False).agg(n_containers=("rho","count"),median_rho=("rho","median"),mean_rho=("rho","mean")) if len(gate) else pd.DataFrame();gsum.to_csv(out/"analysis"/"quality_gated_state_pairs_summary.csv",index=False)
    metrics={"status":"COMPLETE","experiments_successful":int(d.experiment_id.nunique()),"containers":int(d.container_id.nunique()),"cells_rows":int(len(d)),
             "median_test_r2":float(d.drop_duplicates('experiment_id').latent_r2.median()),"positive_r2_fraction":float((d.drop_duplicates('experiment_id').latent_r2>0).mean()),
             "median_delta_vs_persistence":float(d.drop_duplicates('experiment_id').delta_r2_vs_persistence.median()),"beats_persistence_fraction":float((d.drop_duplicates('experiment_id').delta_r2_vs_persistence>0).mean()),
             "median_raw_state_pair_rho":float(raw.rho.median()) if len(raw) else np.nan,"median_residual_state_pair_rho":float(res.rho.median()) if len(res) else np.nan,
             "selection_rule":"lag and ridge alpha selected only on middle validation block; final R2 reported on untouched last 20% test block; PCA/refit use first 80% after selection.",
             "leverage_rule":"direction from train+validation refit; activity amplitude computed on train+validation only; held-out test is reserved for model quality."}
    # Optional old baseline comparison
    oldp=Path(args.old_leverage).resolve() if args.old_leverage else root/"biological_results"/"stage5b_dynamic_leverage_v1"/"allen_vbo"/"leverage_long.csv"
    if oldp.exists():
        old=pd.read_csv(oldp);
        if 'state' in old.columns and 'experiment_state' in old.columns: old=old[old.state.astype(str)==old.experiment_state.astype(str)].copy()
        om=old.drop_duplicates("experiment_id")[["experiment_id","latent_r2"]].rename(columns={"latent_r2":"old_r2"});nm=d.drop_duplicates("experiment_id")[["experiment_id","latent_r2","chosen_lag_seconds","delta_r2_vs_persistence"]].rename(columns={"latent_r2":"new_r2"});cmp=om.merge(nm,on="experiment_id");cmp["delta_r2_new_minus_old"]=cmp.new_r2-cmp.old_r2;cmp.to_csv(out/"analysis"/"old_vs_lagaware_r2.csv",index=False);metrics["median_old_r2"]=float(cmp.old_r2.median());metrics["median_new_r2_matched"]=float(cmp.new_r2.median());metrics["median_r2_improvement"]=float(cmp.delta_r2_new_minus_old.median());metrics["matched_experiments_old_new"]=int(len(cmp))
    write_json(out/"analysis"/"lagaware_key_metrics.json",metrics)
    report=f'''# Allen Visual Behavior lag-aware latent-dynamics refit\n\nLag/alpha selection is performed on a blocked validation interval only. The final 20% time block is untouched until the final model-quality evaluation. PCA and the final Ridge model are refit using the first 80% after hyperparameter selection. Leverage amplitude is also estimated from the first 80%, so the final test block is used only for predictive quality.\n\nCandidate lags: {cfg['lag_seconds']} s. Candidate ridge alpha: {cfg['alphas']}.\n\nMedian held-out test R2: {metrics['median_test_r2']:.4f}; positive fraction: {metrics['positive_r2_fraction']:.1%}.\nMedian delta R2 relative to persistence: {metrics['median_delta_vs_persistence']:.4f}; fraction beating persistence: {metrics['beats_persistence_fraction']:.1%}.\nMedian same-cell state-pair rho: raw {metrics['median_raw_state_pair_rho']:.4f}; residual {metrics['median_residual_state_pair_rho']:.4f}.\n\nManuscript rule: call Allen leverage dynamical only where held-out dynamics quality is adequate; retain quality-gated results and the persistence baseline.\n'''
    (out/"analysis"/"lagaware_report.md").write_text(report,encoding="utf-8")

    setup_mpl();import matplotlib.pyplot as plt
    e=d.drop_duplicates("experiment_id").copy()
    # lag distribution
    fig,ax=plt.subplots(figsize=(5.3,4.2));freq=e.chosen_lag_seconds.value_counts().sort_index();ax.bar([str(x) for x in freq.index],freq.values,color=COL["blue"]);ax.set_xlabel("selected lag (s)");ax.set_ylabel("experiments");clean_ax(ax);save_panel(fig,out/"figures"/"AL01_selected_lag_distribution")
    # R2 by state
    fig,ax=plt.subplots(figsize=(5.5,4.3));rng=np.random.default_rng(6)
    for i,s in enumerate(PRIMARY_STATES):
        v=e[e.experiment_state==s].latent_r2.dropna();ax.scatter(np.full(len(v),i)+rng.normal(0,.04,len(v)),v,s=32,alpha=.68,color=[COL['blue'],COL['teal'],COL['orange']][i]);
        if len(v):ax.plot([i-.18,i+.18],[v.median()]*2,lw=3,color=COL["dark"])
    ax.axhline(0,ls="--",lw=1,color=COL["gray"]);ax.set_xticks(range(3));ax.set_xticklabels([s.replace('_',' ') for s in PRIMARY_STATES],rotation=25,ha="right");ax.set_ylabel("held-out latent-dynamics R²");clean_ax(ax);save_panel(fig,out/"figures"/"AL02_test_r2_by_state")
    # persistence delta
    fig,ax=plt.subplots(figsize=(5.3,4.2));ax.hist(e.delta_r2_vs_persistence.dropna(),bins=18,color=COL["purple"],alpha=.8);ax.axvline(0,ls="--",lw=1,color=COL["gray"]);ax.set_xlabel("R²(model) − R²(persistence)");ax.set_ylabel("experiments");clean_ax(ax);save_panel(fig,out/"figures"/"AL03_delta_vs_persistence")
    plot_heat(rs,out/"figures"/"AL04_state_reconfiguration_raw");plot_heat(xs,out/"figures"/"AL05_state_reconfiguration_residual")
    if oldp.exists() and 'cmp' in locals():
        fig,ax=plt.subplots(figsize=(5.1,4.6));
        for _,r in cmp.iterrows(): ax.plot([0,1],[r.old_r2,r.new_r2],color=COL["light"],lw=1,alpha=.75)
        ax.scatter(np.zeros(len(cmp)),cmp.old_r2,s=28,color=COL["blue"]);ax.scatter(np.ones(len(cmp)),cmp.new_r2,s=28,color=COL["teal"]);ax.axhline(0,ls="--",lw=1,color=COL["gray"]);ax.set_xticks([0,1]);ax.set_xticklabels(["one-step baseline","lag-aware refit"]);ax.set_ylabel("held-out latent-dynamics R²");clean_ax(ax);save_panel(fig,out/"figures"/"AL06_old_vs_lagaware_r2")
    if len(gsum):
        pv=gsum.groupby("gate",as_index=False)["median_rho"].median().rename(columns={"median_rho":"median_pair_rho"})
        fig,ax=plt.subplots(figsize=(5.8,4.3));ax.bar(range(len(pv)),pv.median_pair_rho,color=COL["teal"]);ax.set_xticks(range(len(pv)));ax.set_xticklabels(pv.gate.str.replace('_',' '),rotation=28,ha="right");ax.set_ylabel("median state-pair Spearman ρ");clean_ax(ax);save_panel(fig,out/"figures"/"AL07_quality_gated_state_stability")
    print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()
