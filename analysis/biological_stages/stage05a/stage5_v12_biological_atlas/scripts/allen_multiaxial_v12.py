#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from common import cfg,ensure,read_csv,write_csv,write_json,fnum,bh_fdr,standardize_train_test,ridge_fit_predict,stratified_folds,now

COST={"GLIF1":0,"GLIF2":1,"GLIF3":1,"GLIF4":2,"GLIF5":3}
KEEP_CATS={"adaptation_history","after_spike_trough","excitability_threshold","firing_input_output","passive_membrane","spike_waveform"}

def temporal_table(path,eps):
    rows=read_csv(path);by=defaultdict(dict)
    for r in rows:by[int(float(r["specimen_id"]))][int(float(r["context_ms"]))]=fnum(r["r2"])
    out={}
    for sid,d in by.items():
        cs=sorted(d); vals=np.array([d[x] for x in cs],float);best=np.nanmax(vals)
        suff=[x for x in cs if np.isfinite(d[x]) and d[x]>=best-eps]
        out[sid]={"C_temporal":min(suff) if suff else max(cs),"best_r2":best,
                  "long_short_gain":d[max(cs)]-d[min(cs)],**{f"r2_{x}":d[x] for x in cs}}
    return out

def c_req(row,eps):
    vals={g:fnum(row.get(f"{g}_EVR")) for g in COST}
    good={g:v for g,v in vals.items() if np.isfinite(v)}
    if len(good)<5:return np.nan,""
    best=max(good.values()); eligible=[g for g,v in good.items() if v>=best-eps]
    mc=min(COST[g] for g in eligible);mech="|".join(sorted([g for g in eligible if COST[g]==mc]))
    return float(mc),mech

def onehot(rows,fields):
    levels={}
    for f in fields:levels[f]=sorted({str(r.get(f,"")) for r in rows if str(r.get(f,"")) not in ("","nan","None")})
    names=[];cols=[]
    for f in fields:
        for lv in levels[f][1:]:
            names.append(f"{f}={lv}");cols.append(np.array([1.0 if str(r.get(f,""))==lv else 0.0 for r in rows]))
    return (np.column_stack(cols) if cols else np.zeros((len(rows),0))),names

def repeated_nested_cv(X,y,cfgcv,seedbase=0):
    X=np.asarray(X,float);y=np.asarray(y,float);n=len(y)
    pred_sum=np.zeros(n);pred_n=np.zeros(n)
    chosen=[];fold_records=[]
    k=int(cfgcv["outer_folds"]);reps=int(cfgcv["repeats"]);alphas=list(map(float,cfgcv["alphas"]))
    for rep in range(reps):
        folds=stratified_folds(y,k,int(cfgcv["seed"])+seedbase+rep*97)
        for fi,test in enumerate(folds):
            train=np.setdiff1d(np.arange(n),test,assume_unique=False)
            inner=stratified_folds(y[train],max(3,k-1),int(cfgcv["seed"])+seedbase+rep*997+fi)
            scores=[]
            for alpha in alphas:
                sc=[]
                for val_local in inner:
                    tr_local=np.setdiff1d(np.arange(len(train)),val_local)
                    pr=ridge_fit_predict(X[train][tr_local],y[train][tr_local],X[train][val_local],alpha)
                    sc.append(np.mean((pr-y[train][val_local])**2))
                scores.append(np.mean(sc))
            alpha=alphas[int(np.argmin(scores))];chosen.append(alpha)
            p=ridge_fit_predict(X[train],y[train],X[test],alpha)
            pred_sum[test]+=p;pred_n[test]+=1
            fold_records.append({"repeat":rep,"fold":fi,"alpha":alpha,"test_n":len(test)})
    pred=pred_sum/np.maximum(pred_n,1)
    rho=float(spearmanr(y,pred).statistic);mae=float(np.mean(np.abs(y-pred)))
    r2=float(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2))
    return pred,{"rho":rho,"mae":mae,"r2":r2,"median_alpha":float(np.median(chosen)),"folds":fold_records}

def fixed_cv(X,y,alpha,cfgcv,seedbase=0):
    X=np.asarray(X,float);y=np.asarray(y,float);n=len(y);pred_sum=np.zeros(n);pred_n=np.zeros(n)
    for rep in range(int(cfgcv["repeats"])):
        folds=stratified_folds(y,int(cfgcv["outer_folds"]),int(cfgcv["seed"])+seedbase+rep*97)
        for test in folds:
            train=np.setdiff1d(np.arange(n),test)
            p=ridge_fit_predict(X[train],y[train],X[test],alpha);pred_sum[test]+=p;pred_n[test]+=1
    return pred_sum/np.maximum(pred_n,1)

def permutation_p(X,y,alpha,cfgcv,observed,seedbase):
    rng=np.random.default_rng(int(cfgcv["seed"])+seedbase);vals=[]
    for i in range(int(cfgcv["permutations"])):
        yp=rng.permutation(y);pred=fixed_cv(X,yp,alpha,cfgcv,seedbase=10000+i*13)
        vals.append(float(spearmanr(yp,pred).statistic))
        if (i+1)%50==0:print(f"permutation {i+1}/{cfgcv['permutations']}",flush=True)
    vals=np.asarray(vals);p=(1+np.sum(vals>=observed))/(len(vals)+1)
    return float(p),vals

def bootstrap_delta(y,p1,p0,n,seed):
    rng=np.random.default_rng(seed);d=[]
    idx=np.arange(len(y))
    for _ in range(n):
        z=rng.choice(idx,len(idx),replace=True)
        r1=spearmanr(y[z],p1[z]).statistic;r0=spearmanr(y[z],p0[z]).statistic
        if np.isfinite(r1) and np.isfinite(r0):d.append(float(r1-r0))
    d=np.asarray(d);return {"median":float(np.median(d)),"ci95":[float(np.percentile(d,2.5)),float(np.percentile(d,97.5))],"p_delta_le_0":float(np.mean(d<=0))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args();c=cfg(a.config)
    out=ensure(Path(c["output_root"])/"20_allen_multiaxial_v12")
    ar=Path(c["allen_root"]); vr=Path(c["v11_result_root"])
    ephys=read_csv(ar/"analysis/ephys_axis_sweep_inventory_v1/ephys_features/ephys_features_specimen_level_clean.csv")
    tax=read_csv(ar/"analysis/ephys_axis_sweep_inventory_v1/ephys_features/feature_taxonomy.csv")
    glif=read_csv(ar/"analysis/glif_performance_atlas_v1/tables/glif_performance_atlas_specimen_level.csv")
    feature_cat={r["feature"]:r["category"] for r in tax if r["category"] in KEEP_CATS}
    features=[f for f in feature_cat if f in ephys[0]]
    ab=temporal_table(vr/"30_temporal_context/AB/context_performance.csv",float(c["c_model_epsilon"]))
    ba=temporal_table(vr/"30_temporal_context/BA/context_performance.csv",float(c["c_model_epsilon"]))
    E={int(float(r["specimen_id"])):r for r in ephys}
    G={int(float(r["specimen_id"])):r for r in glif}
    rows=[]
    for sid in sorted(set(E)&set(G)&set(ab)&set(ba)):
        g=G[sid]
        if str(g.get("species","")).lower()!=str(c["primary_species"]).lower():continue
        creq,mech=c_req(g,float(c["c_model_epsilon"]))
        if not np.isfinite(creq):continue
        r={"specimen_id":sid,"C_req_eps_0.02":creq,"C_req_mechanism":mech,
           "species":g.get("species",""),"layer":g.get("layer",""),"dendrite_type":g.get("dendrite_type",""),
           "structure_parent_acronym":g.get("structure_parent_acronym","")}
        for f in features:r[f]=fnum(E[sid].get(f))
        r["C_temporal_AB"]=ab[sid]["C_temporal"];r["C_temporal_BA"]=ba[sid]["C_temporal"]
        r["C_temporal_consensus"]=max(ab[sid]["C_temporal"],ba[sid]["C_temporal"])
        r["temporal_best_r2_mean"]=np.nanmean([ab[sid]["best_r2"],ba[sid]["best_r2"]])
        r["temporal_long_short_gain_mean"]=np.nanmean([ab[sid]["long_short_gain"],ba[sid]["long_short_gain"]])
        for ctx in c["contexts_ms"]:
            r[f"temporal_r2mean_{ctx}"]=np.nanmean([ab[sid][f"r2_{ctx}"],ba[sid][f"r2_{ctx}"]])
        rows.append(r)
    if len(rows)<300:raise SystemExit(f"Too few primary complete cells: {len(rows)}")
    y=np.asarray([r["C_req_eps_0.02"] for r in rows],float)
    Xe=np.asarray([[r.get(f,np.nan) for f in features] for r in rows],float)
    temporal_cols=["C_temporal_consensus","temporal_best_r2_mean","temporal_long_short_gain_mean"]+[f"temporal_r2mean_{x}" for x in c["contexts_ms"]]
    Xt=np.asarray([[r.get(f,np.nan) for f in temporal_cols] for r in rows],float)
    Xm,meta_cols=onehot(rows,["layer","dendrite_type","structure_parent_acronym"])
    sets={"temporal_profile":Xt,"multiaxial_ephys":Xe,"metadata_only":Xm,
          "ephys_plus_metadata":np.column_stack([Xe,Xm]),"ephys_plus_temporal":np.column_stack([Xe,Xt])}
    results={};preds={}
    for j,(name,X) in enumerate(sets.items()):
        print(f"CV {name}: n={len(y)} p={X.shape[1]}",flush=True)
        pred,res=repeated_nested_cv(X,y,c["cv"],seedbase=j*1000)
        pp,null=permutation_p(X,y,res["median_alpha"],c["cv"],res["rho"],seedbase=50000+j*1000)
        res["permutation_p"]=pp;res["n"]=len(y);res["p_features"]=X.shape[1];results[name]=res;preds[name]=pred
        print(name,res["rho"],res["r2"],res["mae"],"perm_p",pp,flush=True)
    b1=bootstrap_delta(y,preds["multiaxial_ephys"],preds["temporal_profile"],int(c["cv"]["bootstrap"]),int(c["cv"]["seed"])+1)
    b2=bootstrap_delta(y,preds["ephys_plus_metadata"],preds["metadata_only"],int(c["cv"]["bootstrap"]),int(c["cv"]["seed"])+2)
    # Univariate feature associations
    assoc=[]
    for f in features:
        x=np.asarray([r.get(f,np.nan) for r in rows],float);m=np.isfinite(x)
        if m.sum()<50:continue
        rr,pp=spearmanr(x[m],y[m])
        assoc.append({"feature":f,"category":feature_cat[f],"n":int(m.sum()),"rho":float(rr),"p":float(pp)})
    q=bh_fdr([r["p"] for r in assoc])
    for r,qq in zip(assoc,q):r["q_fdr"]=float(qq)
    assoc.sort(key=lambda r:abs(r["rho"]),reverse=True)
    # Category ablation
    abl=[]
    fullrho=results["multiaxial_ephys"]["rho"]
    for cat in sorted(KEEP_CATS):
        cols=[i for i,f in enumerate(features) if feature_cat[f]!=cat]
        if not cols:continue
        pred,res=repeated_nested_cv(Xe[:,cols],y,c["cv"],seedbase=80000+len(abl)*1000)
        abl.append({"removed_category":cat,"rho_without":res["rho"],"delta_rho_full_minus_without":fullrho-res["rho"]})
    for i,r in enumerate(rows):
        for name,p in preds.items():r[f"pred_Creq_{name}"]=float(p[i])
    write_csv(out/"allen_multiaxial_primary_mouse_complete.csv",rows)
    write_csv(out/"feature_Creq_associations.csv",assoc)
    write_csv(out/"category_ablation.csv",abl)
    summary={"created_utc":now(),"primary_species":c["primary_species"],"n":len(rows),"C_req_distribution":{str(int(v)):int(np.sum(y==v)) for v in sorted(set(y))},
             "features":features,"feature_categories":feature_cat,"temporal_columns":temporal_cols,"metadata_columns":meta_cols,
             "models":{k:{kk:vv for kk,vv in v.items() if kk!="folds"} for k,v in results.items()},
             "bootstrap_multiaxial_minus_temporal":b1,"bootstrap_ephys_plus_metadata_minus_metadata":b2}
    write_json(out/"allen_multiaxial_summary.json",summary);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
