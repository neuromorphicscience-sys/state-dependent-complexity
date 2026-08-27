from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.stage4_safeio import atomic_write_csv,atomic_write_json

FEATURES=[
"sel_total_degree","sel_feedback","sel_cycle3","sel_bridge","sel_spectral",
"coverage1","coverage2","out_neighbor_redundancy","selected_dispersion",
"graph_density","degree_cv","max_degree_ratio","inout_corr",
]

def collect_tasks(outdir,phase):
    files=sorted((outdir/"tasks").glob("*.csv"))
    dfs=[]
    for f in files:
        try:
            d=pd.read_csv(f)
            if len(d) and str(d["phase"].iloc[0])==phase: dfs.append(d)
        except Exception: pass
    if not dfs: raise RuntimeError(f"No {phase} task CSVs found")
    return pd.concat(dfs,ignore_index=True)

def ridge_fit(X,y,lam=1e-2):
    mu=X.mean(0); sd=X.std(0)+1e-9
    Z=(X-mu)/sd
    Z1=np.column_stack([np.ones(len(Z)),Z])
    I=np.eye(Z1.shape[1]); I[0,0]=0
    beta=np.linalg.solve(Z1.T@Z1+lam*I,Z1.T@y)
    pred=Z1@beta
    ssr=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)+1e-12
    return beta,mu,sd,1-ssr/sst

def main():
    cfg=json.load(open(ROOT/"configs"/"stage4.json","r",encoding="utf-8"))
    outdir=ROOT/cfg.get("output_dir","results_stage4")
    disc=collect_tasks(outdir,"discovery")
    atomic_write_csv(outdir/"stage4_discovery_methods.csv",disc)

    # method comparison: paired on exact same graph/regime/K
    keys=["topology","regime","graph_seed","k","coupling","connection_prob","noise_sigma"]
    rnd=disc[disc.method=="random"].groupby(keys,as_index=False).agg(
        random_mean=("rhythm_score_mean","mean"),
        random_p90=("rhythm_score_mean",lambda x: float(np.quantile(x,0.9)))
    )
    heur=disc[disc.method.isin(["high_degree","feedback_hub","module_bridge","cycle_proxy","spectral"])]\
        .groupby(keys,as_index=False).agg(best_heuristic=("rhythm_score_mean","max"))
    opt=disc[disc.method=="gpu_optimized"][keys+["rhythm_score_mean","rhythm_score_std"]].rename(
        columns={"rhythm_score_mean":"optimized","rhythm_score_std":"optimized_std"}
    )
    comp=opt.merge(rnd,on=keys,how="left").merge(heur,on=keys,how="left")
    comp["gain_vs_random"]=comp["optimized"]-comp["random_mean"]
    comp["gain_vs_best_heuristic"]=comp["optimized"]-comp["best_heuristic"]
    comp["ratio_vs_random"]=comp["optimized"]/(comp["random_mean"]+1e-9)
    atomic_write_csv(outdir/"stage4_optimization_summary.csv",comp)

    # Absolute threshold complexity saving across K
    thresholds=cfg["analysis_thresholds"]
    save_rows=[]
    condkeys=["topology","regime","graph_seed","coupling","connection_prob","noise_sigma"]
    for cond,sub in comp.groupby(condkeys):
        c=dict(zip(condkeys,cond if isinstance(cond,tuple) else (cond,)))
        for thr in thresholds:
            def min_k(col):
                q=sub[sub[col]>=thr]
                return float(q.k.min()) if len(q) else np.nan
            kr=min_k("random_mean"); kh=min_k("best_heuristic"); ko=min_k("optimized")
            save_rows.append({
                **c,"threshold":thr,"k_random":kr,"k_best_heuristic":kh,"k_optimized":ko,
                "saving_vs_random":(1-ko/kr) if np.isfinite(kr) and kr>0 and np.isfinite(ko) else np.nan,
                "saving_vs_heuristic":(1-ko/kh) if np.isfinite(kh) and kh>0 and np.isfinite(ko) else np.nan,
            })
    saving=pd.DataFrame(save_rows)
    atomic_write_csv(outdir/"stage4_complexity_saving.csv",saving)

    # Interpretable set-level surrogate law.
    # Use all discovery sets, but cap random multiplicity so it does not dominate.
    train=[]
    for _,sub in disc.groupby(keys):
        rr=sub[sub.method=="random"].head(16)
        oo=sub[sub.method!="random"]
        train.append(pd.concat([rr,oo],ignore_index=True))
    train=pd.concat(train,ignore_index=True).dropna(subset=FEATURES+["rhythm_score_mean"])
    X=train[FEATURES].to_numpy(float); y=train["rhythm_score_mean"].to_numpy(float)
    beta,mu,sd,r2=ridge_fit(X,y,float(cfg["surrogate_ridge_lambda"]))
    coef=pd.DataFrame({
        "feature":["intercept"]+FEATURES,
        "coefficient":beta
    })
    atomic_write_csv(outdir/"stage4_allocation_law_coefficients.csv",coef)
    law={"features":FEATURES,"beta":beta.tolist(),"mean":mu.tolist(),"std":sd.tolist(),"train_r2":float(r2)}
    atomic_write_json(outdir/"stage4_allocation_law.json",law)

    print("\n=== Stage 4 optimization ===")
    print(comp.groupby("topology").agg(
        mean_gain_vs_random=("gain_vs_random","mean"),
        mean_gain_vs_heuristic=("gain_vs_best_heuristic","mean"),
        win_rate_vs_random=("gain_vs_random",lambda x: float(np.mean(x>0))),
        win_rate_vs_heuristic=("gain_vs_best_heuristic",lambda x: float(np.mean(x>0))),
    ).to_string())
    print("\n=== Complexity saving ===")
    print(saving.groupby(["topology","threshold"]).agg(
        mean_saving=("saving_vs_random","mean"),
        median_saving=("saving_vs_random","median"),
        positive_rate=("saving_vs_random",lambda x: float(np.mean(x.dropna()>0)) if len(x.dropna()) else np.nan),
    ).to_string())
    print("\nSet-level surrogate train R^2:",round(r2,4))
    print("\nSaved Stage 4 summaries in",outdir)

if __name__=="__main__":
    main()
