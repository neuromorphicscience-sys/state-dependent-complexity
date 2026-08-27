from __future__ import annotations
import argparse, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import *

REPS=["previous_rep0","new_rep1","new_rep2","new_rep3"]


def binary_entropy(p):
    if p<=0 or p>=1: return 0.0
    return -(p*np.log2(p)+(1-p)*np.log2(1-p))


def condition_metrics(g, eps):
    n=int(g.N.iloc[0]);k=int(g.k.iloc[0]); scores=pd.to_numeric(g.rhythm_score_mean,errors="coerce").to_numpy(float)
    masks=[parse_mask_key(x,n) for x in g.mask_key]
    best=float(np.nanmax(scores)); keep=np.isfinite(scores)&((best-scores)<=eps+1e-12)
    km=[masks[i] for i in range(len(masks)) if keep[i]]; ks=scores[keep]
    pairs=[]
    for i,j in combinations(range(len(masks)),2): pairs.append((jaccard_distance(masks[i],masks[j]),abs(scores[i]-scores[j])))
    if len(km)>=2:
        ds=[jaccard_distance(a,b) for a,b in combinations(km,2)]
        union=set().union(*km); inter=set(km[0])
        for x in km[1:]: inter &= x
        ps=[]
        for node in union: ps.append(sum(node in m for m in km)/len(km))
        ent=float(np.mean([binary_entropy(p) for p in ps])) if ps else 0.0
        d=float(np.mean(ds)); maxd=float(np.max(ds)); union_exp=len(union)/k; core=len(inter)/k
    elif len(km)==1:
        d=0.0;maxd=0.0;union_exp=1.0;core=1.0;ent=0.0
    else:
        d=maxd=union_exp=core=ent=np.nan
    eq_pairs=[1 for dist,gap in pairs if gap<=eps+1e-12]
    highdiv=[1 for dist,gap in pairs if gap<=eps+1e-12 and dist>=.5]
    return {"epsilon":eps,"best_score":best,"near_optimal_count":int(keep.sum()),"near_optimal_fraction":float(keep.mean()),
            "D_epsilon_mean_jaccard_distance":d,"max_distant_near_optimal":maxd,"union_expansion":union_exp,"core_fraction":core,
            "selection_entropy_union":ent,"equivalent_pair_count":len(eq_pairs),"high_diversity_equivalent_pair_count":len(highdiv),
            "high_diversity_equivalent_pair_fraction":len(highdiv)/len(eq_pairs) if eq_pairs else np.nan}


def compute_from_table(d, epsilons, score_col="rhythm_score_mean"):
    x=d.copy()
    if score_col!="rhythm_score_mean": x["rhythm_score_mean"]=x[score_col]
    rows=[];pairs=[]
    keys=["graph_seed","regime","k"]
    for key,g in x.groupby(keys,sort=True):
        n=int(g.N.iloc[0]); masks={r:parse_mask_key(m,n) for r,m in zip(g.replicate_label,g.mask_key)}
        scores={r:float(s) for r,s in zip(g.replicate_label,g.rhythm_score_mean)}
        for a,b in combinations(sorted(masks),2):
            pairs.append({"graph_seed":key[0],"regime":key[1],"k":key[2],"rep_a":a,"rep_b":b,
                          "jaccard_distance":jaccard_distance(masks[a],masks[b]),"fresh_score_gap":abs(scores[a]-scores[b]),
                          "score_a":scores[a],"score_b":scores[b]})
        base={"graph_seed":key[0],"regime":key[1],"k":key[2],"N":n}
        for eps in epsilons: rows.append({**base,**condition_metrics(g,float(eps))})
    return pd.DataFrame(rows),pd.DataFrame(pairs)


def load_transfer_home(path):
    p=Path(path)
    if p.is_dir(): p=p/"analysis"/"cross_state_mask_summary.csv"
    if not p.exists(): return pd.DataFrame()
    d=pd.read_csv(p); d=d[d.source_state.astype(str)==d.target_state.astype(str)].copy()
    # convert to Stage4F-like schema
    d["regime"]=d["source_state"]; return d


def main():
    ap=argparse.ArgumentParser(description="Formal empirical functional-degeneracy metrics for Stage4F optimizer solutions.")
    ap.add_argument("--project-root",default="/data/coding/NeuralScience");ap.add_argument("--stage4f-source",default="auto")
    ap.add_argument("--transfer-results",default=None,help="Optional stage4_cross_state_transfer output root for independent fresh-noise replication.")
    ap.add_argument("--output-root",default=None);ap.add_argument("--epsilons",default="0.01,0.02,0.05")
    args=ap.parse_args(); root=Path(args.project_root).resolve(); src=resolve_stage4f_source(root,args.stage4f_source)
    out=ensure_dir(Path(args.output_root) if args.output_root else root/"results_complexity_leverage_closure_v1"/"stage4_functional_degeneracy");ensure_dir(out/"analysis");ensure_dir(out/"figures")
    eps=[float(x) for x in args.epsilons.split(",")]
    d=load_stage4f_reevaluation(src); met,pairs=compute_from_table(d,eps)
    met["evaluation_set"]="stage4f_fresh_6501_6524";pairs["evaluation_set"]="stage4f_fresh_6501_6524"
    met.to_csv(out/"analysis"/"degeneracy_metrics_by_condition.csv",index=False);pairs.to_csv(out/"analysis"/"pairwise_mask_score_geometry.csv",index=False)
    rep=pd.DataFrame(); repmet=pd.DataFrame()
    if args.transfer_results:
        h=load_transfer_home(args.transfer_results)
        if len(h):
            repmet,reppairs=compute_from_table(h,eps);repmet["evaluation_set"]="cross_state_independent_home";reppairs["evaluation_set"]="cross_state_independent_home"
            repmet.to_csv(out/"analysis"/"degeneracy_metrics_independent_home.csv",index=False);reppairs.to_csv(out/"analysis"/"pairwise_geometry_independent_home.csv",index=False)
            q1=met[met.epsilon==0.02].merge(repmet[repmet.epsilon==0.02],on=["graph_seed","regime","k"],suffixes=("_old","_new"))
            r,p,n=spearman_safe(q1.D_epsilon_mean_jaccard_distance_old,q1.D_epsilon_mean_jaccard_distance_new)
            rep={"n_conditions":n,"spearman_D002_old_vs_new":r,"p":p}
            write_json(out/"analysis"/"independent_replication_summary.json",rep)
    primary=met[np.isclose(met.epsilon,.02)].copy()
    byk=primary.groupby("k").agg(n_conditions=("k","size"),mean_D=("D_epsilon_mean_jaccard_distance","mean"),median_D=("D_epsilon_mean_jaccard_distance","median"),
                                       mean_near=("near_optimal_count","mean"),mean_union_expansion=("union_expansion","mean"),mean_core_fraction=("core_fraction","mean"),mean_entropy=("selection_entropy_union","mean")).reset_index()
    byk.to_csv(out/"analysis"/"degeneracy_primary_by_budget.csv",index=False)
    # paired low vs high k by seed/regime
    w=primary.pivot_table(index=["graph_seed","regime"],columns="k",values="D_epsilon_mean_jaccard_distance",aggfunc="first")
    diff=(w[8]-w[64]).dropna() if 8 in w and 64 in w else pd.Series(dtype=float)
    metrics={"status":"COMPLETE","definition":"E_epsilon={A: score(A)>=best-epsilon}; D_epsilon=mean pairwise Jaccard distance within sampled E_epsilon",
             "primary_epsilon":0.02,"n_conditions":int(len(primary)),"optimizer_replicates_per_condition":4,
             "mean_D_epsilon":float(primary.D_epsilon_mean_jaccard_distance.mean()),"median_D_epsilon":float(primary.D_epsilon_mean_jaccard_distance.median()),
             "mean_near_optimal_count":float(primary.near_optimal_count.mean()),"k8_minus_k64_mean_D":float(diff.mean()) if len(diff) else np.nan,
             "k8_gt_k64_exact_signflip_p":exact_signflip_p(diff,"greater") if len(diff) else np.nan,
             "interpretation_guardrail":"Four optimizer replicates provide an empirical lower bound/sample of the equivalence class, not the full combinatorial solution volume."}
    write_json(out/"analysis"/"degeneracy_key_metrics.json",metrics)
    report=f'''# Stage 4 functional degeneracy formalization\n\nDefine the empirical near-optimal allocation class\n\nE_epsilon(G,s,k) = {{A_r : Phi(A_r) >= max_j Phi(A_j) - epsilon}}\n\nand\n\nD_epsilon = mean_{{A,B in E_epsilon, A<B}} [1 - Jaccard(A,B)].\n\nPrimary epsilon = 0.02. We additionally report 0.01 and 0.05 sensitivity.\n\nOther geometry metrics: union expansion |union E|/k, core fraction |intersection E|/k, and mean binary selection entropy over the union.\n\nGuardrail: only four independently searched optimizer solutions are available per condition; therefore these metrics estimate sampled functional degeneracy and cannot be called the full solution-space volume.\n\nPrimary mean D_0.02 = {metrics['mean_D_epsilon']:.4f}; median = {metrics['median_D_epsilon']:.4f}.\n'''
    (out/"analysis"/"functional_degeneracy_report.md").write_text(report,encoding="utf-8")

    setup_mpl();import matplotlib.pyplot as plt
    # 1 pair geometry
    fig,ax=plt.subplots(figsize=(5.6,4.5));
    for k,c in [(8,COL["blue"]),(32,COL["teal"]),(64,COL["orange"])]:
        g=pairs[pairs.k==k];ax.scatter(g.jaccard_distance,g.fresh_score_gap,s=26,alpha=.6,label=f"k={k}",color=c)
    ax.axhline(.02,ls="--",lw=1,color=COL["rose"]);ax.set_xlabel("pairwise mask Jaccard distance");ax.set_ylabel("absolute fresh-score difference");clean_ax(ax);ax.legend(frameon=False)
    save_panel(fig,out/"figures"/"DG01_pairwise_geometry")
    # helper box scatter
    def scatter_by_k(col,name,ylabel):
        fig,ax=plt.subplots(figsize=(5.4,4.3));rng=np.random.default_rng(5)
        for i,k in enumerate([8,32,64]):
            g=primary[primary.k==k];v=pd.to_numeric(g[col],errors="coerce").dropna();ax.scatter(np.full(len(v),i)+rng.normal(0,.04,len(v)),v,s=30,alpha=.68,color=[COL['blue'],COL['teal'],COL['orange']][i]);
            if len(v): ax.plot([i-.18,i+.18],[v.mean()]*2,lw=3,color=COL["dark"])
        ax.set_xticks([0,1,2]);ax.set_xticklabels(["k=8","k=32","k=64"]);ax.set_ylabel(ylabel);clean_ax(ax);save_panel(fig,out/"figures"/name)
    scatter_by_k("D_epsilon_mean_jaccard_distance","DG02_D_epsilon_by_budget","D₀.₀₂: near-optimal mask diversity")
    scatter_by_k("near_optimal_count","DG03_near_optimal_count_by_budget","sampled near-optimal solutions")
    scatter_by_k("union_expansion","DG04_union_expansion_by_budget","union expansion |∪E| / k")
    scatter_by_k("core_fraction","DG05_core_fraction_by_budget","core fraction |∩E| / k")
    scatter_by_k("selection_entropy_union","DG06_selection_entropy_by_budget","selection entropy over ∪E")
    # threshold sensitivity
    tab=met.groupby(["epsilon","k"],as_index=False).D_epsilon_mean_jaccard_distance.mean()
    es=sorted(tab.epsilon.unique());ks=[8,32,64];M=np.full((len(es),3),np.nan)
    for i,e in enumerate(es):
        for j,k in enumerate(ks):
            q=tab[(np.isclose(tab.epsilon,e))&(tab.k==k)];M[i,j]=q.D_epsilon_mean_jaccard_distance.iloc[0] if len(q) else np.nan
    fig,ax=plt.subplots(figsize=(5.2,3.6));im=ax.imshow(M,cmap="viridis",vmin=0,vmax=max(.01,float(np.nanmax(M))));ax.set_xticks(range(3));ax.set_xticklabels([f"k={k}" for k in ks]);ax.set_yticks(range(len(es)));ax.set_yticklabels([f"ε={e:.2f}" for e in es]);
    for i in range(len(es)):
        for j in range(3): ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",color="white" if M[i,j]>.55*np.nanmax(M) else COL["dark"])
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03);cb.set_label("mean Dε");save_panel(fig,out/"figures"/"DG07_epsilon_sensitivity")
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
