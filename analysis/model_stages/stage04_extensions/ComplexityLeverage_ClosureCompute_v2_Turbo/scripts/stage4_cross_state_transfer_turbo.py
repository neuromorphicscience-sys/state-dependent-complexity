from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent; PKG=HERE.parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(PKG))
from common import *
from frozen_stage4.src.stage4_topology import build_topology, to_gpu_adjacency
from frozen_stage4.src.stage4_gpu_simulator import simulate_masks
from frozen_stage4.src.stage4_gpu_simulator_turbo import simulate_masks_seed_batch

REGIME_PARAMS={
    "transition_dense":{"coupling":0.035,"connection_prob":0.10,"noise_sigma":1.0},
    "transition_mid":{"coupling":0.035,"connection_prob":0.05,"noise_sigma":1.0},
    "sparse_drive":{"coupling":0.05,"connection_prob":0.05,"noise_sigma":1.0},
}
REPS=["previous_rep0","new_rep1","new_rep2","new_rep3"]
TARGET_ORDER=["transition_mid","sparse_drive","transition_dense"]

def load_cfg(): return json.loads((PKG/"frozen_stage4"/"stage4a.json").read_text(encoding="utf-8"))

def _metrics_to_rows(met,seeds,actuals,labels):
    arrays={}
    for k,v in met.items(): arrays[k]=v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    rows=[]
    for si,seed in enumerate(seeds):
        for j,lab in enumerate(labels):
            r={"validation_seed":int(seed),"actual_noise_seed":int(actuals[si]),"source_label":lab}
            for k,a in arrays.items(): r[k]=float(np.asarray(a[si,j]).mean())
            rows.append(r)
    return rows

def evaluate_turbo(cfg,A,masks,labels,target_params,noise_seeds,graph_seed,mask_batch,seed_batch):
    rows=[]; masks=np.stack(masks).astype(bool); M=len(masks)
    mask_batch=max(1,min(int(mask_batch),M)); seed_batch=max(1,int(seed_batch))
    for ss in range(0,len(noise_seeds),seed_batch):
        seeds=list(map(int,noise_seeds[ss:ss+seed_batch])); actuals=[s+int(graph_seed)*1000003 for s in seeds]
        for ms in range(0,M,mask_batch):
            me=min(ms+mask_batch,M); labs=labels[ms:me]
            T=torch.as_tensor(masks[ms:me],dtype=torch.bool,device=A.device)
            met=simulate_masks_seed_batch(cfg,A,T,float(target_params["coupling"]),float(target_params["noise_sigma"]),actuals,common_noise=True)
            rows.extend(_metrics_to_rows(met,seeds,actuals,labs))
    return pd.DataFrame(rows)

def evaluate_legacy(cfg,A,masks,labels,target_params,noise_seeds,graph_seed,mask_batch):
    rows=[]; masks=np.stack(masks).astype(bool); M=len(masks)
    for seed in noise_seeds:
        actual=int(seed)+int(graph_seed)*1000003
        for ms in range(0,M,mask_batch):
            me=min(ms+mask_batch,M); labs=labels[ms:me]
            T=torch.as_tensor(masks[ms:me],dtype=torch.bool,device=A.device)
            met=simulate_masks(cfg,A,T,float(target_params["coupling"]),float(target_params["noise_sigma"]),actual,common_noise=True)
            met2={k:v.reshape(1,-1) for k,v in met.items()}
            rows.extend(_metrics_to_rows(met2,[seed],[actual],labs))
    return pd.DataFrame(rows)

def task_dir(outroot,graph_seed,k,target_state): return outroot/"tasks"/f"seed{int(graph_seed)}_k{int(k)}"/target_state

def prepare_masks(reev,graph_seed,k):
    sub=reev[(reev.graph_seed==graph_seed)&(reev.k==k)].copy()
    if len(sub)!=12: raise RuntimeError(f"anchor seed={graph_seed} k={k}: expected 12 source masks, got {len(sub)}")
    labels=[]; masks=[]; meta=[]
    for src_state in TARGET_ORDER:
        q=sub[sub.regime.astype(str)==src_state]
        for rep in REPS:
            z=q[q.replicate_label.astype(str)==rep]
            if len(z)!=1: raise RuntimeError(f"missing unique mask {graph_seed=} {k=} {src_state=} {rep=}")
            row=z.iloc[0]; labels.append(f"{src_state}|{rep}")
            masks.append(mask_bool(parse_mask_key(row.mask_key,int(row.N)),int(row.N)))
            meta.append({"source_state":src_state,"replicate_label":rep,"source_label":labels[-1],"mask_key":row.mask_key})
    return sub,labels,masks,pd.DataFrame(meta)

def run_task(reev,cfg,outroot,graph_seed,k,target_state,noise_seeds,mask_batch,seed_batch,device,mode="turbo"):
    tdir=task_dir(outroot,graph_seed,k,target_state); ensure_dir(tdir)
    done=tdir/"done.json"; summp=tdir/"mask_summary.csv"; seedp=tdir/"seed_metrics.csv"
    if done.exists() and summp.exists() and seedp.exists():
        d=json.loads(done.read_text(encoding="utf-8"))
        if d.get("noise_seeds")==list(map(int,noise_seeds)):
            return pd.read_csv(summp),pd.read_csv(seedp),True
    sub,labels,masks,md=prepare_masks(reev,graph_seed,k)
    N=int(sub.N.iloc[0]); p=REGIME_PARAMS[target_state]["connection_prob"]
    A_np,_=build_topology("scale_free",N,p,int(graph_seed)); A=to_gpu_adjacency(A_np,device)
    if mode=="legacy": raw=evaluate_legacy(cfg,A,masks,labels,REGIME_PARAMS[target_state],noise_seeds,graph_seed,mask_batch)
    else: raw=evaluate_turbo(cfg,A,masks,labels,REGIME_PARAMS[target_state],noise_seeds,graph_seed,mask_batch,seed_batch)
    raw["graph_seed"]=graph_seed; raw["k"]=k; raw["N"]=N; raw["target_state"]=target_state; raw=raw.merge(md,on="source_label",how="left")
    agg=raw.groupby(["graph_seed","k","N","source_state","replicate_label","source_label","target_state","mask_key"],as_index=False).agg(
        rhythm_score_mean=("rhythm_score","mean"),rhythm_score_std=("rhythm_score","std"),mean_rate_hz_mean=("mean_rate_hz","mean"),
        dominant_frequency_hz_mean=("dominant_frequency_hz","mean"),spectral_concentration_mean=("spectral_concentration","mean"),
        spectral_entropy_mean=("spectral_entropy","mean"),synchrony_proxy_mean=("synchrony_proxy","mean"),silent_fraction_mean=("silent_fraction","mean"),
        n_validation_seeds=("validation_seed","nunique"))
    raw.to_csv(seedp,index=False); agg.to_csv(summp,index=False)
    write_json(done,{"status":"COMPLETE","engine":mode,"graph_seed":int(graph_seed),"k":int(k),"target_state":target_state,
                     "noise_seeds":list(map(int,noise_seeds)),"mask_batch":int(mask_batch),"seed_batch":int(seed_batch),"masks":len(masks),"rows":len(raw)})
    del A
    if device.type=="cuda": torch.cuda.empty_cache()
    return agg,raw,False

def analyse(summary,outroot):
    A="transition_mid"; B="sparse_drive"
    m=summary.groupby(["graph_seed","k","source_state","target_state"],as_index=False)["rhythm_score_mean"].mean()
    home=m[m.source_state==m.target_state][["graph_seed","k","target_state","rhythm_score_mean"]].rename(columns={"rhythm_score_mean":"home_mean"})
    centered=m.merge(home,on=["graph_seed","k","target_state"],how="left"); centered["delta_vs_target_home"]=centered["rhythm_score_mean"]-centered["home_mean"]
    centered.to_csv(outroot/"analysis"/"transfer_centered_by_anchor.csv",index=False)
    rows=[]
    for (gs,k,rep),g in summary.groupby(["graph_seed","k","replicate_label"]):
        def score(src,tgt):
            q=g[(g.source_state==src)&(g.target_state==tgt)]; return float(q.rhythm_score_mean.iloc[0]) if len(q)==1 else np.nan
        aa=score(A,A); ba=score(B,A); ab=score(A,B); bb=score(B,B)
        if np.isfinite([aa,ba,ab,bb]).all(): rows.append({"graph_seed":gs,"k":k,"replicate_label":rep,"mid_home_advantage":aa-ba,"sparse_home_advantage":bb-ab,"crossover_interaction":.5*((aa-ba)+(bb-ab)),"mid_in_mid":aa,"sparse_in_mid":ba,"mid_in_sparse":ab,"sparse_in_sparse":bb})
    inter=pd.DataFrame(rows); inter.to_csv(outroot/"analysis"/"pure_state_crossover_by_replicate.csv",index=False)
    anchor=inter.groupby(["graph_seed","k"],as_index=False).agg(mid_home_advantage=("mid_home_advantage","mean"),sparse_home_advantage=("sparse_home_advantage","mean"),crossover_interaction=("crossover_interaction","mean"))
    anchor.to_csv(outroot/"analysis"/"pure_state_crossover_by_anchor.csv",index=False)
    mu,lo,hi=bootstrap_mean_ci(anchor.crossover_interaction,10000,20260820); p=exact_signflip_p(anchor.crossover_interaction,"greater")
    comp=[]
    for X,Y in [("transition_dense",A),("transition_dense",B)]:
        rr=[]
        for (gs,k,rep),g in summary.groupby(["graph_seed","k","replicate_label"]):
            def sc(src,tgt):
                q=g[(g.source_state==src)&(g.target_state==tgt)]; return float(q.rhythm_score_mean.iloc[0]) if len(q)==1 else np.nan
            xx,yx,xy,yy=sc(X,X),sc(Y,X),sc(X,Y),sc(Y,Y)
            if np.isfinite([xx,yx,xy,yy]).all(): rr.append(.5*((xx-yx)+(yy-xy)))
        comp.append({"contrast":f"{X} vs {Y}","interpretation":"topology+dynamics composite because connection_prob differs","mean_interaction":float(np.mean(rr)) if rr else np.nan,"n_replicate_pairs":len(rr)})
    pd.DataFrame(comp).to_csv(outroot/"analysis"/"composite_dense_interactions.csv",index=False)
    mat=centered.groupby(["source_state","target_state"],as_index=False)["delta_vs_target_home"].mean(); mat.to_csv(outroot/"analysis"/"transfer_matrix_mean.csv",index=False)
    metrics={"status":"COMPLETE","engine":"v2_turbo_seed_batch","primary_contrast":"transition_mid vs sparse_drive (same topology density p=0.05)","n_anchors":int(len(anchor)),"n_replicate_pairs":int(len(inter)),"mean_crossover_interaction":mu,"bootstrap95_low":lo,"bootstrap95_high":hi,"exact_anchor_signflip_p_greater":p,"claim_rule":"Positive crossover means each state's own optimized masks outperform foreign-state masks when evaluated in that target state.","dense_warning":"transition_dense uses p=0.10 and is a topology+dynamics composite, not a pure state contrast."}
    write_json(outroot/"analysis"/"cross_state_key_metrics.json",metrics)
    setup_mpl(); import matplotlib.pyplot as plt
    order=[A,B,"transition_dense"]; M=np.zeros((3,3),float)
    for i,s in enumerate(order):
        for j,t in enumerate(order):
            q=mat[(mat.source_state==s)&(mat.target_state==t)]; M[i,j]=q.delta_vs_target_home.iloc[0] if len(q) else np.nan
    vmax=max(.02,float(np.nanmax(np.abs(M)))); fig,ax=plt.subplots(figsize=(5.7,4.8)); im=ax.imshow(M,cmap="RdBu_r",vmin=-vmax,vmax=vmax)
    labs=[STATE_LABEL[x] for x in order]; ax.set_xticks(range(3));ax.set_xticklabels(labs,rotation=30,ha="right");ax.set_yticks(range(3));ax.set_yticklabels(labs);ax.set_xlabel("target state");ax.set_ylabel("source-state optimized masks")
    for i in range(3):
        for j in range(3): ax.text(j,i,f"{M[i,j]:.3f}",ha="center",va="center",color="white" if abs(M[i,j])>.55*vmax else COL["dark"])
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03);cb.set_label("Δ score vs target-state home ensemble"); save_panel(fig,outroot/"figures"/"CT01_transfer_matrix_centered")
    fig,ax=plt.subplots(figsize=(5.7,4.4)); xs={8:0,32:1,64:2}; rng=np.random.default_rng(4)
    for k,g in anchor.groupby("k"):
        x=xs[int(k)]; ax.scatter(np.full(len(g),x)+rng.normal(0,.045,len(g)),g.crossover_interaction,s=34,alpha=.72,color=COL["blue"]);ax.plot([x-.18,x+.18],[g.crossover_interaction.mean()]*2,lw=3,color=COL["rose"])
    ax.axhline(0,ls="--",lw=1,color=COL["gray"]);ax.set_xticks([0,1,2]);ax.set_xticklabels(["k=8","k=32","k=64"]);ax.set_ylabel("pure-state crossover interaction");clean_ax(ax);save_panel(fig,outroot/"figures"/"CT02_pure_state_crossover_by_budget")
    fig,ax=plt.subplots(figsize=(5.7,4.4)); vals=[anchor.mid_home_advantage,anchor.sparse_home_advantage]; labs=["mid target","sparse target"]
    for i,v in enumerate(vals): ax.scatter(np.full(len(v),i)+rng.normal(0,.035,len(v)),v,s=34,alpha=.72,color=[COL["teal"],COL["orange"]][i]);ax.plot([i-.18,i+.18],[np.mean(v)]*2,lw=3,color=COL["dark"])
    ax.axhline(0,ls="--",lw=1,color=COL["gray"]);ax.set_xticks([0,1]);ax.set_xticklabels(labs);ax.set_ylabel("home-mask advantage over foreign mask");clean_ax(ax);save_panel(fig,outroot/"figures"/"CT03_home_advantage")
    return metrics

def build_manifest(reev):
    anchors=reev[["graph_seed","k"]].drop_duplicates().sort_values(["graph_seed","k"]); out=[]
    for _,r in anchors.iterrows():
        for target in TARGET_ORDER: out.append({"graph_seed":int(r.graph_seed),"k":int(r.k),"target_state":target})
    return out

def finalize(outroot,expected_noise_seeds):
    aggs=[]; raws=[]; missing=[]
    man=pd.read_csv(outroot/"analysis"/"task_manifest.csv")
    for _,t in man.iterrows():
        td=task_dir(outroot,int(t.graph_seed),int(t.k),str(t.target_state)); done=td/"done.json"; sp=td/"mask_summary.csv"; rp=td/"seed_metrics.csv"
        if not(done.exists() and sp.exists() and rp.exists()): missing.append(str(td)); continue
        d=json.loads(done.read_text(encoding="utf-8"))
        if d.get("noise_seeds")!=list(map(int,expected_noise_seeds)): missing.append(str(td)+" [seed mismatch]"); continue
        aggs.append(pd.read_csv(sp)); raws.append(pd.read_csv(rp))
    if missing: raise RuntimeError(f"Cannot finalize: {len(missing)} incomplete/mismatched tasks. First: {missing[:5]}")
    summary=pd.concat(aggs,ignore_index=True); raw=pd.concat(raws,ignore_index=True)
    if len(summary)!=27*12: raise RuntimeError(f"Expected 324 mask-summary rows, got {len(summary)}")
    if len(raw)!=27*12*len(expected_noise_seeds): raise RuntimeError(f"Expected {27*12*len(expected_noise_seeds)} seed rows, got {len(raw)}")
    summary.to_csv(outroot/"analysis"/"cross_state_mask_summary.csv",index=False); raw.to_csv(outroot/"analysis"/"cross_state_seed_metrics.csv",index=False)
    return analyse(summary,outroot)

def main():
    ap=argparse.ArgumentParser(description="Stage4 cross-state transfer v2 Turbo: exact science, seed-batched GPU execution and task sharding.")
    ap.add_argument("--project-root",default="/data/coding/NeuralScience");ap.add_argument("--stage4f-source",default="auto");ap.add_argument("--output-root",default=None)
    ap.add_argument("--noise-seed-start",type=int,default=7601);ap.add_argument("--noise-seed-count",type=int,default=24)
    ap.add_argument("--mask-batch",type=int,default=12,help="There are only 12 source masks/task; default already evaluates all at once.")
    ap.add_argument("--seed-batch",type=int,default=4,help="Independent noise seeds jointly vectorized on GPU.")
    ap.add_argument("--task-shard-index",type=int,default=0);ap.add_argument("--task-shard-count",type=int,default=1)
    ap.add_argument("--device",default="cuda");ap.add_argument("--engine",choices=["turbo","legacy"],default="turbo")
    ap.add_argument("--dry-run",action="store_true");ap.add_argument("--no-finalize",action="store_true");ap.add_argument("--finalize-only",action="store_true")
    args=ap.parse_args(); root=Path(args.project_root).resolve(); source=resolve_stage4f_source(root,args.stage4f_source)
    outroot=ensure_dir(Path(args.output_root) if args.output_root else root/"results_complexity_leverage_closure_v2_turbo"/"stage4_cross_state_transfer")
    ensure_dir(outroot/"analysis");ensure_dir(outroot/"figures"); reev=load_stage4f_reevaluation(source); manifest=build_manifest(reev)
    pd.DataFrame(manifest).to_csv(outroot/"analysis"/"task_manifest.csv",index=False)
    noise=list(range(args.noise_seed_start,args.noise_seed_start+args.noise_seed_count))
    if args.dry_run:
        shard=[t for i,t in enumerate(manifest) if i%args.task_shard_count==args.task_shard_index]
        print(json.dumps({"status":"DRY_RUN_OK","engine":args.engine,"stage4f_source":str(source),"conditions":27,"source_masks":108,"transfer_tasks":len(manifest),"this_shard_tasks":len(shard),"seed_batch":args.seed_batch,"mask_batch":args.mask_batch,"noise_seeds":noise},indent=2));return
    if args.finalize_only:
        met=finalize(outroot,noise); print(json.dumps(met,indent=2)); return
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    if args.task_shard_count<1 or not(0<=args.task_shard_index<args.task_shard_count): raise ValueError("invalid shard index/count")
    device=torch.device(args.device); cfg=load_cfg(); shard=[t for i,t in enumerate(manifest) if i%args.task_shard_count==args.task_shard_index]; t0=time.time();skipped=0
    for ii,t in enumerate(shard,1):
        print(f"[shard {args.task_shard_index+1}/{args.task_shard_count}] [{ii}/{len(shard)}] seed={t['graph_seed']} k={t['k']} target={t['target_state']}",flush=True)
        _,_,was=run_task(reev,cfg,outroot,t["graph_seed"],t["k"],t["target_state"],noise,args.mask_batch,args.seed_batch,device,args.engine); skipped+=int(was)
    rep={"status":"SHARD_COMPLETE","shard_index":args.task_shard_index,"shard_count":args.task_shard_count,"tasks":len(shard),"skipped_completed":skipped,"runtime_minutes":(time.time()-t0)/60,"engine":args.engine,"seed_batch":args.seed_batch}
    write_json(outroot/"analysis"/f"shard_{args.task_shard_index:02d}_complete.json",rep); print(json.dumps(rep,indent=2))
    if not args.no_finalize and args.task_shard_count==1:
        met=finalize(outroot,noise); print(json.dumps(met,indent=2))
if __name__=="__main__": main()
