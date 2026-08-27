from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.stage4_topology import build_topology,to_gpu_adjacency,heuristic_mask,set_descriptors,graph_descriptors
from src.stage4_gpu_simulator import simulate_masks
from src.stage4_safeio import atomic_write_csv

HEUR=["high_degree","feedback_hub","module_bridge","cycle_proxy","spectral"]

def load_law(path):
    d=json.load(open(path,"r",encoding="utf-8"))
    return d

def predict_scores(descs,law):
    X=np.asarray([[d[f] for f in law["features"]] for d in descs],float)
    Z=(X-np.asarray(law["mean"]))/(np.asarray(law["std"])+1e-9)
    beta=np.asarray(law["beta"])
    return beta[0]+Z@beta[1:]

def random_library(n,k,size,seed):
    rng=np.random.default_rng(seed)
    arr=np.zeros((size,n),dtype=bool)
    for i in range(size):
        arr[i,rng.choice(n,size=k,replace=False)]=True
    return arr

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=json.load(open(args.config,"r",encoding="utf-8"))
    outdir=ROOT/cfg.get("output_dir","results_stage4")
    law=load_law(outdir/"stage4_allocation_law.json")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("CUDA required")

    rows=[]
    for topo in cfg["topologies"]:
        for reg in cfg["regimes"]:
            for seed in cfg["heldout_graph_seeds"]:
                a,modules=build_topology(topo,cfg["n_neurons"],reg["connection_prob"],seed)
                A=to_gpu_adjacency(a,device)
                gd=graph_descriptors(a)
                for k in cfg["k_values"]:
                    lib=random_library(cfg["n_neurons"],k,int(cfg["prospective_library_size"]),seed+k*111)
                    # Include heuristics in candidate library, but prediction is based only on learned law.
                    extra=np.stack([heuristic_mask(a,modules,k,h,seed) for h in HEUR])
                    lib=np.concatenate([lib,extra],axis=0)
                    descs=[]
                    for m in lib:
                        d=set_descriptors(a,modules,m); d.update(gd); descs.append(d)
                    pred=predict_scores(descs,law)
                    best=int(np.argmax(pred))
                    predicted=lib[best]

                    eval_masks=[predicted]
                    labels=["law_predicted"]
                    for h in HEUR:
                        eval_masks.append(heuristic_mask(a,modules,k,h,seed)); labels.append(h)
                    # paired random controls
                    rr=random_library(cfg["n_neurons"],k,int(cfg["heldout_random_controls"]),seed+k*333)
                    eval_masks.extend(rr); labels.extend(["random"]*len(rr))
                    T=torch.as_tensor(np.stack(eval_masks),dtype=torch.bool,device=device)

                    vals=[]
                    for ns in cfg["validation_noise_seeds"]:
                        met=simulate_masks(cfg,A,T,reg["coupling"],reg["noise_sigma"],int(ns)+seed*1000003,common_noise=True)
                        vals.append(met["rhythm_score"])
                    mat=torch.stack(vals,0)
                    for i,(lab,m) in enumerate(zip(labels,eval_masks)):
                        rows.append({
                            "topology":topo,"regime":reg["name"],"graph_seed":seed,"k":k,
                            "coupling":reg["coupling"],"connection_prob":reg["connection_prob"],"noise_sigma":reg["noise_sigma"],
                            "method":lab,
                            "rhythm_score_mean":float(mat[:,i].mean().item()),
                            "rhythm_score_std":float(mat[:,i].std(unbiased=False).item()),
                            "predicted_surrogate_score":float(pred[best]) if lab=="law_predicted" else np.nan,
                        })
                    print("heldout",topo,reg["name"],seed,k,"done")

    df=pd.DataFrame(rows)
    atomic_write_csv(outdir/"stage4_heldout_prediction.csv",df)

    keys=["topology","regime","graph_seed","k","coupling","connection_prob","noise_sigma"]
    rnd=df[df.method=="random"].groupby(keys,as_index=False).rhythm_score_mean.mean().rename(columns={"rhythm_score_mean":"random_mean"})
    heur=df[df.method.isin(HEUR)].groupby(keys,as_index=False).rhythm_score_mean.max().rename(columns={"rhythm_score_mean":"best_heuristic"})
    lawdf=df[df.method=="law_predicted"][keys+["rhythm_score_mean"]].rename(columns={"rhythm_score_mean":"law_predicted"})
    summ=lawdf.merge(rnd,on=keys).merge(heur,on=keys)
    summ["law_gain_vs_random"]=summ.law_predicted-summ.random_mean
    summ["law_gain_vs_heuristic"]=summ.law_predicted-summ.best_heuristic
    atomic_write_csv(outdir/"stage4_heldout_summary.csv",summ)
    print("\nProspective held-out summary:")
    print(summ.groupby("topology").agg(
        gain_vs_random=("law_gain_vs_random","mean"),
        gain_vs_heuristic=("law_gain_vs_heuristic","mean"),
        win_vs_random=("law_gain_vs_random",lambda x: float(np.mean(x>0))),
    ).to_string())

if __name__=="__main__":
    main()
