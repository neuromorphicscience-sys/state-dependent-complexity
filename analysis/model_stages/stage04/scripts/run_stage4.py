from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.stage4_topology import build_topology,to_gpu_adjacency,heuristic_mask,graph_descriptors,set_descriptors
from src.stage4_optimizer import evolve, random_fixed_k
from src.stage4_gpu_simulator import simulate_masks
from src.stage4_safeio import atomic_write_csv,atomic_write_json,atomic_torch_save

HEURISTICS=["high_degree","feedback_hub","module_bridge","cycle_proxy","spectral"]

def task_id(task):
    return hashlib.sha1(json.dumps(task,sort_keys=True).encode()).hexdigest()[:18]

def make_tasks(cfg,phase):
    seeds=cfg["discovery_graph_seeds"] if phase=="discovery" else cfg["heldout_graph_seeds"]
    tasks=[]
    for topo in cfg["topologies"]:
        for reg in cfg["regimes"]:
            for seed in seeds:
                for k in cfg["k_values"]:
                    t={
                        "phase":phase,"topology":topo,"regime":reg["name"],
                        "coupling":reg["coupling"],"connection_prob":reg["connection_prob"],
                        "noise_sigma":reg["noise_sigma"],"graph_seed":int(seed),"k":int(k)
                    }
                    t["task_id"]=task_id(t); tasks.append(t)
    return tasks

def mask_key(mask):
    return ",".join(map(str,np.flatnonzero(mask).tolist()))

def evaluate_methods(cfg,A,a_np,modules,task,best_mask,device):
    k=task["k"]; n=A.shape[0]
    masks=[]; labels=[]
    # exact same graph for all methods
    for h in HEURISTICS:
        masks.append(heuristic_mask(a_np,modules,k,h,task["graph_seed"])); labels.append(h)
    # many random placements
    rg=np.random.default_rng(task["graph_seed"]+k*10007+31)
    for i in range(int(cfg["validation_random_placements"])):
        m=np.zeros(n,dtype=bool); m[rg.choice(n,size=k,replace=False)]=True
        masks.append(m); labels.append("random")
    masks.append(best_mask.detach().cpu().numpy().astype(bool)); labels.append("gpu_optimized")

    T=torch.as_tensor(np.stack(masks),dtype=torch.bool,device=device)
    # final validation across multiple common-noise seeds
    score_stack=[]; metric_acc={}
    for s in cfg["validation_noise_seeds"]:
        met=simulate_masks(cfg,A,T,task["coupling"],task["noise_sigma"],int(s)+task["graph_seed"]*1000003,common_noise=True)
        score_stack.append(met["rhythm_score"])
        for key,val in met.items():
            metric_acc.setdefault(key,[]).append(val)
    rows=[]
    score_mat=torch.stack(score_stack,0)
    for i,(lab,m) in enumerate(zip(labels,masks)):
        desc=set_descriptors(a_np,modules,m)
        rows.append({
            **task,
            "method":lab,
            "mask_key":mask_key(m),
            "rhythm_score_mean":float(score_mat[:,i].mean().item()),
            "rhythm_score_std":float(score_mat[:,i].std(unbiased=False).item()),
            **desc,
        })
    return rows

def run_task(cfg,task,outdir,device):
    ckpt_dir=outdir/"checkpoints"; ckpt_dir.mkdir(parents=True,exist_ok=True)
    done_dir=outdir/"tasks"; done_dir.mkdir(parents=True,exist_ok=True)
    final_path=done_dir/f"{task['task_id']}.csv"
    if final_path.exists():
        return pd.read_csv(final_path)

    a_np,modules=build_topology(task["topology"],cfg["n_neurons"],task["connection_prob"],task["graph_seed"])
    A=to_gpu_adjacency(a_np,device)

    initial=[heuristic_mask(a_np,modules,task["k"],h,task["graph_seed"]) for h in HEURISTICS]
    ckpt_path=ckpt_dir/f"{task['task_id']}.pt"
    resume=None
    if ckpt_path.exists():
        try: resume=torch.load(ckpt_path,map_location="cpu",weights_only=False)
        except Exception: resume=None

    def checkpoint_cb(state):
        atomic_torch_save(ckpt_path,state)

    best_mask,best_score,history,_=evolve(
        cfg,A,task["k"],task["coupling"],task["noise_sigma"],
        search_seed=task["graph_seed"]*100000+task["k"]*101+len(task["topology"]),
        initial_masks=initial,resume=resume,checkpoint_cb=checkpoint_cb
    )

    rows=evaluate_methods(cfg,A,a_np,modules,task,best_mask,device)
    gd=graph_descriptors(a_np)
    for r in rows:
        r.update(gd)
        r["optimizer_search_best"]=best_score
        r["optimizer_last_generation"]=history[-1]["generation"] if history else -1
        r["optimizer_history_best"]=json.dumps(history,separators=(",",":"))

    df=pd.DataFrame(rows)
    atomic_write_csv(final_path,df)
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--phase",choices=["discovery","heldout"],default="discovery")
    args=ap.parse_args()
    cfg=json.load(open(args.config,"r",encoding="utf-8"))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda":
        raise RuntimeError("Stage 4 formal search requires CUDA")

    outdir=ROOT/cfg.get("output_dir","results_stage4")
    outdir.mkdir(parents=True,exist_ok=True)
    tasks=make_tasks(cfg,args.phase)

    print(f"Stage 4 {args.phase}: {len(tasks)} optimization tasks")
    print("Device:",torch.cuda.get_device_name(0))
    print("Population:",cfg["optimizer"]["population"],"Generations:",cfg["optimizer"]["generations"])
    print("dt/duration:",cfg["dt_ms"],"ms /",cfg["duration_ms"],"ms")

    all_rows=[]
    t0=time.time()
    for i,t in enumerate(tasks,1):
        print(f"\n[{i}/{len(tasks)}] {t}")
        df=run_task(cfg,t,outdir,device)
        all_rows.append(df)
        atomic_write_json(outdir/f"progress_{args.phase}.json",{
            "phase":args.phase,"completed_tasks":i,"total_tasks":len(tasks),
            "elapsed_min":(time.time()-t0)/60.0,"last_task":t["task_id"]
        })
    final=pd.concat(all_rows,ignore_index=True)
    atomic_write_csv(outdir/f"stage4_{args.phase}_methods.csv",final)
    print("\nSaved:",outdir/f"stage4_{args.phase}_methods.csv")

if __name__=="__main__":
    main()
