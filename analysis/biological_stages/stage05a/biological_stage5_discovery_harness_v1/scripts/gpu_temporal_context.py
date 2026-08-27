#!/usr/bin/env python3
import argparse, csv, json, math, os, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from common import load_config, ensure_dir, write_csv, write_json, now

class ContextNet(nn.Module):
    def __init__(self,n_cells,emb=32):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(1,64,9,padding=4,stride=2),
            nn.GELU(),
            nn.Conv1d(64,128,7,padding=3,stride=2),
            nn.GELU(),
            nn.Conv1d(128,128,5,padding=2,stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.emb=nn.Embedding(n_cells,emb)
        self.head=nn.Sequential(nn.Linear(128+emb,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,x,cell):
        h=self.conv(x[:,None,:]).squeeze(-1)
        return self.head(torch.cat([h,self.emb(cell)],dim=1)).squeeze(1)

def load_cells(npzdir,direction,device):
    files=sorted(Path(npzdir).glob("*.npz"))
    data=[]
    for p in files:
        try:
            z=np.load(p)
            sid=int(z["specimen_id"][0])
            if direction=="AB":
                tr_i=z["stim0"].astype(np.float32); tr_v=z["volt0"].astype(np.float32)
                te_i=z["stim1"].astype(np.float32); te_v=z["volt1"].astype(np.float32)
            else:
                tr_i=z["stim1"].astype(np.float32); tr_v=z["volt1"].astype(np.float32)
                te_i=z["stim0"].astype(np.float32); te_v=z["volt0"].astype(np.float32)
            n=min(len(tr_i),len(tr_v)); m=min(len(te_i),len(te_v))
            if n<1000 or m<1000: continue
            tr_i,tr_v=tr_i[:n],tr_v[:n]; te_i,te_v=te_i[:m],te_v[:m]
            im=float(np.mean(tr_i)); isd=float(np.std(tr_i)); isd=max(isd,1e-6)
            vm=float(np.mean(tr_v)); vsd=float(np.std(tr_v)); vsd=max(vsd,1e-3)
            tr_i=(tr_i-im)/isd; te_i=(te_i-im)/isd
            tr_v=(tr_v-vm)/vsd; te_v=(te_v-vm)/vsd
            data.append((sid,tr_i,tr_v,te_i,te_v))
        except Exception:
            pass
    if not data: raise RuntimeError("No usable Allen noise NPZ files.")
    max_tr=max(len(x[1]) for x in data); max_te=max(len(x[3]) for x in data)
    ncell=len(data)
    # CPU packing; converted to BF16 on device to reduce resident memory.
    tr_i=np.zeros((ncell,max_tr),np.float32); tr_v=np.zeros((ncell,max_tr),np.float32)
    te_i=np.zeros((ncell,max_te),np.float32); te_v=np.zeros((ncell,max_te),np.float32)
    tr_len=np.zeros(ncell,np.int64); te_len=np.zeros(ncell,np.int64); sids=[]
    for k,(sid,a,b,c,d) in enumerate(data):
        tr_i[k,:len(a)]=a; tr_v[k,:len(b)]=b; te_i[k,:len(c)]=c; te_v[k,:len(d)]=d
        tr_len[k]=len(a); te_len[k]=len(c); sids.append(sid)
    print(f"Packing {ncell} cells onto GPU: train max={max_tr}, test max={max_te}",flush=True)
    dtype=torch.bfloat16
    pack={
        "train_i":torch.from_numpy(tr_i).to(device=device,dtype=dtype),
        "train_v":torch.from_numpy(tr_v).to(device=device,dtype=dtype),
        "test_i":torch.from_numpy(te_i).to(device=device,dtype=dtype),
        "test_v":torch.from_numpy(te_v).to(device=device,dtype=dtype),
        "train_len":torch.from_numpy(tr_len).to(device),
        "test_len":torch.from_numpy(te_len).to(device),
        "sids":sids
    }
    return pack

def sample_batch(stim,volt,lengths,ctx,batch):
    n=stim.shape[0]
    cells=torch.randint(0,n,(batch,),device=stim.device)
    lens=lengths[cells]
    valid=torch.clamp(lens-ctx-1,min=1)
    offs=(torch.rand(batch,device=stim.device)*valid.float()).long()
    target=ctx+offs
    ar=torch.arange(ctx-1,-1,-1,device=stim.device)
    idx=target[:,None]-ar[None,:]
    x=stim[cells[:,None],idx]
    y=volt[cells,target].float()
    return x,cells,y

@torch.no_grad()
def evaluate(model,pack,ctx,windows,batch):
    stim=pack["test_i"]; volt=pack["test_v"]; lengths=pack["test_len"]
    n=stim.shape[0]
    sse=torch.zeros(n,device=stim.device)
    sy=torch.zeros(n,device=stim.device)
    sy2=torch.zeros(n,device=stim.device)
    cnt=torch.zeros(n,device=stim.device)
    # deterministic grid-ish random sampling, all cells represented equally.
    gen=torch.Generator(device=stim.device); gen.manual_seed(777+ctx)
    all_cells=torch.arange(n,device=stim.device).repeat_interleave(windows)
    for start in range(0,len(all_cells),batch):
        cells=all_cells[start:start+batch]
        lens=lengths[cells]
        valid=torch.clamp(lens-ctx-1,min=1)
        offs=(torch.rand(len(cells),device=stim.device,generator=gen)*valid.float()).long()
        target=ctx+offs
        ar=torch.arange(ctx-1,-1,-1,device=stim.device)
        idx=target[:,None]-ar[None,:]
        x=stim[cells[:,None],idx]
        y=volt[cells,target].float()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            pred=model(x,cells).float()
        err=(pred-y)**2
        sse.index_add_(0,cells,err); sy.index_add_(0,cells,y); sy2.index_add_(0,cells,y*y)
        cnt.index_add_(0,cells,torch.ones_like(y))
    sst=sy2-sy*sy/torch.clamp(cnt,min=1)
    r2=1.0-sse/torch.clamp(sst,min=1e-9)
    return r2.detach().cpu().numpy(),cnt.detach().cpu().numpy()

def train_one(pack,ctx,cfg,compile_model=True):
    device=pack["train_i"].device; n=pack["train_i"].shape[0]
    model=ContextNet(n,int(cfg["cell_embedding_dim"])).to(device)
    if compile_model and hasattr(torch,"compile"):
        try: model=torch.compile(model,mode="reduce-overhead")
        except Exception as e: print("torch.compile disabled after error:",e,flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]))
    batch=int(cfg["batch_size"]); steps=int(cfg["train_steps_per_context"])
    model.train()
    t0=time.time()
    step=0
    while step<steps:
        try:
            x,cell,y=sample_batch(pack["train_i"],pack["train_v"],pack["train_len"],ctx,batch)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                pred=model(x,cell)
                loss=torch.mean((pred.float()-y)**2)
            loss.backward(); opt.step()
            step+=1
            if step%100==0 or step==1:
                dt=time.time()-t0
                print(f"ctx={ctx} step={step}/{steps} batch={batch} loss={loss.item():.5f} "
                      f"steps/s={step/max(dt,1e-6):.2f} mem={torch.cuda.max_memory_allocated()/1024**3:.2f}GiB",flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch//=2
            if batch<256: raise
            print(f"CUDA OOM -> batch reduced to {batch}",flush=True)
    model.eval()
    return model,batch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--direction",choices=["AB","BA"],required=True)
    a=ap.parse_args(); c=load_config(a.config)
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required.")
    device=torch.device("cuda:0")
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=True
    try: torch.backends.cudnn.allow_tf32=True
    except Exception: pass
    torch.set_float32_matmul_precision("high")
    print("GPU:",torch.cuda.get_device_name(0),flush=True)
    print("CUDA:",torch.version.cuda,"Torch:",torch.__version__,flush=True)
    out=ensure_dir(Path(c["output_root"])/"30_temporal_context"/a.direction)
    npzdir=Path(c["output_root"])/"20_allen_ephys"/"noise_npz"
    pack=load_cells(npzdir,a.direction,device)
    contexts=[int(x) for x in c["contexts_ms"]] # target rate is 1 kHz => ms == samples.
    cfg=c["gpu_profile"]
    allrows=[]
    for ci,ctx in enumerate(contexts,1):
        torch.cuda.reset_peak_memory_stats()
        print("="*80,flush=True); print(f"{a.direction} context {ctx} ms ({ci}/{len(contexts)})",flush=True)
        model,batch=train_one(pack,ctx,cfg,bool(cfg.get("compile",True)))
        r2,nobs=evaluate(model,pack,ctx,int(cfg["eval_windows_per_cell"]),batch)
        for sid,rr,nn in zip(pack["sids"],r2,nobs):
            allrows.append({"specimen_id":sid,"direction":a.direction,"context_ms":ctx,
                            "r2":float(rr),"n_eval":int(nn)})
        write_csv(out/"context_performance_partial.csv",allrows)
        del model; torch.cuda.empty_cache()
    write_csv(out/"context_performance.csv",allrows)
    write_json(out/"run_summary.json",{
        "created_utc":now(),"direction":a.direction,"cells":len(pack["sids"]),
        "contexts_ms":contexts,"gpu":torch.cuda.get_device_name(0),"config":cfg
    })
    print(f"{a.direction} COMPLETE rows={len(allrows)}",flush=True)

if __name__=="__main__": main()
