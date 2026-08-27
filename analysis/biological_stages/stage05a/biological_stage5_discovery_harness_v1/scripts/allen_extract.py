#!/usr/bin/env python3
import argparse, csv, os, re, json, math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import h5py
from common import load_config, ensure_dir, write_csv, write_json, now

def decode_scalar(x):
    try:
        if hasattr(x,"shape") and x.shape==(): x=x[()]
    except Exception: pass
    if isinstance(x,(bytes,np.bytes_)): return x.decode("utf-8","replace")
    if isinstance(x,np.ndarray) and x.size==1: return decode_scalar(x.reshape(-1)[0])
    return str(x)

def rate_from_group(g):
    try:
        st=g.get("starting_time")
        if st is not None and "rate" in st.attrs: return float(st.attrs["rate"])
    except Exception: pass
    for obj in [g,g.get("data")]:
        if obj is None: continue
        for k in ("rate","sampling_rate","sampling_rate_hz"):
            if k in obj.attrs:
                try: return float(obj.attrs[k])
                except Exception: pass
    return float("nan")

def unit_string(ds):
    if ds is None: return ""
    for k in ("unit","units"):
        if k in ds.attrs:
            return decode_scalar(ds.attrs[k])
    return ""

def apply_conversion(arr,ds,kind):
    arr=np.asarray(arr,dtype=np.float32)
    conv=1.0
    if ds is not None and "conversion" in ds.attrs:
        try: conv=float(ds.attrs["conversion"])
        except Exception: pass
    arr=arr*conv
    unit=unit_string(ds).lower()
    m=float(np.nanmedian(np.abs(arr))) if arr.size else 0.0
    if kind=="voltage":
        if "mv" in unit: pass
        elif unit in ("v","volt","volts") or (m>0 and m<1.0): arr*=1000.0
    else:
        if "pa" in unit: pass
        elif "na" in unit: arr*=1000.0
        elif unit in ("a","amp","amps","ampere","amperes") or (m>0 and m<1e-6): arr*=1e12
    return arr

def find_stim_name(f,sweep_name,sweep_num,stim_g):
    candidates=[]
    if stim_g is not None:
        for k in ("stimulus_name","stimulus_description","description","comments","source"):
            if k in stim_g.attrs: candidates.append(decode_scalar(stim_g.attrs[k]))
        for k in stim_g.keys():
            if any(t in k.lower() for t in ("name","description","stimulus")):
                try:
                    obj=stim_g[k]
                    if isinstance(obj,h5py.Dataset) and obj.size<=10:
                        candidates.append(decode_scalar(obj[()]))
                except Exception: pass
    ep=f.get("/epochs")
    if ep is not None:
        for key in (sweep_name,f"Sweep_{sweep_num}",str(sweep_num)):
            if key in ep:
                eg=ep[key]
                for k in ("aibs_stimulus_name","stimulus_name","description"):
                    if k in eg:
                        try: candidates.append(decode_scalar(eg[k][()]))
                        except Exception: pass
                    if k in eg.attrs: candidates.append(decode_scalar(eg.attrs[k]))
    for x in candidates:
        if x and x.lower() not in ("none","nan"): return x
    return sweep_name

def load_sweep_lookup(allen_root):
    lookup={}
    root=Path(allen_root)
    for p in root.rglob("*.csv"):
        if "sweep" not in p.name.lower(): continue
        try:
            with p.open("r",encoding="utf-8-sig",newline="") as f:
                rd=csv.DictReader(f)
                cols=rd.fieldnames or []
                sidc=next((c for c in cols if c.lower() in ("specimen_id","specimen__id")),None)
                swc=next((c for c in cols if "sweep" in c.lower() and ("num" in c.lower() or c.lower()=="sweep_number")),None)
                nc=next((c for c in cols if "stimulus" in c.lower() and ("name" in c.lower() or "description" in c.lower())),None)
                if not(sidc and swc and nc): continue
                for r in rd:
                    try: lookup[(str(int(float(r[sidc]))),int(float(r[swc])))] = r[nc]
                    except Exception: pass
            if lookup:
                print(f"Loaded sweep-name lookup: {p} rows={len(lookup)}",flush=True)
                break
        except Exception: pass
    return lookup

def integer_downsample(x,step):
    n=(len(x)//step)*step
    if n<=0: return np.asarray([],np.float32)
    return x[:n].reshape(-1,step).mean(axis=1).astype(np.float32)

def spike_stats(v,rate):
    if len(v)<3 or not np.isfinite(rate) or rate<=0: return (0,float("nan"),float("nan"),float("nan"))
    thr=-20.0
    idx=np.flatnonzero((v[:-1]<thr)&(v[1:]>=thr))+1
    refractory=max(1,int(rate*0.001))
    if len(idx)>1:
        keep=[int(idx[0])]
        for x in idx[1:]:
            if x-keep[-1]>=refractory: keep.append(int(x))
        idx=np.asarray(keep)
    if len(idx)<2: return (len(idx),float("nan"),float("nan"),float("nan"))
    isi=np.diff(idx)/rate
    cv=float(np.std(isi)/np.mean(isi)) if np.mean(isi)>0 else float("nan")
    adapt=float(np.median(isi[len(isi)//2:])/np.median(isi[:max(1,len(isi)//2)])) if len(isi)>=4 else float("nan")
    return (len(idx),float(np.mean(isi)),cv,adapt)

def extract_one(args):
    path_str,sid,out_dir,target_rate,lookup=args
    p=Path(path_str)
    try:
        with h5py.File(p,"r") as f:
            acq=f.get("/acquisition/timeseries")
            stimroot=f.get("/stimulus/presentation")
            if acq is None: raise RuntimeError("missing /acquisition/timeseries")
            sweeps=[]
            for name,g in acq.items():
                if not isinstance(g,h5py.Group) or "data" not in g: continue
                m=re.search(r"(\d+)",name); sw=int(m.group(1)) if m else -1
                rg=g; rd=g["data"]
                sg=None
                if stimroot is not None:
                    if name in stimroot: sg=stimroot[name]
                    elif f"Sweep_{sw}" in stimroot: sg=stimroot[f"Sweep_{sw}"]
                if sg is None or "data" not in sg: continue
                sd=sg["data"]
                rate=rate_from_group(rg)
                if not np.isfinite(rate) or rate<=0: continue
                v=apply_conversion(rd[()],rd,"voltage")
                cur=apply_conversion(sd[()],sd,"current")
                n=min(len(v),len(cur)); v=v[:n]; cur=cur[:n]
                stim_name=lookup.get((sid,sw)) or find_stim_name(f,name,sw,sg)
                sc,meanisi,cv,adapt=spike_stats(v,rate)
                sweeps.append({
                    "number":sw,"name":stim_name,"rate":rate,"v":v,"i":cur,
                    "spike_count":sc,"mean_isi":meanisi,"isi_cv":cv,"adaptation":adapt
                })
        noise=[s for s in sweeps if "noise" in s["name"].lower()]
        if len(noise)<2:
            return {"specimen_id":sid,"status":"no_two_noise","n_sweeps":len(sweeps),"n_noise":len(noise)}
        noise=sorted(noise,key=lambda s:(-len(s["v"]),s["number"]))[:2]
        arrays={}
        meta={"specimen_id":sid,"status":"ok","n_sweeps":len(sweeps),"n_noise":len(noise)}
        for j,s in enumerate(noise):
            step=max(1,int(round(s["rate"]/target_rate)))
            actual=s["rate"]/step
            vv=integer_downsample(np.clip(s["v"],-120,-20),step)
            ii=integer_downsample(s["i"],step)
            n=min(len(vv),len(ii)); vv=vv[:n]; ii=ii[:n]
            arrays[f"stim{j}"]=ii
            arrays[f"volt{j}"]=vv
            meta[f"sweep{j}_number"]=s["number"]
            meta[f"sweep{j}_name"]=s["name"]
            meta[f"sweep{j}_orig_rate"]=s["rate"]
            meta[f"sweep{j}_rate"]=actual
            meta[f"sweep{j}_samples"]=n
            meta[f"sweep{j}_spike_count"]=s["spike_count"]
            meta[f"sweep{j}_mean_isi"]=s["mean_isi"]
            meta[f"sweep{j}_isi_cv"]=s["isi_cv"]
            meta[f"sweep{j}_adaptation"]=s["adaptation"]
        rate0=float(meta["sweep0_rate"]); rate1=float(meta["sweep1_rate"])
        if abs(rate0-target_rate)>5 or abs(rate1-target_rate)>5:
            meta["status"]="rate_warning"
        arrays["specimen_id"]=np.asarray([int(sid)],dtype=np.int64)
        arrays["rate_hz"]=np.asarray([target_rate],dtype=np.float32)
        np.savez_compressed(Path(out_dir)/f"{sid}.npz",**arrays)
        return meta
    except Exception as e:
        return {"specimen_id":sid,"status":"failed","detail":f"{type(e).__name__}: {e}"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True)
    a=ap.parse_args(); c=load_config(a.config)
    root=Path(c["allen_root"]); out=ensure_dir(Path(c["output_root"])/"20_allen_ephys")
    npzdir=ensure_dir(out/"noise_npz")
    lookup=load_sweep_lookup(root)
    files=sorted((root/"ephys_nwb"/"raw").glob("*.nwb"))
    jobs=[]
    for p in files:
        m=re.search(r"(\d+)",p.stem)
        if m: jobs.append((str(p),m.group(1),str(npzdir),int(c["target_rate_hz"]),lookup))
    workers=int(c.get("allen_cpu_workers",16))
    print(f"Allen NWB files={len(jobs)} workers={workers} target_rate={c['target_rate_hz']}Hz",flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(extract_one,j):j[1] for j in jobs}
        done=0
        for fut in as_completed(futs):
            r=fut.result(); rows.append(r); done+=1
            if done%50==0 or done==len(jobs):
                ok=sum(x.get("status") in ("ok","rate_warning") for x in rows)
                print(f"Allen extraction {done}/{len(jobs)} usable_noise={ok}",flush=True)
    rows.sort(key=lambda r:int(r["specimen_id"]))
    write_csv(out/"allen_noise_extraction_summary.csv",rows)
    summary={
        "created_utc":now(),"nwb_files":len(jobs),
        "usable_two_noise":sum(r.get("status") in ("ok","rate_warning") for r in rows),
        "failed":sum(r.get("status")=="failed" for r in rows),
        "no_two_noise":sum(r.get("status")=="no_two_noise" for r in rows),
        "target_rate_hz":c["target_rate_hz"]
    }
    write_json(out/"allen_ephys_summary.json",summary)
    print(summary)

if __name__=="__main__": main()
