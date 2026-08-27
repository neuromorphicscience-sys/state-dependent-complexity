#!/usr/bin/env python3
import argparse, csv, math, os, re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from common import load_config, ensure_dir, write_csv, write_json, now, extract_area_layer_from_path

def entropy_from_counts(vals):
    if not vals: return 0.0
    arr=np.asarray(list(vals),float); arr=arr[arr>0]
    p=arr/arr.sum()
    return float(-(p*np.log2(p)).sum())

def spatial_entropy(xyz, bins=4):
    if len(xyz)<2: return 0.0
    xyz=np.asarray(xyz,float)
    lo=np.nanmin(xyz,axis=0); hi=np.nanmax(xyz,axis=0)
    span=np.maximum(hi-lo,1e-9)
    q=np.floor((xyz-lo)/span*bins).astype(int)
    q=np.clip(q,0,bins-1)
    code=q[:,0]*bins*bins+q[:,1]*bins+q[:,2]
    _,cnt=np.unique(code,return_counts=True)
    return entropy_from_counts(cnt)

def swc_features(path_str):
    p=Path(path_str)
    nodes=[]
    try:
        with p.open("r",encoding="utf-8",errors="replace") as f:
            for line in f:
                s=line.strip()
                if not s or s.startswith("#"): continue
                a=s.split()
                if len(a)<7: continue
                try:
                    nid=int(float(a[0])); typ=int(float(a[1]))
                    x,y,z,r=map(float,a[2:6]); par=int(float(a[6]))
                except Exception: continue
                nodes.append((nid,typ,x,y,z,r,par))
        if not nodes: raise ValueError("no valid SWC nodes")
        byid={n[0]:n for n in nodes}
        child_count=defaultdict(int)
        lengths=[]; axon_len=0.; dend_len=0.; total=0.
        pathlen={}
        soma_xyz=[]
        xyz=[]
        type_counts=defaultdict(int)
        for n in nodes:
            nid,typ,x,y,z,r,par=n
            xyz.append((x,y,z)); type_counts[typ]+=1
            if typ==1: soma_xyz.append((x,y,z))
            if par in byid:
                pn=byid[par]
                d=math.dist((x,y,z),(pn[2],pn[3],pn[4]))
                total+=d
                if typ==2: axon_len+=d
                elif typ in (3,4): dend_len+=d
                child_count[par]+=1
                pathlen[nid]=pathlen.get(par,0.0)+d
            else:
                pathlen[nid]=0.0
        xyz=np.asarray(xyz,float)
        soma=np.mean(np.asarray(soma_xyz,float),axis=0) if soma_xyz else xyz[0]
        radii=np.linalg.norm(xyz-soma,axis=1)
        lo=xyz.min(axis=0); hi=xyz.max(axis=0); span=hi-lo
        centered=xyz-xyz.mean(axis=0)
        rg=float(np.sqrt(np.mean(np.sum(centered*centered,axis=1))))
        branches=sum(v>=2 for v in child_count.values())
        endpoints=sum(child_count.get(n[0],0)==0 for n in nodes)
        axon_nodes=sum(n[1]==2 for n in nodes)
        dend_nodes=sum(n[1] in (3,4) for n in nodes)
        area,layer=extract_area_layer_from_path(p)
        return {
            "source_path":str(p),"file_name":p.name,"status":"ok",
            "n_nodes":len(nodes),"n_axon_nodes":axon_nodes,"n_dendrite_nodes":dend_nodes,
            "total_length":total,"axon_length":axon_len,"dendrite_length":dend_len,
            "branch_points":branches,"endpoints":endpoints,
            "max_path_length":max(pathlen.values()) if pathlen else 0.0,
            "max_euclidean_radius":float(radii.max()),
            "radius_gyration":rg,
            "bbox_dx":float(span[0]),"bbox_dy":float(span[1]),"bbox_dz":float(span[2]),
            "bbox_volume":float(np.prod(np.maximum(span,0.0))),
            "spatial_entropy_4":spatial_entropy(xyz,4),
            "node_type_entropy":entropy_from_counts(type_counts.values()),
            "soma_x":float(soma[0]),"soma_y":float(soma[1]),"soma_z":float(soma[2]),
            "path_area":area,"path_layer":layer,
        }
    except Exception as e:
        return {"source_path":str(p),"file_name":p.name,"status":"failed",
                "detail":f"{type(e).__name__}: {e}"}

def load_subtype_info(path):
    m={}
    rows=0
    if not Path(path).exists(): return m,rows
    with Path(path).open("r",encoding="utf-8",errors="replace") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            mm=re.match(r'^(\d+)\s+"([^"]+)"\s+(\d+)',s)
            if mm:
                arch,cell,sub=mm.groups()
                m[(arch,cell)]=int(sub); rows+=1
    return m,rows

def annotate_pfc2022(row, subtype_map, archive_ids):
    p=Path(row["source_path"])
    arch=""
    for part in p.parts:
        if part in archive_ids: arch=part
    stem=p.stem
    nums=re.findall(r"\d+",stem)
    candidates=[stem]+nums+[x.zfill(3) for x in nums if x.isdigit()]
    subtype=""
    cell_id=""
    for c in candidates:
        if (arch,c) in subtype_map:
            cell_id=c; subtype=subtype_map[(arch,c)]; break
    row["archive_id"]=arch
    row["cell_id"]=cell_id
    row["projection_subtype"]=subtype
    return row

def run_set(label, root, out_csv, workers, subtype_map=None):
    files=sorted(Path(root).rglob("*.swc"))
    print(f"[{label}] SWC files={len(files)} workers={workers}",flush=True)
    rows=[]
    archive_ids=set(a for a,_ in subtype_map) if subtype_map else set()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(swc_features,str(p)):p for p in files}
        done=0
        for fut in as_completed(futs):
            r=fut.result()
            if subtype_map is not None and r.get("status")=="ok":
                r=annotate_pfc2022(r,subtype_map,archive_ids)
            r["dataset"]=label
            rows.append(r); done+=1
            if done%500==0 or done==len(files):
                print(f"[{label}] {done}/{len(files)}",flush=True)
    rows.sort(key=lambda r:r.get("source_path",""))
    write_csv(out_csv,rows)
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True)
    a=ap.parse_args(); c=load_config(a.config)
    out=ensure_dir(Path(c["output_root"])/"10_digital_brain")
    pfc=Path(c["digital_pfc_root"]); whole=Path(c["digital_whole_root"])
    submap,subrows=load_subtype_info(pfc/"pfc_2022"/"metadata"/"SubtypeInfo.txt")
    w=int(c.get("digital_cpu_workers",24))
    all_summary={}
    sets=[
        ("whole_cortex_2025",whole/"extracted_bsdc",out/"whole_cortex_features.csv",None),
        ("pfc_2022",pfc/"pfc_2022"/"extracted_by_archive",out/"pfc2022_features.csv",submap),
        ("pfc_2023",pfc/"pfc_2023"/"extracted",out/"pfc2023_features.csv",None),
    ]
    for label,root,csvout,sm in sets:
        rows=run_set(label,root,csvout,w,sm)
        all_summary[label]={
            "files":len(rows),"ok":sum(r.get("status")=="ok" for r in rows),
            "failed":sum(r.get("status")!="ok" for r in rows),
            "with_path_area":sum(bool(r.get("path_area")) for r in rows),
            "with_path_layer":sum(bool(r.get("path_layer")) for r in rows),
            "with_projection_subtype":sum(r.get("projection_subtype","")!="" for r in rows),
        }
    all_summary["subtypeinfo_rows"]=subrows
    all_summary["created_utc"]=now()
    write_json(out/"digital_brain_summary.json",all_summary)
    print(all_summary)

if __name__=="__main__": main()
