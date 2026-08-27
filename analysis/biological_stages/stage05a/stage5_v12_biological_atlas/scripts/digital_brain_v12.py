#!/usr/bin/env python3
import argparse,math,re,json
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from common import cfg,ensure,write_csv,write_json,now

def entropy_counts(vals):
    x=np.asarray(list(vals),float);x=x[x>0]
    if not len(x):return 0.0
    p=x/x.sum();return float(-(p*np.log2(p)).sum())

def spatial_entropy(xyz,bins=4):
    xyz=np.asarray(xyz,float)
    if xyz.ndim!=2 or xyz.shape[0]<2:return 0.0
    lo=np.nanmin(xyz,axis=0);hi=np.nanmax(xyz,axis=0)
    span=np.maximum(hi-lo,1e-12)
    q=np.floor((xyz-lo)/span*bins).astype(int);q=np.clip(q,0,bins-1)
    code=q[:,0]*bins*bins+q[:,1]*bins+q[:,2]
    _,cnt=np.unique(code,return_counts=True)
    return entropy_counts(cnt)

def parse_swc(pathstr):
    p=Path(pathstr)
    try:
        nodes=[]
        with p.open("r",encoding="utf-8",errors="replace") as f:
            for line in f:
                s=line.strip()
                if not s or s.startswith("#"):continue
                a=s.split()
                if len(a)<7:continue
                try:nodes.append((int(float(a[0])),int(float(a[1])),float(a[2]),float(a[3]),float(a[4]),float(a[5]),int(float(a[6]))))
                except:continue
        if not nodes:raise ValueError("no valid nodes")
        by={n[0]:n for n in nodes};children=defaultdict(list)
        total=axon=dend=0.0; edge_len={}
        xyz=np.asarray([(n[2],n[3],n[4]) for n in nodes],float)
        soma_nodes=[n for n in nodes if n[1]==1];soma=np.mean([[n[2],n[3],n[4]] for n in soma_nodes],axis=0) if soma_nodes else xyz[0]
        tc=Counter(n[1] for n in nodes)
        for n in nodes:
            par=n[6]
            if par in by:
                pn=by[par]; d=math.dist((n[2],n[3],n[4]),(pn[2],pn[3],pn[4]))
                edge_len[n[0]]=d;children[par].append(n[0]);total+=d
                if n[1]==2:axon+=d
                if n[1] in (3,4):dend+=d
        memo={}
        def plen(nid,seen=None):
            if nid in memo:return memo[nid]
            n=by[nid];par=n[6]
            if par not in by or par==nid:memo[nid]=0.0;return 0.0
            seen=set() if seen is None else seen
            if nid in seen:return 0.0
            seen.add(nid);v=plen(par,seen)+edge_len.get(nid,0.0);memo[nid]=v;return v
        pathlens=[plen(n[0]) for n in nodes]
        span=np.nanmax(xyz,axis=0)-np.nanmin(xyz,axis=0)
        rad=np.linalg.norm(xyz-soma,axis=1)
        rg=float(np.sqrt(np.mean(np.sum((xyz-xyz.mean(0))**2,axis=1))))
        branches=sum(len(v)>=2 for v in children.values())
        endpoints=sum(len(children.get(n[0],[]))==0 for n in nodes)
        straight=float(np.linalg.norm(xyz[-1]-soma))
        maxpath=float(max(pathlens) if pathlens else 0.0)
        return {
          "status":"ok","source_path":str(p),"file_name":p.name,
          "n_nodes":len(nodes),"soma_nodes":len(soma_nodes),"axon_nodes":tc.get(2,0),"dendrite_nodes":tc.get(3,0)+tc.get(4,0),
          "total_length":total,"axon_length":axon,"dendrite_length":dend,
          "branch_points":branches,"endpoints":endpoints,"max_path_length":maxpath,
          "max_euclidean_radius":float(np.max(rad)),"radius_gyration":rg,
          "bbox_dx":float(span[0]),"bbox_dy":float(span[1]),"bbox_dz":float(span[2]),
          "bbox_volume":float(np.prod(np.maximum(span,0))),
          "spatial_entropy_4":spatial_entropy(xyz,4),"spatial_entropy_6":spatial_entropy(xyz,6),
          "node_type_entropy":entropy_counts(tc.values()),
          "path_tortuosity_proxy":float(maxpath/max(np.max(rad),1e-9)),
          "soma_x":float(soma[0]),"soma_y":float(soma[1]),"soma_z":float(soma[2])
        }
    except Exception as e:
        return {"status":"failed","source_path":str(p),"file_name":p.name,"detail":f"{type(e).__name__}: {e}"}

def subtype_rows(path):
    out=[]
    with Path(path).open("r",encoding="utf-8",errors="replace") as f:
        for line in f:
            m=re.match(r'^(\d+)\s+"([^"]+)"\s+(\d+)',line.strip())
            if m:out.append(m.groups())
    return out

def pfc_key(p,base,byarch):
    rel=p.relative_to(base);arch=""
    for part in rel.parts:
        if part in byarch:arch=part;break
    stem=p.stem; nums=re.findall(r"\d+",stem)
    cand=[stem]+nums+[x.zfill(3) for x in nums]
    cell=next((x for x in cand if x in byarch.get(arch,set())), "")
    return arch,cell,str(rel)

def run_parallel(label,files,workers,annot=None):
    rows=[];print(f"[{label}] files={len(files)} workers={workers}",flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut={ex.submit(parse_swc,str(p)):p for p in files}
        for i,f in enumerate(as_completed(fut),1):
            r=f.result()
            if annot:r.update(annot(fut[f]))
            r["dataset"]=label;rows.append(r)
            if i%500==0 or i==len(files):
                ok=sum(x["status"]=="ok" for x in rows)
                print(f"[{label}] {i}/{len(files)} ok={ok} failed={i-ok}",flush=True)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args()
    c=cfg(a.config);out=ensure(Path(c["output_root"])/"10_digital_brain_v12");w=int(c["digital_cpu_workers"])
    wholebase=Path(c["digital_whole_root"])/"extracted_bsdc"
    whole=sorted(wholebase.rglob("*.swc"))
    wr=run_parallel("whole_cortex_2025_ccf",whole,w)
    write_csv(out/"whole_cortex_features_v12.csv",wr)

    pfcroot=Path(c["digital_pfc_root"]);base=pfcroot/"pfc_2022/extracted_by_archive"
    info=subtype_rows(pfcroot/"pfc_2022/metadata/SubtypeInfo.txt")
    byarch=defaultdict(set);sub={}
    for arch,cell,st in info:byarch[arch].add(cell);sub[(arch,cell)]=int(st)
    aligned=sorted([p for p in base.rglob("*.swc") if "swc_allen_space" in [x.lower() for x in p.parts]])
    def ann(p):
        arch,cell,rel=pfc_key(p,base,byarch)
        return {"archive_id":arch,"cell_id":cell,"projection_subtype":sub.get((arch,cell),""),"relative_path":rel}
    pr=run_parallel("pfc2022_allen_space",aligned,w,ann)
    write_csv(out/"pfc2022_canonical_6357_features_v12.csv",pr)
    keys={(r.get("archive_id"),r.get("cell_id")) for r in pr if r["status"]=="ok" and r.get("archive_id") and r.get("cell_id")}
    expected=set(sub)
    # PFC 2023 is audited, not treated as independent biological cohort.
    p23base=pfcroot/"pfc_2023/extracted"; p23=sorted(p23base.rglob("*.swc"))
    classes=Counter()
    stems=[]
    for p in p23:
        rel=str(p.relative_to(p23base)).lower()
        first=p.relative_to(p23base).parts[0] if p.relative_to(p23base).parts else ""
        classes[first]+=1;stems.append(p.stem)
    report={
      "created_utc":now(),
      "whole":{"raw_files":len(whole),"ok":sum(r["status"]=="ok" for r in wr),"failed":sum(r["status"]!="ok" for r in wr)},
      "pfc2022":{"canonical_files":len(aligned),"ok":sum(r["status"]=="ok" for r in pr),"failed":sum(r["status"]!="ok" for r in pr),
                 "matched_ids":len(keys),"expected_ids":len(expected),"missing_ids":len(expected-keys),
                 "subtypes":len(set(sub.values()))},
      "pfc2023_audit":{"raw_swc":len(p23),"top_level_dirs":dict(classes),
                       "interpretation":"AUDIT_ONLY: current archive is not accepted as an independent ~2000-neuron 2023 dendrite cohort; no manuscript claim is generated from it."}
    }
    write_json(out/"digital_brain_v12_summary.json",report);print(json.dumps(report,indent=2))
    if report["whole"]["ok"]!=12264:raise SystemExit("Whole-cortex corrected parser did not recover all 12264 SWCs.")
    if report["pfc2022"]["ok"]!=6357 or report["pfc2022"]["matched_ids"]!=6357:raise SystemExit("PFC2022 canonical cohort did not close at 6357.")
    print("DIGITAL BRAIN V1.2 PASS")
if __name__=="__main__":main()
