from __future__ import annotations
from pathlib import Path
import sys,re,json
import pandas as pd, numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage5b.common import read_yaml,ensure_dir,write_json

CFG=read_yaml(Path(__file__).resolve().parents[1]/"config/default.yaml")
ROOT=Path(CFG["project_root"])
OUT=ensure_dir(Path(CFG["output_root"])/"stage5a_recovery")

NAME_PAT=re.compile(r"(glif|neuronal.?model|model.?run|performance|explained|evr)",re.I)
PERF_PAT=re.compile(r"(explained.*variance|variance.*ratio|\bevr\b|noise.?2.*(score|ratio|variance)|held.?out)",re.I)
SPEC_PAT=re.compile(r"(specimen.?id|cell.?specimen.?id)",re.I)
MODEL_PAT=re.compile(r"(glif|model.?template|template.?name|neuronal.?model)",re.I)

def inspect_table(path):
    try:
        if path.suffix.lower()==".csv":
            df=pd.read_csv(path,nrows=30)
        elif path.suffix.lower()==".parquet":
            df=pd.read_parquet(path).head(30)
        else:
            return None
        cols=[str(c) for c in df.columns]
        return {"path":str(path),"kind":"table","columns":cols,
                "specimen_cols":[c for c in cols if SPEC_PAT.search(c)],
                "performance_cols":[c for c in cols if PERF_PAT.search(c)],
                "model_cols":[c for c in cols if MODEL_PAT.search(c)]}
    except Exception as e:
        return {"path":str(path),"error":repr(e)}

def inspect_json(path):
    try:
        if path.stat().st_size>100*1024*1024: return None
        obj=json.loads(path.read_text(encoding="utf-8",errors="replace"))
        keys=set()
        def walk(x,p="",depth=0):
            if depth>6:return
            if isinstance(x,dict):
                for k,v in list(x.items())[:500]:
                    kk=f"{p}.{k}" if p else str(k); keys.add(kk); walk(v,kk,depth+1)
            elif isinstance(x,list):
                for v in x[:30]: walk(v,p,depth+1)
        walk(obj)
        ks=sorted(keys)
        return {"path":str(path),"kind":"json",
                "specimen_keys":[k for k in ks if SPEC_PAT.search(k)],
                "performance_keys":[k for k in ks if PERF_PAT.search(k)],
                "model_keys":[k for k in ks if MODEL_PAT.search(k)]}
    except Exception as e:
        return {"path":str(path),"error":repr(e)}

candidates=[]
for p in ROOT.rglob("*"):
    if not p.is_file(): continue
    if p.suffix.lower() not in {".csv",".parquet",".json"}: continue
    if not NAME_PAT.search(str(p)): continue
    if p.stat().st_size>500*1024*1024: continue
    r=inspect_table(p) if p.suffix.lower() in {".csv",".parquet"} else inspect_json(p)
    if r: candidates.append(r)

# Rank candidates: specimen + performance + model evidence
def score(r):
    return 5*len(r.get("performance_cols",r.get("performance_keys",[]))) + \
           3*len(r.get("specimen_cols",r.get("specimen_keys",[]))) + \
           2*len(r.get("model_cols",r.get("model_keys",[])))
candidates=sorted(candidates,key=score,reverse=True)
write_json(OUT/"glif_recovery_audit.json",{"candidate_count":len(candidates),"candidates":candidates})

# Try to reconstruct only from unambiguous tabular evidence.
made=False
for r in candidates:
    if r.get("kind")!="table": continue
    if not r.get("specimen_cols") or not r.get("performance_cols"): continue
    p=Path(r["path"])
    try:
        df=pd.read_csv(p) if p.suffix.lower()==".csv" else pd.read_parquet(p)
    except Exception:
        continue
    spec=r["specimen_cols"][0]
    perf_cols=r["performance_cols"]
    # Accept wide GLIF1..5 columns directly.
    wide={}
    for c in perf_cols:
        m=re.search(r"glif\s*([1-5])",c,re.I)
        if m: wide[int(m.group(1))]=c
    if len(wide)>=5:
        out=df[[spec]+[wide[i] for i in range(1,6)]].copy()
        out=out.rename(columns={spec:"specimen_id",**{wide[i]:f"GLIF{i}_EVR" for i in range(1,6)}})
        out=out.drop_duplicates("specimen_id")
        out.to_csv(OUT/"glif_performance_atlas_specimen_level.csv",index=False)
        made=True; break

status={"status":"RECOVERED_WIDE_TABLE" if made else "AUDIT_ONLY",
        "output":str(OUT/"glif_performance_atlas_specimen_level.csv") if made else None,
        "instruction":"If AUDIT_ONLY, inspect glif_recovery_audit.json. Do not guess a performance field."}
write_json(OUT/"recovery_status.json",status)
print(json.dumps(status,indent=2))
