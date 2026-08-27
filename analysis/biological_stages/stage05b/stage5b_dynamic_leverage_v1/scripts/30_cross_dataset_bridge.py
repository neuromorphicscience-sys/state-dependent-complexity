from pathlib import Path
import sys,re,json
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage5b.common import *

CFG=read_yaml(Path(__file__).resolve().parents[1]/"config/default.yaml")
ROOT=Path(CFG["project_root"]); OUT=ensure_dir(Path(CFG["output_root"])/"bridge")
vbo=Path(CFG["output_root"])/"allen_vbo"/"role_flexibility.csv"
if not vbo.exists():raise SystemExit("Allen role_flexibility.csv missing")
V=pd.read_csv(vbo)

def driver(x):
    s=str(x)
    markers=["Pvalb","Sst","Vip","Slc17a7","Cux2","Rorb","Rbp4","Tlx3","Ntsr1","Scnn1a","Fezf2","Foxp2","Emx1"]
    for m in markers:
        if re.search(rf"\b{m}\b",s,re.I):return m
    # first driver-like token
    m=re.search(r"([A-Za-z0-9]+)-IRES|([A-Za-z0-9]+)-Cre",s)
    return next((g for g in m.groups() if g),None) if m else None

V["bridge_class"]=V.get("cre_line",pd.Series([None]*len(V))).map(driver)
vsum=V.dropna(subset=["bridge_class"]).groupby("bridge_class").agg(
    vbo_n=("cell_specimen_id","size"),
    median_flexibility=("role_flexibility","median"),
    median_mean_leverage=("mean_leverage","median")).reset_index()

# Search Stage5A specimen-level complexity candidates.
cands=[]
for p in ROOT.rglob("*.csv"):
    if "stage5" not in str(p).lower() and "glif" not in str(p).lower():continue
    try:
        df=pd.read_csv(p,nrows=5)
    except:continue
    cols=[str(c) for c in df.columns]
    if any(re.search(r"(specimen.?id|cell.?specimen.?id)",c,re.I) for c in cols) and \
       any(re.search(r"(c.?req|complexity|mechanism.?cost)",c,re.I) for c in cols):
        cands.append((p,cols))
write_json(OUT/"stage5a_candidate_tables.json",{"candidates":[{"path":str(p),"columns":c} for p,c in cands]})

status={"status":"AUDIT_ONLY","reason":"No unambiguous Stage5A class field resolved automatically."}
# Attempt only if one candidate has an obvious line/transgenic field and complexity field.
for p,cols in cands:
    line=next((c for c in cols if re.search(r"(transgenic|cre.?line|driver.?line)",c,re.I)),None)
    comp=next((c for c in cols if re.search(r"(c.?req|mechanism.?cost)",c,re.I)),None)
    if not line or not comp:continue
    A=pd.read_csv(p)
    A["bridge_class"]=A[line].map(driver)
    A[comp]=pd.to_numeric(A[comp],errors="coerce")
    asum=A.dropna(subset=["bridge_class",comp]).groupby("bridge_class").agg(
        stage5a_n=(comp,"size"),median_complexity=(comp,"median")).reset_index()
    M=asum.merge(vsum,on="bridge_class",how="inner")
    M.to_csv(OUT/"class_conditioned_bridge.csv",index=False)
    usable=M[(M["stage5a_n"]>=10)&(M["vbo_n"]>=20)]
    result={"status":"EXPLORATORY",
            "source_stage5a":str(p),"complexity_col":comp,"line_col":line,
            "overlap_classes":int(len(M)),"usable_classes":int(len(usable))}
    if len(usable)>=4:
        r1,p1,n1=spearman_safe(usable["median_complexity"],usable["median_flexibility"])
        r2,p2,n2=spearman_safe(usable["median_complexity"],usable["median_mean_leverage"])
        result["complexity_vs_flexibility"]={"rho":r1,"p":p1,"n_classes":n1}
        result["complexity_vs_mean_leverage"]={"rho":r2,"p":p2,"n_classes":n2}
    status=result
    break
write_json(OUT/"bridge_status.json",status)
print(json.dumps(status,indent=2))
