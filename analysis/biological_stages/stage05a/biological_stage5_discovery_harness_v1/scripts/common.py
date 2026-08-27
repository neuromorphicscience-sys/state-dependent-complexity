#!/usr/bin/env python3
import csv, json, math, os, re, hashlib
from pathlib import Path
from datetime import datetime, timezone

def now():
    return datetime.now(timezone.utc).isoformat()

def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def ensure_dir(path):
    p=Path(path); p.mkdir(parents=True, exist_ok=True); return p

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def write_csv(path, rows, fieldnames=None):
    rows=list(rows)
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames=sorted({k for r in rows for k in r}) if rows else []
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames: w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fieldnames})

def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def sha256_file(path, chunk=8*1024*1024):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def safe_float(x, default=float("nan")):
    try: return float(x)
    except Exception: return default

def robust_z(x):
    import numpy as np
    x=np.asarray(x, dtype=float)
    med=np.nanmedian(x, axis=0)
    mad=np.nanmedian(np.abs(x-med), axis=0)
    scale=1.4826*mad
    sd=np.nanstd(x, axis=0)
    scale=np.where((~np.isfinite(scale))|(scale<1e-12), sd, scale)
    scale=np.where((~np.isfinite(scale))|(scale<1e-12), 1.0, scale)
    return (x-med)/scale, med, scale

def rankdata_simple(x):
    import numpy as np
    x=np.asarray(x, dtype=float)
    order=np.argsort(x, kind="mergesort")
    ranks=np.empty(len(x), float)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        r=(i+j-1)/2+1
        ranks[order[i:j]]=r
        i=j
    return ranks

def spearman(x,y):
    import numpy as np
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3: return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr
        r,p=spearmanr(x[m],y[m])
        return float(r), float(p)
    except Exception:
        rx=rankdata_simple(x[m]); ry=rankdata_simple(y[m])
        r=float(np.corrcoef(rx,ry)[0,1])
        return r,float("nan")

def extract_area_layer_from_path(path):
    s=str(path)
    areas=[
        "VISp","VISl","VISal","VISam","VISpm","VISrl",
        "MOp","MOs","ACA","PL","ILA","ORB","RSP","AUDp","AUDd","AUDv",
        "SSp","SSs","TEa","PERI","ECT","GU","AI","VISC"
    ]
    area=""
    for a in areas:
        if re.search(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", s, re.I):
            area=a; break
    layer=""
    m=re.search(r"(?:layer|[_/\-]L)\s*([1-6](?:[ab])?)", s, re.I)
    if m: layer=m.group(1)
    return area,layer
