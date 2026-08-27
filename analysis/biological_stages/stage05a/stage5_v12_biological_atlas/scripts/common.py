#!/usr/bin/env python3
from pathlib import Path
import csv, json, hashlib, datetime, math
import numpy as np

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def cfg(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def ensure(path):
    p=Path(path); p.mkdir(parents=True, exist_ok=True); return p

def write_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8")

def read_csv(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path,rows,fields=None):
    rows=list(rows); p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:
        fields=sorted({k for r in rows for k in r}) if rows else []
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        if fields:w.writeheader()
        w.writerows(rows)

def fnum(x):
    try:return float(x)
    except:return float("nan")

def spearman(x,y):
    from scipy.stats import spearmanr
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3:return float("nan"),float("nan")
    r,p=spearmanr(x[m],y[m])
    return float(r),float(p)

def bh_fdr(pvals):
    p=np.asarray(pvals,float); out=np.full(len(p),np.nan)
    m=np.isfinite(p)
    idx=np.flatnonzero(m)
    if not len(idx): return out
    vals=p[idx]; order=np.argsort(vals); ranked=vals[order]
    q=ranked*len(ranked)/(np.arange(len(ranked))+1)
    q=np.minimum.accumulate(q[::-1])[::-1]
    q=np.clip(q,0,1)
    out[idx[order]]=q
    return out

def sha256_file(path,chunk=8<<20):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(chunk),b""): h.update(b)
    return h.hexdigest()

def standardize_train_test(Xtr,Xte):
    Xtr=np.asarray(Xtr,float); Xte=np.asarray(Xte,float)
    med=np.nanmedian(Xtr,axis=0)
    med=np.where(np.isfinite(med),med,0.0)
    a=np.where(np.isfinite(Xtr),Xtr,med)
    b=np.where(np.isfinite(Xte),Xte,med)
    mu=a.mean(axis=0); sd=a.std(axis=0)
    sd=np.where(sd<1e-8,1.0,sd)
    return (a-mu)/sd,(b-mu)/sd

def ridge_fit_predict(Xtr,ytr,Xte,alpha):
    Xtr=np.asarray(Xtr,float); ytr=np.asarray(ytr,float); Xte=np.asarray(Xte,float)
    Xtr,Xte=standardize_train_test(Xtr,Xte)
    ym=ytr.mean(); yy=ytr-ym
    A=Xtr.T@Xtr + float(alpha)*np.eye(Xtr.shape[1])
    try:w=np.linalg.solve(A,Xtr.T@yy)
    except np.linalg.LinAlgError:w=np.linalg.pinv(A)@(Xtr.T@yy)
    return Xte@w+ym

def stratified_folds(y,k,seed):
    rng=np.random.default_rng(seed); y=np.asarray(y)
    folds=[[] for _ in range(k)]
    for cls in np.unique(y):
        ix=np.flatnonzero(y==cls); rng.shuffle(ix)
        for j,v in enumerate(ix): folds[j%k].append(int(v))
    return [np.asarray(sorted(x),int) for x in folds]
