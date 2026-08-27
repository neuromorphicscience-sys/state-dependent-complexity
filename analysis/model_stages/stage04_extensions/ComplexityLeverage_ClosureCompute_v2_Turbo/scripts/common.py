from __future__ import annotations
import hashlib, json, math, tarfile, io, re
from pathlib import Path
import numpy as np
import pandas as pd

COL = {
    "blue": "#355C7D",
    "teal": "#2A9D8F",
    "orange": "#BC6C25",
    "purple": "#8D6A9F",
    "rose": "#B56576",
    "gray": "#7A838B",
    "light": "#C7CED6",
    "dark": "#23313D",
}

STATES = ["transition_mid", "sparse_drive", "transition_dense"]
STATE_LABEL = {
    "transition_mid": "transition mid",
    "sparse_drive": "sparse drive",
    "transition_dense": "transition dense*",
}


def ensure_dir(p):
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def sha256(path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def parse_mask_key(s, n=None):
    if pd.isna(s): return set()
    vals=set(int(x) for x in str(s).split(",") if str(x).strip())
    if n is not None and any((x<0 or x>=n) for x in vals):
        raise ValueError(f"mask index out of range for n={n}: {sorted(vals)[:10]}")
    return vals


def mask_bool(mask_set, n):
    m=np.zeros(int(n), dtype=bool)
    if mask_set: m[list(mask_set)] = True
    return m


def jaccard_distance(a, b):
    a=set(a); b=set(b); u=a|b
    return 0.0 if not u else 1.0 - len(a&b)/len(u)


def bh_fdr(pvals):
    p=np.asarray(pvals,float); out=np.full_like(p,np.nan)
    m=np.isfinite(p); v=p[m]
    if not len(v): return out
    order=np.argsort(v); sv=v[order]; n=len(sv)
    q=sv*n/np.arange(1,n+1)
    q=np.minimum.accumulate(q[::-1])[::-1]; q=np.clip(q,0,1)
    inv=np.empty_like(order); inv[order]=np.arange(n)
    out[np.where(m)[0]]=q[inv]
    return out


def bootstrap_mean_ci(x, n_boot=10000, seed=20260820):
    a=np.asarray(x,float); a=a[np.isfinite(a)]
    if not len(a): return np.nan,np.nan,np.nan
    if len(a)==1: return float(a[0]),float(a[0]),float(a[0])
    rng=np.random.default_rng(seed)
    vals=np.empty(n_boot,float)
    for i in range(n_boot): vals[i]=np.mean(rng.choice(a,len(a),replace=True))
    return float(np.mean(a)), float(np.quantile(vals,.025)), float(np.quantile(vals,.975))


def exact_signflip_p(x, alternative="greater"):
    a=np.asarray(x,float); a=a[np.isfinite(a)]
    n=len(a)
    if n==0: return np.nan
    obs=float(np.mean(a))
    if n<=20:
        total=1<<n; extreme=0
        for bits in range(total):
            s=np.ones(n,float)
            for i in range(n):
                if bits & (1<<i): s[i]=-1.0
            v=float(np.mean(a*s))
            if alternative=="greater": extreme += (v >= obs-1e-15)
            elif alternative=="less": extreme += (v <= obs+1e-15)
            else: extreme += (abs(v) >= abs(obs)-1e-15)
        return extreme/total
    rng=np.random.default_rng(20260820); B=100000
    vals=[]
    for _ in range(B): vals.append(np.mean(a*rng.choice([-1.0,1.0],size=n)))
    vals=np.asarray(vals)
    if alternative=="greater": return float((np.sum(vals>=obs)+1)/(B+1))
    if alternative=="less": return float((np.sum(vals<=obs)+1)/(B+1))
    return float((np.sum(np.abs(vals)>=abs(obs))+1)/(B+1))


def spearman_safe(x,y):
    from scipy.stats import spearmanr
    a=pd.to_numeric(pd.Series(x),errors="coerce").to_numpy(float)
    b=pd.to_numeric(pd.Series(y),errors="coerce").to_numpy(float)
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<4 or np.std(a[m])<1e-12 or np.std(b[m])<1e-12: return np.nan,np.nan,int(m.sum())
    r,p=spearmanr(a[m],b[m]); return float(r),float(p),int(m.sum())


def rank01(x):
    from scipy.stats import rankdata
    a=np.asarray(x,float); out=np.full(a.shape,np.nan,float); m=np.isfinite(a)
    if m.sum(): out[m]=(rankdata(a[m],method="average")-0.5)/m.sum()
    return out


def resolve_stage4f_source(project_root: Path, explicit: str|None=None):
    if explicit and str(explicit).lower()!="auto":
        p=Path(explicit).expanduser().resolve()
        if not p.exists(): raise FileNotFoundError(p)
        return p
    root=Path(project_root).resolve()
    cands=[
        root/"results_stage4f_closure",
        root/"stage4f_closure_results.tar.gz",
        root/"results_stage4f_closure.tar.gz",
        root/"bio data"/"stage4f_closure_results.tar.gz",
    ]
    for p in cands:
        if p.exists(): return p
    hits=list(root.rglob("stage4f_closure_results.tar.gz"))
    if hits: return hits[0]
    hits=[p for p in root.rglob("results_stage4f_closure") if p.is_dir()]
    if hits: return hits[0]
    raise FileNotFoundError("Could not auto-discover Stage4F closure results. Pass --stage4f-source explicitly.")


def load_stage4f_reevaluation(source: Path):
    source=Path(source)
    dfs=[]
    if source.is_dir():
        for p in sorted(source.rglob("reevaluation.csv")):
            try: dfs.append(pd.read_csv(p))
            except Exception: pass
    else:
        with tarfile.open(source,"r:gz") as tf:
            for n in tf.getnames():
                if n.endswith("/reevaluation.csv"):
                    f=tf.extractfile(n)
                    if f is not None: dfs.append(pd.read_csv(io.BytesIO(f.read())))
    if not dfs: raise RuntimeError(f"No Stage4F reevaluation.csv found in {source}")
    d=pd.concat(dfs,ignore_index=True,sort=False)
    # strict de-dup on actual experimental key
    key=[c for c in ["graph_seed","regime","k","replicate_label"] if c in d.columns]
    d=d.drop_duplicates(key,keep="first").copy()
    expected=27*4
    if len(d)!=expected:
        raise RuntimeError(f"Expected 108 Stage4F reevaluation mask rows, got {len(d)}")
    if d[["graph_seed","regime","k"]].drop_duplicates().shape[0]!=27:
        raise RuntimeError("Expected 27 Stage4F reevaluation conditions")
    return d


def setup_mpl():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi":150,"savefig.dpi":600,"font.size":10.5,"axes.labelsize":11,
        "xtick.labelsize":9.5,"ytick.labelsize":9.5,"legend.fontsize":9,
        "axes.linewidth":1.0,"lines.linewidth":1.9,"lines.markersize":5,
        "pdf.fonttype":42,"ps.fonttype":42,
    })


def clean_ax(ax, grid="y"):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if grid: ax.grid(axis=grid,alpha=.18,linewidth=.7)
    ax.set_axisbelow(True); ax.set_title("")


def save_panel(fig, path, pdf=True):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if getattr(fig,"_suptitle",None) is not None: fig._suptitle.set_text("")
    for ax in fig.axes: ax.set_title("")
    fig.tight_layout(); fig.savefig(path.with_suffix(".png"),dpi=600,bbox_inches="tight")
    if pdf: fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    import matplotlib.pyplot as plt; plt.close(fig)
