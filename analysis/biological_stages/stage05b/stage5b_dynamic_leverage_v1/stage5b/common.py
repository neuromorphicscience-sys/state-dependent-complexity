from __future__ import annotations
import json, math, re, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import log_loss, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def ensure_dir(p):
    p=Path(p); p.mkdir(parents=True,exist_ok=True); return p

def read_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=str),encoding="utf-8")

def find_nwb(root):
    return sorted(Path(root).rglob("*.nwb"))

def safe_numeric(x):
    return pd.to_numeric(x,errors="coerce")

def candidate_col(df, names):
    cols={str(c).lower():c for c in df.columns}
    for n in names:
        if n.lower() in cols: return cols[n.lower()]
    # relaxed punctuation-insensitive
    norm=lambda s: re.sub(r"[^a-z0-9]","",str(s).lower())
    nc={norm(c):c for c in df.columns}
    for n in names:
        if norm(n) in nc: return nc[norm(n)]
    return None

def rank01(x):
    a=np.asarray(x,float)
    out=np.full(a.shape,np.nan,float)
    m=np.isfinite(a)
    if m.sum():
        r=rankdata(a[m],method="average")
        out[m]=(r-0.5)/m.sum()
    return out

def robust_z(x):
    a=np.asarray(x,float)
    med=np.nanmedian(a); mad=np.nanmedian(np.abs(a-med))
    if not np.isfinite(mad) or mad<=1e-12:
        sd=np.nanstd(a)
        return (a-np.nanmean(a))/(sd if sd>1e-12 else 1.0)
    return (a-med)/(1.4826*mad)

def normalise_trial_table(df):
    """Handle both ordinary trial tables and variable-by-row 'type/data' tables."""
    if df is None or len(df)==0:
        return pd.DataFrame()
    out=df.copy()
    if "type" in out.columns and "data" in out.columns:
        types=[str(x) for x in out["type"].tolist()]
        wanted={"included","go_cue","visual_stimulus_time","visual_stimulus_left_contrast",
                "visual_stimulus_right_contrast","response_time","response_choice",
                "feedback_time","feedback_type","rep_num"}
        if len(set(types)&wanted)>=4:
            d={}
            for _,r in out.iterrows():
                k=str(r["type"])
                v=r["data"]
                if isinstance(v,(list,tuple,np.ndarray,pd.Series)):
                    d[k]=np.asarray(v).ravel()
            if d:
                n=max(len(v) for v in d.values())
                return pd.DataFrame({k:np.pad(v.astype(object),(0,n-len(v)),constant_values=np.nan)
                                     if len(v)<n else v for k,v in d.items()})
    return out.reset_index(drop=True)

def build_event_mask(times, event_times, rel_window):
    times=np.asarray(times,float)
    mask=np.zeros(times.shape,bool)
    lo,hi=map(float,rel_window)
    for t in np.asarray(event_times,float):
        if np.isfinite(t):
            mask |= (times>=t+lo)&(times<t+hi)
    return mask

def dynamics_leverage(X, state_masks, n_components=20, lag=1, alpha=1.0,
                      min_state_bins=80, random_state=0):
    """
    X: time x neurons, nonnegative activity/counts.
    Returns long dataframe and session diagnostics.
    """
    X=np.asarray(X,np.float32)
    good=np.isfinite(X).all(axis=0) & (np.nanstd(X,axis=0)>1e-8)
    if good.sum()<3 or X.shape[0]<100:
        return pd.DataFrame(), {"status":"insufficient_matrix",
                                "bins":int(X.shape[0]),"neurons":int(good.sum())}
    Xg=X[:,good]
    mu=Xg.mean(0); sd=Xg.std(0); sd[sd<1e-8]=1
    Xz=(Xg-mu)/sd
    k=max(2,min(int(n_components),Xz.shape[1]-1,Xz.shape[0]-2))
    pca=PCA(n_components=k,svd_solver="randomized",random_state=random_state)
    Z=pca.fit_transform(Xz)
    W=pca.components_.T  # neurons x latent
    rows=[]; diagnostics={}
    for state,mask in state_masks.items():
        mask=np.asarray(mask,bool)
        idx=np.flatnonzero(mask[:-lag] & mask[lag:])
        if idx.size<min_state_bins:
            diagnostics[state]={"status":"too_few_bins","pairs":int(idx.size)}
            continue
        # blocked train/test, preserving time order
        cut=max(10,int(idx.size*0.8))
        tr=idx[:cut]; te=idx[cut:]
        if te.size<10:
            tr=idx; te=idx
        model=Ridge(alpha=alpha,fit_intercept=True)
        model.fit(Z[tr],Z[tr+lag])
        pred=model.predict(Z[te])
        r2=float(r2_score(Z[te+lag],pred,multioutput="variance_weighted"))
        B=model.coef_.T
        direction=np.linalg.norm(W @ B,axis=1)
        mean_abs=np.mean(np.abs(Xz[te]),axis=0)
        lev=mean_abs*direction
        mean_activity=Xg[te].mean(0)
        diagnostics[state]={"status":"ok","pairs":int(idx.size),"test_pairs":int(te.size),
                            "latent_r2":r2}
        good_idx=np.flatnonzero(good)
        for j,orig in enumerate(good_idx):
            rows.append({"neuron_index":int(orig),"state":state,
                         "L_dyn":float(lev[j]),
                         "mean_activity":float(mean_activity[j]),
                         "pca_loading_norm":float(np.linalg.norm(W[j])),
                         "latent_r2":r2})
    d=pd.DataFrame(rows)
    if len(d):
        d["L_dyn_rank"]=d.groupby("state")["L_dyn"].transform(rank01)
    return d, {"status":"ok","n_components":k,
               "explained_variance":float(pca.explained_variance_ratio_.sum()),
               "states":diagnostics}

def predictive_ablation_logloss(X,y,n_splits=5,random_state=0,min_samples=40):
    """
    Test-time feature deletion on held-out folds.
    Returns feature leverage = logloss(ablated)-logloss(full).
    """
    X=np.asarray(X,float); y=np.asarray(y)
    m=np.isfinite(X).all(1) & pd.Series(y).notna().to_numpy()
    X=X[m]; y=y[m]
    cls,counts=np.unique(y,return_counts=True)
    if len(y)<min_samples or len(cls)<2 or counts.min()<3:
        return None, {"status":"insufficient_labels","n":int(len(y)),
                      "classes":{str(c):int(n) for c,n in zip(cls,counts)}}
    splits=min(n_splits,int(counts.min()))
    if splits<2:
        return None, {"status":"insufficient_folds"}
    skf=StratifiedKFold(splits,shuffle=True,random_state=random_state)
    delta=np.zeros(X.shape[1],float); full_losses=[]; folds=0
    for tr,te in skf.split(X,y):
        sc=StandardScaler().fit(X[tr])
        Xt=sc.transform(X[tr]); Xe=sc.transform(X[te])
        lr=LogisticRegression(max_iter=1500,C=1.0,solver="lbfgs")
        lr.fit(Xt,y[tr])
        p=lr.predict_proba(Xe)
        labels=lr.classes_
        base=log_loss(y[te],p,labels=labels)
        full_losses.append(base); folds+=1
        # Robust and explicit; typical trial x neuron matrices are manageable.
        for j in range(X.shape[1]):
            Xa=Xe.copy()
            Xa[:,j]=0.0
            pa=lr.predict_proba(Xa)
            delta[j]+=log_loss(y[te],pa,labels=labels)-base
    delta/=folds
    return delta, {"status":"ok","folds":folds,"full_logloss":float(np.mean(full_losses)),
                   "n":int(len(y))}

def summarize_flexibility(long_df,id_cols=("session_id","neuron_id"),lev_col="L_dyn_rank"):
    if long_df is None or len(long_df)==0:
        return pd.DataFrame()
    gcols=list(id_cols)
    rows=[]
    for key,g in long_df.groupby(gcols,dropna=False):
        vals=pd.to_numeric(g[lev_col],errors="coerce").dropna().values
        if not isinstance(key,tuple): key=(key,)
        r={c:v for c,v in zip(gcols,key)}
        r.update({"n_states":int(len(vals)),
                  "mean_leverage":float(np.mean(vals)) if len(vals) else np.nan,
                  "role_flexibility":float(np.var(vals,ddof=0)) if len(vals)>=2 else np.nan,
                  "leverage_range":float(np.ptp(vals)) if len(vals)>=2 else np.nan})
        rows.append(r)
    return pd.DataFrame(rows)

def spearman_safe(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<4: return (np.nan,np.nan,int(m.sum()))
    r,p=spearmanr(x[m],y[m])
    return float(r),float(p),int(m.sum())
