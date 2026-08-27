from __future__ import annotations
from contextlib import contextmanager
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from pynwb.ophys import RoiResponseSeries

@contextmanager
def open_nwb(path):
    io=NWBHDF5IO(str(path),"r",load_namespaces=True)
    try:
        nwb=io.read()
        yield nwb
    finally:
        io.close()

def table_df(obj):
    if obj is None: return pd.DataFrame()
    try: return obj.to_dataframe()
    except Exception: return pd.DataFrame()

def nwb_schema_summary(nwb):
    out={"identifier":str(getattr(nwb,"identifier","")),
         "session_description":str(getattr(nwb,"session_description","")),
         "acquisition":list(getattr(nwb,"acquisition",{}).keys()),
         "processing":list(getattr(nwb,"processing",{}).keys()),
         "intervals":list(getattr(nwb,"intervals",{}).keys())}
    u=table_df(getattr(nwb,"units",None)); t=table_df(getattr(nwb,"trials",None))
    out["units_rows"]=int(len(u)); out["units_columns"]=[str(c) for c in u.columns]
    out["trials_rows"]=int(len(t)); out["trials_columns"]=[str(c) for c in t.columns]
    return out

def unit_spike_times(nwb):
    u=getattr(nwb,"units",None)
    if u is None: return [], pd.DataFrame()
    df=u.to_dataframe()
    spikes=[]
    if "spike_times" in df.columns:
        spikes=[np.asarray(x,float) for x in df["spike_times"]]
    else:
        try:
            spikes=[np.asarray(u["spike_times"][i],float) for i in range(len(u.id))]
        except Exception:
            spikes=[]
    return spikes,df

def bin_spikes(spikes,t0,t1,bin_s):
    edges=np.arange(float(t0),float(t1)+float(bin_s),float(bin_s))
    if len(edges)<3: raise ValueError("recording interval too short")
    X=np.zeros((len(edges)-1,len(spikes)),np.float32)
    for i,s in enumerate(spikes):
        if len(s):
            X[:,i]=np.histogram(s,bins=edges)[0]
    centers=(edges[:-1]+edges[1:])/2
    return X,centers

def find_roi_response_series(nwb):
    candidates=[]
    for obj in nwb.objects.values():
        if isinstance(obj,RoiResponseSeries):
            nm=str(getattr(obj,"name","")).lower()
            score=0
            if "event" in nm: score+=100
            if "dff" in nm: score+=50
            candidates.append((score,nm,obj))
    candidates.sort(key=lambda x:(-x[0],x[1]))
    return candidates

def roi_series_matrix(series):
    data=np.asarray(series.data[:],dtype=np.float32)
    if series.timestamps is not None:
        ts=np.asarray(series.timestamps[:],float)
    else:
        rate=float(series.rate); start=float(series.starting_time)
        # determine time axis after loading
        ntime=max(data.shape)
        ts=start+np.arange(ntime)/rate
    try:
        ridx=np.asarray(series.rois.data[:],int)
        tab=series.rois.table.to_dataframe()
        sub=tab.iloc[ridx].copy()
    except Exception:
        sub=pd.DataFrame(index=np.arange(min(data.shape)))
    # infer orientation using timestamp length
    if data.ndim!=2:
        raise ValueError(f"RoiResponseSeries must be 2D, got {data.shape}")
    if data.shape[0]==len(ts):
        X=data
    elif data.shape[1]==len(ts):
        X=data.T
    else:
        # choose larger dimension as time by convention, but reportable upstream
        X=data if data.shape[0]>=data.shape[1] else data.T
        ts=ts[:X.shape[0]]
    # cell specimen id
    cell_ids=None
    for c in sub.columns:
        if str(c).lower()=="cell_specimen_id":
            cell_ids=pd.to_numeric(sub[c],errors="coerce").to_numpy()
            break
    if cell_ids is None:
        idxname=str(sub.index.name or "").lower()
        if "cell_specimen" in idxname:
            cell_ids=pd.to_numeric(pd.Series(sub.index),errors="coerce").to_numpy()
        else:
            cell_ids=np.asarray(sub.index)
    return X,ts,cell_ids,sub

def bin_continuous(X,ts,bin_s):
    X=np.asarray(X,np.float32); ts=np.asarray(ts,float)
    m=np.isfinite(ts)
    X=X[m]; ts=ts[m]
    order=np.argsort(ts); X=X[order]; ts=ts[order]
    t0=float(ts[0]); t1=float(ts[-1])
    edges=np.arange(t0,t1+bin_s,bin_s)
    if len(edges)<3: raise ValueError("too few bins")
    bid=np.searchsorted(edges,ts,side="right")-1
    good=(bid>=0)&(bid<len(edges)-1)
    bid=bid[good]; X=X[good]
    out=np.zeros((len(edges)-1,X.shape[1]),np.float32)
    cnt=np.bincount(bid,minlength=len(edges)-1).astype(np.float32)
    for j in range(X.shape[1]):
        np.add.at(out[:,j],bid,X[:,j])
    # sum is appropriate for event-like traces; divide later only for dF/F if desired
    centers=(edges[:-1]+edges[1:])/2
    return out,centers,cnt

def find_timeseries(nwb, keywords):
    kws=[k.lower() for k in keywords]
    found=[]
    for obj in nwb.objects.values():
        if hasattr(obj,"data") and (hasattr(obj,"timestamps") or hasattr(obj,"rate")):
            nm=str(getattr(obj,"name","")).lower()
            if any(k in nm for k in kws):
                found.append(obj)
    return found
