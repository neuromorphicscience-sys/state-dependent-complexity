#!/usr/bin/env python3
import argparse,csv,re,json,math
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,h5py

def load_cfg(p): return json.load(open(p))
def ensure(p): p=Path(p);p.mkdir(parents=True,exist_ok=True);return p
def dec(x):
    try:
        if hasattr(x,'shape') and x.shape==(): x=x[()]
    except: pass
    if isinstance(x,(bytes,np.bytes_)): return x.decode('utf-8','replace')
    if isinstance(x,np.ndarray) and x.size==1:return dec(x.reshape(-1)[0])
    return str(x)

def rate(g):
    st=g.get('starting_time')
    if st is not None and 'rate' in st.attrs:
        try:return float(st.attrs['rate'])
        except:pass
    return np.nan

def convert(a,ds,kind):
    a=np.asarray(a,np.float32)
    c=float(ds.attrs.get('conversion',1.0)); a=a*c
    u=dec(ds.attrs.get('unit','')).lower()
    med=float(np.nanmedian(np.abs(a))) if a.size else 0
    if kind=='v':
        if 'mv' in u: pass
        elif u in ('v','volt','volts') or (0<med<1): a*=1000
    else:
        if 'pa' in u: pass
        elif 'na' in u:a*=1000
        elif u in ('a','amp','amps','ampere','amperes') or (0<med<1e-6):a*=1e12
    return a

def down(x,orig,target):
    step=max(1,int(round(orig/target)))
    n=(len(x)//step)*step
    if n<=0:return np.empty(0,np.float32),orig
    return x[:n].reshape(-1,step).mean(1).astype(np.float32),orig/step

def load_lookup(root):
    # IMPORTANT: exact stimulus_name is preferred. Never silently use stimulus_description.
    candidates=[
        Path(root)/'analysis/ephys_axis_sweep_inventory_v1/sweeps/all_glif_overlap_sweeps.csv'
    ]
    candidates += [p for p in Path(root).rglob('*.csv') if 'sweep' in p.name.lower()]
    for p in candidates:
        if not p.exists():continue
        try:
            m={}
            bycell={}
            with p.open('r',encoding='utf-8-sig',newline='') as f:
                rd=csv.DictReader(f); cols=rd.fieldnames or []
                sidc=next((c for c in cols if c.lower() in ('specimen_id','specimen__id')),None)
                swc=next((c for c in cols if c.lower()=='sweep_number'),None)
                namec=next((c for c in cols if c.lower()=='stimulus_name'),None)
                if not(sidc and swc and namec): continue
                for r in rd:
                    try:
                        sid=str(int(float(r[sidc]))); sw=int(float(r[swc])); name=r[namec].strip()
                        m[(sid,sw)]=name
                        bycell.setdefault(sid,[]).append((sw,name))
                    except:pass
            if m:
                print(f'Loaded EXACT stimulus_name lookup: {p} rows={len(m)}',flush=True)
                return m,bycell,str(p)
        except Exception:pass
    raise RuntimeError('No sweep CSV with exact specimen_id+sweep_number+stimulus_name columns.')

def nwb_name(acq,stim,sw):
    for g in (acq,stim):
        if g is not None and 'aibs_stimulus_name' in g:
            try:return dec(g['aibs_stimulus_name'][()])
            except:pass
    return f'Sweep_{sw}'

def spike_stats(v,fs):
    if len(v)<3:return (0,np.nan,np.nan,np.nan)
    idx=np.flatnonzero((v[:-1]<-20)&(v[1:]>=-20))+1
    ref=max(1,int(fs*.001))
    if len(idx)>1:
        keep=[int(idx[0])]
        for q in idx[1:]:
            if q-keep[-1]>=ref:keep.append(int(q))
        idx=np.asarray(keep)
    if len(idx)<2:return (len(idx),np.nan,np.nan,np.nan)
    isi=np.diff(idx)/fs
    cv=float(np.std(isi)/np.mean(isi)) if np.mean(isi)>0 else np.nan
    ad=float(np.median(isi[len(isi)//2:])/np.median(isi[:max(1,len(isi)//2)])) if len(isi)>=4 else np.nan
    return len(idx),float(np.mean(isi)),cv,ad

def aggregate_repeats(group,target):
    # Repeated presentations of the same Noise seed: resample each, trim to common length,
    # verify stimuli agree, then average voltage across repeats. This reduces trial noise.
    ss=[];vv=[]; rates=[]; stats=[]
    for x in group:
        i,ri=down(x['i'],x['rate'],target); v,rv=down(x['v'],x['rate'],target)
        n=min(len(i),len(v)); i=i[:n];v=v[:n]
        if n<1000:continue
        ss.append(i);vv.append(v);rates.append((ri+rv)/2);stats.append(x['stats'])
    if not ss:return None
    n=min(map(len,ss)); ss=np.stack([x[:n] for x in ss]); vv=np.stack([x[:n] for x in vv])
    # stimulus repeats should be essentially identical; retain median trace if not exact.
    stim=np.median(ss,axis=0).astype(np.float32)
    volt=np.mean(vv,axis=0).astype(np.float32)
    stim_repeat_rmse=float(np.sqrt(np.mean((ss-stim[None,:])**2)))
    return stim,volt,float(np.median(rates)),len(ss),stim_repeat_rmse,stats

def one(job):
    pstr,sid,outdir,target,lookup,bycell=job;p=Path(pstr)
    try:
        wanted={sw:name for sw,name in bycell.get(sid,[]) if name.strip().lower() in ('noise 1','noise 2')}
        if not wanted:return {'specimen_id':sid,'status':'inventory_no_noise12'}
        groups={'Noise 1':[],'Noise 2':[]}
        with h5py.File(p,'r') as f:
            A=f.get('/acquisition/timeseries'); S=f.get('/stimulus/presentation')
            if A is None or S is None:raise RuntimeError('missing acquisition/stimulus roots')
            for sw,csvname in sorted(wanted.items()):
                key=f'Sweep_{sw}'
                if key not in A or key not in S:continue
                ag=A[key];sg=S[key]
                fs=rate(ag)
                if not np.isfinite(fs) or fs<=0:continue
                nm=lookup.get((sid,sw),'').strip() or nwb_name(ag,sg,sw)
                if nm not in groups:continue
                v=convert(ag['data'][()],ag['data'],'v')
                i=convert(sg['data'][()],sg['data'],'i')
                n=min(len(v),len(i));v=v[:n];i=i[:n]
                st=spike_stats(v,fs)
                groups[nm].append({'sw':sw,'i':i,'v':v,'rate':fs,'stats':st})
        a=aggregate_repeats(groups['Noise 1'],target); b=aggregate_repeats(groups['Noise 2'],target)
        if a is None or b is None:
            return {'specimen_id':sid,'status':'missing_noise_class',
                    'noise1_repeats':len(groups['Noise 1']),'noise2_repeats':len(groups['Noise 2'])}
        arr={}
        meta={'specimen_id':sid,'status':'ok','noise1_repeats':a[3],'noise2_repeats':b[3],
              'noise1_stim_repeat_rmse_pa':a[4],'noise2_stim_repeat_rmse_pa':b[4]}
        for j,x in enumerate((a,b)):
            stim,volt,fs,nrep,rmse,stats=x
            arr[f'stim{j}']=stim;arr[f'volt{j}']=volt
            meta[f'sweep{j}_name']=f'Noise {j+1}'
            meta[f'sweep{j}_rate']=fs;meta[f'sweep{j}_samples']=len(stim)
            meta[f'sweep{j}_repeat_count']=nrep
        arr['specimen_id']=np.asarray([int(sid)],np.int64);arr['rate_hz']=np.asarray([target],np.float32)
        np.savez_compressed(Path(outdir)/f'{sid}.npz',**arr)
        return meta
    except Exception as e:return {'specimen_id':sid,'status':'failed','detail':f'{type(e).__name__}: {e}'}

def run(config,limit=0,workers=None,smoke=False):
    c=load_cfg(config);root=Path(c['allen_root'])
    out=ensure(Path(c['output_root'])/'20_allen_ephys_v11');npz=ensure(out/'noise_npz')
    lookup,bycell,src=load_lookup(root)
    files=sorted((root/'ephys_nwb'/'raw').glob('*.nwb'))
    jobs=[]
    for p in files:
        m=re.search(r'(\d+)',p.stem)
        if m:jobs.append((str(p),m.group(1),str(npz),int(c['target_rate_hz']),lookup,bycell))
    if limit:jobs=jobs[:limit]
    w=workers or int(c.get('allen_cpu_workers',16))
    print(f'Allen v1.1 files={len(jobs)} workers={w} target={c["target_rate_hz"]}Hz',flush=True)
    rows=[]
    with ProcessPoolExecutor(max_workers=w) as ex:
        futs=[ex.submit(one,j) for j in jobs]
        for k,f in enumerate(as_completed(futs),1):
            rows.append(f.result())
            if k%20==0 or k==len(futs):
                ok=sum(r.get('status')=='ok' for r in rows)
                print(f'Allen v1.1 {k}/{len(futs)} usable_noise12={ok}',flush=True)
    rows.sort(key=lambda r:int(r['specimen_id']))
    fields=sorted({k for r in rows for k in r})
    with (out/('smoke_summary.csv' if smoke else 'allen_noise_extraction_summary.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)
    ok=sum(r.get('status')=='ok' for r in rows)
    summary={'files':len(rows),'usable_noise12':ok,'fraction':ok/max(1,len(rows)),'lookup_source':src}
    (out/('smoke_summary.json' if smoke else 'allen_ephys_summary.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if smoke and summary['fraction']<0.90:
        raise SystemExit(f'SMOKE FAILED usable fraction={summary["fraction"]:.3f}')
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--workers',type=int,default=0);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();run(a.config,a.limit,a.workers or None,a.smoke)
