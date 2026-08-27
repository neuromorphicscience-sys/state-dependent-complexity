#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from common import cfg,read_csv,write_csv,write_json,fnum,now

FEATURES=["axon_length","dendrite_length","branch_points","endpoints","max_path_length","max_euclidean_radius",
          "radius_gyration","bbox_volume","spatial_entropy_4","spatial_entropy_6","node_type_entropy","path_tortuosity_proxy"]

def clean(rows):
    good=[];X=[];g=[]
    for r in rows:
        if r.get("status")!="ok" or str(r.get("projection_subtype",""))=="":continue
        vals=[fnum(r.get(f)) for f in FEATURES]
        if np.isfinite(vals).sum()<len(FEATURES)-2:continue
        good.append(r);X.append(vals);g.append(int(float(r["projection_subtype"])))
    X=np.asarray(X,float)
    for j in range(X.shape[1]):
        med=np.nanmedian(X[:,j]);X[~np.isfinite(X[:,j]),j]=med
        # log-transform positive heavy-tailed geometric features
        if FEATURES[j] not in ("spatial_entropy_4","spatial_entropy_6","node_type_entropy","path_tortuosity_proxy"):
            X[:,j]=np.log1p(np.maximum(X[:,j],0))
    med=np.median(X,axis=0);mad=np.median(np.abs(X-med),axis=0)*1.4826
    sd=np.std(X,axis=0);scale=np.where(mad>1e-8,mad,np.where(sd>1e-8,sd,1))
    return good,(X-med)/scale,np.asarray(g,int)

def pseudoF(X,g):
    n,p=X.shape;ug=np.unique(g);grand=X.mean(0)
    ssb=0.;ssw=0.
    for q in ug:
        z=X[g==q];mu=z.mean(0);ssb+=len(z)*np.sum((mu-grand)**2);ssw+=np.sum((z-mu)**2)
    dfb=len(ug)-1;dfw=n-len(ug);F=(ssb/max(dfb,1))/(ssw/max(dfw,1))
    r2=ssb/max(ssb+ssw,1e-12)
    return float(F),float(r2)

def eta2(x,g):
    x=np.asarray(x,float);grand=x.mean();sst=np.sum((x-grand)**2);ssb=0
    for q in np.unique(g):
        z=x[g==q];ssb+=len(z)*(z.mean()-grand)**2
    return float(ssb/max(sst,1e-12))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args();c=cfg(a.config)
    root=Path(c["output_root"]);out=root/"30_structural_stats_v12";out.mkdir(parents=True,exist_ok=True)
    rows=read_csv(root/"10_digital_brain_v12/pfc2022_canonical_6357_features_v12.csv")
    good,X,g=clean(rows);F,R2=pseudoF(X,g)
    rng=np.random.default_rng(int(c["structural"]["seed"]));nperm=int(c["structural"]["permutations"]);ge=0
    for i in range(nperm):
        fp,_=pseudoF(X,rng.permutation(g));ge+=fp>=F
        if (i+1)%500==0:print(f"structural permutation {i+1}/{nperm}",flush=True)
    p=(ge+1)/(nperm+1)
    effects=[]
    for j,f in enumerate(FEATURES):
        obs=eta2(X[:,j],g);gej=0
        # use 1000 perms per feature to control runtime
        for _ in range(1000):gej+=eta2(X[:,j],rng.permutation(g))>=obs
        effects.append({"feature":f,"eta2_subtype":obs,"perm_p":(gej+1)/1001})
    effects.sort(key=lambda r:r["eta2_subtype"],reverse=True)
    # subtype centroid heterogeneity
    cents=[]
    for q in sorted(np.unique(g)):
        z=X[g==q];rec={"projection_subtype":int(q),"n":len(z)}
        mu=z.mean(0)
        for j,f in enumerate(FEATURES):rec[f"{f}_zmean"]=float(mu[j])
        cents.append(rec)
    summary={"created_utc":now(),"n":len(g),"subtypes":len(np.unique(g)),"multivariate_pseudoF":F,
             "multivariate_R2":R2,"permutation_p":p,"features":FEATURES}
    write_csv(out/"pfc2022_feature_subtype_effects.csv",effects)
    write_csv(out/"pfc2022_subtype_centroids.csv",cents)
    write_json(out/"pfc2022_structural_stats_summary.json",summary);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
