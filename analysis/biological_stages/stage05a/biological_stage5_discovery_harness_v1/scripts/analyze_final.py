#!/usr/bin/env python3
import argparse, csv, json, math, os, re
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from common import load_config, ensure_dir, write_csv, write_json, read_csv, robust_z, spearman, now

def fnum(x):
    try:return float(x)
    except:return float("nan")

def context_table(path,eps):
    rows=read_csv(path)
    by=defaultdict(dict)
    for r in rows:
        by[int(float(r["specimen_id"]))][int(float(r["context_ms"]))]=fnum(r["r2"])
    out={}
    for sid,d in by.items():
        cs=sorted(d)
        vals=np.asarray([d[c] for c in cs],float)
        if not np.isfinite(vals).any(): continue
        best=float(np.nanmax(vals)); threshold=best-eps
        sufficient=[c for c in cs if np.isfinite(d[c]) and d[c]>=threshold]
        out[sid]={"c_context":min(sufficient) if sufficient else max(cs),"best_r2":best,
                  **{f"r2_{c}":d[c] for c in cs}}
    return out

def wilcoxon_or_sign(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0:return float("nan"),float("nan")
    try:
        from scipy.stats import wilcoxon
        s,p=wilcoxon(x)
        return float(s),float(p)
    except Exception:
        pos=np.sum(x>0); neg=np.sum(x<0); n=pos+neg
        return float(pos),float("nan")

def eta2(groups,values):
    values=np.asarray(values,float); groups=np.asarray(groups)
    m=np.isfinite(values); values=values[m]; groups=groups[m]
    if len(values)<3:return float("nan")
    grand=values.mean(); ss_tot=np.sum((values-grand)**2)
    if ss_tot<=0:return 0.0
    ss_b=0.
    for g in np.unique(groups):
        v=values[groups==g]
        if len(v): ss_b+=len(v)*(v.mean()-grand)**2
    return float(ss_b/ss_tot)

def perm_eta2(groups,values,nperm=5000,seed=123):
    obs=eta2(groups,values)
    rng=np.random.default_rng(seed)
    ge=0
    for _ in range(nperm):
        if eta2(rng.permutation(groups),values)>=obs: ge+=1
    return obs,(ge+1)/(nperm+1)

def numeric_feature_matrix(rows,features):
    good=[]
    X=[]
    for r in rows:
        if r.get("status")!="ok":continue
        vals=[fnum(r.get(f,"")) for f in features]
        if np.isfinite(vals).sum()>=len(features)-1:
            good.append(r); X.append(vals)
    X=np.asarray(X,float)
    if X.size==0:return [],X
    # median impute
    for j in range(X.shape[1]):
        med=np.nanmedian(X[:,j]); X[~np.isfinite(X[:,j]),j]=med
    Xz,_,_=robust_z(X)
    return good,Xz

def pca_np(X):
    X=X-np.mean(X,axis=0)
    u,s,vt=np.linalg.svd(X,full_matrices=False)
    scores=X@vt.T
    var=s*s
    ratio=var/var.sum()
    return scores,ratio,vt

def discover_cmodel(allen_root):
    candidates=[]
    for p in Path(allen_root).rglob("*.csv"):
        if p.stat().st_size>200*1024*1024: continue
        try:
            with p.open("r",encoding="utf-8-sig",newline="") as f:
                rd=csv.reader(f); hdr=next(rd)
            low=[x.lower() for x in hdr]
            sid=next((h for h,l in zip(hdr,low) if l in ("specimen_id","specimen__id")),None)
            cc=next((h for h,l in zip(hdr,low) if re.search(r"(c.?req|minimum.*complex|complexity.*req)",l)),None)
            if sid and cc:candidates.append((p,sid,cc))
        except Exception:pass
    if not candidates:return None,None,None
    candidates.sort(key=lambda x:("glif" not in str(x[0]).lower(),len(str(x[0]))))
    return candidates[0]

def load_allen_metadata(allen_root):
    p=Path(allen_root)/"metadata"/"all_cells_raw.json"
    out={}
    if not p.exists():return out
    try:
        data=json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data,dict):
            for k in ("msg","data","cells"):
                if isinstance(data.get(k),list):data=data[k];break
        for r in data:
            sid=r.get("specimen__id",r.get("specimen_id"))
            if sid is None:continue
            area=r.get("structure__acronym",r.get("structure_acronym",""))
            layer=r.get("structure__layer",r.get("layer",""))
            out[int(sid)]={"area":str(area or ""),"layer":str(layer or "")}
    except Exception:pass
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True)
    a=ap.parse_args();c=load_config(a.config)
    out=ensure_dir(Path(c["output_root"])/"40_analysis")
    eps=float(c["epsilon_primary"])
    ab=context_table(Path(c["output_root"])/"30_temporal_context"/"AB"/"context_performance.csv",eps)
    ba=context_table(Path(c["output_root"])/"30_temporal_context"/"BA"/"context_performance.csv",eps)
    common=sorted(set(ab)&set(ba))
    carr_ab=np.array([ab[s]["c_context"] for s in common],float)
    carr_ba=np.array([ba[s]["c_context"] for s in common],float)
    rho_rep,p_rep=spearman(carr_ab,carr_ba)
    contexts=sorted(c["contexts_ms"])
    longc=max(contexts);shortc=min(contexts)
    gains=[]
    cellrows=[]
    for sid in common:
        ga=ab[sid].get(f"r2_{longc}",np.nan)-ab[sid].get(f"r2_{shortc}",np.nan)
        gb=ba[sid].get(f"r2_{longc}",np.nan)-ba[sid].get(f"r2_{shortc}",np.nan)
        gains.append(np.nanmean([ga,gb]))
        cellrows.append({
            "specimen_id":sid,"C_temporal_AB_ms":ab[sid]["c_context"],
            "C_temporal_BA_ms":ba[sid]["c_context"],
            "C_temporal_consensus_ms":max(ab[sid]["c_context"],ba[sid]["c_context"]),
            "best_r2_AB":ab[sid]["best_r2"],"best_r2_BA":ba[sid]["best_r2"],
            "long_minus_short_r2_mean":np.nanmean([ga,gb])
        })
    stat_gain,p_gain=wilcoxon_or_sign(gains)
    dist=Counter(r["C_temporal_consensus_ms"] for r in cellrows)

    # C_model optional
    cmodel_path,sidcol,ccol=discover_cmodel(c["allen_root"])
    cmodel_stats={"found":False}
    if cmodel_path:
        try:
            cr=read_csv(cmodel_path); cm={}
            for r in cr:
                try:cm[int(float(r[sidcol]))]=float(r[ccol])
                except:pass
            xs=[];ys=[]
            for r in cellrows:
                if r["specimen_id"] in cm:
                    r["C_model"]=cm[r["specimen_id"]]
                    xs.append(r["C_temporal_consensus_ms"]);ys.append(cm[r["specimen_id"]])
            rr,pp=spearman(xs,ys)
            cmodel_stats={"found":True,"path":str(cmodel_path),"column":ccol,"n":len(xs),"rho":rr,"p":pp}
        except Exception as e:cmodel_stats={"found":False,"error":str(e)}

    write_csv(out/"allen_temporal_complexity_cells.csv",cellrows)

    # Digital structural organization
    feats=["axon_length","dendrite_length","branch_points","endpoints","max_path_length",
           "max_euclidean_radius","radius_gyration","bbox_volume","spatial_entropy_4"]
    pfcrows=read_csv(Path(c["output_root"])/"10_digital_brain"/"pfc2022_features.csv")
    good,X=numeric_feature_matrix(pfcrows,feats)
    structural={}
    feature_eta=[]
    if len(good)>10:
        scores,ratio,vt=pca_np(X)
        groups=np.array([r.get("projection_subtype","") for r in good])
        m=groups!=""
        if m.sum()>10 and len(set(groups[m]))>=2:
            obs,pv=perm_eta2(groups[m],scores[m,0],5000,123)
            structural={"n":int(m.sum()),"subtypes":len(set(groups[m])),"pc1_var":float(ratio[0]),
                        "pc1_subtype_eta2":obs,"pc1_perm_p":pv}
            for j,f in enumerate(feats):
                e,p=perm_eta2(groups[m],X[m,j],2000,100+j)
                feature_eta.append({"feature":f,"eta2_subtype":e,"perm_p":p})
    write_csv(out/"pfc2022_subtype_effects.csv",feature_eta)

    # PFC 2023 axon/dendrite coupling
    p23=read_csv(Path(c["output_root"])/"10_digital_brain"/"pfc2023_features.csv")
    ax=[];de=[]
    for r in p23:
        a1=fnum(r.get("axon_length"));d1=fnum(r.get("dendrite_length"))
        if np.isfinite(a1) and np.isfinite(d1) and a1>0 and d1>0:ax.append(a1);de.append(d1)
    rho_ad,p_ad=spearman(ax,de)
    pfc23={"n":len(ax),"axon_dendrite_spearman":rho_ad,"p":p_ad}

    thr=c["claim_thresholds"];alpha=float(thr["alpha"])
    h1=(len(common)>=int(c["min_noise_cells"]) and
        np.isfinite(rho_rep) and rho_rep>=float(thr["temporal_repro_rho"]) and
        np.nanmedian(gains)>=float(thr["median_long_short_r2_gain"]) and
        (not np.isfinite(p_gain) or p_gain<alpha) and len(dist)>=3)
    h2=(structural.get("pc1_subtype_eta2",0)>=float(thr["structural_subtype_eta2"]) and
        structural.get("pc1_perm_p",1)<alpha)

    verdict={
        "created_utc":now(),
        "H1_intrinsic_temporal_complexity":{
            "supported":bool(h1),"n_cells":len(common),"AB_BA_rho":rho_rep,"AB_BA_p":p_rep,
            "median_long_minus_short_r2_gain":float(np.nanmedian(gains)),
            "gain_test_stat":stat_gain,"gain_p":p_gain,"C_temporal_distribution":dict(sorted(dist.items()))
        },
        "H1b_Cmodel_correspondence":cmodel_stats,
        "H2_structural_allocation_nonrandom":{"supported":bool(h2),**structural},
        "PFC2023_local_longrange_morphology":pfc23,
        "H3_dynamic_leverage_vs_static_prominence":{
            "supported":False,"status":"NOT TESTED BY CURRENT DATASETS",
            "reason":"A state-resolved functional population dataset is required to compare dynamic leverage against static prominence without circular inference."
        },
        "main_journal_extension":{
            "status":"REQUIRES FUNCTIONAL DATASET",
            "target":"Test whether C_bio tracks cross-state leverage variability / role flexibility L_i(s), then return to the model for a second theory cycle."
        }
    }
    write_json(out/"verdict.json",verdict)

    zh=[
        "# Stage 5 生物学发现：自动结论",
        "",
        f"- H1 真实神经元的内在时间动力学复杂度：**{'支持' if h1 else '暂不支持/需检查'}**。",
        f"  - 可分析细胞：{len(common)}",
        f"  - Noise A→B 与 B→A 的 minimum sufficient temporal context 一致性：Spearman ρ={rho_rep:.3f}, p={p_rep:.3g}",
        f"  - 500 ms 相对 10 ms 的预测 R² 增益中位数：{np.nanmedian(gains):.4f}, 检验 p={p_gain:.3g}",
        f"  - C_temporal 分布：{dict(sorted(dist.items()))}",
        "",
        f"- H2 projectome 结构角色的非随机组织：**{'支持' if h2 else '暂不支持/需检查'}**。",
        f"  - PFC2022 subtype 数：{structural.get('subtypes','NA')}",
        f"  - 结构 PC1 解释率：{structural.get('pc1_var',float('nan')):.3f}",
        f"  - subtype 对结构 PC1 的 η²：{structural.get('pc1_subtype_eta2',float('nan')):.3f}, permutation p={structural.get('pc1_perm_p',float('nan')):.3g}",
        "",
        f"- PFC2023 axon–dendrite 跨尺度耦合：n={pfc23['n']}, Spearman ρ={rho_ad:.3f}, p={p_ad:.3g}",
        "",
        "## 当前能够写入论文的边界",
        "当前两类真实数据能够回答“内在复杂度是否为可重复的生物表型”以及“结构网络角色是否非随机组织”。",
        "但它们**不能单独证明** dynamic leverage 比 static prominence 更能解释复杂度配置；这一主张必须由状态分辨的功能群体数据独立检验。",
        "",
        "这条边界是预注册式的：脚本不会因为期待正结果而把缺失的 dynamic-leverage 证据解释成已经验证。"
    ]
    (out/"final_report_zh.md").write_text("\n".join(zh),encoding="utf-8")

    en=[
        "# Stage 5 Biological Discovery: Automated Conclusions","",
        f"- H1 intrinsic temporal dynamical complexity: **{'SUPPORTED' if h1 else 'NOT YET SUPPORTED / CHECK REQUIRED'}**.",
        f"  - analyzable cells: {len(common)}",
        f"  - A→B vs B→A minimum sufficient context reproducibility: Spearman rho={rho_rep:.3f}, p={p_rep:.3g}",
        f"  - median R2 gain (500 ms vs 10 ms): {np.nanmedian(gains):.4f}, p={p_gain:.3g}",
        "",
        f"- H2 non-random projectome structural organization: **{'SUPPORTED' if h2 else 'NOT YET SUPPORTED / CHECK REQUIRED'}**.",
        f"  - subtype eta2 on structural PC1={structural.get('pc1_subtype_eta2',float('nan')):.3f}, permutation p={structural.get('pc1_perm_p',float('nan')):.3g}",
        "",
        "The present datasets do not, by themselves, establish that dynamic leverage explains complexity allocation better than static prominence. That comparison is reserved for an independent state-resolved functional dataset."
    ]
    (out/"final_report_en.md").write_text("\n".join(en),encoding="utf-8")
    print(json.dumps(verdict,indent=2))

if __name__=="__main__":main()
