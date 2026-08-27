#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,wilcoxon
from collections import defaultdict,Counter
from common import cfg,read_csv,write_json,write_csv,fnum,now

def temporal_boundary(v11,eps):
    def load(d):
        by=defaultdict(dict)
        for r in read_csv(Path(v11)/f"30_temporal_context/{d}/context_performance.csv"):
            by[int(float(r["specimen_id"]))][int(float(r["context_ms"]))]=fnum(r["r2"])
        out={}
        for sid,z in by.items():
            cs=sorted(z);best=max(z.values());s=[c for c in cs if z[c]>=best-eps]
            out[sid]={"C":min(s),"gain":z[max(cs)]-z[min(cs)]}
        return out
    A=load("AB");B=load("BA");ids=sorted(set(A)&set(B))
    rho,p=spearmanr([A[i]["C"] for i in ids],[B[i]["C"] for i in ids])
    gains=np.asarray([(A[i]["gain"]+B[i]["gain"])/2 for i in ids],float)
    try:wp=wilcoxon(gains).pvalue
    except:wp=np.nan
    dist=Counter(max(A[i]["C"],B[i]["C"]) for i in ids)
    return {"n":len(ids),"AB_BA_rho":float(rho),"AB_BA_p":float(p),"median_500_minus_10_R2":float(np.median(gains)),
            "gain_wilcoxon_p":float(wp),"consensus_context_distribution":dict(sorted(dist.items()))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args();c=cfg(a.config)
    root=Path(c["output_root"]);out=root/"40_final_adjudication_v12";out.mkdir(parents=True,exist_ok=True)
    allen=json.loads((root/"20_allen_multiaxial_v12/allen_multiaxial_summary.json").read_text())
    struct=json.loads((root/"30_structural_stats_v12/pfc2022_structural_stats_summary.json").read_text())
    digital=json.loads((root/"10_digital_brain_v12/digital_brain_v12_summary.json").read_text())
    temporal=temporal_boundary(c["v11_result_root"],float(c["c_model_epsilon"]))
    A=c["adjudication"]
    m=allen["models"]
    delta_mt=m["multiaxial_ephys"]["rho"]-m["temporal_profile"]["rho"]
    delta_em=m["ephys_plus_metadata"]["rho"]-m["metadata_only"]["rho"]
    h_mult=(m["multiaxial_ephys"]["permutation_p"]<A["alpha"] and delta_mt>=A["multiaxial_vs_temporal_delta_rho"]
            and allen["bootstrap_multiaxial_minus_temporal"]["ci95"][0]>0)
    h_ind=(m["ephys_plus_metadata"]["permutation_p"]<A["alpha"] and delta_em>=A["ephys_vs_metadata_delta_rho"]
           and allen["bootstrap_ephys_plus_metadata_minus_metadata"]["ci95"][0]>0)
    h_struct=(struct["permutation_p"]<A["alpha"] and struct["multivariate_R2"]>=A["structural_r2_min"])
    temporal_strong=(temporal["AB_BA_rho"]>=0.30 and temporal["median_500_minus_10_R2"]>=0.01)
    verdict={
      "created_utc":now(),
      "H_temporal_memory_as_complexity_axis":{"supported":bool(temporal_strong),**temporal,
        "interpretation":"Boundary test: a single stimulus-history timescale is not promoted to C_bio unless it is reproducible and longer context gives material held-out benefit."},
      "H_multiaxial_ephys_maps_to_model_requirement":{"supported":bool(h_mult),"delta_rho_vs_temporal":delta_mt,
        "multiaxial":m["multiaxial_ephys"],"temporal":m["temporal_profile"],
        "bootstrap_delta":allen["bootstrap_multiaxial_minus_temporal"]},
      "H_ephys_information_beyond_static_metadata":{"supported":bool(h_ind),"delta_rho":delta_em,
        "ephys_plus_metadata":m["ephys_plus_metadata"],"metadata_only":m["metadata_only"],
        "bootstrap_delta":allen["bootstrap_ephys_plus_metadata_minus_metadata"]},
      "H_projectome_structure_nonrandom_by_projection_subtype":{"supported":bool(h_struct),**struct},
      "digital_qc":digital,
      "dynamic_leverage_claim":{"supported":False,"status":"NOT TESTED",
        "reason":"The present datasets do not provide a matched state-resolved functional-population leverage variable L_i(s)."},
      "paper_level_summary":{
        "supported_core":bool(h_mult and h_struct),
        "statement":"If supported_core=true, the data support a multiaxial intrinsic-complexity phenotype linked to mechanistic model requirement, together with non-random allocation of projectome structure across projection-defined biological roles. They do not yet establish dynamic leverage as the organizing variable."
      }
    }
    write_json(out/"verdict_v12.json",verdict)
    zh=[
      "# Stage 5 v1.2 生物学主分析最终裁决","",
      f"## 1. 单一时间记忆轴：{'支持' if temporal_strong else '不支持作为主要复杂度定义'}",
      f"- n={temporal['n']}；A→B vs B→A Spearman ρ={temporal['AB_BA_rho']:.3f}。",
      f"- 500 ms 相对 10 ms 的 held-out R² 增益中位数={temporal['median_500_minus_10_R2']:.4f}。",
      "- 因此该实验作为 boundary result 保留：timescale 不等同于 intrinsic complexity。","",
      f"## 2. 多轴电生理复杂度 → GLIF 最小充分机制需求：{'支持' if h_mult else '当前不支持'}",
      f"- multiaxial ephys CV Spearman ρ={m['multiaxial_ephys']['rho']:.3f}，permutation p={m['multiaxial_ephys']['permutation_p']:.4g}。",
      f"- temporal profile CV ρ={m['temporal_profile']['rho']:.3f}；Δρ={delta_mt:.3f}。",
      f"- bootstrap Δρ 95% CI={allen['bootstrap_multiaxial_minus_temporal']['ci95']}.","",
      f"## 3. 排除静态元数据解释：{'支持' if h_ind else '当前不支持'}",
      f"- metadata-only CV ρ={m['metadata_only']['rho']:.3f}。",
      f"- ephys+metadata CV ρ={m['ephys_plus_metadata']['rho']:.3f}；Δρ={delta_em:.3f}。","",
      f"## 4. PFC 6357 projectome 的 projection-subtype 结构组织：{'支持' if h_struct else '当前不支持'}",
      f"- 64 subtypes；multivariate R²={struct['multivariate_R2']:.4f}；permutation p={struct['permutation_p']:.4g}。","",
      "## 5. 可写入论文的边界",
      "- PFC2022 已固定为 6357 个 `swc_allen_space` canonical neurons；不再把 12714 个坐标表示当作独立细胞。",
      "- 当前所谓 PFC2023 数据只做审计，不作为独立 ~2000-neuron dendrite cohort 写入结论。",
      "- dynamic leverage > static prominence 仍未被这些数据直接检验，必须留给状态分辨的 functional dataset。","",
      "## 综合",
      verdict["paper_level_summary"]["statement"]
    ]
    (out/"final_report_zh.md").write_text("\n".join(zh),encoding="utf-8")
    en=[
      "# Stage 5 v1.2 Biological Adjudication","",
      f"Single temporal-memory axis: {'SUPPORTED' if temporal_strong else 'NOT SUPPORTED AS THE PRIMARY COMPLEXITY DEFINITION'}.",
      f"Multiaxial electrophysiology to minimum sufficient GLIF mechanism requirement: {'SUPPORTED' if h_mult else 'NOT CURRENTLY SUPPORTED'}.",
      f"Projection-subtype structural organization: {'SUPPORTED' if h_struct else 'NOT CURRENTLY SUPPORTED'}.",
      "",
      verdict["paper_level_summary"]["statement"],
      "",
      "Dynamic leverage remains explicitly untested by the present datasets."
    ]
    (out/"final_report_en.md").write_text("\n".join(en),encoding="utf-8")
    print(json.dumps(verdict,indent=2))
if __name__=="__main__":main()
