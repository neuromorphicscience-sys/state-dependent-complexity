#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import cfg,read_csv,fnum

def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/f"{name}.pdf");fig.savefig(out/f"{name}.png",dpi=300);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args();c=cfg(a.config)
    root=Path(c["output_root"]);out=root/"50_figures_v12";out.mkdir(parents=True,exist_ok=True)
    # Prediction
    rows=read_csv(root/"20_allen_multiaxial_v12/allen_multiaxial_primary_mouse_complete.csv")
    y=np.array([fnum(r["C_req_eps_0.02"]) for r in rows]);p=np.array([fnum(r["pred_Creq_multiaxial_ephys"]) for r in rows])
    fig,ax=plt.subplots(figsize=(4,3.6));ax.scatter(y,p,s=10,alpha=.4);ax.set_xlabel("Observed minimum mechanism cost");ax.set_ylabel("Cross-validated multiaxial prediction")
    save(fig,out,"Fig5_multiaxial_predicts_Creq")
    # model comparison
    sm=json.loads((root/"20_allen_multiaxial_v12/allen_multiaxial_summary.json").read_text())
    names=["temporal_profile","metadata_only","multiaxial_ephys","ephys_plus_metadata","ephys_plus_temporal"]
    vals=[sm["models"][x]["rho"] for x in names]
    fig,ax=plt.subplots(figsize=(5,3.5));ax.bar(np.arange(len(names)),vals);ax.set_xticks(np.arange(len(names)));ax.set_xticklabels(names,rotation=35,ha="right");ax.set_ylabel("Cross-validated Spearman ρ")
    save(fig,out,"Fig5_model_comparison")
    # feature associations
    assoc=read_csv(root/"20_allen_multiaxial_v12/feature_Creq_associations.csv")[:15]
    if assoc:
        names=[r["feature"] for r in assoc][::-1];vals=[fnum(r["rho"]) for r in assoc][::-1]
        fig,ax=plt.subplots(figsize=(5.5,4.5));ax.barh(np.arange(len(names)),vals);ax.set_yticks(np.arange(len(names)));ax.set_yticklabels(names);ax.set_xlabel("Spearman ρ with C_req")
        save(fig,out,"Fig5_feature_associations")
    # structural effects
    eff=read_csv(root/"30_structural_stats_v12/pfc2022_feature_subtype_effects.csv")
    names=[r["feature"] for r in eff][::-1];vals=[fnum(r["eta2_subtype"]) for r in eff][::-1]
    fig,ax=plt.subplots(figsize=(5.2,4));ax.barh(np.arange(len(names)),vals);ax.set_yticks(np.arange(len(names)));ax.set_yticklabels(names);ax.set_xlabel("Projection subtype η²")
    save(fig,out,"Fig6_projection_subtype_structural_effects")
    print("FIGURES COMPLETE",out)
if __name__=="__main__":main()
