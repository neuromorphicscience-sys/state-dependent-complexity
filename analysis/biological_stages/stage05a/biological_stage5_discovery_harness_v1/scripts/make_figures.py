#!/usr/bin/env python3
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_config, read_csv, ensure_dir

def f(x):
    try:return float(x)
    except:return np.nan

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True)
    a=ap.parse_args();c=load_config(a.config)
    root=Path(c["output_root"]); out=ensure_dir(root/"50_figures")
    # Context performance curve
    rows=read_csv(root/"30_temporal_context"/"AB"/"context_performance.csv")
    by=defaultdict(list)
    for r in rows:by[int(float(r["context_ms"]))].append(f(r["r2"]))
    cs=sorted(by);med=[];lo=[];hi=[]
    rng=np.random.default_rng(123)
    for x in cs:
        v=np.asarray(by[x],float);v=v[np.isfinite(v)]
        med.append(np.median(v))
        boots=[np.median(rng.choice(v,len(v),replace=True)) for _ in range(1000)] if len(v) else [np.nan]
        lo.append(np.percentile(boots,2.5));hi.append(np.percentile(boots,97.5))
    plt.figure(figsize=(4.5,3.5))
    plt.plot(cs,med,marker="o")
    plt.fill_between(cs,lo,hi,alpha=.2)
    plt.xscale("log");plt.xlabel("Temporal context (ms)");plt.ylabel("Held-out Noise R²")
    plt.tight_layout();plt.savefig(out/"Fig5_temporal_context_curve.pdf");plt.savefig(out/"Fig5_temporal_context_curve.png",dpi=300);plt.close()

    # C temporal AB vs BA
    cells=read_csv(root/"40_analysis"/"allen_temporal_complexity_cells.csv")
    x=np.array([f(r["C_temporal_AB_ms"]) for r in cells]);y=np.array([f(r["C_temporal_BA_ms"]) for r in cells])
    plt.figure(figsize=(3.7,3.7));plt.scatter(x,y,s=8,alpha=.35)
    mn=min(c["contexts_ms"]);mx=max(c["contexts_ms"])
    plt.plot([mn,mx],[mn,mx],linewidth=1)
    plt.xscale("log");plt.yscale("log");plt.xlabel("C temporal A→B (ms)");plt.ylabel("C temporal B→A (ms)")
    plt.tight_layout();plt.savefig(out/"Fig5_temporal_reproducibility.pdf");plt.savefig(out/"Fig5_temporal_reproducibility.png",dpi=300);plt.close()

    # Structural subtype eta2
    effects=read_csv(root/"40_analysis"/"pfc2022_subtype_effects.csv")
    if effects:
        names=[r["feature"] for r in effects];vals=[f(r["eta2_subtype"]) for r in effects]
        order=np.argsort(vals)
        plt.figure(figsize=(5,4));plt.barh(np.arange(len(names)),np.asarray(vals)[order])
        plt.yticks(np.arange(len(names)),np.asarray(names)[order]);plt.xlabel("Variance explained by projection subtype (η²)")
        plt.tight_layout();plt.savefig(out/"Fig6_structural_subtype_eta2.pdf");plt.savefig(out/"Fig6_structural_subtype_eta2.png",dpi=300);plt.close()

    # Axon/dendrite
    p23=read_csv(root/"10_digital_brain"/"pfc2023_features.csv")
    ax=[];de=[]
    for r in p23:
        a1=f(r.get("axon_length"));d1=f(r.get("dendrite_length"))
        if np.isfinite(a1) and np.isfinite(d1) and a1>0 and d1>0:ax.append(a1);de.append(d1)
    if ax:
        plt.figure(figsize=(3.8,3.5));plt.scatter(ax,de,s=5,alpha=.25)
        plt.xlabel("Axon cable length");plt.ylabel("Dendrite cable length")
        plt.tight_layout();plt.savefig(out/"Fig6_axon_dendrite_coupling.pdf");plt.savefig(out/"Fig6_axon_dendrite_coupling.png",dpi=300);plt.close()
    print("Figures written to",out)

if __name__=="__main__":main()
