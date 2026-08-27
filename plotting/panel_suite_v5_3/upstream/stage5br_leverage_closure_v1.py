#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 5B-R — Leverage decomposition and validation
===================================================
Paper-level analysis / plotting closure for the existing Stage 5B baseline.

Default local inputs (Windows):
  D:\\Research\\Neural Science\\bio data\\Stage5B_complete_20260820.tar.gz
  D:\\Research\\Neural Science\\bio data\\stage5b_dynamic_leverage_v1.zip

Default output:
  D:\\Research\\Neural Science\\plot\\Stage5BR_leverage_closure_v1\\
      analysis\\
      figures\\

This script deliberately DOES NOT re-fit the NWB dynamics models. It performs
strict archive-level scientific closure on the outputs already computed:

  * strict Steinmetz duplicate-session audit and de-duplication
  * state-pair leverage-rank stability with hierarchical session/container summaries
  * activity/loading residualized leverage
  * stable-backbone vs state-residual variance decomposition
  * neuron-identity shuffle null for observed flexibility
  * activity-flexibility controls
  * predictive-leverage state specificity and L_pred vs L_dyn separation
  * Allen VBO same-cell primary-state re-ranking
  * Allen VBO latent-R2 quality stratification
  * same-session running / pupil state contrasts
  * cross-dataset stability/flexibility comparison
  * source-code audit of the baseline implementation
  * many standalone, title-free publication-style panels (600-dpi PNG + vector PDF)

Scientific guardrails
---------------------
1. Steinmetz baseline archive contains repeated copies of some sessions because
   multiple NWB paths share the same file stem. The original script appends the
   checkpoint CSV again on every repeated path. This script STRICTLY de-duplicates
   session/neuron/state (and predictive session/neuron/state/target) before any
   inference.
2. Current L_dyn includes activity amplitude by construction. Results are therefore
   reported both raw and after within-state rank residualization against mean activity
   and PCA loading norm.
3. Allen VBO latent-dynamics one-step R2 is used as a validity diagnostic. The script
   never relabels weak-R2 results as validated dynamics.
4. The archived predictive-leverage output does not contain enough label-distribution
   information to reconstruct a principled naive-logloss threshold. Predictive results
   remain exploratory unless the underlying prediction pipeline is re-fit from NWB.
5. No Stage5A-to-Stage5B single-cell mapping is asserted.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import tarfile
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu

DEFAULT_ROOT = Path.cwd()

# User explicitly requested continuation of the established visual language.
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

STEIN_STATES = ["baseline", "visual", "decision", "movement", "outcome"]
ALLEN_PRIMARY = ["familiar_active", "novel_active", "passive"]
SAVE_DPI = 600
SAVE_PDF = True


def style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 10.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.9,
        "lines.markersize": 5.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_ax(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)


def save_panel(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    # No titles / suptitles in publication panels.
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=SAVE_DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def safe_rho(x, y):
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.nanstd(a[m]) < 1e-12 or np.nanstd(b[m]) < 1e-12:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(a[m], b[m])
    return float(r), float(p), int(m.sum())


def rank01(x):
    s = pd.Series(x)
    m = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if m.sum():
        r = s.loc[m].rank(method="average")
        out.loc[m] = (r - 0.5) / m.sum()
    return out.to_numpy(float)


def bootstrap_ci(x, stat="median", n_boot=2000, seed=20260820):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    fn = np.nanmedian if stat == "median" else np.nanmean
    obs = float(fn(a))
    if len(a) == 1:
        return obs, obs, obs
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, float)
    for i in range(n_boot):
        vals[i] = fn(rng.choice(a, len(a), replace=True))
    return obs, float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


class Stage5BArchive:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.tf = tarfile.open(self.path, "r:gz")
        self.names = self.tf.getnames()

    def close(self):
        self.tf.close()

    def member(self, suffix: str):
        hits = [n for n in self.names if n.endswith(suffix)]
        if not hits:
            raise FileNotFoundError(f"Archive member not found: *{suffix}")
        if len(hits) > 1:
            hits.sort(key=len)
        return hits[0]

    def csv(self, suffix: str):
        m = self.tf.extractfile(self.member(suffix))
        if m is None:
            raise FileNotFoundError(suffix)
        return pd.read_csv(io.BytesIO(m.read()))

    def json(self, suffix: str):
        m = self.tf.extractfile(self.member(suffix))
        if m is None:
            raise FileNotFoundError(suffix)
        return json.loads(m.read().decode("utf-8"))

    def text(self, suffix: str):
        m = self.tf.extractfile(self.member(suffix))
        if m is None:
            raise FileNotFoundError(suffix)
        return m.read().decode("utf-8", errors="replace")


def source_audit(zip_path: Path):
    out = {"source_zip": str(zip_path), "status": "NOT_FOUND"}
    if not zip_path.exists():
        return out
    out["status"] = "FOUND"
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        def read_ending(suf):
            h = [n for n in names if n.endswith(suf)]
            return z.read(h[0]).decode("utf-8", errors="replace") if h else ""
        common = read_ending("stage5b/common.py")
        stein = read_ending("scripts/11_steinmetz_leverage.py")
        allen = read_ending("scripts/21_allen_leverage.py")
        cfg = read_ending("config/default.yaml")
        out.update({
            "has_dynamics_leverage": "def dynamics_leverage" in common,
            "Ldyn_multiplies_mean_abs_activity": "lev=mean_abs*direction" in common.replace(" ", ""),
            "predictive_logistic_fixed_C1": "C=1.0" in common,
            "steinmetz_checkpoint_append_pattern": bool(re.search(r"if ck\.exists\(\).*?all_dyn\.append\(pd\.read_csv\(dynfile\)\)", stein, flags=re.S)),
            "allen_primary_rank": "L_primary_rank" in allen,
            "config_excerpt": cfg[:5000],
        })
    return out


def strict_dedup_stein(dyn, pred):
    dyn_key = ["session_id", "neuron_index", "state"]
    pred_key = ["session_id", "neuron_index", "state", "target"]
    audit = {
        "dyn_rows_raw": int(len(dyn)),
        "dyn_duplicate_rows": int(dyn.duplicated(dyn_key).sum()),
        "dyn_rows_strict": int(len(dyn.drop_duplicates(dyn_key, keep="first"))),
        "pred_rows_raw": int(len(pred)),
        "pred_duplicate_rows": int(pred.duplicated(pred_key).sum()),
        "pred_rows_strict": int(len(pred.drop_duplicates(pred_key, keep="first"))),
        "unique_sessions": int(dyn["session_id"].nunique()),
    }
    return dyn.drop_duplicates(dyn_key, keep="first").copy(), pred.drop_duplicates(pred_key, keep="first").copy(), audit


def add_residual_rank(df: pd.DataFrame, group_cols: list[str], y="L_dyn_rank", predictors=("mean_activity", "pca_loading_norm"), out="L_resid_rank"):
    parts = []
    for _, g in df.groupby(group_cols, dropna=False, sort=False):
        h = g.copy()
        yy = pd.to_numeric(h[y], errors="coerce").to_numpy(float)
        cols = []
        for p in predictors:
            if p in h:
                cols.append(rank01(pd.to_numeric(h[p], errors="coerce")))
        if not cols:
            h[out] = rank01(yy)
            parts.append(h); continue
        X = np.column_stack([np.ones(len(h))] + cols)
        m = np.isfinite(yy) & np.isfinite(X).all(1)
        resid = np.full(len(h), np.nan)
        if m.sum() >= max(8, X.shape[1] + 2):
            beta, *_ = np.linalg.lstsq(X[m], yy[m], rcond=None)
            resid[m] = yy[m] - X[m] @ beta
            h[out] = rank01(resid)
        else:
            h[out] = np.nan
        parts.append(h)
    return pd.concat(parts, ignore_index=True) if parts else df.copy()


def pairwise_within_groups(df, group_col, unit_col, state_col, value_col, states: list[str]):
    rows=[]
    for group, g in df.groupby(group_col, dropna=False):
        for a,b in combinations(states,2):
            ga=g[g[state_col].astype(str)==a][[unit_col,value_col]].dropna().drop_duplicates(unit_col)
            gb=g[g[state_col].astype(str)==b][[unit_col,value_col]].dropna().drop_duplicates(unit_col)
            m=ga.merge(gb,on=unit_col,suffixes=("_a","_b"))
            r,p,n=safe_rho(m[f"{value_col}_a"],m[f"{value_col}_b"])
            rows.append({group_col:group,"state_a":a,"state_b":b,"rho":r,"p":p,"n_units":n})
    return pd.DataFrame(rows)


def summarize_pairs(pair_df, n_boot=2000):
    rows=[]
    if pair_df.empty: return pd.DataFrame()
    for (a,b),g in pair_df.groupby(["state_a","state_b"],sort=False):
        x=pd.to_numeric(g["rho"],errors="coerce").dropna().to_numpy(float)
        med,lo,hi=bootstrap_ci(x,"median",n_boot=n_boot,seed=abs(hash((a,b)))%(2**32))
        rows.append({"state_a":a,"state_b":b,"n_groups":len(x),"median_rho":med,"ci95_low":lo,"ci95_high":hi,"mean_rho":float(np.mean(x)) if len(x) else np.nan})
    return pd.DataFrame(rows)


def make_heatmap(summary, states, out, label="Median within-group Spearman ρ", cmap="RdBu_r", vmin=-1, vmax=1):
    M=np.eye(len(states),dtype=float)
    for i,a in enumerate(states):
        for j,b in enumerate(states):
            if i==j: continue
            q=summary[((summary.state_a==a)&(summary.state_b==b))|((summary.state_a==b)&(summary.state_b==a))]
            M[i,j]=q.iloc[0].median_rho if len(q) else np.nan
    fig,ax=plt.subplots(figsize=(5.8,5.0))
    im=ax.imshow(M,cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_xticks(range(len(states))); ax.set_xticklabels([s.replace("_"," ") for s in states],rotation=35,ha="right")
    ax.set_yticks(range(len(states))); ax.set_yticklabels([s.replace("_"," ") for s in states])
    for i in range(len(states)):
        for j in range(len(states)):
            if np.isfinite(M[i,j]):
                ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=8.5,color="white" if abs(M[i,j])>.55 else COL["dark"])
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03); cb.set_label(label)
    save_panel(fig,out)


def compute_cell_flex(df, group_col, unit_col, state_col, state_filter, rank_col="L_dyn_rank"):
    d=df[df[state_col].astype(str).isin(state_filter)].copy()
    d["activity_rank"]=d.groupby([group_col,state_col])["mean_activity"].transform(rank01)
    rows=[]
    for key,g in d.groupby([group_col,unit_col],dropna=False):
        vals=pd.to_numeric(g[rank_col],errors="coerce").dropna().to_numpy(float)
        act=pd.to_numeric(g["activity_rank"],errors="coerce").dropna().to_numpy(float)
        raw_act=pd.to_numeric(g["mean_activity"],errors="coerce").dropna().to_numpy(float)
        if len(vals)<2: continue
        rows.append({group_col:key[0],unit_col:key[1],"n_states":len(vals),
                     "mean_leverage":float(np.mean(vals)),"role_flexibility":float(np.var(vals)),"leverage_range":float(np.ptp(vals)),
                     "mean_activity":float(np.mean(raw_act)) if len(raw_act) else np.nan,
                     "activity_flexibility":float(np.var(act)) if len(act)>=2 else np.nan})
    return pd.DataFrame(rows)


def backbone_fraction(df, group_col, unit_col, state_col, states, value_col):
    rows=[]
    for group,g in df[df[state_col].astype(str).isin(states)].groupby(group_col):
        p=g.pivot_table(index=unit_col,columns=state_col,values=value_col,aggfunc="first")
        cols=[s for s in states if s in p.columns]
        if len(cols)<2: continue
        p=p[cols].dropna(axis=0,how="any")
        if len(p)<5: continue
        A=p.to_numpy(float)
        cell_mean=A.mean(1)
        between=float(np.var(cell_mean))
        within=float(np.mean(np.var(A,axis=1)))
        frac=between/(between+within) if between+within>0 else np.nan
        rows.append({group_col:group,"n_cells":len(p),"n_states":len(cols),"between_var":between,"within_var":within,"backbone_fraction":frac})
    return pd.DataFrame(rows)


def identity_shuffle(df, group_col, unit_col, state_col, states, value_col, n_perm=300, seed=20260820):
    rng=np.random.default_rng(seed); rows=[]
    for group,g in df[df[state_col].astype(str).isin(states)].groupby(group_col):
        p=g.pivot_table(index=unit_col,columns=state_col,values=value_col,aggfunc="first")
        cols=[s for s in states if s in p.columns]
        if len(cols)<2: continue
        p=p[cols].dropna(axis=0,how="any")
        if len(p)<8: continue
        A=p.to_numpy(float)
        obs=float(np.mean(np.var(A,axis=1)))
        null=np.empty(n_perm,float)
        for k in range(n_perm):
            B=np.empty_like(A)
            for j in range(A.shape[1]): B[:,j]=A[rng.permutation(A.shape[0]),j]
            null[k]=np.mean(np.var(B,axis=1))
        rows.append({group_col:group,"n_cells":len(p),"observed_mean_flex":obs,"null_mean_flex":float(np.mean(null)),
                     "observed_to_null":obs/np.mean(null) if np.mean(null)>0 else np.nan,
                     "p_less_than_shuffle":float((1+np.sum(null<=obs))/(n_perm+1))})
    return pd.DataFrame(rows)


def confound_correlations(df, group_cols):
    rows=[]
    for key,g in df.groupby(group_cols,dropna=False):
        if not isinstance(key,tuple): key=(key,)
        base={c:v for c,v in zip(group_cols,key)}
        for x in ["mean_activity","pca_loading_norm"]:
            r,p,n=safe_rho(g["L_dyn_rank"],g[x])
            rows.append({**base,"confound":x,"rho":r,"p":p,"n":n})
    return pd.DataFrame(rows)


def allen_primary_table(a):
    p=a[a["state"].astype(str)==a["experiment_state"].astype(str)].copy()
    if "L_primary_rank" not in p or p["L_primary_rank"].isna().all():
        p["L_primary_rank"]=p.groupby("experiment_id")["L_dyn"].transform(rank01)
    # Use one record per cell within experiment.
    return p.drop_duplicates(["experiment_id","cell_specimen_id"]).copy()


def allen_pairwise(primary, value_col="L_primary_rank", r2_threshold=None):
    d=primary.copy()
    if r2_threshold is not None:
        d=d[pd.to_numeric(d["latent_r2"],errors="coerce")>float(r2_threshold)]
    return pairwise_within_groups(d,"container_id","cell_specimen_id","experiment_state",value_col,ALLEN_PRIMARY)


def allen_arousal_pairs(a, value_col):
    rows=[]
    for pair_name,(sa,sb) in {"running":("running_low","running_high"),"pupil":("pupil_low","pupil_high")}.items():
        q=pairwise_within_groups(a,"experiment_id","cell_specimen_id","state",value_col,[sa,sb])
        if len(q):
            q["contrast"]=pair_name; rows.append(q)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def predictive_pairwise(pred):
    d=pred.copy()
    d["L_pred_rank"]=d.groupby(["session_id","state","target"])["L_pred"].transform(rank01)
    rows=[]
    for target,g in d.groupby("target"):
        q=pairwise_within_groups(g,"session_id","neuron_index","state","L_pred_rank",STEIN_STATES)
        q["target"]=target; rows.append(q)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(),d


def pred_vs_dyn(pred_rank,dyn):
    m=pred_rank.merge(dyn[["session_id","neuron_index","state","L_dyn_rank"]],on=["session_id","neuron_index","state"],how="inner")
    rows=[]
    for key,g in m.groupby(["session_id","state","target"]):
        r,p,n=safe_rho(g["L_pred_rank"],g["L_dyn_rank"])
        rows.append({"session_id":key[0],"state":key[1],"target":key[2],"rho":r,"p":p,"n":n,"full_logloss":float(pd.to_numeric(g["full_logloss"],errors="coerce").median())})
    return pd.DataFrame(rows)


def plot_box_by_category(df, cat, val, out, ylabel, order=None, hline=None, palette=None):
    if order is None: order=[x for x in df[cat].dropna().astype(str).unique()]
    arrays=[pd.to_numeric(df[df[cat].astype(str)==str(c)][val],errors="coerce").dropna().to_numpy(float) for c in order]
    fig,ax=plt.subplots(figsize=(max(5.4,1.05*len(order)+2.2),4.2))
    bp=ax.boxplot(arrays,patch_artist=True,showfliers=False,widths=.55,medianprops={"color":COL["dark"],"linewidth":1.3})
    cols=palette or [COL["blue"],COL["teal"],COL["orange"],COL["purple"],COL["rose"]]
    for i,b in enumerate(bp["boxes"]): b.set_facecolor(cols[i%len(cols)]); b.set_alpha(.80); b.set_edgecolor("none")
    rng=np.random.default_rng(20260820)
    for i,a in enumerate(arrays,1):
        if len(a): ax.scatter(i+rng.normal(0,.045,len(a)),a,s=12,alpha=.35,color=COL["dark"],edgecolor="none")
    if hline is not None: ax.axhline(hline,color=COL["gray"],ls="--",lw=1)
    ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels([str(x).replace("_"," ") for x in order],rotation=28,ha="right")
    ax.set_ylabel(ylabel); clean_ax(ax,"y"); save_panel(fig,out)


def plot_scatter(x,y,out,xlabel,ylabel,annotation=None):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    fig,ax=plt.subplots(figsize=(5.0,4.5))
    ax.scatter(x[m],y[m],s=13,alpha=.28,color=COL["blue"],edgecolor="none",rasterized=True)
    if annotation:
        ax.text(.03,.97,annotation,transform=ax.transAxes,ha="left",va="top",fontsize=9.5)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); clean_ax(ax,"both"); save_panel(fig,out)


def plot_paired_raw_resid(summary_raw,summary_res,out):
    m=summary_raw.merge(summary_res,on=["state_a","state_b"],suffixes=("_raw","_resid"))
    if m.empty:return
    fig,ax=plt.subplots(figsize=(5.3,4.3))
    for i,r in m.reset_index(drop=True).iterrows():
        ax.plot([0,1],[r.median_rho_raw,r.median_rho_resid],color=COL["light"],lw=1.3,zorder=1)
        ax.scatter([0],[r.median_rho_raw],color=COL["blue"],s=35,zorder=2)
        ax.scatter([1],[r.median_rho_resid],color=COL["teal"],s=35,zorder=2)
    ax.set_xticks([0,1]);ax.set_xticklabels(["Raw leverage","Activity/loading-residual"])
    ax.set_ylabel("Median state-pair Spearman ρ"); clean_ax(ax,"y"); save_panel(fig,out)


def plot_shuffle(shuf,out):
    if shuf.empty:return
    fig,ax=plt.subplots(figsize=(4.8,4.5))
    x=shuf["null_mean_flex"].to_numpy(float); y=shuf["observed_mean_flex"].to_numpy(float)
    ax.scatter(x,y,s=30,alpha=.7,color=COL["purple"],edgecolor="white",linewidth=.4)
    lim=[min(np.nanmin(x),np.nanmin(y)),max(np.nanmax(x),np.nanmax(y))]
    ax.plot(lim,lim,ls="--",color=COL["gray"],lw=1)
    ax.set_xlabel("Neuron-identity shuffle mean flexibility");ax.set_ylabel("Observed mean flexibility");clean_ax(ax,"both");save_panel(fig,out)


def plot_bar_values(labels,values,out,ylabel,colors=None,hline=None):
    fig,ax=plt.subplots(figsize=(max(5.2,.9*len(labels)+2),4.1))
    cols=colors or [COL["blue"]]*len(labels)
    ax.bar(range(len(labels)),values,color=cols,alpha=.9,width=.65)
    if hline is not None: ax.axhline(hline,color=COL["gray"],ls="--",lw=1)
    ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels,rotation=28,ha="right")
    ax.set_ylabel(ylabel);clean_ax(ax,"y");save_panel(fig,out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    ap.add_argument("--results",type=Path,default=None)
    ap.add_argument("--source",type=Path,default=None)
    ap.add_argument("--output",type=Path,default=None)
    ap.add_argument("--bootstrap",type=int,default=2000)
    ap.add_argument("--shuffle",type=int,default=300)
    ap.add_argument("--dpi",type=int,default=600)
    ap.add_argument("--no-pdf",action="store_true",help="Skip vector PDF export (useful for a quick smoke test)")
    args=ap.parse_args()
    global SAVE_DPI, SAVE_PDF
    SAVE_DPI=int(args.dpi); SAVE_PDF=not args.no_pdf

    style()
    root=args.root.resolve()
    results=(args.results.resolve() if args.results else root/"bio data"/"Stage5B_complete_20260820.tar.gz")
    source=(args.source.resolve() if args.source else root/"bio data"/"stage5b_dynamic_leverage_v1.zip")
    outroot=(args.output.resolve() if args.output else root/"plot"/"Stage5BR_leverage_closure_v1")
    AOUT=outroot/"analysis"; FOUT=outroot/"figures"; AOUT.mkdir(parents=True,exist_ok=True);FOUT.mkdir(parents=True,exist_ok=True)
    if not results.exists(): raise SystemExit(f"Missing results archive: {results}")

    print("[1/9] Loading Stage5B archive ...")
    ar=Stage5BArchive(results)
    try:
        s_dyn=ar.csv("steinmetz/leverage_long.csv")
        s_pred=ar.csv("steinmetz/predictive_leverage_long.csv")
        s_inv=ar.csv("steinmetz/inventory.csv")
        a_dyn=ar.csv("allen_vbo/leverage_long.csv")
        a_inv=ar.csv("allen_vbo/inventory.csv")
        bridge_status=ar.json("bridge/bridge_status.json")
        recovery_status=ar.json("stage5a_recovery/recovery_status.json")
    finally:
        ar.close()

    print("[2/9] Strict Steinmetz de-duplication and source audit ...")
    s_dyn,s_pred,dedup_audit=strict_dedup_stein(s_dyn,s_pred)
    dedup_audit.update({"inventory_rows":int(len(s_inv)),"inventory_unique_file_stems":int(s_inv["file"].astype(str).str.replace(r"\\.nwb$","",regex=True).nunique()) if "file" in s_inv else None,"inventory_unique_identifiers":int(s_inv["identifier"].nunique()) if "identifier" in s_inv else None})
    (AOUT/"steinmetz_duplicate_audit.json").write_text(json.dumps(dedup_audit,indent=2),encoding="utf-8")
    sa=source_audit(source); (AOUT/"source_code_audit.json").write_text(json.dumps(sa,indent=2),encoding="utf-8")

    print("[3/9] Steinmetz stable-backbone / residual analyses ...")
    s_dyn=add_residual_rank(s_dyn,["session_id","state"],out="L_resid_rank")
    s_raw_pairs=pairwise_within_groups(s_dyn,"session_id","neuron_index","state","L_dyn_rank",STEIN_STATES)
    s_res_pairs=pairwise_within_groups(s_dyn,"session_id","neuron_index","state","L_resid_rank",STEIN_STATES)
    s_raw_sum=summarize_pairs(s_raw_pairs,args.bootstrap);s_res_sum=summarize_pairs(s_res_pairs,args.bootstrap)
    s_raw_pairs.to_csv(AOUT/"steinmetz_state_pairs_raw_by_session.csv",index=False);s_res_pairs.to_csv(AOUT/"steinmetz_state_pairs_residual_by_session.csv",index=False)
    s_raw_sum.to_csv(AOUT/"steinmetz_state_pairs_raw_summary.csv",index=False);s_res_sum.to_csv(AOUT/"steinmetz_state_pairs_residual_summary.csv",index=False)

    s_conf=confound_correlations(s_dyn,["session_id","state"]);s_conf.to_csv(AOUT/"steinmetz_confound_correlations.csv",index=False)
    s_flex=compute_cell_flex(s_dyn,"session_id","neuron_index","state",STEIN_STATES);s_flex.to_csv(AOUT/"steinmetz_role_flexibility_corrected.csv",index=False)
    s_back=backbone_fraction(s_dyn,"session_id","neuron_index","state",STEIN_STATES,"L_dyn_rank");s_back.to_csv(AOUT/"steinmetz_backbone_fraction.csv",index=False)
    s_shuf=identity_shuffle(s_dyn,"session_id","neuron_index","state",STEIN_STATES,"L_dyn_rank",args.shuffle);s_shuf.to_csv(AOUT/"steinmetz_identity_shuffle.csv",index=False)
    s_r2=s_dyn.groupby(["session_id","state"],as_index=False)["latent_r2"].median();s_r2.to_csv(AOUT/"steinmetz_latent_r2.csv",index=False)

    print("[4/9] Steinmetz predictive-leverage audit ...")
    pred_pairs,pred_rank=predictive_pairwise(s_pred); pred_pairs.to_csv(AOUT/"steinmetz_predictive_state_pairs_by_session.csv",index=False)
    pred_summary=pred_pairs.groupby(["target","state_a","state_b"])["rho"].agg(n_groups="count",median_rho="median",mean_rho="mean").reset_index();pred_summary.to_csv(AOUT/"steinmetz_predictive_state_pairs_summary.csv",index=False)
    pvd=pred_vs_dyn(pred_rank,s_dyn);pvd.to_csv(AOUT/"steinmetz_predictive_vs_dynamic.csv",index=False)
    pred_quality=s_pred.drop_duplicates(["session_id","state","target"])[["session_id","state","target","full_logloss"]].copy();pred_quality.to_csv(AOUT/"steinmetz_predictive_model_logloss.csv",index=False)

    print("[5/9] Allen VBO same-cell primary-state analyses ...")
    a_dyn=add_residual_rank(a_dyn,["experiment_id","state"],out="L_resid_rank")
    ap=allen_primary_table(a_dyn)
    # Primary residual ranks are inherited from the corresponding experiment-state rows.
    a_raw_pairs=allen_pairwise(ap,"L_primary_rank",None);a_res_pairs=allen_pairwise(ap,"L_resid_rank",None)
    a_raw_sum=summarize_pairs(a_raw_pairs,args.bootstrap);a_res_sum=summarize_pairs(a_res_pairs,args.bootstrap)
    a_raw_pairs.to_csv(AOUT/"allen_primary_state_pairs_raw_by_container.csv",index=False);a_res_pairs.to_csv(AOUT/"allen_primary_state_pairs_residual_by_container.csv",index=False)
    a_raw_sum.to_csv(AOUT/"allen_primary_state_pairs_raw_summary.csv",index=False);a_res_sum.to_csv(AOUT/"allen_primary_state_pairs_residual_summary.csv",index=False)
    # R2-gated sensitivity, pairwise rather than requiring all three states simultaneously.
    gated=[]
    for thr in [0.0,0.05]:
        q=allen_pairwise(ap,"L_primary_rank",thr);q["r2_threshold"]=thr;gated.append(q)
    a_gate=pd.concat(gated,ignore_index=True);a_gate.to_csv(AOUT/"allen_primary_state_pairs_r2_gated.csv",index=False)
    gate_sum=a_gate.groupby(["r2_threshold","state_a","state_b"])["rho"].agg(n_groups="count",median_rho="median",mean_rho="mean").reset_index();gate_sum.to_csv(AOUT/"allen_primary_state_pairs_r2_gated_summary.csv",index=False)

    a_flex=compute_cell_flex(ap,"container_id","cell_specimen_id","experiment_state",ALLEN_PRIMARY,rank_col="L_primary_rank");a_flex.to_csv(AOUT/"allen_role_flexibility_corrected.csv",index=False)
    a_back=backbone_fraction(ap,"container_id","cell_specimen_id","experiment_state",ALLEN_PRIMARY,"L_primary_rank");a_back.to_csv(AOUT/"allen_backbone_fraction.csv",index=False)
    a_shuf=identity_shuffle(ap,"container_id","cell_specimen_id","experiment_state",ALLEN_PRIMARY,"L_primary_rank",args.shuffle,seed=20260821);a_shuf.to_csv(AOUT/"allen_identity_shuffle.csv",index=False)
    a_r2=ap.drop_duplicates(["experiment_id","experiment_state"])[["experiment_id","container_id","experiment_state","latent_r2","cre_line","targeted_structure","imaging_depth"]];a_r2.to_csv(AOUT/"allen_primary_latent_r2.csv",index=False)

    print("[6/9] Allen within-session running/pupil analyses ...")
    ar_raw=allen_arousal_pairs(a_dyn,"L_dyn_rank");ar_res=allen_arousal_pairs(a_dyn,"L_resid_rank")
    ar_raw.to_csv(AOUT/"allen_within_session_arousal_pairs_raw.csv",index=False);ar_res.to_csv(AOUT/"allen_within_session_arousal_pairs_residual.csv",index=False)
    ar_sum=ar_raw.groupby("contrast")["rho"].agg(n_experiments="count",median_rho="median",mean_rho="mean").reset_index(); ar_sum.to_csv(AOUT/"allen_within_session_arousal_raw_summary.csv",index=False)
    arr_sum=ar_res.groupby("contrast")["rho"].agg(n_experiments="count",median_rho="median",mean_rho="mean").reset_index(); arr_sum.to_csv(AOUT/"allen_within_session_arousal_residual_summary.csv",index=False)

    print("[7/9] Cross-dataset summary and statistics ...")
    # Scalar summaries used for the report and cross-dataset panels.
    metrics={}
    metrics["steinmetz_raw_pair_median"]=float(pd.to_numeric(s_raw_pairs.rho,errors="coerce").median())
    metrics["steinmetz_residual_pair_median"]=float(pd.to_numeric(s_res_pairs.rho,errors="coerce").median())
    metrics["steinmetz_latent_r2_median"]=float(pd.to_numeric(s_r2.latent_r2,errors="coerce").median())
    metrics["steinmetz_latent_r2_positive_fraction"]=float((pd.to_numeric(s_r2.latent_r2,errors="coerce")>0).mean())
    metrics["steinmetz_Ldyn_activity_rho_median"]=float(s_conf[s_conf.confound=="mean_activity"].rho.median())
    metrics["steinmetz_Ldyn_loading_rho_median"]=float(s_conf[s_conf.confound=="pca_loading_norm"].rho.median())
    metrics["steinmetz_flex_mean_activity_rho"]=safe_rho(s_flex.role_flexibility,s_flex.mean_activity)[0]
    metrics["steinmetz_flex_activity_flex_rho"]=safe_rho(s_flex.role_flexibility,s_flex.activity_flexibility)[0]
    metrics["steinmetz_backbone_fraction_median"]=float(s_back.backbone_fraction.median()) if len(s_back) else np.nan
    metrics["steinmetz_observed_shuffle_ratio_median"]=float(s_shuf.observed_to_null.median()) if len(s_shuf) else np.nan
    metrics["allen_primary_pair_median"]=float(pd.to_numeric(a_raw_pairs.rho,errors="coerce").median())
    metrics["allen_primary_residual_pair_median"]=float(pd.to_numeric(a_res_pairs.rho,errors="coerce").median())
    metrics["allen_primary_latent_r2_median"]=float(pd.to_numeric(a_r2.latent_r2,errors="coerce").median())
    metrics["allen_primary_latent_r2_positive_fraction"]=float((pd.to_numeric(a_r2.latent_r2,errors="coerce")>0).mean())
    metrics["allen_flex_mean_activity_rho"]=safe_rho(a_flex.role_flexibility,a_flex.mean_activity)[0]
    metrics["allen_flex_activity_flex_rho"]=safe_rho(a_flex.role_flexibility,a_flex.activity_flexibility)[0]
    metrics["allen_backbone_fraction_median"]=float(a_back.backbone_fraction.median()) if len(a_back) else np.nan
    metrics["allen_observed_shuffle_ratio_median"]=float(a_shuf.observed_to_null.median()) if len(a_shuf) else np.nan
    for c in ["running","pupil"]:
        q=ar_raw[ar_raw.contrast==c]; qr=ar_res[ar_res.contrast==c]
        metrics[f"allen_{c}_raw_pair_median"]=float(q.rho.median()) if len(q) else np.nan
        metrics[f"allen_{c}_residual_pair_median"]=float(qr.rho.median()) if len(qr) else np.nan
    (AOUT/"stage5br_key_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")

    # Bridge/recovery status preserved exactly from baseline archive.
    (AOUT/"baseline_bridge_status.json").write_text(json.dumps(bridge_status,indent=2),encoding="utf-8")
    (AOUT/"baseline_stage5a_recovery_status.json").write_text(json.dumps(recovery_status,indent=2),encoding="utf-8")

    print("[8/9] Drawing standalone figure atlas ...")
    catalog=[]
    def reg(name,category,claim): catalog.append({"figure":name,"category":category,"intended_message":claim})

    # 01 strict duplicate audit
    labels=["Raw NWB inventory","Unique session stems","Duplicate file aliases"]
    vals=[len(s_inv),s_inv["file"].nunique(),len(s_inv)-s_inv["file"].nunique()]
    plot_bar_values(labels,vals,FOUT/"S5BR_01_steinmetz_duplicate_audit","Count",[COL["gray"],COL["blue"],COL["rose"]]);reg("S5BR_01","QC","Steinmetz archive contains repeated file aliases; all inference is strictly session-deduplicated.")

    plot_box_by_category(s_r2,"state","latent_r2",FOUT/"S5BR_02_steinmetz_latent_r2","Held-out latent-dynamics R²",STEIN_STATES,hline=0);reg("S5BR_02","MAIN_CANDIDATE","Steinmetz latent dynamics are predictively valid across task epochs.")
    make_heatmap(s_raw_sum,STEIN_STATES,FOUT/"S5BR_03_steinmetz_state_stability_raw");reg("S5BR_03","MAIN_CANDIDATE","Raw dynamical leverage has a strong cross-state backbone.")
    make_heatmap(s_res_sum,STEIN_STATES,FOUT/"S5BR_04_steinmetz_state_stability_residual");reg("S5BR_04","MAIN_CANDIDATE","After activity/loading removal, a weaker state-specific residual structure remains.")
    plot_paired_raw_resid(s_raw_sum,s_res_sum,FOUT/"S5BR_05_steinmetz_raw_vs_residual_state_stability");reg("S5BR_05","MAIN_CANDIDATE","Residualization separates stable prominence from dynamical-state structure.")
    plot_box_by_category(s_conf,"confound","rho",FOUT/"S5BR_06_steinmetz_activity_loading_confound","Spearman ρ with Ldyn rank",["mean_activity","pca_loading_norm"],hline=0);reg("S5BR_06","SI_CANDIDATE","Current Ldyn contains substantial activity/loading prominence by construction.")
    plot_scatter(s_flex.mean_activity,s_flex.role_flexibility,FOUT/"S5BR_07_steinmetz_flex_vs_mean_activity","Mean activity","Role flexibility",f"ρ = {metrics['steinmetz_flex_mean_activity_rho']:.2f}");reg("S5BR_07","SI_CANDIDATE","Steinmetz role flexibility is not simply mean activity.")
    plot_scatter(s_flex.activity_flexibility,s_flex.role_flexibility,FOUT/"S5BR_08_steinmetz_flex_vs_activity_flexibility","Activity-rank flexibility","Leverage-rank flexibility",f"ρ = {metrics['steinmetz_flex_activity_flex_rho']:.2f}");reg("S5BR_08","MAIN_CANDIDATE","Much of fast-timescale leverage flexibility covaries with activity flexibility.")
    plot_box_by_category(s_back.assign(dataset="Steinmetz"),"dataset","backbone_fraction",FOUT/"S5BR_09_steinmetz_backbone_fraction","Rank-variance backbone fraction",["Steinmetz"],hline=.5);reg("S5BR_09","MAIN_CANDIDATE","Quantifies stable-backbone dominance within fast task states.")
    plot_shuffle(s_shuf,FOUT/"S5BR_10_steinmetz_identity_shuffle");reg("S5BR_10","MAIN_CANDIDATE","Observed role flexibility is far below a neuron-identity shuffle null.")

    for target,idx in [("choice",11),("outcome",12),("engagement",13)]:
        q=pred_summary[pred_summary.target==target].copy()
        make_heatmap(q,STEIN_STATES,FOUT/f"S5BR_{idx:02d}_steinmetz_predictive_{target}",label="Median predictive-leverage rank ρ")
        reg(f"S5BR_{idx:02d}","EXPLORATORY","Predictive contribution has target/state-specific organization; baseline predictor quality remains an explicit caveat.")
    plot_box_by_category(pvd,"target","rho",FOUT/"S5BR_14_predictive_vs_dynamic","Spearman ρ: predictive vs dynamical leverage",sorted(pvd.target.unique()),hline=0);reg("S5BR_14","EXPLORATORY","Task-predictive importance and dynamical influence are distinct quantities.")

    plot_box_by_category(a_r2,"experiment_state","latent_r2",FOUT/"S5BR_15_allen_primary_latent_r2","Held-out latent-dynamics R²",ALLEN_PRIMARY,hline=0);reg("S5BR_15","MAIN_CANDIDATE","Allen VBO baseline one-step latent dynamics are weak and heterogeneous.")
    frac=a_r2.groupby("experiment_state")["latent_r2"].apply(lambda x:float((pd.to_numeric(x,errors='coerce')>0).mean())).reindex(ALLEN_PRIMARY)
    plot_bar_values([x.replace("_"," ") for x in ALLEN_PRIMARY],frac.values,FOUT/"S5BR_16_allen_positive_r2_fraction","Fraction of experiments with R² > 0",[COL["blue"],COL["teal"],COL["orange"]],hline=.5);reg("S5BR_16","QC","Only a subset of Allen experiments support the current one-step dynamics model.")
    make_heatmap(a_raw_sum,ALLEN_PRIMARY,FOUT/"S5BR_17_allen_primary_state_reconfiguration_raw");reg("S5BR_17","MAIN_CANDIDATE","Same-cell leverage is less stable across experience/context than fast Steinmetz task epochs.")
    make_heatmap(a_res_sum,ALLEN_PRIMARY,FOUT/"S5BR_18_allen_primary_state_reconfiguration_residual");reg("S5BR_18","MAIN_CANDIDATE","State reconfiguration remains after activity/loading residualization.")
    plot_paired_raw_resid(a_raw_sum,a_res_sum,FOUT/"S5BR_19_allen_raw_vs_residual_state_stability");reg("S5BR_19","MAIN_CANDIDATE","Allen primary-state stability decreases after static prominence removal.")

    # gated sensitivity medians
    if len(gate_sum):
        labels=[];vals=[];cols=[]
        for thr in [0.0,0.05]:
            q=gate_sum[gate_sum.r2_threshold==thr]
            labels.append(f"R² > {thr:g}");vals.append(float(q.median_rho.median()) if len(q) else np.nan);cols.append(COL["teal"] if thr==0 else COL["purple"])
        plot_bar_values(labels,vals,FOUT/"S5BR_20_allen_r2_gated_state_stability","Median primary-state Spearman ρ",cols,hline=0);reg("S5BR_20","MAIN_CANDIDATE","Tests whether same-cell re-ranking survives dynamics-quality gating.")

    plot_scatter(a_flex.mean_activity,a_flex.role_flexibility,FOUT/"S5BR_21_allen_flex_vs_mean_activity","Mean activity","Role flexibility",f"ρ = {metrics['allen_flex_mean_activity_rho']:.2f}");reg("S5BR_21","MAIN_CANDIDATE","Allen role flexibility is not explained by mean activity.")
    plot_scatter(a_flex.activity_flexibility,a_flex.role_flexibility,FOUT/"S5BR_22_allen_flex_vs_activity_flexibility","Activity-rank flexibility","Leverage-rank flexibility",f"ρ = {metrics['allen_flex_activity_flex_rho']:.2f}");reg("S5BR_22","MAIN_CANDIDATE","Activity-state variation explains part, but not necessarily all, of leverage flexibility.")

    # Within-session arousal raw vs residual distributions
    if len(ar_raw):
        z=pd.concat([ar_raw.assign(metric="Raw"),ar_res.assign(metric="Residual")],ignore_index=True)
        for contrast,idx in [("running",23),("pupil",24)]:
            q=z[z.contrast==contrast]
            plot_box_by_category(q,"metric","rho",FOUT/f"S5BR_{idx:02d}_allen_{contrast}_within_session","Within-experiment Spearman ρ",["Raw","Residual"],hline=0,palette=[COL["blue"],COL["teal"]])
            reg(f"S5BR_{idx:02d}","MAIN_CANDIDATE",f"Within-session {contrast} state shifts provide a session-drift-resistant leverage test.")

    # Cross-dataset panels
    labels=["Steinmetz\nfast task epochs","Allen\nexperience/context","Allen\nrunning low/high","Allen\npupil low/high"]
    vals=[metrics["steinmetz_raw_pair_median"],metrics["allen_primary_pair_median"],metrics.get("allen_running_raw_pair_median",np.nan),metrics.get("allen_pupil_raw_pair_median",np.nan)]
    plot_bar_values(labels,vals,FOUT/"S5BR_25_cross_dataset_state_stability","Median state-pair Spearman ρ",[COL["blue"],COL["orange"],COL["teal"],COL["purple"]],hline=0);reg("S5BR_25","MAIN_CANDIDATE","Leverage stability depends on the scale/type of state change.")
    plot_box_by_category(pd.concat([s_back[["backbone_fraction"]].assign(dataset="Steinmetz"),a_back[["backbone_fraction"]].assign(dataset="Allen VBO")]),"dataset","backbone_fraction",FOUT/"S5BR_26_cross_dataset_backbone_fraction","Rank-variance backbone fraction",["Steinmetz","Allen VBO"],hline=.5,palette=[COL["blue"],COL["orange"]]);reg("S5BR_26","MAIN_CANDIDATE","Directly compares stable-backbone dominance across datasets.")

    pd.DataFrame(catalog).to_csv(AOUT/"figure_catalog.csv",index=False,encoding="utf-8-sig")

    print("[9/9] Writing manuscript-oriented closure report ...")
    report=f"""# Stage 5B-R leverage decomposition and validation\n\n## Integrity correction\n\n- Steinmetz NWB inventory rows: **{len(s_inv)}**.\n- Unique file stems / sessions in the inventory: **{s_inv['file'].nunique()}**.\n- Raw `leverage_long.csv` rows: **{dedup_audit['dyn_rows_raw']}**; exact repeated session–neuron–state rows removed: **{dedup_audit['dyn_duplicate_rows']}**.\n- Raw predictive rows: **{dedup_audit['pred_rows_raw']}**; repeated session–neuron–state–target rows removed: **{dedup_audit['pred_duplicate_rows']}**.\n\nThe repeated rows arise from duplicate NWB paths sharing the same file stem. The baseline script correctly reuses an existing checkpoint, but appends that checkpoint table again for each repeated path. All Stage5B-R inference is therefore performed after strict de-duplication.\n\n## Steinmetz: fast task states\n\n- Median held-out latent-dynamics R²: **{metrics['steinmetz_latent_r2_median']:.3f}**; fraction of session-state models with R² > 0: **{metrics['steinmetz_latent_r2_positive_fraction']:.1%}**.\n- Median raw state-pair leverage-rank correlation: **{metrics['steinmetz_raw_pair_median']:.3f}**.\n- After within-state activity/loading residualization: **{metrics['steinmetz_residual_pair_median']:.3f}**.\n- Median Ldyn/activity correlation: **{metrics['steinmetz_Ldyn_activity_rho_median']:.3f}**; median Ldyn/PCA-loading correlation: **{metrics['steinmetz_Ldyn_loading_rho_median']:.3f}**.\n- Role-flexibility vs mean activity: ρ = **{metrics['steinmetz_flex_mean_activity_rho']:.3f}**.\n- Role-flexibility vs activity-flexibility: ρ = **{metrics['steinmetz_flex_activity_flex_rho']:.3f}**.\n- Median rank-variance backbone fraction: **{metrics['steinmetz_backbone_fraction_median']:.3f}**.\n- Median observed/shuffled flexibility ratio: **{metrics['steinmetz_observed_shuffle_ratio_median']:.3f}**.\n\nInterpretation: rapid task-epoch changes occur around a strong stable leverage scaffold. The raw leverage definition contains activity/loading prominence by construction; residualization isolates the more state-specific component.\n\n## Allen Visual Behavior 2P: experience/context states\n\n- Median primary-state latent-dynamics R²: **{metrics['allen_primary_latent_r2_median']:.3f}**; fraction R² > 0: **{metrics['allen_primary_latent_r2_positive_fraction']:.1%}**.\n- Median same-cell primary-state leverage correlation: **{metrics['allen_primary_pair_median']:.3f}**.\n- After activity/loading residualization: **{metrics['allen_primary_residual_pair_median']:.3f}**.\n- Role-flexibility vs mean activity: ρ = **{metrics['allen_flex_mean_activity_rho']:.3f}**.\n- Role-flexibility vs activity-flexibility: ρ = **{metrics['allen_flex_activity_flex_rho']:.3f}**.\n- Median backbone fraction: **{metrics['allen_backbone_fraction_median']:.3f}**.\n- Running low/high median raw correlation: **{metrics.get('allen_running_raw_pair_median',np.nan):.3f}**; residual: **{metrics.get('allen_running_residual_pair_median',np.nan):.3f}**.\n- Pupil low/high median raw correlation: **{metrics.get('allen_pupil_raw_pair_median',np.nan):.3f}**; residual: **{metrics.get('allen_pupil_residual_pair_median',np.nan):.3f}**.\n\nInterpretation: same-cell functional rankings are substantially more labile across experience/context than Steinmetz rapid task epochs, and within-session running/pupil contrasts test this without cross-day identity drift. However, the current one-step Allen latent model is weak in many experiments, so any manuscript use of the word *dynamical* should be conditioned on R²-gated robustness or a later NWB-level lag/model refit.\n\n## Predictive leverage\n\nThe baseline source uses a fixed `LogisticRegression(C=1.0)` and archives full-model log-loss but not sufficient class-distribution information to reconstruct a principled naive-logloss gate. Stage5B-R therefore treats predictive leverage as exploratory. It is useful for testing whether task-predictive importance and Ldyn are separable, but is not yet a confirmatory biological endpoint.\n\n## Current model suggested by the data\n\nA manuscript-safe decomposition is:\n\n`L_i(s) = L_i^core + δL_i(s)`\n\nwith a strong stable component over fast task epochs and a larger residual reconfiguration over experience/context changes. This is more precise than claiming that leverage is either completely static or completely reassigned across states.\n\n## Claims NOT established by this archive\n\n- No single-cell Stage5A Cbio ↔ Stage5B leverage mapping.\n- No causal leverage.\n- No activity-independent leverage claim without residualized analyses.\n- No fully validated Allen dynamical-leverage claim where latent R² is weak.\n- No confirmatory predictive-leverage claim without refitting the predictive model with nested regularization / baseline gates.\n\n## Baseline status carried forward\n\n- Bridge status: `{bridge_status.get('status', bridge_status)}`\n- Stage5A GLIF recovery status: `{recovery_status.get('status', recovery_status)}`\n\n## Recommended next compute, not performed by this archive-level script\n\n1. Allen NWB-level lag/model selection (0.25–2 s effective horizon) using held-out dynamics prediction only.\n2. Steinmetz predictive model refit with nested regularization and naive/class-entropy quality gates.\n3. If desired, parse the recovered nested GLIF JSON into specimen × GLIF1–5 held-out EVR for Stage5A epsilon sensitivity.\n"""
    (AOUT/"stage5br_closure_report.md").write_text(report,encoding="utf-8")
    manifest={"results_archive":str(results),"source_archive":str(source),"output_root":str(outroot),"strict_steinmetz_dedup":True,"bootstrap":args.bootstrap,"shuffle":args.shuffle,"figures_png":len(list(FOUT.glob('*.png'))),"figures_pdf":len(list(FOUT.glob('*.pdf')))}
    (AOUT/"stage5br_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    print("\n=== Stage 5B-R closure complete ===")
    print(json.dumps({"status":"COMPLETE","output_root":str(outroot),"figures_png":manifest['figures_png'],"steinmetz_raw_pair_median":metrics['steinmetz_raw_pair_median'],"steinmetz_residual_pair_median":metrics['steinmetz_residual_pair_median'],"allen_primary_pair_median":metrics['allen_primary_pair_median'],"allen_primary_residual_pair_median":metrics['allen_primary_residual_pair_median']},indent=2))
    print("\nRead first:")
    print(AOUT/"stage5br_closure_report.md")
    print(AOUT/"stage5br_key_metrics.json")
    print(AOUT/"figure_catalog.csv")

if __name__=="__main__":
    main()
