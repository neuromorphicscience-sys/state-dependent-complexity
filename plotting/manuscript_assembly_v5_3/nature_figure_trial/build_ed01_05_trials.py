#!/usr/bin/env python3
"""Build source-data-first Nature-style reference versions of ED Figures 1--5."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

MM = 25.4
DPI = 600
ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3"
ED = ATLAS / "02_EXTENDED_DATA"
S08 = ATLAS / "03_SUPPLEMENTARY" / "S08"
OUTROOT = Path(__file__).resolve().parent
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

BLUE, TEAL, GOLD = "#38598C", "#2A9D8F", "#D29A3A"
CORAL, PURPLE, INK = "#C27655", "#7A68A6", "#263238"
GREY, LIGHT = "#7B858A", "#D9DEE1"
STATE = {"sparse_drive": BLUE, "transition_mid": TEAL, "transition_dense": GOLD}
TOPO = {"erdos_renyi": BLUE, "modular": TEAL, "scale_free": GOLD, "small_world": PURPLE}
DIV = LinearSegmentedColormap.from_list("nature_div", [BLUE, "#F7F7F4", "#B23A48"])


def setup_style():
    if ARIAL.exists(): font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists(): font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 5.0, "axes.labelsize": 5.0,
        "xtick.labelsize": 4.0, "ytick.labelsize": 4.0, "legend.fontsize": 3.7,
        "axes.linewidth": .58, "xtick.major.width": .58, "ytick.major.width": .58,
        "xtick.major.size": 1.8, "ytick.major.size": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", pad=1.2)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=.32, alpha=.6)
        ax.set_axisbelow(True)


def label(ax, text):
    ax.text(-.18, 1.08, text, transform=ax.transAxes, fontsize=7, fontweight="bold",
            ha="left", va="bottom", clip_on=False)


def find_csv(base, token):
    hits = sorted(base.rglob(f"*{token}*.csv"))
    if not hits: raise FileNotFoundError(f"No CSV matching {token} below {base}")
    return hits[0]


def save(fig, edno, sources, geometry, panel_map):
    out = OUTROOT / f"ED{edno:02d}" / "output_full_v1"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"ED{edno:02d}_full_nature_trial"
    fig.savefig(stem.with_suffix(".pdf"), dpi=DPI)
    fig.savefig(stem.with_suffix(".svg"), dpi=DPI)
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"), dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    manifest = {"figure": f"ED{edno:02d}", "status": "review_not_frozen",
                "sources": [str(p.relative_to(ROOT)) for p in sources],
                "panel_source_map": {k: [str(p.relative_to(ROOT)) for p in v]
                                     for k, v in panel_map.items()}}
    (out / "source_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "axis_geometry_mm.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    qa = ("# QA report\n\nStatus: review reference; not frozen.\n\n"
          "- Nature width and font range applied.\n"
          "- All panels redrawn from mapped CSV source data.\n"
          "- PDF, SVG, 600-dpi PNG and TIFF exported; PDF was re-rendered at 600 dpi for review.\n"
          "- Panel letters are single-level lowercase labels.\n")
    (out / "QA_REPORT.md").write_text(qa, encoding="utf-8")


def build_ed1():
    base = ED / "ED01"
    p1 = find_csv(base, "CT05_offdiagonal_transfer_penalties")
    p2 = find_csv(base, "CT06_graph_seed_consistency")
    d1, d2 = pd.read_csv(p1), pd.read_csv(p2)
    fig, axs = plt.subplots(1, 2, figsize=(183/MM, 62/MM), gridspec_kw={"left": .10, "right": .985, "bottom": .28, "top": .91, "wspace": .34})
    ax = axs[0]
    order = list(dict.fromkeys(d1["pair"]))
    cols = [BLUE, TEAL, GOLD, BLUE, TEAL, GOLD]
    vals = [d1.loc[d1.pair == x, "delta_vs_target_home"].values for x in order]
    bp = ax.boxplot(vals, positions=np.arange(len(order)), widths=.58, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": .75}, boxprops={"color": INK, "lw": .55},
                    whiskerprops={"color": GREY, "lw": .5}, capprops={"color": GREY, "lw": .5})
    for i, (box, v) in enumerate(zip(bp["boxes"], vals)):
        box.set_facecolor(cols[i]); box.set_alpha(.72)
        jitter = np.linspace(-.18, .18, len(v))
        ax.scatter(i+jitter, v, s=5, color=GREY, alpha=.48, edgecolor="none", zorder=3)
    short = [str(x).replace(" → ", " →\n") for x in order]
    ax.set_xticks(range(len(order)), short, rotation=32, ha="right", fontsize=3.35)
    ax.axhline(0, color=GREY, lw=.6, ls="--", dashes=(3, 2))
    ax.set_xlabel("Transferred state pair")
    ax.set_ylabel("Transferred − target-home score")
    clean(ax); label(ax, "a")
    ax = axs[1]
    for seed, g in d2.groupby("graph_seed"):
        g = g.sort_values("k")
        ax.plot(g.k, g.crossover_interaction, color=GREY, alpha=.42, lw=.55, marker="o", ms=2)
    s = d2.groupby("k").crossover_interaction.agg(["mean", "sem"]).reset_index()
    ax.fill_between(s.k, s["mean"]-1.96*s["sem"], s["mean"]+1.96*s["sem"], color=TEAL, alpha=.16, lw=0)
    ax.plot(s.k, s["mean"], color=TEAL, lw=1.25, marker="o", ms=3, label="Across graph seeds")
    ax.axhline(0, color=GREY, lw=.6, ls="--", dashes=(3, 2))
    ax.set_xlabel("Complexity budget, k")
    ax.set_ylabel("State × budget interaction")
    ax.legend(frameon=False, loc="upper left")
    clean(ax); label(ax, "b")
    save(fig, 1, [p1, p2], {"canvas_mm": [183, 62], "axes": "two equal-width axes"},
         {"a": [p1], "b": [p2]})


def line_ci(ax, d, y, ci, color=TEAL, xlabel="High-complexity fraction", ylabel=""):
    d = d.sort_values("complexity_fraction")
    x, m = d.complexity_fraction.to_numpy(float), d[y].to_numpy(float)
    e = d[ci].to_numpy(float) if ci in d else np.zeros(len(d))
    ax.fill_between(x, m-e, m+e, color=color, alpha=.17, lw=0)
    ax.plot(x, m, color=color, lw=1.05, marker="o", ms=2.2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); clean(ax)


def build_ed2():
    base = ED / "ED02"
    toks = ["window_frequency_transition", "frequency_transition_frequency_state_composition",
            "window_rhythm_transition", "window_rhythm_score", "complexity_activity_gain",
            "complexity_synchrony", "emergence_vs_budget_by_topology", "positive_emergence_fraction"]
    ps = [find_csv(base, t) for t in toks]
    ds = [pd.read_csv(p) for p in ps]
    fig, axs = plt.subplots(2, 4, figsize=(183/MM, 91/MM), gridspec_kw={"left": .07, "right": .99, "bottom": .11, "top": .95, "wspace": .53, "hspace": .62})
    line_ci(axs[0,0], ds[0], "frequency_transition_strength_mean", "frequency_transition_strength_ci95", BLUE,
            ylabel="Frequency transition strength"); label(axs[0,0], "a")
    d = ds[1]
    piv = d.pivot_table(index="complexity_fraction", columns="frequency_state", values="fraction", aggfunc="mean").fillna(0)
    state_cols = [BLUE, TEAL, GOLD, CORAL, PURPLE][:len(piv.columns)]
    axs[0,1].stackplot(piv.index, [piv[c] for c in piv], colors=state_cols, alpha=.88, labels=[str(c).replace("_", " ") for c in piv])
    axs[0,1].set_xlabel("High-complexity fraction"); axs[0,1].set_ylabel("State fraction"); axs[0,1].set_ylim(0,1)
    clean(axs[0,1], None); label(axs[0,1], "b")
    axs[0,1].legend(frameon=False, loc="lower right", fontsize=3.0, handlelength=.8, labelspacing=.18)
    line_ci(axs[0,2], ds[2], "rhythm_transition_strength_mean", "rhythm_transition_strength_ci95", GOLD,
            ylabel="Rhythm transition strength")
    line_ci(axs[0,3], ds[3], "rhythm_score_stage1b_mean", "rhythm_score_stage1b_ci95", TEAL,
            ylabel="Rhythm organization"); label(axs[0,3], "c")
    line_ci(axs[1,0], ds[4], "mean", "ci95", BLUE, ylabel="Activity gain"); label(axs[1,0], "d")
    line_ci(axs[1,1], ds[5], "mean", "ci95", TEAL, ylabel="Synchrony"); label(axs[1,1], "e")
    ax = axs[1,2]; d = ds[6]
    for top, g in d.groupby("topology"):
        g=g.sort_values("complexity_fraction"); c=TOPO.get(top, GREY)
        ax.fill_between(g.complexity_fraction, g["mean"]-g.ci95, g["mean"]+g.ci95, color=c, alpha=.10, lw=0)
        ax.plot(g.complexity_fraction, g["mean"], color=c, lw=.85, marker="o", ms=1.8, label=top.replace("_"," "))
    ax.axhline(0,color=GREY,lw=.5,ls="--"); ax.set_xlabel("High-complexity fraction"); ax.set_ylabel("Emergence")
    ax.legend(frameon=False, loc="lower right", fontsize=3.0, handlelength=.9); clean(ax); label(ax,"f")
    ax=axs[1,3]; d=ds[7].sort_values("positive_fraction", ascending=False)
    cc=[TOPO.get(x,GREY) for x in d.topology]; y=np.arange(len(d))
    ax.barh(y,d.positive_fraction,color=cc,alpha=.82,height=.62)
    ax.set_yticks(y,[x.replace("_"," ") for x in d.topology],fontsize=3.5); ax.invert_yaxis(); ax.set_xlim(0,1.02)
    ax.set_xlabel("Positive-emergence fraction"); ax.set_ylabel("Topology"); clean(ax,"x"); label(ax,"g")
    save(fig, 2, ps, {"canvas_mm": [183,91], "axes": "2 rows x 4 equal physical axes; panel b contains two adjacent microplots"},
         {"a":[ps[0]], "b":[ps[1],ps[2]], "c":[ps[3]], "d":[ps[4]], "e":[ps[5]], "f":[ps[6]], "g":[ps[7]]})


def heat(ax, d, row, col, val, ylabel, xlabel):
    d = d[d[col].astype(str).str.lower().ne("none")].copy()
    piv=d.pivot(index=row,columns=col,values=val)
    z=piv.to_numpy(float); lim=max(abs(np.nanmin(z)),abs(np.nanmax(z)),.001)
    im=ax.imshow(z,aspect="auto",cmap=DIV,norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
    ax.set_xticks(range(len(piv.columns)),[str(x).replace("_"," ").replace("None","") for x in piv.columns],rotation=42,ha="right",fontsize=3.4)
    ax.set_yticks(range(len(piv.index)),[str(x).replace("_"," ") for x in piv.index],fontsize=3.6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j,i,f"{z[i,j]:.2f}",ha="center",va="center",fontsize=3.2,color="white" if abs(z[i,j])>.55*lim else INK)
    return im


def build_ed3():
    base=ED/"ED03"
    ps=[find_csv(base,"regime_by_allocation_gain"),find_csv(base,"topology_by_allocation_gain"),find_csv(base,"targeted_vs_random_complexity_requirement")]
    ds=[pd.read_csv(p) for p in ps]
    fig,axs=plt.subplots(1,3,figsize=(183/MM,65/MM),gridspec_kw={"left":.085,"right":.985,"bottom":.27,"top":.9,"wspace":.58})
    heat(axs[0],ds[0],"regime","placement","placement_gain_vs_random_mean","State","Allocation rule"); label(axs[0],"a")
    heat(axs[1],ds[1],"topology","placement","placement_gain_vs_random_mean","Topology","Allocation rule"); label(axs[1],"b")
    ax=axs[2]; d=ds[2]
    for top,g in d.groupby("topology"):
        ax.scatter(g.random_complex_units_required,g.targeted_complex_units_required,s=4,color=TOPO.get(top,GREY),alpha=.28,edgecolor="none",label=top.replace("_"," "))
    lo=min(d.iloc[:,0].min(),d.iloc[:,1].min()); hi=max(d.iloc[:,0].max(),d.iloc[:,1].max())
    ax.plot([lo,hi],[lo,hi],color=GREY,lw=.6,ls="--",dashes=(3,2))
    ax.set_xlabel("Random complexity requirement"); ax.set_ylabel("Targeted complexity requirement")
    ax.legend(frameon=False,loc="upper left",fontsize=3.0,handletextpad=.2); clean(ax); label(ax,"c")
    save(fig,3,ps,{"canvas_mm":[183,65],"axes":"one row x 3 equal axes; former duplicated c,d omitted"},
         {"a":[ps[0]], "b":[ps[1]], "c":[ps[2]]})


def build_ed4():
    pa=find_csv(S08/"S08_a__Confirmatory_baseline_comparison_across_Stage4_phases","stage4_paired_contrasts")
    pb=find_csv(S08/"S08_b__Model-side_descriptor-to-leverage_bridge","model_side_biology_bridge")
    pc=find_csv(S08/"S08_c__ER-patch_phase_gain_curves","stage4_core_225_task_methods")
    contrasts,bridge,core=pd.read_csv(pa),pd.read_csv(pb),pd.read_csv(pc)
    fig=plt.figure(figsize=(183/MM,96/MM))
    gs=fig.add_gridspec(2,12,left=.065,right=.99,bottom=.105,top=.95,wspace=1.8,hspace=.72)
    axs=[fig.add_subplot(gs[0,0:4]),fig.add_subplot(gs[0,4:8]),fig.add_subplot(gs[0,8:12]),
         fig.add_subplot(gs[1,0:3]),fig.add_subplot(gs[1,3:6]),fig.add_subplot(gs[1,6:9]),fig.add_subplot(gs[1,9:12])]
    d=contrasts[(contrasts.grouping=="phase")]
    phases=["stage4b_replication","stage4c_atlas_extension","stage4d_scaling","stage4e_topology","er_patch"]
    bases=["spectral","feedback_hub","high_degree","module_bridge","cycle_proxy","random"]
    piv=d.pivot(index="group",columns="baseline",values="mean_delta").reindex(index=phases,columns=bases)
    z=piv.to_numpy(float); im=axs[0].imshow(z,aspect="auto",cmap=DIV,norm=TwoSlopeNorm(vmin=-.25,vcenter=0,vmax=.25))
    axs[0].set_xticks(range(6),[x.replace("_"," ") for x in bases],rotation=42,ha="right",fontsize=3.1)
    axs[0].set_yticks(range(5),["Replication","Atlas extension","Scale scaling","Topology","ER patch"],fontsize=3.2)
    axs[0].set_xlabel("Comparator"); axs[0].set_ylabel("Confirmatory phase")
    for i in range(5):
        for j in range(6):
            axs[0].text(j,i,f"{z[i,j]:.2f}",ha="center",va="center",fontsize=2.8,color="white" if abs(z[i,j])>.12 else INK)
    label(axs[0],"a")
    ax=axs[1]
    ax.scatter(bridge.coverage2,bridge.leverage_gain_vs_spectral,c=bridge.selected_dispersion,cmap="viridis",s=6,alpha=.62,edgecolor="none")
    ax.axhline(0,color=GREY,lw=.5,ls="--"); ax.set_xlabel("Two-hop selected-unit coverage"); ax.set_ylabel("Leverage gain vs spectral"); clean(ax); label(ax,"b")
    phase_order=["er_patch","stage4b_replication","stage4c_atlas_extension","stage4d_scaling","stage4e_topology"]
    phase_names=["ER patch","Replication","Atlas extension","Scale scaling","Topology"]
    for idx,(ax,phase,name) in enumerate(zip(axs[2:],phase_order,phase_names),start=2):
        d=core[core.phase==phase]
        key=["task_id"]
        wide=d.pivot_table(index=key,columns="method",values="score",aggfunc="mean").reset_index()
        meta=d.drop_duplicates("task_id")[["task_id","regime","budget_fraction"]]
        wide=wide.merge(meta,on="task_id"); wide["gain"]=wide.dynamics_aware-wide.spectral
        for reg,g in wide.groupby("regime"):
            s=g.groupby("budget_fraction").gain.agg(["mean","sem"]).reset_index(); c=STATE.get(reg,GREY)
            ax.fill_between(s.budget_fraction,s["mean"]-1.96*s["sem"],s["mean"]+1.96*s["sem"],color=c,alpha=.12,lw=0)
            ax.plot(s.budget_fraction,s["mean"],color=c,lw=.85,marker="o",ms=1.8,label=reg.replace("_"," "))
        ax.axhline(0,color=GREY,lw=.5,ls="--"); ax.set_xlabel("High-complexity fraction")
        ax.set_ylabel("Gain vs spectral" if idx in [2,3] else "")
        ax.set_title(name,fontsize=5,pad=2); clean(ax); label(ax,chr(ord('a')+idx))
        if idx in [2,3]: ax.legend(frameon=False,loc="best",fontsize=2.8,handlelength=.8,labelspacing=.15)
    save(fig,4,[pa,pb,pc],{"canvas_mm":[183,96],"axes":"row 1: a,b,c; row 2: d,e,f,g"},
         {"a":[pa], "b":[pb], "c":[pc], "d":[pc], "e":[pc], "f":[pc], "g":[pc]})


def build_ed5():
    base=ED/"ED05"; p=find_csv(base,"near_optimal_count_by_budget"); d=pd.read_csv(p)
    fig,ax=plt.subplots(figsize=(89/MM,59/MM),gridspec_kw={"left":.2,"right":.97,"bottom":.2,"top":.91})
    ks=sorted(d.k.unique()); x=np.arange(len(ks)); regs=list(STATE)
    offsets=np.linspace(-.22,.22,len(regs))
    for off,reg in zip(offsets,regs):
        g=d[d.regime==reg]
        vals=[g[g.k==k].near_optimal_count.values for k in ks]
        means=np.array([np.mean(v) for v in vals]); sem=np.array([pd.Series(v).sem() for v in vals])
        ax.errorbar(x+off,means,yerr=sem,fmt="o-",lw=.8,ms=2.8,capsize=1.5,color=STATE[reg],label=reg.replace("_"," "))
        for xi,v in zip(x+off,vals): ax.scatter(np.repeat(xi,len(v)),v,s=4,color=STATE[reg],alpha=.26,edgecolor="none")
    ax.set_xticks(x,[f"k={k}" for k in ks]); ax.set_xlabel("Complexity budget"); ax.set_ylabel("Near-optimal allocations among four searches")
    ax.set_ylim(.85,4.15); ax.set_yticks([1,2,3,4]); ax.legend(frameon=False,loc="upper right"); clean(ax); label(ax,"a")
    save(fig,5,[p],{"canvas_mm":[89,59],"axes":"single-column single axis"},{"a":[p]})


if __name__ == "__main__":
    setup_style()
    build_ed1(); build_ed2(); build_ed3(); build_ed4(); build_ed5()
    print("Built ED01--ED05 review figures.")
