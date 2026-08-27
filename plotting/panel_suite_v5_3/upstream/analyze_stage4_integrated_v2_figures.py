#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage 4 integrated analysis + manuscript figure builder (v2)
=============================================================

Purpose
-------
1. Rebuild the authoritative Stage 4 core evidence using:
   - stage4_complete_results.tar.gz  (4B/4C/4D/4E old overnight archive)
   - stage4f_closure_results.tar.gz  (ER repair + stability / closure)
2. Optionally load the cleaned Stage 4A discovery canonical table.
3. Preserve the statistical guardrails from analyze_stage4_integrated_v1.py.
4. In addition, generate manuscript-oriented figure bundles that explicitly
   keep 4A as discovery and 4B/4C as independent replication.

Important framing
-----------------
- 4A is shown as discovery evidence; it is NOT pooled into the confirmatory
  p-value that should be attributed to replication.
- 4B/4C serve as independent replication / atlas extension.
- 4D/4E/ER patch are generalization evidence.
- 4F closure contributes optimizer degeneracy / fresh validation evidence.
- Search maxima are never used as final performance.

Suggested output structure
--------------------------
plot/Stage4_integrated_full/
    analysis/
    figures_main/
    figures_si/

The script writes both tables and figures. Every figure is exported as a
600-dpi PNG and a vector PDF; `--dpi` can override the PNG resolution.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DEFAULT_ROOT = Path.cwd()
EXPORT_DPI = 600

# Soft, publication-friendly palette
COLORS = {
    "discovery": "#B56576",
    "replication": "#355C7D",
    "generalization": "#2A9D8F",
    "closure": "#8D6A9F",
    "pooled": "#6C757D",
    "dynamics": "#264653",
    "spectral": "#577590",
    "feedback_hub": "#8C564B",
    "high_degree": "#E09F3E",
    "module_bridge": "#4D908E",
    "cycle_proxy": "#B565A7",
    "random": "#A9A9A9",
}

BASELINE_LABELS = {
    "spectral": "Spectral",
    "feedback_hub": "Feedback hub",
    "high_degree": "High degree",
    "module_bridge": "Module bridge",
    "cycle_proxy": "Cycle proxy",
    "random": "Random",
}

TOPOLOGY_LABELS = {
    "scale_free": "Scale-free",
    "erdos_renyi": "ER",
    "modular": "Modular",
    "small_world": "Small-world",
}

REGIME_LABELS = {
    "sparse_drive": "Sparse drive",
    "transition_mid": "Transition mid",
    "transition_dense": "Transition dense",
}


def load_v1_module(path: Path):
    spec = importlib.util.spec_from_file_location("stage4v1", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure_pair(fig, outpath: Path):
    """Export every Stage4 integrated asset as 600-dpi PNG plus vector PDF."""
    base = outpath.with_suffix("")
    fig.savefig(base.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


def prettify_group_name(group: str) -> str:
    mapping = {
        "stage4a_discovery": "Discovery (4A)",
        "stage4b_replication": "Replication (4B)",
        "stage4c_atlas_extension": "Atlas extension (4C)",
        "stage4d_scaling": "Scale scaling (4D)",
        "stage4e_topology": "Topology generalization (4E)",
        "er_patch": "ER repair / closure patch (4F-ER)",
        "all_225_core_tasks": "All confirmatory core tasks",
        "scale_free_N256_replication_atlas": "Replication + atlas (4B+4C)",
        "ER_N256_repair": "ER repair",
        "modular_N256": "Modular topology",
        "small_world_N256": "Small-world topology",
        "scale_free_scaling_N128_N512": "Scale-free scaling",
    }
    return mapping.get(group, str(group))


def evidence_family(group: str) -> str:
    if group in ["stage4a_discovery"]:
        return "discovery"
    if group in ["stage4b_replication", "stage4c_atlas_extension", "scale_free_N256_replication_atlas"]:
        return "replication"
    if group in ["stage4d_scaling", "stage4e_topology", "er_patch", "ER_N256_repair", "modular_N256", "small_world_N256", "scale_free_scaling_N128_N512"]:
        return "generalization"
    if group in ["closure"]:
        return "closure"
    return "pooled"


def write_tables(v1, out_analysis: Path, complete, closure_path, stage4a_path):
    """Re-run the v1 logic and additionally return all core tables in memory."""
    complete_tm, complete_manifest, complete_integrity = v1.load_complete_archive(complete)
    closure = v1.load_closure_archive(closure_path)

    core = pd.concat([complete_tm, closure["er"]], ignore_index=True, sort=False)
    v1.validate_core(core)
    core.to_csv(out_analysis / "stage4_core_225_task_methods.csv", index=False, encoding="utf-8-sig")

    stage4a_tm = v1.load_optional_stage4a(stage4a_path)
    stage4a_tm.to_csv(out_analysis / "stage4a_discovery_task_methods.csv", index=False, encoding="utf-8-sig")

    contrasts = v1.build_all_contrasts(core)
    contrasts.to_csv(out_analysis / "stage4_paired_contrasts.csv", index=False, encoding="utf-8-sig")

    gain_spectral = v1.gain_table(core, "spectral", ["phase", "topology", "N", "regime", "k"])
    gain_spectral.to_csv(out_analysis / "stage4_gain_landscape_vs_spectral.csv", index=False, encoding="utf-8-sig")

    gain_random = v1.gain_table(core, "random", ["phase", "topology", "N", "regime", "k"])
    gain_random.to_csv(out_analysis / "stage4_gain_landscape_vs_random.csv", index=False, encoding="utf-8-sig")

    fixed_state, fixed_gain = v1.fixed_topology_state_analysis(core)
    fixed_state.to_csv(out_analysis / "fixed_topology_state_shift_analysis.csv", index=False, encoding="utf-8-sig")
    fixed_gain.to_csv(out_analysis / "fixed_topology_state_gain_shift.csv", index=False, encoding="utf-8-sig")

    pairdeg, degsummary, degcorr = v1.pairwise_functional_degeneracy(closure["analysis"])
    pairdeg.to_csv(out_analysis / "optimizer_pairwise_mask_vs_fresh_score.csv", index=False, encoding="utf-8-sig")
    degsummary.to_csv(out_analysis / "optimizer_functional_degeneracy_summary.csv", index=False, encoding="utf-8-sig")
    degcorr.to_csv(out_analysis / "optimizer_mask_score_correlation.csv", index=False, encoding="utf-8-sig")

    bridge = v1.model_side_biology_bridge(core)
    bridge.to_csv(out_analysis / "model_side_biology_bridge.csv", index=False, encoding="utf-8-sig")

    for fn, d in closure["analysis"].items():
        d.to_csv(out_analysis / f"stage4f_{fn}", index=False, encoding="utf-8-sig")
    closure["fresh_reevaluation"].to_csv(out_analysis / "stage4f_fresh_reevaluation_24seeds.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "complete_archive": str(complete),
        "closure_archive": str(closure_path),
        "stage4a_path": str(stage4a_path),
        "stage4a_loaded": bool(len(stage4a_tm)),
        "authoritative_core_tasks": int(core["task_id"].nunique()),
        "authoritative_core_rows": int(len(core)),
        "complete_old_tasks": int(complete_tm["task_id"].nunique()),
        "er_repair_tasks": int(closure["er"]["task_id"].nunique()),
    }
    (out_analysis / "integrated_figure_manifest_v2.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    v1.write_summary(
        outdir=out_analysis,
        complete_integrity=complete_integrity,
        closure=closure,
        core=core,
        contrasts=contrasts,
        fixed_state=fixed_state,
        fixed_gain=fixed_gain,
        degeneracy_summary=degsummary,
    )

    return {
        "core": core,
        "stage4a": stage4a_tm,
        "contrasts": contrasts,
        "gain_spectral": gain_spectral,
        "gain_random": gain_random,
        "fixed_state": fixed_state,
        "fixed_gain": fixed_gain,
        "pairdeg": pairdeg,
        "degsummary": degsummary,
        "degcorr": degcorr,
        "bridge": bridge,
        "closure": closure,
    }


def _setup_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)


def plot_forest_discovery_replication(stage4a, contrasts, outpath: Path):
    rows = []

    # discovery: recompute 4A DA vs spectral from raw task-method table
    if stage4a is not None and not stage4a.empty:
        wide = stage4a.pivot_table(index="task_id", columns="method", values="score", aggfunc="mean")
        if "dynamics_aware" in wide.columns and "spectral" in wide.columns:
            delta = (wide["dynamics_aware"] - wide["spectral"]).dropna().to_numpy(float)
            if len(delta):
                mean = float(np.mean(delta))
                lo = float(np.quantile(np.random.default_rng(20260818).choice(delta, size=(4000, len(delta)), replace=True).mean(axis=1), 0.025)) if len(delta) > 1 else mean
                hi = float(np.quantile(np.random.default_rng(20260818).choice(delta, size=(4000, len(delta)), replace=True).mean(axis=1), 0.975)) if len(delta) > 1 else mean
                rows.append({
                    "label": "Discovery (4A)",
                    "mean_delta": mean,
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "n_tasks": len(delta),
                    "family": "discovery",
                })

    use = contrasts[(contrasts["baseline"] == "spectral") & (contrasts["grouping"].isin(["evidence_stratum"]))].copy()
    order = [
        "scale_free_N256_replication_atlas",
        "ER_N256_repair",
        "modular_N256",
        "small_world_N256",
        "scale_free_scaling_N128_N512",
    ]
    use["_ord"] = use["group"].map({g: i for i, g in enumerate(order)})
    use = use.sort_values(["_ord", "group"]).drop(columns="_ord")
    for _, r in use.iterrows():
        rows.append({
            "label": prettify_group_name(r["group"]),
            "mean_delta": r["mean_delta"],
            "bootstrap95_low": r["bootstrap95_low"],
            "bootstrap95_high": r["bootstrap95_high"],
            "n_tasks": int(r["n_tasks"]),
            "family": evidence_family(r["group"]),
        })

    pooled = contrasts[(contrasts["baseline"] == "spectral") & (contrasts["grouping"] == "pooled")]
    if not pooled.empty:
        r = pooled.iloc[0]
        rows.append({
            "label": "All confirmatory core tasks",
            "mean_delta": r["mean_delta"],
            "bootstrap95_low": r["bootstrap95_low"],
            "bootstrap95_high": r["bootstrap95_high"],
            "n_tasks": int(r["n_tasks"]),
            "family": "pooled",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=240)
    y = np.arange(len(df))[::-1]
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1.0)

    for yi, (_, r) in zip(y, df.iterrows()):
        c = COLORS.get(r["family"], COLORS["pooled"])
        ax.plot([r["bootstrap95_low"], r["bootstrap95_high"]], [yi, yi], color=c, linewidth=2.6, solid_capstyle="round")
        ax.scatter(r["mean_delta"], yi, s=48, color=c, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(r["bootstrap95_high"] + 0.004, yi, f"n={int(r['n_tasks'])}", va="center", ha="left", fontsize=9, color="#444444")

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=10)
    ax.set_xlabel("Dynamics-aware gain vs spectral baseline", fontsize=11)
    ax.set_title("Stage 4 integrated evidence: discovery, replication, and generalization", fontsize=12, pad=12)
    _setup_ax(ax)

    xmax = max(float(df["bootstrap95_high"].max()), float(df["mean_delta"].max()))
    xmin = min(float(df["bootstrap95_low"].min()), float(df["mean_delta"].min()))
    pad = max(0.01, 0.12 * (xmax - xmin + 1e-9))
    ax.set_xlim(xmin - pad, xmax + 4.0 * pad)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COLORS["discovery"], lw=2.6, marker="o", markersize=5.5, label="Discovery"),
        Line2D([0], [0], color=COLORS["replication"], lw=2.6, marker="o", markersize=5.5, label="Replication"),
        Line2D([0], [0], color=COLORS["generalization"], lw=2.6, marker="o", markersize=5.5, label="Generalization"),
        Line2D([0], [0], color=COLORS["pooled"], lw=2.6, marker="o", markersize=5.5, label="Pooled estimate"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower right", fontsize=9)
    fig.tight_layout()
    save_figure_pair(fig, outpath)
    plt.close(fig)


def plot_phase_baseline_heatmap(contrasts, outpath: Path):
    use = contrasts[contrasts["grouping"] == "phase"].copy()
    if use.empty:
        return
    order = ["stage4b_replication", "stage4c_atlas_extension", "stage4d_scaling", "stage4e_topology", "er_patch"]
    use = use[use["group"].isin(order)].copy()
    use["phase_label"] = use["group"].map(prettify_group_name)
    use["baseline_label"] = use["baseline"].map(BASELINE_LABELS)
    mat = use.pivot_table(index="phase_label", columns="baseline_label", values="mean_delta", aggfunc="mean")
    row_order = [prettify_group_name(x) for x in order if prettify_group_name(x) in mat.index]
    col_order = [BASELINE_LABELS[x] for x in ["spectral", "feedback_hub", "high_degree", "module_bridge", "cycle_proxy", "random"] if BASELINE_LABELS[x] in mat.columns]
    mat = mat.reindex(index=row_order, columns=col_order)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=240)
    data = mat.to_numpy(float)
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=28, ha="right", fontsize=9)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist(), fontsize=10)
    ax.set_title("Dynamics-aware advantage across Stage 4 confirmatory phases", fontsize=12, pad=10)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = data[i, j]
            txtc = "white" if abs(val) > 0.06 else "#111111"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=txtc)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Mean score gain", fontsize=10)
    fig.tight_layout()
    save_figure_pair(fig, outpath)
    plt.close(fig)


def plot_gain_landscape(gain_spectral, outpath: Path):
    use = gain_spectral.copy()
    use = use[use["topology"].isin(["scale_free", "erdos_renyi", "modular", "small_world"])]
    if use.empty:
        return
    use["budget_fraction"] = use["k"] / use["N"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=240, sharex=True, sharey=True)
    axes = axes.ravel()
    topologies = ["scale_free", "erdos_renyi", "modular", "small_world"]
    regime_order = ["sparse_drive", "transition_mid", "transition_dense"]
    regime_colors = {
        "sparse_drive": "#4D908E",
        "transition_mid": "#577590",
        "transition_dense": "#BC6C25",
    }

    for ax, topo in zip(axes, topologies):
        sub = use[use["topology"] == topo].copy()
        for reg in regime_order:
            g = sub[sub["regime"] == reg].sort_values("budget_fraction")
            if g.empty:
                continue
            ax.plot(g["budget_fraction"], g["mean_gain"], marker="o", linewidth=2.1, markersize=4.5, color=regime_colors[reg], label=REGIME_LABELS.get(reg, reg))
            ax.fill_between(g["budget_fraction"], g["bootstrap95_low"], g["bootstrap95_high"], color=regime_colors[reg], alpha=0.16)
        ax.axhline(0, color="#999999", linestyle="--", linewidth=0.9)
        ax.set_title(TOPOLOGY_LABELS.get(topo, topo), fontsize=11)
        _setup_ax(ax)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))

    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.supxlabel("High-complexity fraction (k / N)", fontsize=11)
    fig.supylabel("Dynamics-aware gain vs spectral", fontsize=11)
    fig.suptitle("Gain landscape across topologies and collective regimes", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_pair(fig, outpath)
    plt.close(fig)


def plot_state_and_degeneracy(fixed_state, degsummary, outpath: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=240)

    # Panel A: fixed-topology state shifts
    ax = axes[0]
    fs = fixed_state.copy()
    if not fs.empty:
        x = np.arange(len(fs))
        ax.axhline(0, color="#999999", linestyle="--", linewidth=0.9)
        ax.bar(x, fs["mean_shift"], color="#577590", width=0.64, alpha=0.9)
        err_low = fs["mean_shift"] - fs["bootstrap95_low"]
        err_high = fs["bootstrap95_high"] - fs["mean_shift"]
        ax.errorbar(x, fs["mean_shift"], yerr=[err_low, err_high], fmt="none", ecolor="#2F4858", elinewidth=1.2, capsize=3)
        labels = [str(m).replace("sel_", "").replace("_", " ") for m in fs["metric"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_title("Pure fixed-topology state contrast\n(transition mid − sparse drive)", fontsize=11)
        ax.set_ylabel("Mean shift", fontsize=10)
        _setup_ax(ax)

    # Panel B: optimizer degeneracy
    ax = axes[1]
    ds = degsummary.copy()
    if not ds.empty:
        ds = ds[ds["grouping"] == "k"].copy()
        order = []
        for k in [8, 32, 64]:
            if str(k) in ds["group"].astype(str).tolist():
                order.append(str(k))
        if order:
            ds["group"] = ds["group"].astype(str)
            ds = ds.set_index("group").loc[order].reset_index()
        x = np.arange(len(ds))
        ax.plot(x, ds["median_jaccard_distance"], marker="o", linewidth=2.2, color="#8D6A9F", label="Mask dissimilarity")
        ax.plot(x, ds["median_abs_fresh_score_difference"], marker="s", linewidth=2.2, color="#2A9D8F", label="Fresh-score difference")
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={g}" for g in ds["group"].astype(str)], fontsize=10)
        ax.set_ylabel("Median value", fontsize=10)
        ax.set_title("Optimizer degeneracy after fresh reevaluation", fontsize=11)
        _setup_ax(ax)
        ax.legend(frameon=False, loc="upper left", fontsize=9)

    fig.suptitle("State selectivity is modest, but the allocation landscape is degenerate", fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure_pair(fig, outpath)
    plt.close(fig)


def plot_bridge_scatter(bridge, outpath: Path):
    if bridge.empty or "leverage_gain_vs_spectral" not in bridge.columns:
        return
    x = pd.to_numeric(bridge.get("coverage2"), errors="coerce")
    y = pd.to_numeric(bridge.get("leverage_gain_vs_spectral"), errors="coerce")
    c = pd.to_numeric(bridge.get("selected_dispersion"), errors="coerce")
    m = x.notna() & y.notna() & c.notna()
    if m.sum() < 10:
        return

    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=240)
    sc = ax.scatter(x[m], y[m], c=c[m], cmap="viridis", s=28, alpha=0.8, edgecolor="none")
    ax.set_xlabel("Two-hop coverage of selected high-complexity units", fontsize=10)
    ax.set_ylabel("Leverage gain vs spectral", fontsize=10)
    ax.set_title("Model-side biology bridge", fontsize=12)
    _setup_ax(ax)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Selected dispersion", fontsize=9)
    fig.tight_layout()
    save_figure_pair(fig, outpath)
    plt.close(fig)


def plot_si_phase_curves(core, outdir: Path):
    """Supplementary: per-phase paired gain curves vs budget for each topology."""
    if core.empty:
        return
    df = core.copy()
    wide = df.pivot_table(index=[c for c in ["task_id", "phase", "topology", "regime", "k", "N"] if c in df.columns], columns="method", values="score", aggfunc="mean").reset_index()
    if "dynamics_aware" not in wide.columns or "spectral" not in wide.columns:
        return
    wide["budget_fraction"] = wide["k"] / wide["N"]
    wide["gain"] = wide["dynamics_aware"] - wide["spectral"]

    for phase, gphase in wide.groupby("phase", dropna=False):
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), dpi=220, sharey=True)
        regs = ["sparse_drive", "transition_mid", "transition_dense"]
        topologies = sorted(gphase["topology"].dropna().unique())
        topo_colors = {
            "scale_free": "#355C7D",
            "erdos_renyi": "#6C757D",
            "modular": "#2A9D8F",
            "small_world": "#8D6A9F",
        }
        for ax, reg in zip(axes, regs):
            s = gphase[gphase["regime"] == reg]
            for topo in topologies:
                g = s[s["topology"] == topo].groupby("budget_fraction", as_index=False)["gain"].mean().sort_values("budget_fraction")
                if g.empty:
                    continue
                ax.plot(g["budget_fraction"], g["gain"], marker="o", linewidth=2.0, markersize=4.0, color=topo_colors.get(topo, "#333333"), label=TOPOLOGY_LABELS.get(topo, topo))
            ax.axhline(0, color="#999999", linestyle="--", linewidth=0.8)
            ax.set_title(REGIME_LABELS.get(reg, reg), fontsize=10)
            _setup_ax(ax)
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.supxlabel("High-complexity fraction (k / N)", fontsize=10)
        fig.supylabel("Mean gain vs spectral", fontsize=10)
        fig.suptitle(f"Supplementary: {prettify_group_name(str(phase))}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fname = f"si_phase_curve_{str(phase)}.png".replace("/", "_")
        save_figure_pair(fig, outdir / fname)
        plt.close(fig)


def build_figure_bundle(tables: dict, fig_main: Path, fig_si: Path):
    plot_forest_discovery_replication(tables["stage4a"], tables["contrasts"], fig_main / "stage4_main_discovery_replication_forest.png")
    plot_phase_baseline_heatmap(tables["contrasts"], fig_main / "stage4_main_phase_baseline_heatmap.png")
    plot_gain_landscape(tables["gain_spectral"], fig_main / "stage4_main_gain_landscape_vs_spectral.png")
    plot_state_and_degeneracy(tables["fixed_state"], tables["degsummary"], fig_main / "stage4_main_state_and_degeneracy.png")
    plot_bridge_scatter(tables["bridge"], fig_main / "stage4_main_model_biology_bridge.png")
    plot_si_phase_curves(tables["core"], fig_si)


def write_readme(outroot: Path):
    txt = """
Stage 4 integrated figure bundle (v2)
=====================================

Main figures
------------
1. stage4_main_discovery_replication_forest.png
   - 4A is shown as discovery only.
   - 4B+4C are shown as independent replication / atlas extension.
   - 4D/4E/ER patch are shown as generalization.
   - pooled core estimate is descriptive only.

2. stage4_main_phase_baseline_heatmap.png
   - Dynamics-aware gain against each baseline across the confirmatory phases.

3. stage4_main_gain_landscape_vs_spectral.png
   - Gain curves across topologies and collective regimes.

4. stage4_main_state_and_degeneracy.png
   - Left: pure fixed-topology state shift test.
   - Right: optimizer degeneracy after fresh reevaluation.

5. stage4_main_model_biology_bridge.png
   - Model-side descriptor-to-leverage bridge for later biological integration.

Supplementary figures
---------------------
- si_phase_curve_*.png: phase-specific gain curves.

Narrative principle
-------------------
Do not describe Stage 4A as a separate project report section.
Use: discovery -> replication -> generalization -> closure / degeneracy.
""".strip()
    (outroot / "README_stage4_integrated_v2_figures.txt").write_text(txt, encoding="utf-8")


def main():
    global EXPORT_DPI
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--complete-archive", type=Path, default=None)
    ap.add_argument("--closure-archive", type=Path, default=None)
    ap.add_argument("--stage4a-canonical", type=Path, default=None)
    ap.add_argument("--v1-script", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    EXPORT_DPI = int(args.dpi)

    root = args.root.resolve()
    complete = args.complete_archive.resolve() if args.complete_archive else root / "stage4_complete_results.tar.gz"
    closure_path = args.closure_archive.resolve() if args.closure_archive else root / "stage4f_closure_results.tar.gz"
    stage4a = args.stage4a_canonical.resolve() if args.stage4a_canonical else root / "plot" / "Stage4A" / "analysis" / "stage4a_canonical_methods.csv"
    outroot = args.output.resolve() if args.output else root / "plot" / "Stage4_integrated_full"
    v1_path = args.v1_script.resolve() if args.v1_script else Path(__file__).resolve().with_name("analyze_stage4_integrated_v1.py")

    if not complete.exists():
        raise SystemExit(f"Missing complete archive: {complete}")
    if not closure_path.exists():
        raise SystemExit(f"Missing closure archive: {closure_path}")
    if not v1_path.exists():
        raise SystemExit(f"Missing helper v1 script: {v1_path}")

    out_analysis = ensure_dir(outroot / "analysis")
    fig_main = ensure_dir(outroot / "figures_main")
    fig_si = ensure_dir(outroot / "figures_si")

    v1 = load_v1_module(v1_path)

    print("[1/3] Rebuilding Stage 4 integrated tables ...")
    tables = write_tables(v1, out_analysis, complete, closure_path, stage4a)

    print("[2/3] Building figure bundle ...")
    build_figure_bundle(tables, fig_main, fig_si)

    print("[3/3] Writing README ...")
    write_readme(outroot)

    print("\n=== DONE ===")
    print(f"Output root : {outroot}")
    print("Read first :")
    print(f"  {out_analysis / 'integrated_stage4_summary.md'}")
    print(f"  {fig_main / 'stage4_main_discovery_replication_forest.png'}")
    print(f"  {fig_main / 'stage4_main_gain_landscape_vs_spectral.png'}")


if __name__ == "__main__":
    main()
