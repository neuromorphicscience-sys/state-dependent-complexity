#!/usr/bin/env python3
"""Build the source-data-first Nature-width Fig.5 reference figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MM_PER_INCH = 25.4
DPI = 600
ROOT = Path(__file__).resolve().parents[3]
ATLAS = ROOT / "Manuscript_Figures_Nature_DataFirst_V5_3" / "01_MAIN" / "Fig05"
DEFAULT_OUT = Path(__file__).resolve().parent / "output_full_v1"
ARIAL = Path("/mnt/c/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("/mnt/c/Windows/Fonts/arialbd.ttf")

TEAL = "#2A9D8F"
BLUE = "#38598C"
GOLD = "#D29A3A"
CORAL = "#D06F82"
PURPLE = "#7A68A6"
INK = "#263238"
GREY = "#778187"
LIGHT = "#D9DEE1"
PRIMARY = TEAL
CONSERVATIVE = BLUE
MODEL_COLORS = [TEAL, BLUE, GOLD, CORAL, PURPLE]


def setup_style() -> None:
    if ARIAL.exists():
        font_manager.fontManager.addfont(str(ARIAL))
    if ARIAL_BOLD.exists():
        font_manager.fontManager.addfont(str(ARIAL_BOLD))
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 5.0,
        "axes.labelsize": 5.0,
        "xtick.labelsize": 4.2,
        "ytick.labelsize": 4.2,
        "legend.fontsize": 3.7,
        "axes.linewidth": .58,
        "xtick.major.width": .58,
        "ytick.major.width": .58,
        "xtick.major.size": 1.8,
        "ytick.major.size": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def add_axes_mm(fig, fw, fh, x, y, w, h):
    return fig.add_axes([x/fw, y/fh, w/fw, h/fh])


def clean_axis(ax, grid=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.25)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=.34, alpha=.58)
        ax.set_axisbelow(True)


def source(panel: str, token: str, suffix="csv") -> Path:
    hits = sorted((ATLAS / panel / "source_data").glob(f"*{token}*.{suffix}"))
    if not hits:
        raise FileNotFoundError(f"No source for {panel}: {token}")
    return hits[0]


def short_model(s):
    return {"Temporal profile": "Temporal", "Multiaxial ephys": "Multiaxial",
            "Metadata only": "Metadata", "Ephys + metadata": "Ephys+meta",
            "Ephys + temporal": "Ephys+temp", "Naive median": "Naive"}.get(s, s)


def plot_a(ax):
    p = source("Fig05_a__Operational_definition_of_biological_intrinsic_complexity",
               "epsilon_creq_per_cell")
    d = pd.read_csv(p)
    vals = [(d.ev_best - d[f"ev_cost{k}"]).dropna().to_numpy(float) for k in range(4)]
    bp = ax.boxplot(vals, positions=np.arange(4), widths=.56, patch_artist=True,
                    showfliers=False, medianprops={"color": INK, "lw": .75},
                    whiskerprops={"color": GREY, "lw": .55},
                    capprops={"color": GREY, "lw": .55},
                    boxprops={"color": GREY, "lw": .55})
    for patch, color in zip(bp["boxes"], [BLUE, TEAL, GOLD, CORAL]):
        patch.set_facecolor(color); patch.set_alpha(.72)
    ax.axhline(.02, color=CORAL, lw=.7, ls="--", dashes=(3, 2), label="epsilon = 0.02")
    ax.set_xticks(range(4), ["0", "1", "2", "3"])
    ax.set(xlabel="Mechanism cost", ylabel="EV gap to best")
    ax.legend(loc="upper right", frameon=False, handlelength=1.2,
              borderaxespad=.15, fontsize=3.6)
    clean_axis(ax, "y")
    return [p]


def plot_b(ax_rho, ax_mae):
    p = source("Fig05_b__Held-out_rank_information_and_absolute-prediction_boundary",
               "model_comparison_rho")
    d = pd.read_csv(p).set_index("model")
    order_rho = ["Temporal profile", "Multiaxial ephys", "Metadata only",
                 "Ephys + metadata", "Ephys + temporal"]
    order_mae = ["Naive median", "Temporal profile", "Metadata only",
                 "Multiaxial ephys", "Ephys + metadata"]
    for ax, order, metric, ylabel in [
        (ax_rho, order_rho, "spearman_rho", "Held-out Spearman rho"),
        (ax_mae, order_mae, "mae", "Held-out MAE")]:
        x = np.arange(len(order))
        colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(order))]
        if order[0] == "Naive median": colors[0] = "#9AA2A6"
        ax.bar(x, d.loc[order, metric], color=colors, width=.72, edgecolor="none")
        ax.set_xticks(x, [short_model(v) for v in order], rotation=45, ha="right",
                      fontsize=3.55)
        ax.set_xlabel("Prediction model")
        ax.set_ylabel(ylabel)
        clean_axis(ax, "y")
    ax_rho.axhline(0, color=GREY, lw=.5)
    p2 = source("Fig05_b__Held-out_rank_information_and_absolute-prediction_boundary",
                "model_comparison_mae")
    return [p, p2]


def plot_c(ax):
    p = source("Fig05_c__Full-pipeline_permutation_validation", "full_pipeline_permutation")
    summary = source("Fig05_c__Full-pipeline_permutation_validation",
                     "full_pipeline_permutation_summary", "json")
    d = pd.read_csv(p)
    s = json.loads(summary.read_text(encoding="utf-8"))
    obs = s["observed"]["spearman_rho"]
    ax.hist(d.rho, bins=28, color="#B9C2C6", edgecolor="white", lw=.25)
    ax.axvline(obs, color=CORAL, lw=1.15)
    ax.text(.92, .95, f"Observed rho = {obs:.2f}\np = {s['full_pipeline_p']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=4.0)
    ax.set(xlabel="Permuted held-out rho", ylabel="Count")
    clean_axis(ax)
    return [p, summary]


def plot_d(ax):
    p = source("Fig05_d__Biological_group_generalization", "group_generalization_summary")
    d = pd.read_csv(p)
    labels = ["Repeated CV", "Leave layer", "Leave dendrite", "VISp only"]
    ax.bar(np.arange(4), d.spearman_rho, color=[TEAL, BLUE, CORAL, GOLD], width=.7)
    ax.axhline(0, color=GREY, lw=.5)
    ax.set_xticks(range(4), labels, rotation=45, ha="right", fontsize=3.75)
    ax.set_xlabel("Cross-validation scheme")
    ax.set_ylabel("Held-out Spearman rho")
    clean_axis(ax, "y")
    return [p]


FEATURE_NAMES = {
    "trough_v_long_square": "Trough V, long",
    "fast_trough_v_long_square": "Fast trough V, long",
    "fast_trough_v_ramp": "Fast trough V, ramp",
    "f_i_curve_slope": "F-I slope",
    "upstroke_downstroke_ratio_short_square": "Up/down ratio, short",
    "adaptation": "Adaptation",
    "fast_trough_v_short_square": "Fast trough V, short",
    "trough_v_ramp": "Trough V, ramp",
    "fast_trough_t_short_square": "Fast trough t, short",
    "avg_isi": "Mean ISI",
    "peak_v_short_square": "Peak V, short",
    "upstroke_downstroke_ratio_long_square": "Up/down ratio, long",
    "slow_trough_t_long_square": "Slow trough t, long",
    "peak_v_ramp": "Peak V, ramp",
    "upstroke_downstroke_ratio_ramp": "Up/down ratio, ramp",
    "peak_v_long_square": "Peak V, long",
}


def plot_e(ax_features, ax_ablation):
    p1 = source("Fig05_e__Multiaxial_physiology_and_feature-category_ablation",
                "feature_Creq_associations")
    d = pd.read_csv(p1)
    d = d.loc[d.rho.abs().nlargest(16).index].sort_values("rho")
    y = np.arange(len(d))
    colors = [BLUE if v < 0 else TEAL for v in d.rho]
    ax_features.hlines(y, 0, d.rho, color=colors, lw=.7)
    ax_features.scatter(d.rho, y, c=colors, s=8, zorder=3, edgecolor="none")
    ax_features.axvline(0, color=GREY, lw=.5)
    ax_features.set_yticks(y, [FEATURE_NAMES.get(v, v.replace("_", " ")) for v in d.feature],
                          fontsize=3.1)
    ax_features.set_xlabel("Spearman rho")
    clean_axis(ax_features, "x")

    p2 = source("Fig05_e__Multiaxial_physiology_and_feature-category_ablation",
                "category_ablation")
    a = pd.read_csv(p2).sort_values("delta_rho_full_minus_without")
    labs = [v.replace("_", " ").replace("after spike trough", "after-spike trough")
            for v in a.removed_category]
    yy = np.arange(len(a))
    cc = [CORAL if v < 0 else TEAL for v in a.delta_rho_full_minus_without]
    ax_ablation.barh(yy, a.delta_rho_full_minus_without, color=cc, height=.66)
    ax_ablation.axvline(0, color=GREY, lw=.5)
    ax_ablation.set_yticks(yy, labs, fontsize=3.4)
    ax_ablation.set_xlabel("Delta rho, full - ablated")
    clean_axis(ax_ablation, "x")
    return [p1, p2]


def plot_f(ax):
    p = source("Fig05_f__Definition_stability_around_the_primary_epsilon",
               "epsilon_sensitivity_summary")
    d = pd.read_csv(p)
    ax.plot(d.epsilon, d.spearman_vs_eps_0p02, color=TEAL, marker="o", ms=2.0,
            lw=.9, label="Rank correlation")
    ax.plot(d.epsilon, d.exact_agreement_vs_eps_0p02, color=BLUE, marker="o", ms=2.0,
            lw=.9, label="Exact agreement")
    ax.axvline(.02, color=CORAL, lw=.65, ls="--", dashes=(3, 2))
    ax.set(xlabel="Tolerance epsilon", ylabel="Agreement with epsilon = 0.02", ylim=(.45, 1.04))
    ax.legend(loc="lower right", frameon=False, fontsize=3.55,
              handlelength=1.1, labelspacing=.15, borderaxespad=.2)
    clean_axis(ax, "y")
    return [p]


def plot_mapping(ax, panel_token, title, color):
    p = source("Fig05_g__Primary_and_conservative_cross-dataset_mapping", panel_token)
    d = pd.read_csv(p)
    rng = np.random.default_rng(53 if "primary" in panel_token else 54)
    jitter = rng.uniform(-.16, .16, len(d))
    ax.scatter(d.observed + jitter, d.predicted, s=3.2, color=color, alpha=.24,
               edgecolor="none", rasterized=True)
    meds = d.groupby("observed").predicted.median()
    ax.plot(meds.index, meds.values, color=INK, marker="o", ms=2.2, lw=.75)
    ax.set_xticks(range(4), ["0", "1", "2", "3"])
    ax.set(xlabel="Observed complexity", ylabel="Transferred complexity")
    ax.text(.04, .95, title, transform=ax.transAxes, ha="left", va="top",
            fontsize=4.2, fontweight="bold")
    clean_axis(ax, "y")
    return [p]


ENDPOINT = {
    "psp_amplitude": "PSP amp.", "psc_amplitude": "PSC amp.",
    "paired_pulse_ratio_50hz": "PPR, 50 Hz", "stp_initial_50hz": "Initial STP",
    "stp_induction_50hz": "STP induction", "stp_recovery_250ms": "Recovery",
    "stp_recovery_single_250ms": "Single recovery", "variability_resting_state": "Rest variability",
    "variability_second_pulse_50hz": "2nd-pulse var.",
    "variability_stp_induced_state_50hz": "Induced-state var.",
    "G_norm_50hz": "Gain, 50 Hz", "connection": "Connection",
}


def definition_legend(ax, loc="best"):
    handles = [Line2D([0], [0], marker="o", color=PRIMARY, lw=.8, ms=2.7, label="Primary"),
               Line2D([0], [0], marker="o", color=CONSERVATIVE, lw=.8, ms=2.7,
                      label="Conservative")]
    return ax.legend(handles=handles, loc=loc, frameon=False, fontsize=3.45,
                     handlelength=1.0, labelspacing=.12, handletextpad=.3,
                     borderaxespad=.15)


def plot_h_absorption(ax):
    p = source("Fig05_h__Identity_absorption_and_within_between_decomposition",
               "identity_absorption_fraction")
    d = pd.read_csv(p)
    eps = list(dict.fromkeys(d.endpoint))
    y = np.arange(len(eps))
    for definition, off, color in [("primary_shared", -.13, PRIMARY),
                                    ("protocol_conservative", .13, CONSERVATIVE)]:
        g = d[d.definition == definition].set_index("endpoint").reindex(eps)
        ax.scatter(g.absorption_fraction, y+off, color=color, s=7, edgecolor="none")
    ax.axvline(1, color=GREY, lw=.55, ls="--", dashes=(3, 2))
    ax.set_yticks(y, [ENDPOINT.get(v, v) for v in eps], fontsize=3.05)
    ax.set_xlabel("Identity absorption fraction")
    definition_legend(ax, "upper left")
    clean_axis(ax, "x")
    return [p]


def plot_h_decomp(ax):
    p = source("Fig05_h__Identity_absorption_and_within_between_decomposition",
               "within_between_identity_decomposition")
    d = pd.read_csv(p)
    y = np.arange(len(d))
    colors = [TEAL if "between" in v else BLUE for v in d.label]
    lo, hi = d.beta - 1.96*d.se, d.beta + 1.96*d.se
    ax.hlines(y, lo, hi, color=colors, lw=.8)
    ax.scatter(d.beta, y, c=colors, s=9, edgecolor="none", zorder=3)
    ax.axvline(0, color=GREY, lw=.55)
    labels = [v.replace("cell class | ", "Class: ").replace("cre type | ", "Cre: ")
              for v in d.label]
    ax.set_yticks(y, labels, fontsize=3.5)
    ax.set_xlabel("Standardized effect (95% CI)")
    clean_axis(ax, "x")
    return [p]


def plot_i_generalization(ax):
    p = source("Fig05_i__Local_dynamic_boundary", "fold_local_continuous_generalization")
    d = pd.read_csv(p)
    eps = ["psp_amplitude", "psc_amplitude", "G_norm_50hz"]
    x = np.arange(3)
    for definition, off, color, marker in [("primary_shared", -.1, PRIMARY, "o"),
                                            ("protocol_conservative", .1, CONSERVATIVE, "s")]:
        g = d[d.definition == definition].set_index("endpoint").reindex(eps)
        ax.scatter(x+off, g.fold_local_increment_after, color=color, marker=marker,
                   s=10, edgecolor="none")
    ax.axhline(0, color=GREY, lw=.55)
    ax.set_xticks(x, [ENDPOINT[v] for v in eps], rotation=35, ha="right", fontsize=3.7)
    ax.set_xlabel("Local synaptic phenotype")
    ax.set_ylabel("Fold-local delta CV R²")
    definition_legend(ax, "upper left")
    clean_axis(ax, "y")
    return [p]


def plot_i_stp(ax):
    p = source("Fig05_i__Local_dynamic_boundary", "measured_STP_conditioned_forest")
    d = pd.read_csv(p)
    eps = list(dict.fromkeys(d.endpoint))
    y = np.arange(len(eps))
    for definition, off, color in [("primary_shared", -.13, PRIMARY),
                                    ("protocol_conservative", .13, CONSERVATIVE)]:
        g = d[d.definition == definition].set_index("endpoint").reindex(eps)
        ax.hlines(y+off, g.lo, g.hi, color=color, lw=.65)
        ax.scatter(g.beta_after_identity, y+off, color=color, s=6.5, edgecolor="none")
    ax.axvline(0, color=GREY, lw=.55)
    ax.set_yticks(y, [ENDPOINT.get(v, v) for v in eps], fontsize=2.95)
    ax.set_xlabel("Identity-conditioned beta (95% CI)")
    clean_axis(ax, "x")
    return [p]


def plot_i_frequency(ax):
    p = source("Fig05_i__Local_dynamic_boundary", "local_temporal_gain_frequency_boundary")
    d = pd.read_csv(p)
    for definition, color, marker, label in [("primary_shared", PRIMARY, "o", "Primary"),
                                              ("protocol_conservative", CONSERVATIVE, "s", "Conservative")]:
        g = d[d.definition == definition].sort_values("freq")
        y = g.beta_after_identity.to_numpy(float)
        e = 1.96*g.se_after_identity.to_numpy(float)
        ax.errorbar(g.freq, y, yerr=e, color=color, marker=marker, ms=2.2,
                    lw=.8, elinewidth=.55, capsize=1.2, label=label)
    ax.axhline(0, color=GREY, lw=.55)
    ax.set(xlabel="Stimulation frequency (Hz)", ylabel="Identity-conditioned beta")
    ax.set_xticks([5, 10, 20, 50])
    definition_legend(ax, "upper left")
    clean_axis(ax, "y")
    return [p]


def plot_j(ax):
    p = source("Fig05_j__Integrated_local-circuit_adjudication", "narrative_closure_forest")
    d = pd.read_csv(p)
    eps = list(dict.fromkeys(d.endpoint))
    y = np.arange(len(eps))
    for definition, off, color in [("primary_shared", -.13, PRIMARY),
                                    ("protocol_conservative", .13, CONSERVATIVE)]:
        g = d[d.definition == definition].set_index("endpoint").reindex(eps)
        beta = g.beta_after_identity.to_numpy(float)
        err = 1.96*g.se_after_identity.to_numpy(float)
        ax.hlines(y+off, beta-err, beta+err, color=color, lw=.65)
        ax.scatter(beta, y+off, color=color, s=7, edgecolor="none", zorder=3)
    ax.axvline(0, color=GREY, lw=.55)
    ax.set_yticks(y, [ENDPOINT.get(v, v) for v in eps], fontsize=3.25)
    ax.set_xlabel("Identity-conditioned effect (95% CI)")
    definition_legend(ax, "upper left")
    clean_axis(ax, "x")
    return [p]


def build(out: Path):
    fw, fh = 183.0, 132.0
    fig = plt.figure(figsize=(fw/MM_PER_INCH, fh/MM_PER_INCH))
    xs = [10.0, 54.0, 98.0, 142.0]
    ys = [104.0, 72.0, 40.0, 10.5]
    aw, ah = 34.0, 20.5
    sources = []

    sources += plot_a(add_axes_mm(fig, fw, fh, xs[0], ys[0], aw, ah))
    sources += plot_b(add_axes_mm(fig, fw, fh, xs[1], ys[0], aw, ah),
                      add_axes_mm(fig, fw, fh, xs[2], ys[0], aw, ah))
    sources += plot_c(add_axes_mm(fig, fw, fh, xs[3], ys[0], aw, ah))

    sources += plot_d(add_axes_mm(fig, fw, fh, xs[0], ys[1], aw, ah))
    sources += plot_e(add_axes_mm(fig, fw, fh, xs[1], ys[1], aw, ah),
                      add_axes_mm(fig, fw, fh, xs[2], ys[1], aw, ah))
    sources += plot_f(add_axes_mm(fig, fw, fh, xs[3], ys[1], aw, ah))

    sources += plot_mapping(add_axes_mm(fig, fw, fh, xs[0], ys[2], aw, ah),
                            "primary_mapper_oof", "Primary", PRIMARY)
    sources += plot_mapping(add_axes_mm(fig, fw, fh, xs[1], ys[2], aw, ah),
                            "conservative_mapper_oof", "Conservative", CONSERVATIVE)
    sources += plot_h_absorption(add_axes_mm(fig, fw, fh, xs[2], ys[2], aw, ah))
    sources += plot_h_decomp(add_axes_mm(fig, fw, fh, xs[3], ys[2], aw, ah))

    sources += plot_i_generalization(add_axes_mm(fig, fw, fh, xs[0], ys[3], aw, ah))
    sources += plot_i_stp(add_axes_mm(fig, fw, fh, xs[1], ys[3], aw, ah))
    sources += plot_i_frequency(add_axes_mm(fig, fw, fh, xs[2], ys[3], aw, ah))
    sources += plot_j(add_axes_mm(fig, fw, fh, xs[3], ys[3], aw, ah))

    panel_labels = [
        ("a", 3, ys[0]+22), ("b", 47, ys[0]+22), ("c", 135, ys[0]+22),
        ("d", 3, ys[1]+22), ("e", 47, ys[1]+22), ("f", 135, ys[1]+22),
        ("g", 3, ys[2]+22), ("h", 91, ys[2]+22),
        ("i", 3, ys[3]+22), ("j", 135, ys[3]+22),
    ]
    for label, x, y in panel_labels:
        fig.text(x/fw, y/fh, label, ha="left", va="bottom", fontsize=8,
                 fontweight="bold")

    out.mkdir(parents=True, exist_ok=True)
    stem = out / "Fig05_full_nature_trial"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    fig.savefig(stem.with_suffix(".tiff"), dpi=DPI,
                pil_kwargs={"compression": "tiff_lzw"})
    (out / "axis_geometry_mm.json").write_text(json.dumps({
        "figure_size_mm": [fw, fh],
        "grid": "4 x 4",
        "standard_axis_mm": [aw, ah],
        "row_order": ["a+b+b+c", "d+e+e+f", "g+g+h+h", "i+i+i+j"],
        "no_wrapped_panels": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    }, indent=2), encoding="utf-8")
    (out / "source_data_manifest.json").write_text(json.dumps({
        "figure": "Fig05",
        "source_data": sorted({str(Path(p).relative_to(ROOT)) for p in sources}),
    }, indent=2), encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"status": "COMPLETE", "output": str(out)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    setup_style()
    build(args.out)


if __name__ == "__main__":
    main()
