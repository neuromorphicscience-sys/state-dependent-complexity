from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.safeio import atomic_write_csv, consolidate_chunks

outdir = ROOT / "results_stage3"
runs_path = outdir / "runs_stage3.csv"
if not runs_path.exists():
    runs = consolidate_chunks(outdir / "chunks", runs_path)
else:
    runs = pd.read_csv(runs_path)

group_cols = [
    "topology","placement","n_hh","hh_fraction_actual",
    "coupling","connection_prob","noise_sigma"
]

agg = runs.groupby(group_cols, as_index=False).agg(
    rhythm_score_mean=("rhythm_score","mean"),
    rhythm_score_std=("rhythm_score","std"),
    dominant_frequency_hz=("dominant_frequency_hz","mean"),
    spectral_concentration=("spectral_concentration","mean"),
    synchrony_proxy=("synchrony_proxy","mean"),
    mean_rate_hz=("mean_rate_hz","mean"),
    selected_total_degree_mean=("selected_total_degree_mean","mean"),
    selected_feedback_score_mean=("selected_feedback_score_mean","mean"),
    selected_cycle3_mean=("selected_cycle3_mean","mean"),
    selected_module_bridge_mean=("selected_module_bridge_mean","mean"),
    selected_spectral_centrality_mean=("selected_spectral_centrality_mean","mean"),
    n=("run_id","count"),
)

# absolute threshold complexity saving
thresholds = [0.20, 0.30, 0.40, 0.50]
condition_keys = ["topology","coupling","connection_prob","noise_sigma"]

rows = []
for cond, sub in agg.groupby(condition_keys):
    cond_map = dict(zip(condition_keys, cond if isinstance(cond, tuple) else (cond,)))
    placements = [p for p in sub["placement"].unique() if p != "none"]

    def min_n_for(placement, thr):
        q = sub[(sub["placement"] == placement) & (sub["rhythm_score_mean"] >= thr)]
        if len(q) == 0:
            return np.nan
        return int(q["n_hh"].min())

    for thr in thresholds:
        n_rand = min_n_for("random", thr)
        for placement in placements:
            n_target = min_n_for(placement, thr)
            saving = np.nan
            if np.isfinite(n_rand) and n_rand > 0 and np.isfinite(n_target):
                saving = 1.0 - float(n_target)/float(n_rand)

            rows.append({
                **cond_map,
                "threshold": thr,
                "placement": placement,
                "n_hh_random": n_rand,
                "n_hh_targeted": n_target,
                "complexity_saving": saving,
            })

saving_df = pd.DataFrame(rows)

# placement gain at exact n_hh
base_keys = ["topology","n_hh","coupling","connection_prob","noise_sigma"]
rnd = (
    agg[agg["placement"] == "random"]
    [base_keys + ["rhythm_score_mean"]]
    .rename(columns={"rhythm_score_mean":"random_score"})
)
agg = agg.merge(rnd, on=base_keys, how="left")
agg["placement_gain_vs_random"] = agg["rhythm_score_mean"] - agg["random_score"]
agg["placement_ratio_vs_random"] = agg["rhythm_score_mean"]/(agg["random_score"]+1e-9)

# endpoints for emergence/efficiency
zero_keys = ["topology","coupling","connection_prob","noise_sigma"]
lif = (
    agg[agg["n_hh"] == 0]
    .groupby(zero_keys, as_index=False)["rhythm_score_mean"].mean()
    .rename(columns={"rhythm_score_mean":"lif_score"})
)
hh = (
    agg[agg["n_hh"] == 256]
    .groupby(zero_keys, as_index=False)["rhythm_score_mean"].mean()
    .rename(columns={"rhythm_score_mean":"hh_score"})
)
agg = agg.merge(lif, on=zero_keys, how="left").merge(hh, on=zero_keys, how="left")

p = agg["hh_fraction_actual"]
agg["linear_expectation"] = (1-p)*agg["lif_score"] + p*agg["hh_score"]
agg["emergence_gain"] = agg["rhythm_score_mean"] - agg["linear_expectation"]
agg["gain_per_hh_neuron"] = np.where(
    agg["n_hh"] > 0,
    (agg["rhythm_score_mean"]-agg["lif_score"])/agg["n_hh"],
    np.nan,
)

# Interpretable feature association
feature_cols = [
    "selected_total_degree_mean",
    "selected_feedback_score_mean",
    "selected_cycle3_mean",
    "selected_module_bridge_mean",
    "selected_spectral_centrality_mean",
]
assoc_rows = []
valid = agg[(agg["n_hh"] > 0) & agg["placement_gain_vs_random"].notna()].copy()
for topo, sub in valid.groupby("topology"):
    for feat in feature_cols:
        if sub[feat].std() > 0:
            corr = sub[[feat,"placement_gain_vs_random"]].corr().iloc[0,1]
        else:
            corr = np.nan
        assoc_rows.append({
            "topology": topo,
            "feature": feat,
            "pearson_r_with_placement_gain": corr,
        })
assoc = pd.DataFrame(assoc_rows)

atomic_write_csv(outdir / "stage3_parameter_summary.csv", agg)
atomic_write_csv(outdir / "stage3_complexity_saving.csv", saving_df)
atomic_write_csv(outdir / "stage3_feature_associations.csv", assoc)

cand = agg[
    (agg["n_hh"] > 0)
    & (agg["placement"] != "none")
].sort_values(
    ["placement_gain_vs_random","emergence_gain"],
    ascending=False
)
atomic_write_csv(outdir / "stage3_candidates.csv", cand)

print("\nTop absolute-threshold complexity savings:")
print(
    saving_df.dropna(subset=["complexity_saving"])
    .sort_values("complexity_saving", ascending=False)
    .head(30)
    .to_string(index=False)
)

print("\nTop placement gains:")
cols = [
    "topology","placement","n_hh","hh_fraction_actual",
    "coupling","connection_prob","noise_sigma",
    "rhythm_score_mean","random_score","placement_gain_vs_random",
    "emergence_gain","gain_per_hh_neuron","n"
]
print(cand.head(30)[cols].to_string(index=False))

print("\nFeature associations:")
print(assoc.sort_values("pearson_r_with_placement_gain", ascending=False).to_string(index=False))

print("\nSaved:")
for f in [
    "stage3_parameter_summary.csv",
    "stage3_complexity_saving.csv",
    "stage3_feature_associations.csv",
    "stage3_candidates.csv",
]:
    print(outdir / f)
