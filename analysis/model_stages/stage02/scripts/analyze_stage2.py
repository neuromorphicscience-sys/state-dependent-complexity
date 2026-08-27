from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.safeio import atomic_write_csv, consolidate_chunks

outdir = ROOT / "results_stage2"
runs_path = outdir / "runs_stage2.csv"
if not runs_path.exists():
    runs = consolidate_chunks(outdir / "chunks", runs_path)
else:
    runs = pd.read_csv(runs_path)

group_cols = [
    "regime","topology","placement","n_hh","hh_fraction_actual",
    "coupling","connection_prob","noise_sigma"
]

agg = runs.groupby(group_cols, as_index=False).agg(
    rhythm_score_mean=("rhythm_score","mean"),
    rhythm_score_std=("rhythm_score","std"),
    dominant_frequency_hz=("dominant_frequency_hz","mean"),
    spectral_concentration=("spectral_concentration","mean"),
    synchrony_proxy=("synchrony_proxy","mean"),
    mean_rate_hz=("mean_rate_hz","mean"),
    actual_density=("actual_density","mean"),
    n=("run_id","count"),
)

# Random-placement baseline under identical structural/biophysical condition.
base_keys = [
    "regime","topology","n_hh","coupling","connection_prob","noise_sigma"
]
rnd = (
    agg[agg["placement"] == "random"]
    [base_keys + ["rhythm_score_mean"]]
    .rename(columns={"rhythm_score_mean":"random_placement_score"})
)

agg = agg.merge(rnd, on=base_keys, how="left")
agg["placement_gain_vs_random"] = (
    agg["rhythm_score_mean"] - agg["random_placement_score"]
)
agg["placement_ratio_vs_random"] = (
    agg["rhythm_score_mean"] / (agg["random_placement_score"] + 1e-9)
)

# Complexity efficiency relative to n_hh=0 within same topology & biophysical condition.
zero_keys = ["regime","topology","coupling","connection_prob","noise_sigma"]
zero = (
    agg[agg["n_hh"] == 0]
    .groupby(zero_keys, as_index=False)["rhythm_score_mean"]
    .mean()
    .rename(columns={"rhythm_score_mean":"lif_only_score"})
)
agg = agg.merge(zero, on=zero_keys, how="left")
agg["gain_over_lif"] = agg["rhythm_score_mean"] - agg["lif_only_score"]
agg["gain_per_hh_neuron"] = np.where(
    agg["n_hh"] > 0,
    agg["gain_over_lif"] / agg["n_hh"],
    np.nan,
)

# For each condition, minimum n_hh reaching 80% of that condition's best score.
best_keys = ["regime","topology","placement","coupling","connection_prob","noise_sigma"]
best = (
    agg.groupby(best_keys, as_index=False)["rhythm_score_mean"]
    .max()
    .rename(columns={"rhythm_score_mean":"condition_best_score"})
)
agg = agg.merge(best, on=best_keys, how="left")
agg["fraction_of_condition_best"] = (
    agg["rhythm_score_mean"] / (agg["condition_best_score"] + 1e-9)
)

min_rows = []
for key, sub in agg.groupby(best_keys):
    ok = sub[(sub["n_hh"] > 0) & (sub["fraction_of_condition_best"] >= 0.80)]
    if len(ok):
        r = ok.sort_values("n_hh").iloc[0]
        row = dict(zip(best_keys, key if isinstance(key, tuple) else (key,)))
        row.update({
            "minimum_n_hh_for_80pct_best": int(r["n_hh"]),
            "minimum_fraction_for_80pct_best": float(r["hh_fraction_actual"]),
            "score_at_minimum": float(r["rhythm_score_mean"]),
            "condition_best_score": float(r["condition_best_score"]),
        })
        min_rows.append(row)
minimal = pd.DataFrame(min_rows)

atomic_write_csv(outdir / "stage2_parameter_summary.csv", agg)
atomic_write_csv(outdir / "stage2_minimal_complexity.csv", minimal)

candidates = agg[
    (agg["n_hh"] > 0) &
    (
        (agg["placement_gain_vs_random"] > 0) |
        (agg["gain_per_hh_neuron"] > 0)
    )
].copy()

candidates["candidate_score"] = (
    candidates["placement_gain_vs_random"].fillna(0)
    + 0.25 * candidates["gain_over_lif"].fillna(0)
    + 8.0 * candidates["gain_per_hh_neuron"].fillna(0)
)
candidates = candidates.sort_values("candidate_score", ascending=False)
atomic_write_csv(outdir / "stage2_candidates.csv", candidates)

print("\nTop placement gains vs random:")
cols = [
    "regime","topology","placement","n_hh","hh_fraction_actual",
    "coupling","connection_prob","noise_sigma",
    "rhythm_score_mean","random_placement_score",
    "placement_gain_vs_random","placement_ratio_vs_random",
    "gain_per_hh_neuron","n"
]
print(
    agg[agg["placement"].isin(["high_degree","feedback_hub","module_bridge","cycle_proxy","coverage_greedy"])]
    .sort_values("placement_gain_vs_random", ascending=False)
    .head(25)[cols].to_string(index=False)
)

print("\nMinimal complexity examples:")
if len(minimal):
    print(minimal.sort_values("minimum_n_hh_for_80pct_best").head(25).to_string(index=False))

print("\nSaved:")
print(outdir / "stage2_parameter_summary.csv")
print(outdir / "stage2_minimal_complexity.csv")
print(outdir / "stage2_candidates.csv")
