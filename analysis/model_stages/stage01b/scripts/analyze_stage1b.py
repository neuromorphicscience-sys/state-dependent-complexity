from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
outdir = ROOT / "results_stage1b"
runs = outdir / "runs_stage1b.csv"

if not runs.exists():
    raise SystemExit("results_stage1b/runs_stage1b.csv not found")

df = pd.read_csv(runs)

from src.search import (
    aggregate_parameter_points,
    attach_baselines_and_emergence,
    detect_transitions,
    build_candidates,
)

agg = aggregate_parameter_points(df)
agg = attach_baselines_and_emergence(agg)
agg = detect_transitions(agg)
cand = build_candidates(agg)

agg.to_csv(outdir / "stage1b_parameter_summary.csv", index=False)
cand.to_csv(outdir / "stage1b_candidates.csv", index=False)

# Frequency-state composition
state = (
    df.groupby(
        ["window", "hh_fraction", "frequency_state"],
        as_index=False
    )
    .size()
)
state["fraction"] = state.groupby(
    ["window", "hh_fraction"]
)["size"].transform(lambda x: x / x.sum())
state.to_csv(outdir / "frequency_state_composition.csv", index=False)

print("\nTop positive-emergence regimes:")
cols = [
    "window","hh_fraction","coupling","connection_prob","noise_sigma",
    "rhythm_score_mean","lif_baseline_score","hh_baseline_score",
    "linear_mixture_expectation","emergence_gain","complexity_efficiency",
    "dominant_frequency_hz","frequency_transition_strength","n"
]
print(
    cand.sort_values("emergence_gain", ascending=False)
    .head(20)[cols]
    .to_string(index=False)
)

print("\nTop sparse-complexity efficiency regimes:")
print(
    cand[cand["hh_fraction"].between(0.0001, 0.05)]
    .sort_values("complexity_efficiency", ascending=False)
    .head(20)[cols]
    .to_string(index=False)
)

print("\nSaved:")
print(outdir / "stage1b_parameter_summary.csv")
print(outdir / "stage1b_candidates.csv")
print(outdir / "frequency_state_composition.csv")
