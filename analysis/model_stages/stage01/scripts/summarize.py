from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "results" / "runs.csv"
if not path.exists():
    raise SystemExit("results/runs.csv not found")

df = pd.read_csv(path)

cols = ["hh_fraction", "coupling", "connection_prob", "noise_sigma"]
agg = df.groupby(cols).agg(
    rhythm_score_mean=("rhythm_score", "mean"),
    rhythm_score_std=("rhythm_score", "std"),
    mean_rate_hz=("mean_rate_hz", "mean"),
    dominant_frequency_hz=("dominant_frequency_hz", "mean"),
    spectral_concentration=("spectral_concentration", "mean"),
    silent_fraction=("silent_fraction", "mean"),
    n=("run_id", "count"),
).reset_index()

agg = agg.sort_values("rhythm_score_mean", ascending=False)
out = ROOT / "results" / "summary_by_parameter.csv"
agg.to_csv(out, index=False)

print("\nTop 20 parameter regimes:")
print(agg.head(20).to_string(index=False))
print(f"\nSaved: {out}")
