from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import simulate_batch
from src.metrics import compute_metrics
from src.search import parameter_grid, save_incremental

def run_id(params, seed):
    s = json.dumps({**params, "seed": int(seed)}, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    outdir = ROOT / cfg.get("output_dir", "results")
    outdir.mkdir(parents=True, exist_ok=True)

    runs_path = outdir / "runs.csv"
    done = set()
    if runs_path.exists():
        try:
            old = pd.read_csv(runs_path)
            if "run_id" in old.columns:
                done = set(old["run_id"].astype(str))
        except Exception:
            pass

    base_seed = int(cfg.get("seed", 0))
    batch_size = int(cfg["batch_size"])
    R = int(cfg["realizations_per_point"])

    grid = list(parameter_grid(cfg))
    pbar = tqdm(grid, desc="parameter points")

    global_counter = 0
    pending_rows = []

    for params in pbar:
        seeds = [base_seed + global_counter + i for i in range(R)]
        global_counter += R

        # Skip completed individual runs
        todo = [(s, run_id(params, s)) for s in seeds if run_id(params, s) not in done]
        if not todo:
            continue

        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start+batch_size]
            chunk_seeds = [x[0] for x in chunk]
            chunk_ids = [x[1] for x in chunk]

            sim = simulate_batch(
                cfg=cfg,
                hh_fraction=params["hh_fraction"],
                coupling=params["coupling"],
                connection_prob=params["connection_prob"],
                noise_sigma=params["noise_sigma"],
                seeds=chunk_seeds,
            )

            analysis_duration_s = (cfg["duration_ms"] - cfg["warmup_ms"]) / 1000.0

            rows = []
            for i, seed in enumerate(chunk_seeds):
                metrics = compute_metrics(
                    sim["pop_rate_hz"][i],
                    sim["spike_counts"][i],
                    analysis_duration_s,
                    cfg["record_bin_ms"],
                )
                row = {
                    "run_id": chunk_ids[i],
                    "seed": int(seed),
                    **params,
                    "n_neurons": int(cfg["n_neurons"]),
                    "hh_count": int(sim["hh_count"][i]),
                    "device": sim["device"],
                    **metrics,
                }
                rows.append(row)
                done.add(chunk_ids[i])

            pending_rows.extend(rows)
            if len(pending_rows) >= 64:
                df, cand = save_incremental(pending_rows, str(outdir))
                pending_rows.clear()
                best = cand.iloc[0] if len(cand) else None
                if best is not None:
                    pbar.set_postfix(
                        best_score=f"{best['rhythm_score']:.3g}",
                        best_hh=f"{best['hh_fraction']:.3g}"
                    )

    if pending_rows:
        save_incremental(pending_rows, str(outdir))

    print(f"\nDone. Results: {runs_path}")
    print(f"Candidates: {outdir / 'candidates.csv'}")

if __name__ == "__main__":
    main()
