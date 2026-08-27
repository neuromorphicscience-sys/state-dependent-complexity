from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import simulate_batch
from src.metrics import compute_metrics, classify_frequency_state
from src.search import (
    save_incremental,
    aggregate_parameter_points,
    attach_baselines_and_emergence,
    detect_transitions,
    build_candidates,
)


def run_id(job):
    payload = {
        "window": job["window"],
        "hh_fraction": float(job["hh_fraction"]),
        "coupling": float(job["coupling"]),
        "connection_prob": float(job["connection_prob"]),
        "noise_sigma": float(job["noise_sigma"]),
        "seed": int(job["seed"]),
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def expand_jobs(cfg):
    jobs = []
    seed = int(cfg.get("seed", 20260817))
    counter = 0
    R = int(cfg["realizations_per_point"])

    for window_name, spec in cfg["windows"].items():
        for p_hh in spec["hh_fractions"]:
            for g in spec["couplings"]:
                for pc in spec["connection_probs"]:
                    for sigma in spec["noise_sigmas"]:
                        for _ in range(R):
                            job = {
                                "window": window_name,
                                "hh_fraction": float(p_hh),
                                "coupling": float(g),
                                "connection_prob": float(pc),
                                "noise_sigma": float(sigma),
                                "seed": seed + counter,
                            }
                            job["run_id"] = run_id(job)
                            jobs.append(job)
                            counter += 1

    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    outdir = ROOT / cfg.get("output_dir", "results_stage1b")
    outdir.mkdir(parents=True, exist_ok=True)
    runs_path = outdir / "runs_stage1b.csv"

    done = set()
    if runs_path.exists():
        old = pd.read_csv(runs_path)
        if "run_id" in old.columns:
            done = set(old["run_id"].astype(str))

    all_jobs = expand_jobs(cfg)
    jobs = [j for j in all_jobs if j["run_id"] not in done]

    print(f"Total Stage 1B realizations: {len(all_jobs):,}")
    print(f"Already completed: {len(done):,}")
    print(f"Pending: {len(jobs):,}")
    print(f"GPU super-batch size: {cfg['batch_size']}")

    batch_size = int(cfg["batch_size"])
    pending_rows = []

    pbar = tqdm(
        range(0, len(jobs), batch_size),
        total=(len(jobs) + batch_size - 1) // batch_size,
        desc="Stage1B GPU batches",
    )

    for start in pbar:
        chunk = jobs[start:start + batch_size]

        sim_jobs = [
            {
                "hh_fraction": j["hh_fraction"],
                "coupling": j["coupling"],
                "connection_prob": j["connection_prob"],
                "noise_sigma": j["noise_sigma"],
                "seed": j["seed"],
            }
            for j in chunk
        ]

        sim = simulate_batch(cfg=cfg, jobs=sim_jobs)
        duration_s = (
            float(cfg["duration_ms"]) - float(cfg["warmup_ms"])
        ) / 1000.0

        rows = []
        for i, job in enumerate(chunk):
            m = compute_metrics(
                sim["pop_rate_hz"][i],
                sim["spike_counts"][i],
                duration_s,
                float(cfg["record_bin_ms"]),
            )

            freq_state = classify_frequency_state(
                m["dominant_frequency_hz"],
                m["rhythm_score"],
                m["spectral_concentration"],
            )

            rows.append({
                "run_id": job["run_id"],
                "window": job["window"],
                "seed": int(job["seed"]),
                "hh_fraction": float(job["hh_fraction"]),
                "coupling": float(job["coupling"]),
                "connection_prob": float(job["connection_prob"]),
                "noise_sigma": float(job["noise_sigma"]),
                "n_neurons": int(cfg["n_neurons"]),
                "hh_count": int(sim["hh_count"][i]),
                "device": sim["device"],
                "frequency_state": freq_state,
                **m,
            })

        pending_rows.extend(rows)

        if len(pending_rows) >= max(batch_size, 128):
            df = save_incremental(pending_rows, str(outdir))
            pending_rows.clear()
            best = df["rhythm_score"].max()
            pbar.set_postfix(best=f"{best:.3f}")

    if pending_rows:
        save_incremental(pending_rows, str(outdir))

    print("\nSimulation complete.")
    print("Run analysis with:")
    print(r"python scripts\analyze_stage1b.py")


if __name__ == "__main__":
    main()
