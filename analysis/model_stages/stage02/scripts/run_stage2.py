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
from src.metrics import compute_metrics
from src.safeio import (
    atomic_write_csv,
    atomic_write_json,
    scan_completed_run_ids,
    consolidate_chunks,
)


def make_run_id(job):
    keys = [
        "regime","topology","placement","n_hh","coupling",
        "connection_prob","noise_sigma","seed"
    ]
    payload = {k: job[k] for k in keys}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:18]


def expand_jobs(cfg):
    jobs = []
    seed0 = int(cfg["seed"])
    counter = 0
    reps = int(cfg["realizations_per_condition"])
    placements = cfg["placements"]
    topologies = cfg["topologies"]

    for regime_name, spec in cfg["regimes"].items():
        for g in spec["couplings"]:
            for pc in spec["connection_probs"]:
                for sigma in spec["noise_sigmas"]:
                    for topology in topologies:
                        for n_hh in spec["n_hh_values"]:
                            use_placements = ["none"] if n_hh in (0, cfg["n_neurons"]) else placements
                            for placement in use_placements:
                                for _ in range(reps):
                                    job = {
                                        "regime": regime_name,
                                        "topology": topology,
                                        "placement": placement,
                                        "n_hh": int(n_hh),
                                        "coupling": float(g),
                                        "connection_prob": float(pc),
                                        "noise_sigma": float(sigma),
                                        "seed": seed0 + counter,
                                    }
                                    if placement == "none":
                                        # Simulator still requires a valid strategy for all-HH/all-LIF.
                                        job["placement_sim"] = "random"
                                    else:
                                        job["placement_sim"] = placement
                                    job["run_id"] = make_run_id(job)
                                    jobs.append(job)
                                    counter += 1
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    outdir = ROOT / cfg.get("output_dir", "results_stage2")
    chunks_dir = outdir / "chunks"
    outdir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_jobs = expand_jobs(cfg)
    done = scan_completed_run_ids(chunks_dir)
    jobs = [j for j in all_jobs if j["run_id"] not in done]

    manifest = {
        "stage": "Stage 2 Topology & Minimal Complexity Discovery",
        "config": str(cfg_path),
        "total_jobs": len(all_jobs),
        "completed_before_start": len(done),
        "pending_at_start": len(jobs),
        "batch_size": int(cfg["batch_size"]),
    }
    atomic_write_json(outdir / "manifest.json", manifest)

    print(f"Total Stage 2 realizations: {len(all_jobs):,}")
    print(f"Already safely stored: {len(done):,}")
    print(f"Pending: {len(jobs):,}")
    print(f"GPU super-batch size: {cfg['batch_size']}")
    print("Persistence mode: atomic chunk-per-batch + manifest")

    if not jobs:
        df = consolidate_chunks(chunks_dir, outdir / "runs_stage2.csv")
        print(f"All complete. Consolidated rows: {len(df):,}")
        return

    batch_size = int(cfg["batch_size"])
    duration_s = (float(cfg["duration_ms"]) - float(cfg["warmup_ms"])) / 1000.0
    batch_counter = 0

    pbar = tqdm(
        range(0, len(jobs), batch_size),
        total=(len(jobs) + batch_size - 1) // batch_size,
        desc="Stage2 GPU batches",
    )

    for start in pbar:
        chunk = jobs[start:start + batch_size]

        sim_jobs = []
        for j in chunk:
            sim_jobs.append({
                "topology": j["topology"],
                "placement": j["placement_sim"],
                "n_hh": j["n_hh"],
                "coupling": j["coupling"],
                "connection_prob": j["connection_prob"],
                "noise_sigma": j["noise_sigma"],
                "seed": j["seed"],
            })

        sim = simulate_batch(cfg, sim_jobs)

        rows = []
        for i, job in enumerate(chunk):
            m = compute_metrics(
                sim["pop_rate_hz"][i],
                sim["spike_counts"][i],
                duration_s,
                float(cfg["record_bin_ms"]),
            )
            meta = sim["network_meta"][i]
            rows.append({
                "run_id": job["run_id"],
                "regime": job["regime"],
                "topology": job["topology"],
                "placement": job["placement"],
                "n_hh": int(job["n_hh"]),
                "hh_fraction_actual": float(job["n_hh"]) / int(cfg["n_neurons"]),
                "coupling": float(job["coupling"]),
                "connection_prob": float(job["connection_prob"]),
                "noise_sigma": float(job["noise_sigma"]),
                "seed": int(job["seed"]),
                "n_neurons": int(cfg["n_neurons"]),
                "actual_density": meta["actual_density"],
                "mean_in_degree": meta["mean_in_degree"],
                "mean_out_degree": meta["mean_out_degree"],
                "device": sim["device"],
                **m,
            })

        df_chunk = pd.DataFrame(rows)

        # Unique file name derived from first/last run IDs, safe under resume.
        chunk_name = f"batch_{chunk[0]['run_id']}_{chunk[-1]['run_id']}.csv"
        atomic_write_csv(chunks_dir / chunk_name, df_chunk)

        # Only after the chunk is durably written do we update progress.
        done.update(df_chunk["run_id"].astype(str))
        batch_counter += 1
        atomic_write_json(outdir / "progress.json", {
            "completed_run_ids_count": len(done),
            "total_jobs": len(all_jobs),
            "fraction_complete": len(done) / max(len(all_jobs), 1),
            "last_chunk": chunk_name,
            "batches_written_this_session": batch_counter,
        })

        pbar.set_postfix(
            stored=f"{len(done):,}/{len(all_jobs):,}",
            best=f"{df_chunk['rhythm_score'].max():.3f}",
        )

    df = consolidate_chunks(chunks_dir, outdir / "runs_stage2.csv")
    atomic_write_json(outdir / "progress.json", {
        "completed_run_ids_count": len(df),
        "total_jobs": len(all_jobs),
        "fraction_complete": len(df) / max(len(all_jobs), 1),
        "status": "simulation_complete",
    })

    print(f"\nSimulation complete and consolidated: {len(df):,} rows")
    print(outdir / "runs_stage2.csv")
    print("Now run: python scripts\\analyze_stage2.py")


if __name__ == "__main__":
    main()
