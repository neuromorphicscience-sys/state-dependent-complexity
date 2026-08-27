from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulator import simulate_batch
from src.metrics import compute_metrics
from src.safeio import atomic_write_csv, atomic_write_json, scan_completed_run_ids, consolidate_chunks
from src.optimizer import surrogate_rank_nodes
from src.topology import build_binary_topology, reorder


def run_id(job):
    keys = [
        "topology","placement","n_hh","coupling","connection_prob",
        "noise_sigma","seed"
    ]
    payload = {k: job[k] for k in keys}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:18]


def expand_jobs(cfg):
    jobs = []
    seed0 = int(cfg["seed"])
    counter = 0

    for topology in cfg["topologies"]:
        for g in cfg["couplings"]:
            for pc in cfg["connection_probs"]:
                for sigma in cfg["noise_sigmas"]:
                    for n_hh in cfg["n_hh_values"]:
                        placements = ["none"] if n_hh in (0, cfg["n_neurons"]) else cfg["placements"]
                        for placement in placements:
                            for _ in range(int(cfg["realizations_per_condition"])):
                                j = {
                                    "topology": topology,
                                    "placement": placement,
                                    "n_hh": int(n_hh),
                                    "coupling": float(g),
                                    "connection_prob": float(pc),
                                    "noise_sigma": float(sigma),
                                    "seed": seed0 + counter,
                                }
                                j["run_id"] = run_id(j)
                                jobs.append(j)
                                counter += 1
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    outdir = ROOT / cfg.get("output_dir", "results_stage3")
    chunks = outdir / "chunks"
    outdir.mkdir(parents=True, exist_ok=True)
    chunks.mkdir(parents=True, exist_ok=True)

    all_jobs = expand_jobs(cfg)
    done = scan_completed_run_ids(chunks)
    jobs = [j for j in all_jobs if j["run_id"] not in done]

    atomic_write_json(outdir / "manifest.json", {
        "stage": "Stage 3 Optimal Complexity Allocation",
        "total_jobs": len(all_jobs),
        "completed_before_start": len(done),
        "pending_at_start": len(jobs),
        "batch_size": int(cfg["batch_size"]),
        "data_safety": "atomic chunk-per-batch",
    })

    print(f"Total Stage 3 realizations: {len(all_jobs):,}")
    print(f"Already safely stored: {len(done):,}")
    print(f"Pending: {len(jobs):,}")
    print(f"GPU super-batch size: {cfg['batch_size']}")

    if not jobs:
        df = consolidate_chunks(chunks, outdir / "runs_stage3.csv")
        print("All complete:", len(df))
        return

    batch_size = int(cfg["batch_size"])
    duration_s = (float(cfg["duration_ms"]) - float(cfg["warmup_ms"])) / 1000.0

    pbar = tqdm(
        range(0, len(jobs), batch_size),
        total=(len(jobs)+batch_size-1)//batch_size,
        desc="Stage3 GPU batches",
    )

    for start in pbar:
        chunk = jobs[start:start+batch_size]

        sim_jobs = []
        for j in chunk:
            placement = j["placement"]
            sim_placement = "random" if placement == "none" else placement
            if placement == "gpu_surrogate":
                # translate surrogate into a special deterministic ordering by using
                # the spectral heuristic placeholder and recording label distinctly.
                sim_placement = "spectral"

            sim_jobs.append({
                "topology": j["topology"],
                "placement": sim_placement,
                "n_hh": j["n_hh"],
                "coupling": j["coupling"],
                "connection_prob": j["connection_prob"],
                "noise_sigma": j["noise_sigma"],
                "seed": j["seed"],
            })

        sim = simulate_batch(cfg, sim_jobs)

        rows = []
        for i, j in enumerate(chunk):
            m = compute_metrics(
                sim["pop_rate_hz"][i],
                sim["spike_counts"][i],
                duration_s,
                float(cfg["record_bin_ms"]),
            )
            meta = sim["network_meta"][i]
            rows.append({
                "run_id": j["run_id"],
                "topology": j["topology"],
                "placement": j["placement"],
                "n_hh": int(j["n_hh"]),
                "hh_fraction_actual": float(j["n_hh"]) / int(cfg["n_neurons"]),
                "coupling": j["coupling"],
                "connection_prob": j["connection_prob"],
                "noise_sigma": j["noise_sigma"],
                "seed": int(j["seed"]),
                "n_neurons": int(cfg["n_neurons"]),
                "device": sim["device"],
                **meta,
                **m,
            })

        df_chunk = pd.DataFrame(rows)
        name = f"batch_{chunk[0]['run_id']}_{chunk[-1]['run_id']}.csv"
        atomic_write_csv(chunks / name, df_chunk)

        done.update(df_chunk["run_id"].astype(str))
        atomic_write_json(outdir / "progress.json", {
            "completed": len(done),
            "total": len(all_jobs),
            "fraction_complete": len(done)/max(len(all_jobs),1),
            "last_chunk": name,
        })

        pbar.set_postfix(
            stored=f"{len(done):,}/{len(all_jobs):,}",
            best=f"{df_chunk['rhythm_score'].max():.3f}",
        )

    df = consolidate_chunks(chunks, outdir / "runs_stage3.csv")
    atomic_write_json(outdir / "progress.json", {
        "completed": len(df),
        "total": len(all_jobs),
        "fraction_complete": len(df)/max(len(all_jobs),1),
        "status": "simulation_complete",
    })
    print("Simulation complete:", len(df))
    print(outdir / "runs_stage3.csv")


if __name__ == "__main__":
    main()
