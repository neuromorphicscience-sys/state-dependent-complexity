# Stage 2 — Topology & Minimal Complexity Discovery

Copy/merge this package into:

D:\Research\Neural Science

Stage 1A/1B results are untouched. Stage 2 writes only to:

results_stage2\
results_stage2_smoke\

## Scientific question

Does the location of biophysically complex HH neurons matter more than their abundance?

Stage 2 varies:
- topology: ER / small-world / scale-free / modular
- exact HH neuron count (not rounded nominal fractions)
- placement strategy:
  - random
  - high_degree
  - feedback_hub
  - module_bridge
  - cycle_proxy
  - coverage_greedy

Primary outputs:
- placement_gain_vs_random
- placement_ratio_vs_random
- gain_per_hh_neuron
- minimum_n_hh_for_80pct_best

## DATA SAFETY

This stage does NOT wait until the end to save results.

Each completed GPU batch is written immediately to:
results_stage2\chunks\batch_<first_runid>_<last_runid>.csv

The write is atomic:
1. write temporary file
2. flush + fsync
3. os.replace into final chunk filename

Then progress.json is atomically updated.

On restart, the runner scans all valid chunk files and skips every stored run_id.
A failed final analysis therefore cannot erase simulation results.

At the end, chunk files are consolidated to:
results_stage2\runs_stage2.csv

Even if consolidation or analysis fails, the chunk files remain the source of truth.

## First run smoke test

powershell -ExecutionPolicy Bypass -File .\run_stage2_smoke.ps1

Check that:
results_stage2_smoke\chunks\
contains batch CSV files.

Then run the formal stage:

powershell -ExecutionPolicy Bypass -File .\run_stage2.ps1

## Batch size

Default formal batch_size = 256 because topology construction is more memory/CPU intensive
than Stage 1B. If GPU memory is comfortably below ~12 GB and the smoke test is stable,
try 512. Do not jump directly to 1024 before confirming throughput.

## Important methodological note

Selected HH nodes are reordered to the prefix of each adjacency matrix before GPU integration.
This does NOT change graph structure; the adjacency matrix is permuted consistently. It enables
the simulator to retain HH-only gating kinetics while supporting arbitrary placement strategies.
