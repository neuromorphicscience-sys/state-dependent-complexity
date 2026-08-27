# Stage 3 — Topology–Dynamics Interaction Atlas + Optimal Complexity Allocation

Merge this package into:

D:\Research\Neural Science

## Core scientific goals

1. Absolute-threshold complexity saving

For a common target rhythm score R*:

S(R*) = 1 - N_targeted(R*) / N_random(R*)

This fixes the Stage 2 "80% of own best" problem.

Thresholds analyzed:
- 0.20
- 0.30
- 0.40
- 0.50

2. Topology × cellular-complexity interaction

Compare the same exact HH counts across:
- ER
- small-world
- scale-free
- modular

3. Placement strategies

- random
- high_degree
- feedback_hub
- module_bridge
- cycle_proxy
- spectral
- gpu_surrogate

4. Interpretable structural features

Each run records the mean selected-node:
- total degree
- feedback score
- 3-cycle participation
- module-bridge score
- spectral centrality

The analysis measures which structural features correlate with placement gain.

5. Full homogeneous endpoints

Every condition contains:
- n_hh = 0
- n_hh = 256

So emergence gain and gain-per-HH are defined everywhere.

## Data safety

Every completed GPU batch is immediately atomically saved to:

results_stage3\chunks\batch_*.csv

Only after durable save is progress.json updated.

On restart, all completed run_ids are scanned from chunk files and skipped.

The chunk directory is the source of truth. runs_stage3.csv is only a consolidated copy.

## Performance

Formal batch_size defaults to 2048, based on the user's successful Stage 2 run.

The simulation timestep and duration remain unchanged:
- dt = 0.05 ms
- duration = 2500 ms
- warmup = 500 ms

No reduced-resolution shortcut is used.

Most recurrent numerical work stays on GPU. CPU topology construction occurs only once
per realization before time integration; there is no CPU↔GPU traffic inside the 50,000-step
recurrent loop except final result transfer.

## Smoke test

powershell -ExecutionPolicy Bypass -File .\run_stage3_smoke.ps1

Then check:

results_stage3_smoke\chunks\

## Formal run

powershell -ExecutionPolicy Bypass -File .\run_stage3.ps1

Expected outputs:

results_stage3\runs_stage3.csv
results_stage3\stage3_parameter_summary.csv
results_stage3\stage3_complexity_saving.csv
results_stage3\stage3_feature_associations.csv
results_stage3\stage3_candidates.csv

## Important

The current `gpu_surrogate` is a first-pass structural surrogate label. This package is designed
to establish the absolute-threshold allocation law and feature associations first. A subsequent
Stage 3B can replace it with true evolutionary/swap-based GPU optimization after we identify the
most informative topology/parameter regions.
