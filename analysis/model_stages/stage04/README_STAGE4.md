# Stage 4 — GPU Optimal Cellular-Complexity Allocation

Target project root:

D:\Research\Neural Science

This stage is the transition from phenomenon discovery to principle discovery.

## Scientific question

For a fixed network G and fixed number K of complex (HH) neurons,

V_HH* = argmax R(V_HH)

where R is the collective rhythm score.

HH and LIF remain the only neuron models:
- LIF = low-dimensional phenomenological cell
- HH = biophysically resolved complex cell

No additional neuron model is introduced.

## What Stage 4 fixes from Stage 3

### 1. `gpu_surrogate` is removed

Stage 3's `gpu_surrogate` label was not a true optimizer. Stage 4 performs a real
population-based GPU evolutionary search over HH-node sets.

### 2. Placement comparisons are paired on the exact same graph

All random, heuristic, and optimized placements for a task use the same `graph_seed`
and the same adjacency matrix. Placement is now the only structural variable.

### 3. Small-world and scale-free directed densities are corrected

Stage 3 oriented undirected graphs after constructing them at target mean degree,
which reduced directed edge density by roughly one half. Stage 4 constructs the
undirected precursor at approximately twice the desired degree so that directed
density after orientation matches the requested connection probability.

### 4. Fair stochastic comparison

Within each optimization generation all candidate placements receive the same
noise trace (`common_noise=True`). The trace changes between generations.
Final comparisons are repeated over multiple common-noise validation seeds.

### 5. No node reordering

Stage 4 uses arbitrary HH masks directly on GPU. Node identity is preserved, which
is important for paired placement/noise comparisons.

## GPU design

A single adjacency matrix is kept on GPU for an entire optimization task.

For a population of P placements:
- masks: P x N
- states: P x N
- shared A: N x N

The graph is NOT duplicated P times.

All 50,000 recurrent integration steps remain on GPU.
Population mutation, elite selection, and candidate generation are also on GPU.

Final metrics use GPU FFT and stay on GPU until summary values are written.

The numerical resolution remains unchanged:
- dt = 0.05 ms
- duration = 2500 ms
- warmup = 500 ms

## Search

Formal default:
- population = 256
- elite = 32
- generations = 6
- random injection = 32
- 2 swap mutations per offspring

Each population is initialized with heuristic solutions plus random solutions.

Heuristics retained only as baselines:
- high degree
- feedback hub
- module bridge
- cycle proxy
- spectral

## Formal regimes

Three Stage-3-informed dynamical regions are used:
1. transition_dense: g=0.035, p=0.10, sigma=1.0
2. transition_mid:   g=0.035, p=0.05, sigma=1.0
3. sparse_drive:     g=0.05,  p=0.05, sigma=1.0

Four topologies:
ER / small-world / scale-free / modular

K:
8, 16, 24, 32, 48, 64 HH cells

Two discovery graph seeds are used for every topology/regime/K.

## Data safety

Every optimization task has:

results_stage4\checkpoints\<task_id>.pt

The checkpoint is atomically replaced after every generation and contains:
- current population
- best mask
- best score
- generation number
- optimization history

Completed task comparisons are atomically saved to:

results_stage4\tasks\<task_id>.csv

Restarting `run_stage4.ps1` automatically skips completed tasks and resumes
unfinished tasks from their latest generation checkpoint.

## Stage 4 outputs

After discovery + analysis:

results_stage4\stage4_discovery_methods.csv
results_stage4\stage4_optimization_summary.csv
results_stage4\stage4_complexity_saving.csv
results_stage4\stage4_allocation_law_coefficients.csv
results_stage4\stage4_allocation_law.json

The learned law uses set-level descriptors, not just single-node centrality:
- selected degree / feedback / cycle / bridge / spectral features
- one-step and two-step coverage
- outgoing-neighborhood redundancy
- selected-node dispersion
- graph density / degree heterogeneity

## Prospective held-out prediction

After the discovery law is fitted, `run_stage4_heldout.py` constructs completely
new graphs using `heldout_graph_seeds`.

It does NOT perform evolutionary optimization to choose the predicted set.

Instead:
1. generate a candidate library,
2. score candidate sets using the learned allocation law,
3. select the predicted set,
4. only then run the HH/LIF simulation,
5. compare against random and hand-crafted heuristics.

This gives a genuine:
discovery -> interpretable law -> prospective prediction

test.

## Smoke test

From the project root:

powershell -ExecutionPolicy Bypass -File .\run_stage4_smoke.ps1

Expected:
results_stage4_smoke\tasks\<task_id>.csv

The smoke test intentionally uses a short duration only to check the software path.
Formal Stage 4 never uses this shortened duration.

## Formal run

powershell -ExecutionPolicy Bypass -File .\run_stage4.ps1

## Go / No-Go criteria

A. Optimization:
optimized > best heuristic on a substantial fraction of scale-free tasks.

B. Complexity economy:
S_opt(R*) >= 0.5 across broad regions; 0.7-0.9 would be especially strong.

C. Set-level law:
optimized sets show reproducible set-level structural regularities.

D. Prospective prediction:
the learned law beats random on held-out graphs and approaches the best heuristic
or exceeds it.

If A-D all pass, Stage 4 completes the core computational backbone needed before
moving to Stage 5 scaling/general-law work.
