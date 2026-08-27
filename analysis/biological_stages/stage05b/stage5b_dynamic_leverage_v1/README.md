# Stage 5B — State-resolved dynamical leverage in vivo

This package is a **discovery-first** pipeline for the two downloaded Stage 5B datasets:

1. Steinmetz / DANDI 000017 — brain-wide state-resolved leverage.
2. Allen Visual Behavior 2P — same-cell cross-state role flexibility.

It also contains a Stage 5A recovery utility for the missing specimen-level GLIF
performance atlas needed for epsilon sensitivity.

## Scientific guardrails

- Do **not** claim single-cell correspondence between Allen Cell Types patch clamp
  and Allen Visual Behavior 2P. The bridge is class-conditioned and exploratory.
- `role_flexibility` is computed from **within-session rank-normalized leverage**
  before taking variance across states; this avoids trivial scale differences
  between sessions.
- Primary in-vivo outputs are:
  - `L_dyn(i,s)` — model-based dynamical perturbation leverage.
  - `L_pred(i,s)` — held-out predictive ablation leverage when labels permit.
  - `mean_leverage(i)` and `role_flexibility(i) = Var_s(rank(L_i(s)))`.
- Mean firing/activity, PCA loading norm, region/class, and task/stimulus tuning are
  controls, not alternative definitions of complexity.
- Stage 5A complexity ↔ Stage 5B flexibility is a **hypothesis test**, not a
  prespecified conclusion.

## Expected data roots

```text
/data/coding/NeuralScience/biological_data/stage5b_dynamic/
├── steinmetz_dandi_000017/
└── allen_visual_behavior_2p_official_s3/
```

Expected output root:

```text
/data/coding/NeuralScience/biological_results/stage5b_dynamic_leverage_v1/
```

## Recommended workflow

### 0. Install lightweight analysis environment

```bash
cd /data/coding/NeuralScience/stage5b_dynamic_leverage_v1
bash install_env.sh
```

### 1. Run schema/data audit first

```bash
bash run_audit.sh
```

Read:

```text
.../stage5b_dynamic_leverage_v1/reports/AUDIT_SUMMARY.md
```

### 2. Recover the missing Stage 5A GLIF performance atlas if possible

```bash
. /data/coding/NeuralScience/.venvs/stage5b/bin/activate
python scripts/01_recover_glif_performance.py
```

This searches existing Stage 5A data/results first. It does **not** redownload NWB.

### 3. Run the baseline Stage 5B analysis

```bash
bash run_baseline.sh
```

This runs one session/file at a time and is intentionally conservative. It writes
checkpoint CSVs, so rerunning skips finished files.

### 4. Inspect the final report

```text
.../stage5b_dynamic_leverage_v1/reports/STAGE5B_BASELINE_REPORT.md
```

## Baseline leverage definition

For a binned standardized population matrix `X`, a global PCA encoder gives

```text
Z = X W
```

Within each state `s`, a ridge linear dynamics operator is fit:

```text
Z(t+lag) ≈ Z(t) B_s
```

Deleting neuron `i` from the encoded state changes the next-latent prediction by

```text
Δz_pred = -X_i(t) W_i B_s
```

so the state-resolved dynamical leverage is

```text
L_dyn(i,s) = mean_t |X_i(t)| * ||W_i B_s||_2 .
```

This is scalable, state-specific, and separates a fixed population encoder from
state-specific latent dynamics.

## Output structure

```text
stage5b_dynamic_leverage_v1/
├── audit/
├── stage5a_recovery/
├── steinmetz/
│   ├── inventory.csv
│   ├── leverage_long.csv
│   ├── role_flexibility.csv
│   └── checkpoints/
├── allen_vbo/
│   ├── inventory.csv
│   ├── same_cell_map.csv
│   ├── leverage_long.csv
│   ├── role_flexibility.csv
│   └── checkpoints/
├── bridge/
└── reports/
```

## Important

The first baseline is deliberately **not** the final paper analysis. Its job is to
answer, with real data:

1. Is leverage measurably state-dependent?
2. Is role flexibility reproducible and nontrivial after activity controls?
3. Does the Allen same-cell result replicate the Steinmetz principle?
4. Is there enough class overlap to justify a Stage 5A ↔ Stage 5B bridge?

Only after those answers are known should we lock the final statistical model and
figure plan.
