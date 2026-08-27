# Stage 1B — Emergence & Criticality Search

This package is designed to be copied into the existing project root:

D:\Research\Neural Science

It does NOT overwrite Stage 1A results. New results are written to:

results_stage1b\

## Scientific goals

Stage 1B tests four questions:

1. Sparse-complexity efficiency
   How much collective rhythmic organization is gained per unit HH fraction?

2. Emergence gain
   Does a mixed HH–LIF network outperform the linear expectation interpolated
   between the all-LIF and all-HH homogeneous endpoints?

3. Frequency transition
   Where does the collective network switch between low-frequency and
   high-frequency dynamical regimes?

4. Heterogeneity optimum
   Are there parameter regions where mixed networks show stronger or more
   efficient organization than homogeneous endpoints?

## New quantities

For each parameter regime:

linear_mixture_expectation
    (1-pHH)*R_LIF + pHH*R_HH

emergence_gain
    R_mixed - linear_mixture_expectation

complexity_efficiency
    (R_mixed - R_LIF) / pHH

frequency_transition_strength
    |Δ dominant frequency| / ΔpHH

rhythm_transition_strength
    |Δ rhythm score| / ΔpHH

## Run

Copy the package files into the existing project root, preserving folders.

Then:

powershell -ExecutionPolicy Bypass -File .\run_stage1b.ps1

Outputs:

results_stage1b\runs_stage1b.csv
results_stage1b\stage1b_parameter_summary.csv
results_stage1b\stage1b_candidates.csv
results_stage1b\frequency_state_composition.csv

## Important

Stage 1B intentionally includes pHH=0 and pHH=1 controls in each parameter
condition. This is required to estimate the homogeneous endpoint baseline.

This is still a discovery-stage numerical framework. A positive emergence_gain
is a hypothesis-generating signal, not yet a mechanistic proof.
