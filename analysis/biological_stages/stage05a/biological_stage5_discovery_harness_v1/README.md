# Biological Stage 5 Discovery Harness v1

Purpose
=======
This package asks whether the organization principle discovered in computational
Stages 1–4 has a biological counterpart.

It does NOT hard-code a positive conclusion.

Primary biological questions:
1. Does raw electrophysiology contain a reproducible intrinsic temporal
   dynamical-complexity phenotype?
2. Is whole-cortex / PFC projectome structure non-randomly organized?
3. Does raw temporal complexity correspond to precomputed minimum sufficient
   GLIF complexity when a compatible C_model table is present?
4. What is NOT yet testable without a state-resolved functional population
   dataset?

Allen primary metric
====================
For each neuron and each direction Noise A->B and B->A, train shared
stimulus-history predictors with contexts:

10, 25, 50, 100, 200, 500 ms

C_temporal(epsilon) is the minimum context reaching:
    R2 >= best_R2 - epsilon
Primary epsilon = 0.02, sensitivity concept matches the existing C_model logic.

This is intentionally analogous to:
    minimum sufficient mechanistic model complexity

but is independently defined from raw electrophysiological traces.

GPU design
==========
One CUDA training process at a time.
Noise arrays are padded and moved to GPU memory.
Training-window sampling is performed on GPU.
Default RTX 5090 profile:
- BF16 autocast
- batch 4096
- torch.compile(reduce-overhead) when supported
- TF32 enabled where applicable
- 2200 optimizer steps per context
- 6 contexts x 2 independent directions

Do NOT launch 8 CUDA copies of this workload. Multi-process CUDA was useful for
independent Stage4 simulation jobs; deep temporal learning saturates a single
GPU more efficiently with a large batch and resident data.

Digital Brain
=============
Parallel CPU SWC parsing:
- whole cortex 2025
- PFC 2022 + projection subtype labels
- PFC 2023 dendrite+axon cohort

Extracted metrics include axon/dendrite cable length, branches, endpoints,
path length, spatial extent, radius of gyration, spatial entropy and soma CCF
coordinates.

Outputs
=======
output_root/
  00_preflight/
  10_digital_brain/
  20_allen_ephys/
  30_temporal_context/
  40_analysis/
    verdict.json
    final_report_zh.md
    final_report_en.md
  50_figures/
  logs/
  certificate_manifest.json
  PIPELINE_COMPLETE

Scientific guardrail
====================
Current Allen + Digital Brain data can establish cellular temporal-complexity
phenotypes and structural organization. They cannot by themselves establish
that dynamic leverage outperforms static prominence. The report explicitly
marks that claim as untested until an independent state-resolved functional
dataset is added.
