# Stage 5 Biological Atlas v1.2

This package is the next-stage correction and adjudication package.

It intentionally REUSES the completed v1.1 Allen GPU temporal-context experiment.
It does not waste GPU time rerunning AB/BA.

What v1.2 fixes / adds
======================
1. Correct Digital Brain SWC parser (fixes NumPy truth-value bug).
2. Whole-cortex 2025 must close at exactly 12,264 valid SWCs.
3. PFC2022 is canonicalized to exactly 6,357 `swc_allen_space` biological neurons.
   The paired raw-coordinate SWCs are not treated as independent neurons.
4. Current PFC2023 download is audit-only and is not used as an independent cohort.
5. Allen Biological Complexity Atlas is upgraded from a single temporal-memory axis
   to a preregistered multiaxial ephys feature system.
6. Minimum sufficient GLIF mechanism requirement C_req(eps=0.02) is computed from the
   complete GLIF1–5 cohort using the mechanism lattice costs:
      GLIF1=0; GLIF2=GLIF3=1; GLIF4=2; GLIF5=3.
7. Repeated nested cross-validation compares:
      temporal profile
      static metadata only
      multiaxial ephys
      ephys + metadata
      ephys + temporal
8. Permutation tests and paired bootstrap deltas adjudicate whether multiaxial ephys
   contains information beyond timescale and static biological labels.
9. PFC projection-subtype structural organization is tested with a multivariate
   centroid pseudo-F / R² and 5,000 label permutations.
10. Final reports explicitly preserve negative results and leave dynamic leverage
    untested until a state-resolved functional dataset is added.

Scientific primary cohort
=========================
Mouse-only complete GLIF1–5 cells are primary for C_req association.
This avoids confounding the primary analysis by unequal human/mouse GLIF coverage.

No claim is hard-coded positive.
