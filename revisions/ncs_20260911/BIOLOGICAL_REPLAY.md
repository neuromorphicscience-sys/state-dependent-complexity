# Biological processed-data replay

`reproduce_biology.py` uses NumPy only and the existing, checksum-bound companion data revision. It adds eight clearly delimited result groups to the separate 31-check computational replay. It does not fit decoders, reconstruct states, rerun permutations or fit GLIF models from raw data.

## Independent units and exact schedules

The primary OpenScope, same-image and disjoint-unit tables must contain the same 12 unique mouse identities. Split repeats are not additional mice. Every home-minus-cross value is checked against the paired AUC table. The formal primary interval uses 50,000 mouse resamples with seed 20261821; the same-image effect interval uses seed 20261827. These are `20260821 + 1000 + i`, for endpoint indices 0 and 6 in the archived formal endpoint sequence. Both intervals reproduce the original percentile bounds to numerical precision. Exact sign-flip tests use the archived biological tolerance of 1e-15.

The three high-precision null tables instead fix repeat 0. Their observed statistics are not the 12-repeat average. For decoder transfer, coefficient-landscape reconfiguration and top-set Jaccard, the numbers of null values at least as large as the matched observed value are respectively 4, 1168 and 808 out of 2000. Plus-one corrected one-sided P values are respectively 5/2001, 1169/2001 and 809/2001. The last two tests remain null; no positive repeat-averaged description replaces them.

GLIF mechanism costs are reconstructed from five archived explained-variance fits. Cost 1 uses the better of GLIF2 and GLIF3; the four cost levels are 0–3. The original sufficiency expression is `EV_cost >= EV_best - epsilon - 1e-15`. All 400 cells and nine archived tolerances are checked. Baseline cost counts are 105, 144, 85 and 66. These are model-derived requirements, not directly measured universal biological complexity.

The nested-rerun table, not the earlier descriptive multiaxial table, supplies the observed GLIF rank statistic matched to the full-pipeline permutation. Average ranks reproduce Spearman rho = 0.2527190709987035. None of 300 archived permutation correlations reaches that value, giving P = 1/301. The observed MAE remains 0.8378933723085638, worse than the naive-median MAE of 0.805; this boundary is explicitly tested.

## Source relationships

- The original OpenScope closure functions `bootstrap_mean_ci`, `formal_inference`, `build_formal_closure` and `aggregate_high_precision_null` specify the schedules above. The established repository contains the historical closure under `analysis/biological_stages/openscope/final_closure/`.
- Processed GLIF source aliases include `nested_rerun_multiaxial_predictions`, `full_pipeline_permutation_multiaxial` and `epsilon_creq_per_cell`. Every exact source-content identifier and SHA-256 is reported by the replay command and resolves through the companion `DATASET_INDEX.json`.
- Historical fitted/null values are inputs, not independently regenerated models. This check does not settle the exact analysis-matched raw accession versions or certify all biological plots. Existing GPL-3.0-only code licensing does not grant rights to the raw or processed data.

## Tests

Eight additional CPU tests cover tied ranks, plus-one resolution, null ties, duplicate mouse rejection, bootstrap constants/seeds and nonfinite-input rejection. Together with the existing 13 tests, 21 tests pass in the NumPy-only replay environment. The original 31 computational checks are unchanged.
