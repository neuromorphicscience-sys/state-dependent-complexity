# Validation performed for this code revision

## Additional task-controller audit, 11 September 2026

Twelve optional CPU control-flow tests passed locally with synthetic features,
using the final challenge_v3 task generator and all 32 candidates per search
method on a 32-node fixture. They verify held-out-data independence, source
fingerprints, role order, budget/cardinality, independent-selection rules,
group-count controls, protocol-bound completed resume, cache mismatch rejection
and the 576-entry noise registry. Five original controller methods and three
helpers have identical ASTs to the frozen source. No reservoir dynamics or new
scientific search was executed. This is separate from the 21 NumPy-only tests;
no portable production launcher is claimed. See simulation/WORKFLOW_AUDIT.md.

Validated on 11 September 2026 before publication:

- A fresh Python 3.10.12 virtual environment containing NumPy 2.2.6 replayed 31 listed endpoint checks from the companion processed tables, including exact tests, frozen bootstrap intervals, applicable Holm families and shared-sufficiency bounds. All checks passed. The inventory/hash check includes every file in that data-revision directory.
- Thirteen CPU unit tests passed: exact-test resolution/null handling, Holm ordering, seed-role separation, aggregation order, state-specific selection, graph weighting, sufficiency language, fixed-cardinality search-interface behavior and evaluation accounting.
- Optional task-core tests passed for task generators, final `challenge_v3` role separation and reproducibility, and the training-only ridge readout.
- The optional CUDA test passed with PyTorch 2.5.1+cu121, NumPy 2.2.6, SciPy 1.15.3 and NetworkX 3.4.2. Its 32-node/four-mask/400-ms same-batch-shape comparison at 0.05-ms time steps gave maximum absolute difference 0 for rhythm score, mean rate, dominant frequency, spectral concentration, spectral entropy, synchrony proxy and silent fraction. No-drive and zero-drive cases both matched the original reference.
- Original R2 building-block source bytes are unchanged. All function ASTs in the public task core match the frozen source. Two portability-only global definitions are documented in `SOURCE_IMPLEMENTATION_PROVENANCE.json`.

These checks do **not** repeat the original long search, establish batch-reshaping invariance, rerun all biological model fitting, or certify publication/author approval. Remote-download replay is recorded separately after release; this file describes local pre-publication testing only.

## Biological replay extension, 11 September 2026

Eight processed biological result groups additionally pass in the same clean NumPy-only environment. They include the original 50,000-resample mouse intervals, matched fixed-repeat null tests (including null outcomes), disjoint-unit pairing, the four-level GLIF cost definition at nine tolerances, and the matched nested GLIF rank/permutation result. All 21 CPU unit tests pass. No raw recording or model is refitted. See `BIOLOGICAL_REPLAY.md` for exact coverage.
