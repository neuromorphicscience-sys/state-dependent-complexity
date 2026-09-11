# Validation performed for this code revision

Validated on 11 September 2026 before publication:

- A fresh Python 3.10.12 virtual environment containing NumPy 2.2.6 replayed 31 listed endpoint checks from the companion processed tables, including exact tests, frozen bootstrap intervals, applicable Holm families and shared-sufficiency bounds. All checks passed. The inventory/hash check includes every file in that data-revision directory.
- Thirteen CPU unit tests passed: exact-test resolution/null handling, Holm ordering, seed-role separation, aggregation order, state-specific selection, graph weighting, sufficiency language, fixed-cardinality search-interface behavior and evaluation accounting.
- Optional task-core tests passed for task generators, final `challenge_v3` role separation and reproducibility, and the training-only ridge readout.
- The optional CUDA test passed with PyTorch 2.5.1+cu121, NumPy 2.2.6, SciPy 1.15.3 and NetworkX 3.4.2. Its 32-node/four-mask/400-ms same-batch-shape comparison at 0.05-ms time steps gave maximum absolute difference 0 for rhythm score, mean rate, dominant frequency, spectral concentration, spectral entropy, synchrony proxy and silent fraction. No-drive and zero-drive cases both matched the original reference.
- Original R2 building-block source bytes are unchanged. All function ASTs in the public task core match the frozen source. Two portability-only global definitions are documented in `SOURCE_IMPLEMENTATION_PROVENANCE.json`.

These checks do **not** repeat the original long search, establish batch-reshaping invariance, rerun all biological model fitting, or certify publication/author approval. Remote-download replay is recorded separately after release; this file describes local pre-publication testing only.
