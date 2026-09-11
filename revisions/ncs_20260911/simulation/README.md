# Optional task simulation core

The additive [task workflow audit](WORKFLOW_AUDIT.md) now exposes the five
unchanged scientific controller methods with 12 synthetic CPU control-flow
tests. It is not a production launcher; direct instantiation fails closed.

`benchmark_core.py` preserves the task generator, driven HH/LIF dynamics, graph construction, input projection, ridge readout, prediction and scoring functions from the frozen six-task implementation. Its only implementation edits are repository-relative frozen-source lookup and replacing the expired workstation deadline with an explicit `NCS_RUN_DEADLINE_EPOCH` environment input. It launches no workload on import and does not itself repeat the complete candidate search.

The final frozen task version is `challenge_v3`; the retained `v1` and `challenge_v2` generator branches are development history, not alternate final outcomes. The graph and task settings are in the companion processed-data protocol (`sd_eab3d59c11bcbc7fbc32`). Binary scores returned here are accuracies; published task contrasts use `2 * accuracy - 1`. Recall uses `1 - MSE / Var(y)`.

Run from the code repository root, in an environment with NumPy, SciPy, NetworkX and PyTorch installed:

```bash
python revisions/ncs_20260911/simulation/test_core.py
```

The CPU tests validate task generation and the training-only readout. The optional CUDA test compares four masks on 32 nodes at the original 0.05-ms integration step, for 400 ms, to the unchanged frozen simulator already included at `analysis/model_stages/stage04a/formal_frozen/src`. It tests exact same-batch-shape no-drive and zero-drive parity. It is **not** a claim of invariance under arbitrary batch reshaping or of reproducing a complete task search. A CUDA-unavailable skip must not be reported as a passed GPU test.

The standalone statistics replay in the parent directory does not require these heavier dependencies. Original R2 objective/statistics/search-interface building blocks are in `../ncs_r2_core`; the interface is not misrepresented as the complete production search controller. Private workstation scheduling, approval and Photoshop scripts are intentionally excluded.
