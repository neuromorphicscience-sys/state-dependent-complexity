# Task orchestration: exact source, synthetic CPU audit

Five scientific methods and three helpers in `task_controller.py` are copied
without AST changes from frozen `run_formal.py`, SHA-256
`4483b25c52f2725c419ae30fc7a448363f4b9ca23a3b2bcfe7b6ebee6f8c469a`.
This is the controller entry in the original frozen protocol
(`sd_eab3d59c11bcbc7fbc32` in the companion data revision).
`TASK_CONTROLLER_PROVENANCE.json` binds exported bytes and method/helper
fingerprints. Task generation, readout and dynamics remain in `benchmark_core.py`.

## Run without a GPU

In the optional simulation environment (NumPy, SciPy, NetworkX and PyTorch):

```bash
CUDA_VISIBLE_DEVICES='' python revisions/ncs_20260911/simulation/test_task_controller.py
```

The 12 tests use temporary directories and synthetic features on 32 nodes.
They retain real task generation, readout fitting and orchestration, including
8 initial + 3 generations by 8 offspring = 32 candidates per search method.
No neuron dynamics or new scientific reservoir search is run. Synthetic labels
deliberately influence synthetic features: this is a software fixture, not a
prediction experiment or manuscript evidence.

The checks cover:

- Source fingerprints and fail-closed default instantiation.
- Changes to final-test labels/features leave search records, independent
  selections and frozen readout arrays unchanged, while final scores change.
- Role order and frozen selection/readouts written before final-feature access.
- Equal unique candidate budgets, shared initial masks and fixed cardinality.
- Search maximin, shortlists, statewise specialist selection and independent
  selection winners, recalculated from the resulting records.
- Four randomized controls retain the selected input-group counts.
- Completed-task resume checks protocol identity; invalid cache metadata fails
  before simulation; all 576 formal noise role/block seeds are distinct.

## Boundary

This closes a source-inspection and software-test gap, not the entire production
portability gap. The archived workstation constructor, lock/QA-gate CLI and
failure handler are excluded. Default construction deliberately raises; no new
production launcher is provided. The test subclass substitutes synthetic
features, bypasses resource checks only inside test patches and blocks the
simulator. Those patches must not be used to claim production authorization or
validation.

A future portable launcher still needs a fresh technical QA gate bound to the
published implementation, explicit output/deadline/disk limits, exclusive worker
lock, source-bound resume and driven-batch validation. Historical protocol
hashes must not be replaced with modified-source hashes while claiming unchanged
historical execution. This release neither resumes an old worker nor proves
end-to-end production replay. R3/R4 controllers remain separate coverage gaps.
The processed-table numerical replay in the parent directory remains the
supported lightweight reproduction route.
