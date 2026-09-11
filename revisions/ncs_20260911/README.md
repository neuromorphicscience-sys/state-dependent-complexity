# NCS revision: source-data inference reproduction

This additive revision accompanies the six-main/eight-Extended-Data/six-Supplementary figure version of **Collective state conditions the computational value of neuronal complexity**. Earlier stage-level code remains intact elsewhere in this repository.

## Run the lightweight numerical checks

Tested with Python 3.10.12. Install `requirements-replay.txt` in a separate environment. Clone the companion [data repository](https://github.com/neuromorphicscience-sys/state-dependent-complexity-data) alongside this code repository. From the code repository root:

```bash
python -m pip install -r revisions/ncs_20260911/requirements-replay.txt
python -m unittest discover -s revisions/ncs_20260911 -p 'test_*.py'
python revisions/ncs_20260911/reproduce_revision.py \
  --data-root ../state-dependent-complexity-data/revisions/ncs_20260911 \
  --output local_outputs/reproduction.json
python revisions/ncs_20260911/reproduce_biology.py \
  --data-root ../state-dependent-complexity-data/revisions/ncs_20260911 \
  --output local_outputs/biology.json
```

The output path must not exist. The script verifies every data-revision checksum before analysis. It uses explicit source-content identifiers and no hard-coded workstation path. It needs no GPU, network access, raw recordings, Photoshop or Word.

## Precisely what is reproduced

- The full six-task battery from 3,024 per-mask/state score rows: 22 task/aggregate endpoints, including the null primary aggregate and secondary state interaction; exact sign flips; 20,000-resample graph-bootstrap intervals; three six-task Holm families.
- Six reference-corrected shared-policy endpoints from the 12 graph-level tables: means, exact tests, 10,000-resample intervals, two Holm families and the frozen normal-approximation sufficiency bounds.
- Uniform 15-condition local-stress gain: 12 graph-level means, exact test and the original 10,000-resample interval.
- Primary and rate-matched crossover: means, exact one-sided tests and positive-graph counts on the same 12 graphs.
- Eight additional biological result groups: OpenScope primary transfer and same-image imprint with their original 50,000-resample mouse-bootstrap intervals; disjoint-unit transfer; three matched fixed-repeat-0 permutation tests (including the two null landscape/top-set endpoints); GLIF mechanism-cost construction across nine tolerances; and the nested out-of-fold GLIF rank statistic with its matched 300-draw permutation P value and naive-median MAE comparison. See `BIOLOGICAL_REPLAY.md` for the distinct estimands and exact source schedules.

Control masks are minimized over states **before** mask averaging. Search repeats, state pairs and grid cells are not converted into independent graph samples. Fixed seeds and original RNG call order are retained for the frozen intervals. R4 uses only the corrected reference-faithful evaluation, not its superseded layout-dependent output.

## Scope and limitations

This is a processed-data inference replay, not a claim to rerun the complete study from raw recordings or to repeat every historical simulation. The original Stage 1-5 and biological code remains in the established analysis directories. The verified command above does not establish reproduction of every descriptive biological panel or of the new search trajectory. See [the optional simulation core](simulation/README.md) for the preserved task generator, readout, dynamics and CUDA reference-parity test; `ncs_r2_core/` retains the original objective, statistical and search-interface modules. No paper acceptance, archival DOI or universal-law claim is implied.

The repository's existing GPL-3.0-only license applies to this code. It does not assign rights to third-party raw data or to the companion processed-data repository.
