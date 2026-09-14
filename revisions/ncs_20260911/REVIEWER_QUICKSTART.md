# Reader verification: scope and executable entry points

This guide concerns the versioned revision, not an untested promise to rerun the
complete study. Earlier Stage 1-5 analysis and plotting source remains in the
repository. No additional scientific result is introduced by these checks.

## Installation and requirements

Use Python 3.10.12 and NumPy 2.2.6 for the tested numerical replay. Linux/WSL is
the tested environment; Windows/macOS portability of the complete study has not
been certified. The metadata checker uses only the Python standard library.
The replay needs no GPU, raw recordings, Photoshop or Word. Internet access is
needed only to obtain repositories and install the pinned dependency.

Clone the companion data and code repositories as siblings. Check out the exact
commit IDs in the accompanying manuscript before running the commands. Do not
use changing branch tips as a substitute for those immutable identities.

From the code repository root:

```bash
python3 -m venv .venv-review
. .venv-review/bin/activate
python -m pip install -r revisions/ncs_20260911/requirements-replay.txt
python -m unittest discover -s revisions/ncs_20260911 -p 'test_*.py'
python revisions/ncs_20260911/verify_panel_sources.py \
  --data-root ../state-dependent-complexity-data/revisions/ncs_20260911 \
  --output local_outputs/panel_sources.json
python revisions/ncs_20260911/reproduce_revision.py \
  --data-root ../state-dependent-complexity-data/revisions/ncs_20260911 \
  --output local_outputs/computational.json
python revisions/ncs_20260911/reproduce_biology.py \
  --data-root ../state-dependent-complexity-data/revisions/ncs_20260911 \
  --output local_outputs/biological.json
```

Use new output filenames on each run: existing reports are deliberately not
overwritten. Place outputs outside the data revision, whose exact file inventory
is checksum-verified. No command above modifies the scientific input tables.

## Expected outputs

- 30 lightweight tests: 21 numerical tests plus nine synthetic metadata guard
  tests. Synthetic failures exercise duplicates, missing IDs, invalid paths,
  inconsistent hashes and wrong coverage counts; they are not paper evidence.
- Panel integrity: `passed: true`, 296 datasets, 202 quantitative lettered panels,
  246 microplots, 20 figures. All current revision files and dataset hashes are
  checked, including the crosswalk records. This does not re-examine the plotting
  code or certify an independently rendered figure.
- Computational replay: 22 task/aggregate endpoints, six shared-policy endpoints,
  one local-stress endpoint and two crossover checks, for 31 in total.
- Biological replay: eight groups, with distinct repeat-averaged and fixed-split
  estimands. The landscape and top-set nulls must remain null, not disappear.

## Measured example time

On 14 September 2026, one fresh Python environment on the author Linux/WSL
workstation took 19.57 s to create and 27.97 s to install the pinned dependency.
Package downloads may have used pip's cache; repository download is excluded.
The 30 tests took 0.84 s, complete panel/file integrity 8.89 s, computational
replay 5.12 s and biological replay 5.19 s. These are single local measurements,
not hardware-normalized benchmarks or guaranteed runtimes. Storage and download
conditions can dominate such small runs. None measures the original GPU search.

## Reproduction tiers and limits

The commands above demonstrate processed-table inference and metadata integrity.
Original data providers, versions, hashes and access paths are specified in the
data revision's `resource_provenance/` manifests. Large raw recordings remain
with their providers under their terms; raw input extraction and every fitted
historical model are not rerun by this demo.

`simulation/README.md` documents optional frozen generators, readouts, dynamics
and a short CUDA same-batch-shape reference test. Its environment and hardware
requirements are separate. `simulation/WORKFLOW_AUDIT.md` explains the 12 optional
synthetic controller tests. These do not establish batch-reshaping invariance,
complete production-launcher portability or a new long-search trajectory.

Checksums bind the distributed bytes, not scientific truth or full provenance
of every upstream raw asset. Shared-design inference uses the corrected frozen
evaluations; historical superseded evaluations are not interchangeable with
them. Complete six-task families and negative outcomes remain included.

Code is GPL-3.0-only. No processed-data reuse licence, archival DOI, journal
approval or independent unfamiliar-colleague testing is asserted here. Final
licensing, maintenance commitments and the journal's code/software form remain
author responsibilities.
