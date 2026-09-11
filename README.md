# State-dependent neuronal complexity: code

## September 2026 manuscript revision

The additive [NCS revision](revisions/ncs_20260911/README.md) provides a lightweight, processed-data statistical replay for the updated six-main/eight-Extended-Data/six-Supplementary figure version, plus numerical building blocks and an optional task-simulator parity test. Use its explicit reproduction scope and requirements; it does not assert that every historical simulation or biological model fit was rerun. The companion [processed-data revision](https://github.com/neuromorphicscience-sys/state-dependent-complexity-data/tree/main/revisions/ncs_20260911) preserves complete comparison families, including null results. Earlier code is retained below and in its original directories.

## Historical initial-release documentation

This directory is a code-only staging area assembled from the completed research
archive. It separates analysis code from plotting code and excludes raw data,
computed results, figure binaries, checkpoints, virtual environments and upload
utilities.

This is a code-only public-release candidate. Exact Linux/Python 3.11 dependency
locks, portable cross-platform launchers and consolidated dependency inputs are
included. The analysis and full locks use the PyTorch CPU wheel.

## Quick start

```bash
# Audit the code-only bundle
python release_tools/audit_release.py

# Inspect a model-stage command without executing it
python release_tools/run_model_stage.py stage4a --dry-run

# Dry-run final stage-level analysis/plotting against an external data root
python release_tools/run_panel_suite.py --data-root /path/to/data-root --dry-run
```

See `docs/REPRODUCIBILITY.md` for environment creation, model stages, Stage 5C
phases and plotting commands.

## Directory layout

- `analysis/model_stages/`: self-contained Stage 1 through Stage 4 source
  packages, the confirmed Stage 4A frozen configuration and provenance snapshot,
  and the frozen Stage 4 closure-extension package.
- `analysis/biological_stages/`: Stage 5A, Stage 5B, Stage 5C/SynPhys and
  OpenScope analysis code.
- `plotting/panel_suite_v5_3/`: the V5.3 stage-level analysis and plotting
  orchestrator used to generate the curated panel atlas.
- `plotting/manuscript_assembly_v5_3/`: the frozen main, Extended Data and
  Supplementary figure assembly scripts and their freeze/QA records.
- `plotting/supplementary_table_1/`: code and documentation for Supplementary
  Table 1; generated DOCX/PDF/XLSX files and the source-data CSV are omitted.
- `docs/`: stage-to-code map and provenance notes.
- `release_tools/`: portable model, Stage 5C and panel-suite launchers plus the
  release integrity audit.
- `requirements/`: consolidated dependency inputs and exact Linux/Python 3.11
  lock files.

## Important status notes

1. The Stage 4A provenance audit confirmed the frozen search configuration as
   population 4,096, 512 elites, 3,072 offspring, 512 random injections, six
   generations and two swaps per offspring. The release copy is under
   `analysis/model_stages/stage04a/formal_frozen/`.
2. The matching archived Stage 4A runtime snapshot is retained under
   `analysis/model_stages/stage04a/archived_snapshot/` for reconciliation.
3. The general Stage 4 configuration is a legacy/general configuration and must
   not be used to reproduce Stage 4A.
4. The V5.3 panel-suite `upstream/` directory is the canonical location for
   final Stage 4A, Stage 4 integrated, Stage 5 and OpenScope summary/plot scripts.
   Those scripts are not duplicated into every stage directory.
5. Figure scripts expect processed source-data inputs from the separate Source
   Data release. No raw third-party dataset is included here.

See `docs/STAGE_CODE_MAP.md` for the per-stage entry points and
`docs/PROVENANCE.md` for the original source locations. Current verification
results and environment blockers are recorded in `docs/VALIDATION_STATUS.md`.
Release-only portability edits are listed in `docs/RELEASE_MODIFICATIONS.md`.

## License

Copyright (c) 2026 Authors and their affiliated institutions.

This repository is licensed under the GNU General Public License, version 3
only (`GPL-3.0-only`); see `LICENSE`. Any distributed modified version or larger
work based on this code must comply with GPLv3's corresponding-source and
same-license requirements. The code license does not cover third-party datasets;
see `docs/THIRD_PARTY_DATA.md`.
