# Code provenance

This staging area was assembled without modifying the completed research
archive. Files were copied or selectively extracted from the sources below.

| Release area | Original source |
|---|---|
| Stage 1 | `neural_science_stage1_gpu_harness.zip` |
| Stage 1B | `neural_science_stage1b_emergence_criticality.zip` |
| Stage 2 | `neural_science_stage2_topology_minimal_complexity.zip` |
| Stage 3 | `neural_science_stage3_topology_dynamics_allocation.zip` |
| Stage 4 | `neural_science_stage4_gpu_optimal_allocation.zip` |
| Stage 4A confirmed frozen config | `configs/stage4a.json` after independent provenance verification |
| Stage 4A archived runtime snapshot | `stage4_complete_results.tar.gz`, `results_stage4_complete/source_snapshot/` |
| Stage 4 closure extension | `bio data/ComplexityLeverage_ClosureCompute_v2_Turbo.zip` |
| Stage 5A discovery | `bio data/biological_stage5_discovery_harness_v1.zip` |
| Stage 5A v11 hotfix | `bio data/stage5_v11_hotfix.zip` |
| Stage 5A v12 atlas | `bio data/stage5_v12_biological_atlas.zip` |
| Stage 5B dynamic leverage | `bio data/stage5b_dynamic_leverage_v1.zip` |
| Stage 5C / SynPhys phases | `bio data/scripts/stage5c_*.py` and matching launchers |
| OpenScope V2.3 | `plot/OpenScope_Illusion_Analysis_v2_3/` |
| OpenScope final closure | `plot/OpenScope_FinalClosure_v1/` |
| Final stage-level analysis/plots | `plot/neural_science_panel_suite_v5_3_unified_redraw/` |
| Frozen manuscript assembly | `plot/manuscript_figure_assembly_v5_3/` |

## Selection rules

- Exact historical copies found in old redraw, capture/replay and renderer
  packages were not collected again.
- The V5.3 panel-suite `upstream/` scripts were retained as the canonical final
  stage-level analysis/plotting copies because they match the successful V5.3
  run manifest.
- Source-only/reproduction archives were preferred over result directories for
  Stage 4 extensions and Stage 5 computational packages.
- The Stage 4A formal config was reconciled against all 36 formal checkpoints,
  the main and shard logs, the archived runtime snapshots and the optimizer's
  config-loading path. The confirmed values are 4,096 population, 512 elites,
  3,072 offspring, 512 random injections, six generations and two swaps.
- The legacy/general `configs/stage4.json` remains part of the historical Stage 4
  package but is explicitly not a Stage 4A reproduction config.
- Freeze manifests and QA/figure-contract documents were retained alongside the
  final manuscript builders because they identify the approved script versions.

## Deliberately excluded

- raw Allen, SynPhys, Steinmetz, OpenScope and other third-party datasets;
- processed result tables and the separate public Source Data release;
- checkpoints, run logs, generated figures and manuscript binaries;
- virtual environments, IDE metadata, caches and compiled Python files;
- cloud upload/password scripts and temporary GitHub working directories;
- historical cloud command notes containing machine-specific SSH endpoints;
- related-work PDFs and historical figure pools;
- author/contact files.

Generated Supplementary Table binaries and its source-data CSV were also removed
from this code-only staging copy. Their originals remain unchanged in the main
research archive.
