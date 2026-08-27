# Stage-to-code map

The paths below are relative to the root of this staging directory.

| Stage | Analysis / simulation entry points | Final analysis and plotting |
|---|---|---|
| Stage 1 | `analysis/model_stages/stage01/scripts/run_sweep.py`, `scripts/summarize.py`, `configs/stage1_baseline.json` | `plotting/panel_suite_v5_3/upstream/neural_plot_pipeline_v2_4.py stage1` |
| Stage 1B | `analysis/model_stages/stage01b/scripts/run_stage1b.py`, `scripts/analyze_stage1b.py`, `configs/stage1b.json` | `plotting/panel_suite_v5_3/upstream/neural_plot_pipeline_v2_4.py stage1b` |
| Stage 2 | `analysis/model_stages/stage02/scripts/run_stage2.py`, `scripts/analyze_stage2.py`, `configs/stage2.json` | `plotting/panel_suite_v5_3/upstream/neural_plot_pipeline_v2_4.py stage2` |
| Stage 3 | `analysis/model_stages/stage03/scripts/run_stage3.py`, `scripts/analyze_stage3.py`, `configs/stage3.json` | `plotting/panel_suite_v5_3/upstream/neural_plot_pipeline_v2_4.py stage3` |
| Stage 4 | `analysis/model_stages/stage04/scripts/run_stage4.py`, `scripts/run_stage4_heldout.py`, `scripts/analyze_stage4.py` (legacy/general for Stage 4A reproduction) | Stage 4A/integrated scripts in the V5.3 panel suite |
| Stage 4A | Confirmed config and launcher under `analysis/model_stages/stage04a/formal_frozen/`; archived runtime/source snapshots under `archived_snapshot/` | `analyze_stage4a_manuscript_v1_1.py` and `plot_stage4a_final.py` in `plotting/panel_suite_v5_3/upstream/` |
| Stage 4 integrated | Uses frozen Stage 4 result tables | `analyze_stage4_integrated_v1.py` and `analyze_stage4_integrated_v2_figures.py` in the V5.3 panel suite |
| Stage 4 closure | `analysis/model_stages/stage04_extensions/ComplexityLeverage_ClosureCompute_v2_Turbo/` | `plotting/panel_suite_v5_3/upstream/analyze_closure_extensions_v1.py` |
| Stage 5A | Discovery v1, v11 hotfix and v12 atlas packages under `analysis/biological_stages/stage05a/` | `stage5a_robustness_closure_v1.py` and `stage5a_epsilon_sensitivity_patch_v1.py` in the V5.3 panel suite |
| Stage 5B | `analysis/biological_stages/stage05b/stage5b_dynamic_leverage_v1/` | `plotting/panel_suite_v5_3/upstream/stage5br_leverage_closure_v1.py` |
| Stage 5C / SynPhys | Phase scripts under `analysis/biological_stages/stage05c_synphys/phase_scripts/` | `stage5c_synphys_integrated_v1.py`, `stage5c_synphys_narrative_closure_v2.py`, and `stage5c_synphys_figure_atlas_v1.py` in the V5.3 panel suite |
| OpenScope | V2.3 analysis and final closure under `analysis/biological_stages/openscope/` | `plotting/panel_suite_v5_3/upstream/plot_openscope_final.py` |
| Final manuscript figures | Processed panel source data are external | `plotting/manuscript_assembly_v5_3/` contains Fig01-Fig06, ED01-ED06 and SI01-SI08 builders |
| Supplementary Table 1 | Source-data CSV is external | `plotting/supplementary_table_1/Supplementary_Table_1/build_supplementary_table_1.py` |

## Stage-specific computational modules

Each Stage 1-4 directory retains its matching `src/`, `scripts/` and `configs/`
snapshot. This is intentional: later stages changed shared modules, and the
merged project-root copies no longer import cleanly for Stage 1/1B. Keeping the
small source packages self-contained preserves the code version that belonged
to each stage.

Stage 4A is the one exception to using the historical Stage 4 default config:
its frozen outputs were produced with `formal_frozen/configs/stage4a.json`
(`4096/512/3072/512/6`, with two swaps), not the legacy/general
`stage04/configs/stage4.json`.

## V5.3 plotting orchestrator

The complete stage-level plotting command remains:

```text
python plotting/panel_suite_v5_3/build_panel_suite_v5_3.py --root <project-root>
```

It expects the original project/result directory layout. Path portability and a
code-release-native input layout will be addressed in the next cleanup pass.
