# Path portability policy

The completed project was developed under `D:\Research\Neural Science` and WSL,
so historical README files, launchers and final figure builders contain local
path examples. They are retained where changing them would obscure provenance or
invalidate frozen script hashes.

Public execution must follow these rules:

1. Model Stage 1-4A code is launched through
   `release_tools/run_model_stage.py`, which resolves code paths relative to the
   release directory.
2. Data-dependent panel analysis is launched through
   `release_tools/run_panel_suite.py` with `--data-root`, or with the
   `NEURAL_SCIENCE_DATA_ROOT` environment variable.
3. Stage 5C phase scripts expose `--db`, `--root`, `--phase1`, `--transfer`,
   `--audit` and/or `--out`. `release_tools/run_stage5c_phase.py` supplies all of
   these from one explicit `--synphys-root`.
4. Hard-coded Arial paths in frozen manuscript figure scripts are part of the
   approved figure provenance. Exact figure-layout reproduction therefore needs
   Arial installed. The submitted figure binaries themselves are not included
   in this code-only repository.

The release audit inventories remaining absolute-path references. They are not
used by the portable entry points and should not be interpreted as bundled-data
locations.

For exact typography outside the frozen builders, set `ARIAL_FONT` and
`ARIAL_BOLD_FONT` to the corresponding font files. Otherwise the affected
document/table builders use their documented fallback fonts.
