# Non-scientific release modifications

The stage packages and frozen figure builders were collected from the completed
research archive. The following release-only changes improve portability without
changing model equations, statistical methods, random seeds or frozen numerical
parameters:

- added cross-platform launchers under `release_tools/`;
- replaced executable machine-specific default roots in Stage 5C, OpenScope and
  V5.3 upstream scripts with the current working directory;
- made the V5.3 batch launcher require an explicit root or
  `NEURAL_SCIENCE_DATA_ROOT`;
- made Stage 5C PowerShell launchers use `NEURAL_SCIENCE_SYNPHYS_ROOT` and their
  own script directory;
- added optional `ARIAL_FONT` and `ARIAL_BOLD_FONT` environment variables to the
  non-frozen assembly and Supplementary Table builders;
- added consolidated environment inputs, upload exclusions and release-audit
  tooling.

The 22 Python scripts referenced by manuscript `FREEZE_MANIFEST.json` files were
not edited. Historical README examples and frozen figure font paths remain as
provenance and are inventoried by the release audit.

