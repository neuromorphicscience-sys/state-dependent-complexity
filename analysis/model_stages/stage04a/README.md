# Stage 4A frozen configuration

The independent provenance audit confirmed that the frozen Stage 4A results used
the following evolutionary-search parameters:

- population: 4,096 masks;
- elites retained per generation: 512;
- offspring: 3,072;
- independent random injections: 512;
- generations: 6;
- selected/unselected-node swaps per offspring: 2.

`formal_frozen/` contains the confirmed project-level Stage 4A config, README and
launcher. `archived_snapshot/` retains the runtime config and source snapshots
stored inside `stage4_complete_results.tar.gz`. The formal config is semantically
identical to the archived runtime snapshot; formatting and the final newline are
not computational differences.

The `formal_frozen/` directory is self-contained: it includes the exact Stage 4
runner and source modules used by the archived snapshot, plus
`configs/stage4a.json` and its dependency list.

The general Stage 4 package in `../stage04/` is retained for historical
provenance. Its legacy `configs/stage4.json` is not the Stage 4A reproduction
configuration. Reproduce Stage 4A with
`formal_frozen/configs/stage4a.json`.
