# Staging validation status

Validation performed after assembling the code-only directory:

- 161 Python source files were parsed with Python's AST parser.
- All 161 parsed successfully; there were no syntax errors.
- The Stage 1, Stage 1B, Stage 2, Stage 3, Stage 4 and confirmed Stage 4A primary runners all
  completed their `--help` import/argument-parser checks from their self-contained
  stage directories.
- No `__pycache__`, `.pyc` or `.pyo` files remain.
- No file in the staging directory is larger than 1 MB.
- The 22 Python-script hashes referenced by the manuscript figure freeze
  manifests all match their frozen values.
- Seventy-five retained files referenced by original source-package manifests
  were checked; all 75 hashes match.
- The Stage 4 closure turbo parity self-test passed on CPU with a maximum
  absolute difference of 0.0.
- The Stage 5B core test passed.
- The lag-aware closure core self-test passed, selecting a 0.25-s lag with
  held-out R² = 0.5577 in its synthetic test.
- The confirmed Stage 4A release config matches the project-level
  `configs/stage4a.json` byte-for-byte and is semantically identical to both
  archived runtime config snapshots. Its values are population 4,096, 512
  elites, 3,072 offspring, 512 random injections, six generations and two swaps.

The portable launcher dry-runs passed for Stage 1, Stage 1B, Stage 2, Stage 3,
legacy/general Stage 4, confirmed Stage 4A and all seven Stage 5C phases. A V5.3
panel-suite dry-run against a temporary external root also completed and wrote
the expected dry-run manifests without relying on the original project path.

Exact `analysis-lock.txt`, `qa-lock.txt` and `full-lock.txt` files were resolved
on 2026-08-27 with uv 0.11.19 for Linux x86_64 / manylinux_2_28 and CPython
3.11. The analysis and full locks use PyTorch's CPU build, while the QA lock
omits PyTorch. Do not merge the CPU environment with the separate project GPU
environment; a CUDA host needs its own recorded CUDA-specific lock.

A newly created CPython 3.11.15 QA environment was synchronized from
`requirements/qa-lock.txt`. It imported its full QA dependency set, and the
Stage 5B and lag-aware closure core tests both passed. Recompiling each of the
three lock inputs offline selected identical package entries after excluding the
generated command header.

The release audit currently passes all code/provenance checks: zero syntax
errors, cache files, large files, potential secrets, frozen-script hash
mismatches or package-manifest mismatches. It reports remaining historical path
examples and frozen Arial paths as warnings. Executable Stage 5C, OpenScope and
V5.3 defaults have been made portable, and the wrappers require explicit external
data roots.

The repository is licensed `GPL-3.0-only`, with copyright held by “Authors and
their affiliated institutions”. The audit verifies the top-level GPLv3 text and
the matching `CITATION.cff` license identifier.

There are 21 redundant Python copies across ten hash groups. They are retained
intentionally because they belong to self-contained frozen packages or snapshots:

- compatibility shims with distinct required filenames in the V5.3 runtime;
- small stage-specific source snapshots required to keep Stage 1-4 independently
  runnable;
- the confirmed, self-contained Stage 4A runtime package and its archived source
  snapshot;
- empty package `__init__.py` files;
- frozen Stage 4 closure modules;
- the OpenScope V2.3 and final-closure package copies.

Historical package manifests may list compiled cache files that were present in
the original zip archives. Those cache files were deliberately omitted here, so
the original package manifests serve as provenance records rather than whole-
directory release checksums. `SHA256SUMS` provides release-level checksums after
the final cleanup; the mutable audit report itself is intentionally excluded.
