# Reproducibility guide

## Code and data separation

This repository is code-only. Raw third-party data, frozen result archives,
checkpoints and processed Source Data are distributed separately. Commands that
need those files must be given an external data/project root.

Set the root once if desired:

```bash
export NEURAL_SCIENCE_DATA_ROOT=/path/to/neural-science-data
```

PowerShell equivalent:

```powershell
$env:NEURAL_SCIENCE_DATA_ROOT = "D:\path\to\neural-science-data"
```

## Environment

Python 3.11 is the release target. The consolidated dependency inputs are:

- `requirements/analysis.in`: analysis and simulation;
- `requirements/qa.in`: CPU-only validation without PyTorch;
- `requirements/full.in`: analysis plus figure/manuscript generation.

For convenience, top-level `requirements.txt` and `environment.yml` point to
the exact full lock. The committed locks were resolved for Linux x86_64 /
manylinux_2_28 with CPython 3.11. The analysis and full locks use PyTorch's CPU
wheel; the QA lock omits PyTorch.

Create a CPU validation environment with:

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements/qa-lock.txt
```

On Windows, use `.venv\Scripts\python.exe` in the second command. GPU stages
require a separately resolved CUDA-specific lock for the local NVIDIA
driver/CUDA stack; do not modify the CPU lock in place. Do not mix binaries from
separate Python environments, because that can cause NumPy/scikit-learn ABI
errors.

## Model stages

The portable launcher discovers the release root from its own location:

```bash
python release_tools/run_model_stage.py stage1 --dry-run
python release_tools/run_model_stage.py stage2 --smoke
python release_tools/run_model_stage.py stage4a
```

Stage 4 is explicitly legacy/general. The confirmed frozen Stage 4A workflow is
selected with `stage4a` and uses population 4,096, 512 elites, 3,072 offspring,
512 random injections, six generations and two swaps.

## Final stage-level analysis and plots

The panel suite always receives the external data root explicitly:

```bash
python release_tools/run_panel_suite.py \
  --data-root /path/to/neural-science-data \
  --dry-run
```

Additional arguments are passed to the frozen V5.3 orchestrator. For example,
`--stages stage4a_analysis stage4a_figures` restricts the run.

## Stage 5C/SynPhys phases

The portable Stage 5C launcher replaces the historical machine-specific
PowerShell launchers:

```bash
python release_tools/run_stage5c_phase.py phase1 \
  --synphys-root /path/to/allen_synap --dry-run
python release_tools/run_stage5c_phase.py all \
  --synphys-root /path/to/allen_synap
```

## Release audit

Run before every upload or tagged release:

```bash
python release_tools/audit_release.py --report docs/release_audit.json
```

Historical scripts retain original local-path examples for provenance. These
are reported as warnings; the release launchers above do not rely on them.
