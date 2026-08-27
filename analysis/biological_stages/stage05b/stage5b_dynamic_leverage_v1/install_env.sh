#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/coding/NeuralScience
VENV=$ROOT/.venvs/stage5b
PY=/data/miniconda/envs/torch/bin/python

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/python" -m pip install -U pip
"$VENV/bin/python" -m pip install -r "$ROOT/stage5b_dynamic_leverage_v1/requirements.txt"

"$VENV/bin/python" - <<'PY'
import numpy,pandas,scipy,sklearn,h5py,pynwb,yaml
print("STAGE5B ENV PASS")
print("numpy",numpy.__version__)
print("pandas",pandas.__version__)
print("sklearn",sklearn.__version__)
print("pynwb",pynwb.__version__)
PY
