#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/coding/NeuralScience
PY=$ROOT/.venvs/stage5b/bin/python
cd $ROOT/stage5b_dynamic_leverage_v1
export PYTHONPATH=$PWD
$PY scripts/11_steinmetz_leverage.py
$PY scripts/21_allen_leverage.py
$PY scripts/02_stage5a_epsilon_sensitivity.py || true
$PY scripts/30_cross_dataset_bridge.py || true
$PY scripts/40_report.py
echo "STAGE5B BASELINE COMPLETE"
