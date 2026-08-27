#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/coding/NeuralScience
PY=$ROOT/.venvs/stage5b/bin/python
cd $ROOT/stage5b_dynamic_leverage_v1
export PYTHONPATH=$PWD
$PY scripts/00_env_check.py
$PY scripts/10_steinmetz_inventory.py
$PY scripts/20_allen_inventory.py
$PY scripts/01_recover_glif_performance.py || true
$PY scripts/41_audit_report.py
echo "AUDIT COMPLETE"
