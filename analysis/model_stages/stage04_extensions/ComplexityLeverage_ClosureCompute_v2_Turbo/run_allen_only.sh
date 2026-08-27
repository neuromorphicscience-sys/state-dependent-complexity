#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/data/coding/NeuralScience}";PKG="$(cd "$(dirname "$0")" && pwd)";OUT="$ROOT/results_complexity_leverage_closure_v1"
python "$PKG/scripts/allen_lag_aware_refit.py" --project-root "$ROOT" --output-root "$OUT/allen_lag_aware_refit" --workers 4
