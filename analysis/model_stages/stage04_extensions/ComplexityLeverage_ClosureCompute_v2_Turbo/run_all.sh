#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/data/coding/NeuralScience}"
PKG="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/results_complexity_leverage_closure_v1"

echo "[0/4] Environment check"
python "$PKG/scripts/check_env.py"

echo "[1/4] Stage4 cross-state transfer (GPU)"
python "$PKG/scripts/stage4_cross_state_transfer.py" --project-root "$ROOT" --stage4f-source auto --output-root "$OUT/stage4_cross_state_transfer" --device cuda --noise-seed-count 24

echo "[2/4] Stage4 functional degeneracy (CPU)"
python "$PKG/scripts/stage4_functional_degeneracy.py" --project-root "$ROOT" --stage4f-source auto --transfer-results "$OUT/stage4_cross_state_transfer" --output-root "$OUT/stage4_functional_degeneracy"

echo "[3/4] Allen Visual Behavior lag-aware refit (CPU)"
python "$PKG/scripts/allen_lag_aware_refit.py" --project-root "$ROOT" --output-root "$OUT/allen_lag_aware_refit" --workers 4

echo "[4/4] Closure summary"
python "$PKG/scripts/build_closure_summary.py" --project-root "$ROOT" --output-root "$OUT"

echo "DONE: $OUT"
