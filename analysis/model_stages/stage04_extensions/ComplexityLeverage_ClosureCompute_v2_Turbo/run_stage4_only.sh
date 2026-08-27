#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/data/coding/NeuralScience}";PKG="$(cd "$(dirname "$0")" && pwd)";OUT="$ROOT/results_complexity_leverage_closure_v1"
python "$PKG/scripts/stage4_cross_state_transfer.py" --project-root "$ROOT" --stage4f-source auto --output-root "$OUT/stage4_cross_state_transfer" --device cuda --noise-seed-count 24
python "$PKG/scripts/stage4_functional_degeneracy.py" --project-root "$ROOT" --stage4f-source auto --transfer-results "$OUT/stage4_cross_state_transfer" --output-root "$OUT/stage4_functional_degeneracy"
