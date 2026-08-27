#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/data/coding/NeuralScience}"
OUT="$ROOT/results_complexity_leverage_closure_v1"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$ROOT/ComplexityLeverage_ClosureCompute_results_${STAMP}.tar.gz"
if [ ! -d "$OUT" ]; then echo "Missing $OUT"; exit 2; fi
tar -czf "$DEST" -C "$ROOT" "$(basename "$OUT")"
sha256sum "$DEST" > "$DEST.sha256"
echo "$DEST"
