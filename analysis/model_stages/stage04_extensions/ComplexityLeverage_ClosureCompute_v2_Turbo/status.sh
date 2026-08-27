#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/data/coding/NeuralScience}"
OUT="$ROOT/results_complexity_leverage_closure_v1"
echo "=== Complexity–leverage closure status ==="
if [ -d "$OUT/stage4_cross_state_transfer/tasks" ]; then
  echo -n "Stage4 cross-state completed target tasks: "
  find "$OUT/stage4_cross_state_transfer/tasks" -name done.json -type f | wc -l
else echo "Stage4 cross-state: not started"; fi
if [ -d "$OUT/allen_lag_aware_refit/checkpoints" ]; then
  echo -n "Allen lag-aware successful experiments: "
  find "$OUT/allen_lag_aware_refit/checkpoints" -name '*_diag.json' -type f | wc -l
  echo -n "Allen lag-aware failed experiments: "
  find "$OUT/allen_lag_aware_refit/checkpoints" -name '*_failed.json' -type f | wc -l
else echo "Allen lag-aware: not started"; fi
[ -f "$OUT/stage4_functional_degeneracy/analysis/degeneracy_key_metrics.json" ] && echo "Functional degeneracy: COMPLETE" || echo "Functional degeneracy: pending"
[ -f "$OUT/closure_summary.json" ] && echo "Closure summary: COMPLETE" || true
