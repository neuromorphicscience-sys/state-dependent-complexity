#!/usr/bin/env bash
set -u
ROOT="${1:-/data/coding/NeuralScience}";PKG="$(cd "$(dirname "$0")" && pwd)";OUT="$ROOT/results_complexity_leverage_closure_v2_turbo/stage4_cross_state_transfer"
echo "================ STAGE4 TURBO ================"
pgrep -af 'stage4_cross_state_transfer_turbo.py|run_stage4_turbo_6way.sh' | grep -v 'pgrep -af' || true
n=0; [[ -d "$OUT/tasks" ]] && n=$(find "$OUT/tasks" -name done.json -type f 2>/dev/null | wc -l)
echo "completed tasks: $n / 27"
for f in "$PKG"/logs_stage4_turbo/shard_*.log; do [[ -f "$f" ]] || continue; echo "--- $(basename "$f") ---"; tail -n 3 "$f"; done
echo "================ GPU ================"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true
