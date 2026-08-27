#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/data/coding/NeuralScience}"
SHARDS="${2:-6}"
SEED_BATCH="${3:-4}"
PKG="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/results_complexity_leverage_closure_v2_turbo"
XFER="$OUT/stage4_cross_state_transfer"
LOGDIR="$PKG/logs_stage4_turbo"
mkdir -p "$LOGDIR" "$XFER"

# Reuse only fully completed v1 tasks with the exact 7601-7624 seed set.
python "$PKG/scripts/import_v1_completed_stage4_tasks.py" \
  --v1-root "$ROOT/results_complexity_leverage_closure_v1/stage4_cross_state_transfer" \
  --v2-root "$XFER" || true

echo "[MASTER] root=$ROOT shards=$SHARDS seed_batch=$SEED_BATCH"
echo "[MASTER] output=$OUT"
pids=()
for ((i=0;i<SHARDS;i++)); do
  log="$LOGDIR/shard_${i}.log"
  echo "[MASTER] launch shard $i -> $log"
  python "$PKG/scripts/stage4_cross_state_transfer_turbo.py" \
    --project-root "$ROOT" --stage4f-source auto --output-root "$XFER" \
    --noise-seed-start 7601 --noise-seed-count 24 --mask-batch 12 --seed-batch "$SEED_BATCH" \
    --task-shard-index "$i" --task-shard-count "$SHARDS" --engine turbo --device cuda --no-finalize \
    >"$log" 2>&1 &
  pids+=("$!")
done

fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=1; fi
done
if [[ "$fail" -ne 0 ]]; then
  echo "[MASTER] one or more shards failed; not finalizing" >&2
  exit 2
fi

echo "[MASTER] all shards complete; finalizing transfer"
python "$PKG/scripts/stage4_cross_state_transfer_turbo.py" \
  --project-root "$ROOT" --stage4f-source auto --output-root "$XFER" \
  --noise-seed-start 7601 --noise-seed-count 24 --finalize-only

echo "[MASTER] transfer finalized; computing functional degeneracy"
python "$PKG/scripts/stage4_functional_degeneracy.py" \
  --project-root "$ROOT" --stage4f-source auto --transfer-results "$XFER" \
  --output-root "$OUT/stage4_functional_degeneracy"

echo "[MASTER] COMPLETE"
