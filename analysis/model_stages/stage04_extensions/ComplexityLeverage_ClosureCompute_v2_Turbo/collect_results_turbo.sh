#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/data/coding/NeuralScience}"
OUT="$ROOT/results_complexity_leverage_closure_v2_turbo"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$ROOT/ComplexityLeverage_ClosureCompute_v2_Turbo_results_${STAMP}.tar.gz"
if [[ ! -d "$OUT" ]]; then echo "Missing $OUT" >&2; exit 2; fi
python - <<PY
from pathlib import Path
import json,sys
root=Path(r'''$OUT''')/'stage4_cross_state_transfer'/'tasks'
done=list(root.rglob('done.json')) if root.exists() else []
print(f'completed Stage4 transfer tasks: {len(done)}/27')
if len(done)!=27:
    print('Refusing final result pack because transfer is incomplete.',file=sys.stderr);sys.exit(3)
PY
tar -czf "$DEST" -C "$ROOT" "$(basename "$OUT")"
sha256sum "$DEST" > "$DEST.sha256"
echo "$DEST"
echo "$DEST.sha256"
