#!/usr/bin/env bash
set -euo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${CONFIG:-$PKG/config.json}"
PY="${PYTHON_BIN:-/data/miniconda/envs/torch/bin/python}"
OUT=$("$PY" - "$CFG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["output_root"])
PY
)
mkdir -p "$OUT/logs"

export PYTHONPATH="$PKG/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "======================================================================"
echo "Stage 5 Biological Discovery Harness v1"
echo "======================================================================"
date -Is
echo "PY=$PY"
echo "CFG=$CFG"
echo "OUT=$OUT"

echo "[0] Preflight"
"$PY" "$PKG/scripts/preflight.py" --config "$CFG" | tee "$OUT/logs/00_preflight.log"

echo "[1] CPU preprocessing lanes in parallel"
"$PY" "$PKG/scripts/digital_brain_features.py" --config "$CFG" > "$OUT/logs/10_digital_brain.log" 2>&1 &
PID_DB=$!
"$PY" "$PKG/scripts/allen_extract.py" --config "$CFG" > "$OUT/logs/20_allen_extract.log" 2>&1 &
PID_AL=$!

echo "DigitalBrain PID=$PID_DB"
echo "AllenExtract  PID=$PID_AL"

wait "$PID_DB"
wait "$PID_AL"

echo "[2] GPU temporal-context discovery A->B"
"$PY" "$PKG/scripts/gpu_temporal_context.py" --config "$CFG" --direction AB \
  2>&1 | tee "$OUT/logs/30_gpu_AB.log"

echo "[3] GPU temporal-context discovery B->A"
"$PY" "$PKG/scripts/gpu_temporal_context.py" --config "$CFG" --direction BA \
  2>&1 | tee "$OUT/logs/31_gpu_BA.log"

echo "[4] Statistical synthesis + claim adjudication"
"$PY" "$PKG/scripts/analyze_final.py" --config "$CFG" \
  2>&1 | tee "$OUT/logs/40_analysis.log"

echo "[5] Publication figures"
"$PY" "$PKG/scripts/make_figures.py" --config "$CFG" \
  2>&1 | tee "$OUT/logs/50_figures.log"

echo "[6] Reproducibility certificate"
"$PY" "$PKG/scripts/certify.py" --root "$OUT"

touch "$OUT/PIPELINE_COMPLETE"
echo "======================================================================"
echo "PIPELINE COMPLETE"
echo "Final Chinese report: $OUT/40_analysis/final_report_zh.md"
echo "Verdict:              $OUT/40_analysis/verdict.json"
echo "Figures:              $OUT/50_figures/"
echo "======================================================================"
