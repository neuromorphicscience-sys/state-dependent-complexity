#!/usr/bin/env bash
set -u
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${CONFIG:-$PKG/config.json}"
PY="${PYTHON_BIN:-/data/miniconda/envs/torch/bin/python}"
OUT=$("$PY" - "$CFG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["output_root"])
PY
)
clear
echo "======================================================================"
echo "Stage 5 Biological Discovery — live status"
echo "======================================================================"
date -Is
echo
echo "[processes]"
pgrep -af 'digital_brain_features.py|allen_extract.py|gpu_temporal_context.py|analyze_final.py|run_all.sh' || echo none
echo
echo "[GPU]"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
 --format=csv,noheader 2>/dev/null || true
echo
echo "[Allen extraction]"
if [ -f "$OUT/20_allen_ephys/allen_ephys_summary.json" ]; then cat "$OUT/20_allen_ephys/allen_ephys_summary.json"; else
  echo "NPZ cells: $(find "$OUT/20_allen_ephys/noise_npz" -name '*.npz' 2>/dev/null | wc -l)"
fi
echo
echo "[Digital Brain]"
cat "$OUT/10_digital_brain/digital_brain_summary.json" 2>/dev/null || echo "still running"
echo
echo "[GPU AB tail]"
tail -n 8 "$OUT/logs/30_gpu_AB.log" 2>/dev/null || true
echo
echo "[GPU BA tail]"
tail -n 8 "$OUT/logs/31_gpu_BA.log" 2>/dev/null || true
echo
echo "[final]"
if [ -f "$OUT/PIPELINE_COMPLETE" ]; then
  echo COMPLETE
  cat "$OUT/40_analysis/final_report_zh.md"
else
  echo "not complete"
fi
echo
echo "[disk]"
df -h /data | tail -n 1
