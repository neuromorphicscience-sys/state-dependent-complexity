#!/usr/bin/env bash
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${CONFIG:-$PKG/config.json}";PY="${PYTHON_BIN:-/data/miniconda/envs/torch/bin/python}"
OUT=$("$PY" - "$CFG" <<'PY'
import json,sys;print(json.load(open(sys.argv[1]))["output_root"])
PY
)
clear
echo "===== Stage5 v1.2 =====";date -Is
cat "$OUT/99_status/pipeline_status_v12.json" 2>/dev/null || true
echo;echo "[processes]";pgrep -af 'run_v12.sh|digital_brain_v12.py|allen_multiaxial_v12.py|structural_stats_v12.py|final_adjudication_v12.py' || echo none
echo;echo "[GPU - v1.2 normally CPU/statistics dominated; v1.1 GPU results are reused]"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null || true
echo;echo "[latest log]"
F=$(ls -1t "$OUT"/logs/*.log 2>/dev/null | head -1);[ -n "$F" ] && { echo "$F";tail -n 16 "$F"; }
echo;echo "[outputs]"
[ -f "$OUT/10_digital_brain_v12/digital_brain_v12_summary.json" ] && cat "$OUT/10_digital_brain_v12/digital_brain_v12_summary.json"
[ -f "$OUT/20_allen_multiaxial_v12/allen_multiaxial_summary.json" ] && echo "Allen multiaxial summary ready"
[ -f "$OUT/40_final_adjudication_v12/verdict_v12.json" ] && echo "FINAL VERDICT READY"
