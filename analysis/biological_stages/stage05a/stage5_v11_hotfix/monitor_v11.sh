#!/usr/bin/env bash
BASE="/data/coding/NeuralScience/biological_stage5_discovery_harness_v1"
CFG="$BASE/config.json";PY=/data/miniconda/envs/torch/bin/python
OUT=$("$PY" - "$CFG" <<'PY'
import json,sys;print(json.load(open(sys.argv[1]))["output_root"])
PY
)
echo "===== Stage5 v1.1 =====";date -Is
cat "$OUT/99_status/pipeline_status.json" 2>/dev/null || true
echo;echo "[processes]";pgrep -af 'run_v11.sh|allen_extract_v11.py|pfc_reconcile_v11.py|gpu_temporal_context.py' || echo none
echo;echo "[GPU]";nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null || true
echo;echo "[Allen NPZ]";find "$OUT/20_allen_ephys_v11/noise_npz" -name '*.npz' 2>/dev/null | wc -l
echo;echo "[recent log]";ls -1t "$OUT/logs/"*.log 2>/dev/null | head -1 | xargs -r tail -n 12
