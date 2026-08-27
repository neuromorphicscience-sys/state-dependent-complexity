#!/usr/bin/env bash
set -uo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="/data/coding/NeuralScience/biological_stage5_discovery_harness_v1"
CFG="${CONFIG:-$BASE/config.json}"
PY="${PYTHON_BIN:-/data/miniconda/envs/torch/bin/python}"
OUT=$("$PY" - "$CFG" <<'PY'
import json,sys;print(json.load(open(sys.argv[1]))["output_root"])
PY
)
mkdir -p "$OUT/logs" "$OUT/99_status"
STATUS="$OUT/99_status/pipeline_status.json"
export PYTHONPATH="$BASE/scripts:$PKG/scripts:${PYTHONPATH:-}"

status () {
  "$PY" - "$STATUS" "$1" "$2" <<'PY'
import json,sys,datetime,pathlib
p=pathlib.Path(sys.argv[1]);stage=sys.argv[2];state=sys.argv[3]
x={}
if p.exists():
    try:x=json.load(open(p))
    except:pass
x.update({"stage":stage,"state":state,"updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
p.write_text(json.dumps(x,indent=2))
PY
}
run_stage () {
  NAME="$1"; shift
  LOG="$OUT/logs/${NAME}.log"
  status "$NAME" "running"
  echo "===== $NAME ====="
  "$@" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    status "$NAME" "failed_exit_$rc"
    echo "FAILED stage=$NAME rc=$rc"
    exit "$rc"
  fi
  status "$NAME" "complete"
}

echo "Stage5 v1.1 hotfix runner"
date -Is
run_stage "01_allen_smoke_v11" "$PY" "$PKG/scripts/allen_extract_v11.py" --config "$CFG" --limit 24 --workers 6 --smoke
run_stage "02_pfc_reconcile_v11" "$PY" "$PKG/scripts/pfc_reconcile_v11.py" --config "$CFG"
run_stage "03_allen_full_v11" "$PY" "$PKG/scripts/allen_extract_v11.py" --config "$CFG" --workers 16

# Make the fixed Allen NPZ directory visible to the original GPU code.
rm -rf "$OUT/20_allen_ephys/noise_npz"
mkdir -p "$OUT/20_allen_ephys"
ln -s "$OUT/20_allen_ephys_v11/noise_npz" "$OUT/20_allen_ephys/noise_npz"

# Reuse completed Digital Brain v1 output if present; v1.1 reconciliation report must be reviewed
# before subtype-dependent claims are accepted.
if [ ! -f "$OUT/10_digital_brain/whole_cortex_features.csv" ]; then
  echo "Whole-cortex v1 feature CSV missing; rerunning original Digital Brain extractor."
  run_stage "10_digital_brain_original" "$PY" "$BASE/scripts/digital_brain_features.py" --config "$CFG"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
run_stage "30_gpu_AB_v11" "$PY" "$BASE/scripts/gpu_temporal_context.py" --config "$CFG" --direction AB
run_stage "31_gpu_BA_v11" "$PY" "$BASE/scripts/gpu_temporal_context.py" --config "$CFG" --direction BA

# Do NOT silently run the old subtype statistics on a 2x SWC representation.
# Allen temporal analysis remains valid; final integrated paper adjudication is held until
# the reconciliation report identifies how the paired SWCs should be composed.
status "40_final_adjudication" "blocked_pending_pfc_pair_semantics"
echo "GPU discovery complete."
echo "PFC reconciliation report: $OUT/11_pfc_reconciliation/pfc_reconciliation_report.json"
echo "Allen AB: $OUT/30_temporal_context/AB/context_performance.csv"
echo "Allen BA: $OUT/30_temporal_context/BA/context_performance.csv"
echo "Integrated subtype verdict intentionally NOT run until paired-SWC semantics are resolved."
