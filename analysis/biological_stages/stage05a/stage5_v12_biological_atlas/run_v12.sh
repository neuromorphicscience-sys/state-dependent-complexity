#!/usr/bin/env bash
set -uo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${CONFIG:-$PKG/config.json}"
PY="${PYTHON_BIN:-/data/miniconda/envs/torch/bin/python}"
OUT=$("$PY" - "$CFG" <<'PY'
import json,sys;print(json.load(open(sys.argv[1]))["output_root"])
PY
)
mkdir -p "$OUT/logs" "$OUT/99_status" "$OUT/checkpoints"
STATUS="$OUT/99_status/pipeline_status_v12.json"
export PYTHONPATH="$PKG/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

status(){
 "$PY" - "$STATUS" "$1" "$2" <<'PY'
import json,sys,datetime,pathlib
p=pathlib.Path(sys.argv[1]);x={}
if p.exists():
 try:x=json.load(open(p))
 except:pass
x.update({"stage":sys.argv[2],"state":sys.argv[3],"updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
p.write_text(json.dumps(x,indent=2))
PY
}
run_stage(){
 NAME="$1";shift;DONE="$OUT/checkpoints/$NAME.done";LOG="$OUT/logs/$NAME.log"
 if [ -f "$DONE" ] && [ "${FORCE:-0}" != "1" ]; then echo "SKIP completed $NAME";return 0;fi
 status "$NAME" running;echo "===== $NAME ====="
 "$@" 2>&1 | tee "$LOG";rc=${PIPESTATUS[0]}
 if [ "$rc" -ne 0 ];then status "$NAME" "failed_exit_$rc";echo "FAILED $NAME rc=$rc";exit "$rc";fi
 touch "$DONE";status "$NAME" complete
}
echo "Stage5 Biological Atlas v1.2";date -Is
run_stage 00_preflight_v12 "$PY" "$PKG/scripts/preflight_v12.py" --config "$CFG"
run_stage 10_digital_brain_v12 "$PY" "$PKG/scripts/digital_brain_v12.py" --config "$CFG"
run_stage 20_allen_multiaxial_v12 "$PY" "$PKG/scripts/allen_multiaxial_v12.py" --config "$CFG"
run_stage 30_structural_stats_v12 "$PY" "$PKG/scripts/structural_stats_v12.py" --config "$CFG"
run_stage 40_final_adjudication_v12 "$PY" "$PKG/scripts/final_adjudication_v12.py" --config "$CFG"
run_stage 50_figures_v12 "$PY" "$PKG/scripts/make_figures_v12.py" --config "$CFG"
run_stage 90_certificate_v12 "$PY" "$PKG/scripts/certify_v12.py" --root "$OUT"
touch "$OUT/PIPELINE_COMPLETE_V12"
status COMPLETE complete
echo "================================================================"
echo "STAGE5 V1.2 COMPLETE"
echo "REPORT: $OUT/40_final_adjudication_v12/final_report_zh.md"
echo "VERDICT: $OUT/40_final_adjudication_v12/verdict_v12.json"
echo "================================================================"
