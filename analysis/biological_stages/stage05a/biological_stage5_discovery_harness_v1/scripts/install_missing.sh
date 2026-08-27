#!/usr/bin/env bash
set -euo pipefail
PY="${PYTHON_BIN:-/data/miniconda/envs/torch/bin/python}"
echo "Using $PY"
MISSING=$("$PY" - <<'PY'
mods=["numpy","h5py","torch","scipy","matplotlib"]
miss=[]
for m in mods:
    try:__import__(m)
    except Exception:miss.append(m)
print(" ".join(miss))
PY
)
if [ -z "$MISSING" ]; then
  echo "All required packages are already installed. Torch/CUDA untouched."
  exit 0
fi
echo "Missing: $MISSING"
echo "Installing missing packages only; torch is never installed/upgraded by this script."
PKGS=""
for p in $MISSING; do
  if [ "$p" != "torch" ]; then PKGS="$PKGS $p"; fi
done
if echo "$MISSING" | grep -qw torch; then
  echo "ERROR: torch missing from the designated CUDA environment. Refusing to modify CUDA stack."
  exit 12
fi
"$PY" -m pip install --no-cache-dir $PKGS
