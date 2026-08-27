$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (!(Test-Path ".venv")) {
    python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\check_gpu.py
python scripts\run_sweep.py --config configs\smoke.json
python scripts\summarize.py
