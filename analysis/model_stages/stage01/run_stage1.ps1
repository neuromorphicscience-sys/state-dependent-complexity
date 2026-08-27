$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
.\.venv\Scripts\Activate.ps1
python scripts\check_gpu.py
python scripts\run_sweep.py --config configs\stage1_baseline.json
python scripts\summarize.py
