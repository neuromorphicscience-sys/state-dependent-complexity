$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
.\.venv\Scripts\Activate.ps1
python scripts\check_gpu.py
python scripts\run_stage4.py --config configs\stage4_smoke.json --phase discovery
