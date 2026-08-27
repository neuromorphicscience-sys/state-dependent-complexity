$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
.\.venv\Scripts\Activate.ps1
python scripts\check_gpu.py
python scripts\run_stage3.py --config configs\stage3.json
python scripts\analyze_stage3.py
