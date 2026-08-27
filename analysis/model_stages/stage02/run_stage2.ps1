$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
.\.venv\Scripts\Activate.ps1
python scripts\check_gpu.py
python scripts\run_stage2.py --config configs\stage2.json
python scripts\analyze_stage2.py
