$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
.\.venv\Scripts\Activate.ps1
python scripts\check_gpu.py
python scripts\run_stage1b.py --config configs\stage1b.json
python scripts\analyze_stage1b.py
