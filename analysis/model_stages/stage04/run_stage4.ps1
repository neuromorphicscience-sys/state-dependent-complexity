$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
.\.venv\Scripts\Activate.ps1
python scripts\check_gpu.py
python scripts\run_stage4.py --config configs\stage4.json --phase discovery
python scripts\analyze_stage4.py
python scripts\run_stage4_heldout.py --config configs\stage4.json
