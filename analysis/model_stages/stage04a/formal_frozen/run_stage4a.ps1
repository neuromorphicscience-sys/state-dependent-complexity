$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
.\.venv\Scripts\Activate.ps1

Write-Host "=== Stage 4A: Scale-Free Optimal Allocation Discovery ==="
python scripts\check_gpu.py

python scripts\run_stage4.py --config configs\stage4a.json --phase discovery

Write-Host ""
Write-Host "Stage 4A discovery completed."
Write-Host "Results: results_stage4a\tasks\"
Write-Host ""
Write-Host "Do NOT run the full Stage 4 held-out pipeline yet."
Write-Host "First inspect Stage 4A optimization results and decide whether to proceed to Stage 4B."
