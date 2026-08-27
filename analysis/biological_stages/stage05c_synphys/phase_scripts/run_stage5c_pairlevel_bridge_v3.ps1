$ErrorActionPreference="Stop"
$SynphysRoot=$env:NEURAL_SCIENCE_SYNPHYS_ROOT
if ([string]::IsNullOrWhiteSpace($SynphysRoot)) { throw "Set NEURAL_SCIENCE_SYNPHYS_ROOT." }

python -m pip install -q pandas numpy scipy statsmodels

python (Join-Path $PSScriptRoot "stage5c_pairlevel_bridge_v3.py") `
  --phase1 (Join-Path $SynphysRoot "stage5c_synphys_phase1") `
  --transfer (Join-Path $SynphysRoot "stage5c_synphys_transfer_v2") `
  --out (Join-Path $SynphysRoot "stage5c_synphys_pairlevel_v3")
