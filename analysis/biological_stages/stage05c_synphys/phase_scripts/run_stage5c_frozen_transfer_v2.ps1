$ErrorActionPreference="Stop"
$SynphysRoot=$env:NEURAL_SCIENCE_SYNPHYS_ROOT
if ([string]::IsNullOrWhiteSpace($SynphysRoot)) { throw "Set NEURAL_SCIENCE_SYNPHYS_ROOT." }

python -m pip install -q pandas numpy scipy statsmodels

python (Join-Path $PSScriptRoot "stage5c_frozen_transfer_v2.py") `
  --root (Split-Path $SynphysRoot -Parent) `
  --phase1 (Join-Path $SynphysRoot "stage5c_synphys_phase1") `
  --out (Join-Path $SynphysRoot "stage5c_synphys_transfer_v2")
