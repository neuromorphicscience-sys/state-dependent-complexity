$ErrorActionPreference="Stop"
$SynphysRoot=$env:NEURAL_SCIENCE_SYNPHYS_ROOT
if ([string]::IsNullOrWhiteSpace($SynphysRoot)) { throw "Set NEURAL_SCIENCE_SYNPHYS_ROOT." }

python -m pip install -q pandas numpy scipy statsmodels

python (Join-Path $PSScriptRoot "stage5c_dynamic_leverage_v4.py") `
  --audit (Join-Path $SynphysRoot "stage5c_synapse_model_audit_v1") `
  --phase1 (Join-Path $SynphysRoot "stage5c_synphys_phase1") `
  --out (Join-Path $SynphysRoot "stage5c_dynamic_leverage_v4")
