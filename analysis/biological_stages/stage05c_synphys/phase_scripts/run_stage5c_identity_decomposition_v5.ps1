$ErrorActionPreference="Stop"
$SynphysRoot=$env:NEURAL_SCIENCE_SYNPHYS_ROOT
if ([string]::IsNullOrWhiteSpace($SynphysRoot)) { throw "Set NEURAL_SCIENCE_SYNPHYS_ROOT." }

python -m pip install -q pandas numpy scipy statsmodels scikit-learn

python (Join-Path $PSScriptRoot "stage5c_identity_decomposition_v5.py") `
  --v4 (Join-Path $SynphysRoot "stage5c_dynamic_leverage_v4") `
  --out (Join-Path $SynphysRoot "stage5c_identity_decomposition_v5")
