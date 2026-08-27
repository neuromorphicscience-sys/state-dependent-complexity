$ErrorActionPreference="Stop"
$SynphysRoot=$env:NEURAL_SCIENCE_SYNPHYS_ROOT
if ([string]::IsNullOrWhiteSpace($SynphysRoot)) { throw "Set NEURAL_SCIENCE_SYNPHYS_ROOT." }

python (Join-Path $PSScriptRoot "stage5c_synapse_model_audit_v1.py") `
  --db (Join-Path $SynphysRoot "synphys_r2.1_full.sqlite") `
  --transfer (Join-Path $SynphysRoot "stage5c_synphys_transfer_v2\synphys_frozen_complexity_transfer.csv") `
  --out (Join-Path $SynphysRoot "stage5c_synapse_model_audit_v1")
