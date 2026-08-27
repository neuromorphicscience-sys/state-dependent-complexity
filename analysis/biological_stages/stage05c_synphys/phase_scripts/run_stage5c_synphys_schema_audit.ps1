$ErrorActionPreference="Stop"
$SynphysRoot=$env:NEURAL_SCIENCE_SYNPHYS_ROOT
if ([string]::IsNullOrWhiteSpace($SynphysRoot)) { throw "Set NEURAL_SCIENCE_SYNPHYS_ROOT." }
python (Join-Path $PSScriptRoot "stage5c_synphys_schema_audit.py") `
  --db (Join-Path $SynphysRoot "synphys_r2.1_full.sqlite")
