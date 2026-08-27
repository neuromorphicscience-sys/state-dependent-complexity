@echo off
setlocal
set ROOT=%~1
if "%ROOT%"=="" set ROOT=%NEURAL_SCIENCE_DATA_ROOT%
if "%ROOT%"=="" (
  echo Usage: RUN_V5_3_FULL.bat ^<external-data-root^>
  echo Or set NEURAL_SCIENCE_DATA_ROOT.
  exit /b 2
)
python build_panel_suite_v5_3.py --root "%ROOT%"
endlocal
