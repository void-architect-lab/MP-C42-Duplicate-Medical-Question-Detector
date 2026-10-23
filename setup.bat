@echo off
setlocal
cd /d "%~dp0"
title MP-C42 - Setup

echo ============================================================
echo MP-C42 Duplicate Medical Question Detector - Setup
echo ============================================================
where py >nul 2>&1
if not errorlevel 1 (
  py -3.11 --version >nul 2>&1
  if not errorlevel 1 goto use_py311
)
python --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python is not installed or is not on PATH.
  echo Install Python 3.11 from https://www.python.org/downloads/release/python-3119/
  echo During setup, tick "Add python.exe to PATH".
  pause
  exit /b 1
)
set "PYTHON_CMD=python"
goto make_venv
:use_py311
set "PYTHON_CMD=py -3.11"
:make_venv
%PYTHON_CMD% --version
if not exist ".venv\Scripts\python.exe" %PYTHON_CMD% -m venv .venv
if errorlevel 1 goto fail
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto fail

echo.
echo Installing CUDA-enabled PyTorch for NVIDIA GPUs (may take several minutes)...
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
  echo CUDA install failed. Trying the standard PyTorch package instead...
  python -m pip install torch
  if errorlevel 1 goto fail
)

echo.
echo Installing project libraries...
python -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
if errorlevel 1 goto fail

echo.
echo SETUP COMPLETE. Next run: run_train.bat
pause
exit /b 0
:fail
echo.
echo SETUP FAILED. Copy the final error message and send it for troubleshooting.
pause
exit /b 1
