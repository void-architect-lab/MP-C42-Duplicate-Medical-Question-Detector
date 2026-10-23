@echo off
setlocal
cd /d "%~dp0"
title MP-C42 - Train GPT-2 classifier
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
echo Training GPT-2 Small on all cleaned MQP pairs with validation-selected checkpoint/threshold.
echo First run downloads the MQP dataset and about 548 MB of GPT-2 weights.
echo Keep this window open. Do not close it while training is running.
python train.py --max-samples 0 --epochs 4 --batch-size 8 --max-length 96 --learning-rate 2e-5
if errorlevel 1 (
  echo.
  echo Training failed. Copy the error text and send it for troubleshooting.
  pause
  exit /b 1
)
echo.
echo TRAINING COMPLETE. Results, model checkpoint, charts and report are in outputs\
echo Next run: run_demo.bat
pause
