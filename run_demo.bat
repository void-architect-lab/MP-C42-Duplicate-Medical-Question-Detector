@echo off
setlocal
cd /d "%~dp0"
title MP-C42 - Demo App
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python app.py
pause
