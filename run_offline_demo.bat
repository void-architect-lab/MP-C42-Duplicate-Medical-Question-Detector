@echo off
setlocal
cd /d "%~dp0"
title MP-C42 - Offline Emergency Demo
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first. The offline demo still needs Gradio.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python offline_demo.py
pause
