@echo off
setlocal
cd /d "%~dp0"
title MP-C42 - Complete Setup and Demo
call setup.bat
if errorlevel 1 exit /b 1
call run_train.bat
if errorlevel 1 exit /b 1
call run_demo.bat
