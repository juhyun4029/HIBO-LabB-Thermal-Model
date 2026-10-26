@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run SETUP.bat first.
  exit /b 1
)
".venv\Scripts\python.exe" -u run_pipeline.py status %*
exit /b %errorlevel%
