@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo FPV Sesh needs its project environment.
  echo Double-click install.cmd in this folder first, then open this launcher again.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m fpvsesh.launcher
exit /b 0
