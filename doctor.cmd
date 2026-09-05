@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto project
where py >nul 2>nul
if errorlevel 1 goto missing
py -3.13 -c "import sys" >nul 2>nul
if not errorlevel 1 goto py313
py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 goto py312
:missing
echo FPV Sesh needs 64-bit Python 3.12 or 3.13 with Tk. Install Python, then run install.cmd.
if "%~1"=="" pause
exit /b 1
:project
".venv\Scripts\python.exe" -m fpvsesh.doctor --output "logs\readiness.json" %*
goto finished
:py313
py -3.13 -m fpvsesh.doctor --output "logs\readiness.json" %*
goto finished
:py312
py -3.12 -m fpvsesh.doctor --output "logs\readiness.json" %*
:finished
set "doctorExit=%errorlevel%"
if "%~1"=="" pause
exit /b %doctorExit%
