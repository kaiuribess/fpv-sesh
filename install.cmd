@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set "installResult=%errorlevel%"
if not "%installResult%"=="0" echo Setup did not finish. Read the message above, then run install.cmd again.
pause
exit /b %installResult%
