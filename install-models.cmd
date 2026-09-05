@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run install.cmd first to prepare FPV Sesh.
  pause
  exit /b 1
)
echo FPV Sesh optional models
echo.
echo These downloads run locally after installation. Close active editing jobs first.
echo AI detail needs a compatible NVIDIA GPU. Video understanding needs at least 7 GiB of NVIDIA VRAM.
echo Scene context can also run on the CPU.
echo Flight understanding downloads about 4.3 GB of model weights, plus a separate AI runtime.
echo.
echo 1 - AI detail restoration
echo 2 - Flight understanding and scene context
echo 3 - All optional models
echo Q - Exit without installing
choice /C 123Q /N /M "Choose 1, 2, 3 or Q: "
set "modelChoice=%errorlevel%"
if "%modelChoice%"=="4" exit /b 0
if "%modelChoice%"=="0" exit /b 1
if "%modelChoice%"=="255" exit /b 1
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-ai.ps1" -Upgrade
if errorlevel 1 goto failed
if "%modelChoice%"=="1" goto done
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-vision.ps1"
if errorlevel 1 goto failed
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-video.ps1" -Upgrade
if errorlevel 1 goto failed
:done
echo.
echo Optional setup finished. Reopen FPV Sesh to refresh its status.
echo AI detail still requires a short local sample check; see Help and setup or docs\user-guide.md.
pause
exit /b 0
:failed
echo.
echo Optional setup did not finish. Read the message above and try again after fixing it.
echo Your recordings and saved jobs remain in their existing folders.
pause
exit /b 1
