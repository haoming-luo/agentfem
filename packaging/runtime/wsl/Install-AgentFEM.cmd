@echo off
setlocal
cd /d "%~dp0"
echo AgentFEM Windows installation
echo Extract the complete ZIP before opening this launcher.
if not exist "%~dp0Install-AgentFEM.ps1" (
  echo ERROR: Installer files are missing. Extract the entire ZIP first.
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-AgentFEM.ps1" %*
set "AFM_INSTALL_EXIT=%ERRORLEVEL%"
echo.
if "%AFM_INSTALL_EXIT%"=="3010" (
  echo Windows requires a restart. Save your work, restart, then reopen this launcher.
) else if not "%AFM_INSTALL_EXIT%"=="0" (
  echo Installation did not complete. The error above and LocalAppData\AgentFEM\Installations logs describe the next step.
)
echo This window stays open so you can read or copy the result.
pause
exit /b %AFM_INSTALL_EXIT%
