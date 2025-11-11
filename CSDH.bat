@echo off
setlocal
set "SCRIPT=%~dp0CSDH_script.ps1"

if exist "%SCRIPT%" goto :run
echo PowerShell script not found: "%SCRIPT%"
pause
exit /b 1

:run
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%"
endlocal
exit /b %errorlevel%