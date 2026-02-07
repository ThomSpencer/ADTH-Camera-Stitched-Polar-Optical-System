REM @echo off
REM REM Run as Administrator to attach USB devices
REM net session >nul 2>&1
REM if %errorlevel% neq 0 (
REM     echo This script must run as Administrator. Restarting...
REM     powershell -Command "Start-Process cmd.exe -ArgumentList '/c %~s0' -Verb RunAs"
REM     exit /b
REM )

echo Attaching USB devices to WSL2...

REM usbipd bind --busid 2-1
usbipd bind --busid 7-1 
usbipd bind --busid 7-4

REM usbipd attach --wsl --busid 2-1
usbipd attach --wsl --busid 7-1
usbipd attach --wsl --busid 7-4

echo.
echo Starting Docker container in WSL...
wsl -d Ubuntu -- bash -ic "cd /mnt/c/Users/ducmi/ADTH-Camera-Stitched-Polar-Optical-System && docker compose up -d --build && docker compose exec app /bin/bash"

pause
