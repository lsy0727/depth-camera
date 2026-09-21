@echo off
setlocal

rem Camera plugged directly into this Windows PC.
rem Requires dcw2_capture_daemon.exe to be built first:
rem   3rd_party\openni\src\build_windows.bat
rem (needs the Windows Orbbec SDK + Visual Studio C++ build tools -- see
rem  that file for details, since it's not bundled in this repo.)
rem
rem Usage:
rem   capture_dataset.bat [class_name] [res]

set CLASS_NAME=%~1
set RES=%~2

if "%CLASS_NAME%"=="" set /p CLASS_NAME=Class name (e.g. banner):
if "%RES%"=="" set RES=640x480

echo.
echo class=%CLASS_NAME% res=%RES%
echo.

python "%~dp0examples\capture_dataset.py" --class-name %CLASS_NAME% --res %RES%

pause
