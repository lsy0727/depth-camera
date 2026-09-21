@echo off
setlocal

rem Builds a standalone capture_dataset.exe (via PyInstaller) that can run
rem on another Windows PC without a Python install.
rem
rem IMPORTANT: this uses its own throwaway virtual environment (.build_venv)
rem with the STANDARD opencv-python wheel from PyPI, not whatever cv2 you
rem may have installed system-wide. If this machine's global Python has a
rem custom/locally-built OpenCV (e.g. built from source for CUDA), building
rem with it breaks the .exe on other PCs with:
rem   ImportError: ERROR: recursion is detected during loading of "cv2"
rem   binary extensions. Check OpenCV installation.
rem because a custom build's cv2\config.py / config-3.py bundle this
rem machine's own absolute file paths, which don't exist elsewhere and
rem confuse OpenCV's loader when frozen. The standard PyPI wheel has no
rem such hardcoded paths and is the one PyInstaller's cv2 hook is built
rem for, so always build from it via this venv -- do not switch this
rem script to use the system Python/pip.
rem
rem Output: capture_dataset.exe, written to this repo's root (next to lib\).
rem To deploy to another PC, copy the whole folder structure below there
rem (paths are resolved relative to the .exe, same as this repo's layout):
rem   capture_dataset.exe
rem   lib\                (dcw2_capture_daemon.exe + Orbbec SDK DLLs)

cd /d "%~dp0"

if not exist .build_venv (
    echo Creating build venv (.build_venv)...
    python -m venv .build_venv
    if errorlevel 1 goto :error
)

.build_venv\Scripts\python.exe -m pip install --quiet --upgrade pip
.build_venv\Scripts\python.exe -m pip install --quiet pyinstaller opencv-python numpy
if errorlevel 1 goto :error

.build_venv\Scripts\python.exe -m PyInstaller --onefile --console --name capture_dataset ^
    --distpath . --workpath build --specpath build ^
    examples\capture_dataset.py
if errorlevel 1 goto :error

echo.
echo Done: capture_dataset.exe created at "%~dp0capture_dataset.exe"
echo Deploy it together with the lib\ folder.
pause
goto :eof

:error
echo.
echo Build failed -- see the error above.
pause
