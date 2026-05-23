@echo off
setlocal
cd /d "%~dp0"

if not exist .venv (
  py -3 -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install pystray pillow pyinstaller

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist HermesVoiceTray.spec del HermesVoiceTray.spec

python -m PyInstaller --noconsole --onefile --clean --name HermesVoiceTray windows_tray_client.py

echo.
echo Done: dist\HermesVoiceTray.exe
pause
