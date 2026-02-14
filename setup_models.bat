@echo off
REM Quick Start Script for LLaMA-3 & Phi-3 Setup
echo.
echo ============================================
echo LLaMA-3 ^& Phi-3 Quick Setup
echo ============================================
echo.

cd /d "%~dp0"

echo [Step 1/3] Installing dependencies...
pip install llama-cpp-python huggingface-hub

echo.
echo [Step 2/3] Downloading models...
echo This may take 10-30 minutes depending on your internet speed
echo.
set /p download="Download models now? (y/n): "
if /i "%download%"=="y" (
    python models\download_models.py
) else (
    echo Skipping download. Run later with: python models\download_models.py
)

echo.
echo [Step 3/3] Setup complete!
echo.
echo Next steps:
echo   1. Test models: python models\run_models.py
echo   2. Run experiments: python run_experiments.py
echo.
echo For help, see: models\README.md
echo.
pause
