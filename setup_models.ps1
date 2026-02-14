# Automated Setup Script for LLaMA-3 & Phi-3 Models
# Run this script to set everything up automatically

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "LLaMA-3 & Phi-3 Setup Script" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

# Change to project directory
$projectDir = "c:\Users\Leore\Downloads\disertation_2026-main\disertation_2026-main"
Set-Location $projectDir

# Step 1: Install dependencies
Write-Host "`n[Step 1/3] Installing dependencies..." -ForegroundColor Yellow
Write-Host "Installing llama-cpp-python (this may take a few minutes)..." -ForegroundColor Gray

try {
    pip install llama-cpp-python --quiet
    Write-Host "✅ llama-cpp-python installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install llama-cpp-python" -ForegroundColor Red
    Write-Host "Try manually: pip install llama-cpp-python" -ForegroundColor Yellow
}

try {
    pip install huggingface-hub --quiet
    Write-Host "✅ huggingface-hub installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install huggingface-hub" -ForegroundColor Red
}

# Step 2: Download models
Write-Host "`n[Step 2/3] Downloading models..." -ForegroundColor Yellow
Write-Host "This will download ~8GB of data and may take 10-30 minutes" -ForegroundColor Gray
Write-Host "Models: LLaMA-3 8B (~5GB) and Phi-3 Mini (~2.5GB)" -ForegroundColor Gray

$downloadChoice = Read-Host "`nDownload models now? (y/n)"
if ($downloadChoice -eq 'y') {
    python models/download_models.py
} else {
    Write-Host "⚠️  Skipping model download" -ForegroundColor Yellow
    Write-Host "Models are required to run. Download later with: python models/download_models.py" -ForegroundColor Gray
}

# Step 3: Test
Write-Host "`n[Step 3/3] Testing setup..." -ForegroundColor Yellow

if (Test-Path "models/LLaMA-3-8B/*.gguf" -PathType Leaf) {
    Write-Host "✅ LLaMA-3 model found" -ForegroundColor Green
} else {
    Write-Host "❌ LLaMA-3 model not found" -ForegroundColor Red
}

if (Test-Path "models/Phi-3-Mini/*.gguf" -PathType Leaf) {
    Write-Host "✅ Phi-3 model found" -ForegroundColor Green
} else {
    Write-Host "❌ Phi-3 model not found" -ForegroundColor Red
}

# Summary
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run models interactively:" -ForegroundColor White
Write-Host "     python models/run_models.py`n" -ForegroundColor Gray
Write-Host "  2. Use in your code:" -ForegroundColor White
Write-Host "     from src.llm_interface import ModelFactory`n" -ForegroundColor Gray
Write-Host "  3. Run experiments:" -ForegroundColor White
Write-Host "     python run_experiments.py`n" -ForegroundColor Gray

Write-Host "For help, see: models/README.md or models/SETUP_GUIDE.md" -ForegroundColor Gray
Write-Host "`nPress any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
