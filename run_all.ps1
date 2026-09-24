[CmdletBinding()]
param(
    [switch]$FullExperiments
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "OptionMC verification"
Write-Host "Project: $PSScriptRoot"

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "`nCreating virtual environment..."
    python -m venv .venv
}

Write-Host "`nUsing: $venvPython"
& $venvPython --version

Write-Host "`nChecking project dependencies..."
& $venvPython -c "import matplotlib, numpy, optionmc, pytest, scipy" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing project dependencies..."
    & $venvPython -m pip install -e ".[dev]"
}

Write-Host "`nRunning the automated test suite..."
& $venvPython -m pytest -q -p no:cacheprovider

Write-Host "`nRunning quick method comparison..."
& $venvPython scripts\run_comparison.py

Write-Host "`nRunning quick parameter sensitivity analysis..."
& $venvPython scripts\run_sensitivity.py

if ($FullExperiments) {
    Write-Host "`nRunning the complete repeated experiment..."
    & $venvPython scripts\run_repeated_experiments.py --repetitions 30 --output-dir artifacts\data

    Write-Host "`nGenerating final tables and figures..."
    & $venvPython scripts\generate_report_figures.py --data-dir artifacts\data --output-dir report_figures
}
else {
    Write-Host "`nSkipping the longer repeated experiment."
    Write-Host "Run '.\run_all.ps1 -FullExperiments' to regenerate final report data and figures."
}

Write-Host "`nAll requested OptionMC checks completed successfully."
