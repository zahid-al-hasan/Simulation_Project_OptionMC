[CmdletBinding()]
param([switch]$FullExperiments)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$projectPython = if (Test-Path -LiteralPath ".venv\Scripts\python.exe") {
    ".venv\Scripts\python.exe"
} else { "python" }

& $projectPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }

if ($FullExperiments) {
    & $projectPython -m scripts.run_repeated_experiments --scenarios
} else {
    & $projectPython -m scripts.run_repeated_experiments --repetitions 5 --path-counts 256,1024,4096 --output-dir artifacts/quick_data
}
if ($LASTEXITCODE -ne 0) { throw "Experiments failed." }

if ($FullExperiments) {
    & $projectPython -m scripts.generate_report_figures
} else {
    & $projectPython -m scripts.generate_report_figures --data-dir artifacts/quick_data --output-dir artifacts/quick_figures
}
if ($LASTEXITCODE -ne 0) { throw "Figure generation failed." }
