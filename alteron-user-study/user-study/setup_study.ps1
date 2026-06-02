$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "The study setup will:"
Write-Host "  1. install uv if it is missing,"
Write-Host "  2. create a local .venv, and"
Write-Host "  3. install the study runtime dependency from requirements.txt."
Write-Host ""
$Consent = Read-Host "Do you want to continue? [y/N]"
if ($Consent -notmatch '^(?i:y|yes)$') {
    Write-Host "Setup cancelled. No installation was performed."
    exit 0
}

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Host "uv not found. Installing uv..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$HOME\.local\bin;$HOME\.cargo\bin;$env:Path"
    $uv = Get-Command uv -ErrorAction SilentlyContinue
}

if (-not $uv) {
    throw "Failed to install uv automatically. Please install uv manually and rerun .\user-study\setup_study.ps1"
}

Write-Host "Using uv: $(& uv --version)"
Write-Host "Provisioning Python 3.10+ and creating .venv ..."
uv venv --python 3.10 .venv
Write-Host "Installing study runtime dependencies ..."
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next, run: .\user-study\run_study.ps1"
