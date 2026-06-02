$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

$PythonBin = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonBin)) {
    $PythonBin = "python"
}

& $PythonBin -m alteron.ci.runner `
  --config "$ScriptDir\study_ci.yml" `
  --profile pr-fast `
  --candidate-model-dir "$ScriptDir\study_model\candidate" `
  --candidate-version candidate `
  --baseline-snapshot-dir "$ScriptDir\study_input\snapshots\stable" `
  --baseline-version stable `
  --corpus-dir "$ScriptDir\study_input\corpus" `
  --output-dir "$ScriptDir\study_output" `
  --model-loader ".\user-study\study_loader.py:load_model" `
  --force
$ExitCode = $LASTEXITCODE

Write-Host ""
if ($ExitCode -eq 0) {
    Write-Host "Study run completed with a passing CI result."
}
elseif ($ExitCode -eq 1) {
    Write-Host "Study run completed with a blocking CI result."
    Write-Host "For this bundled study scenario, that result is expected."
    Write-Host "Inspect:"
    Write-Host "  user-study\study_output\ci_summary.json"
    Write-Host "  user-study\study_output\regression_reports\regression_report_stable_to_candidate.csv"
}
else {
    Write-Host "Study run exited with unexpected code $ExitCode."
}

exit $ExitCode
