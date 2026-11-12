# Submit the already frozen copy of successful experiment 55384.
[CmdletBinding()]
param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$pythonCommand = Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1
if ($pythonCommand.Source -match '[/\\](\.venv|venv)[/\\]') {
    throw 'Select a system Python with -Python; do not use the project virtual environment.'
}
& $pythonCommand.Source (Join-Path $PSScriptRoot 'tools/submit_unchanged_control.py')
if ($LASTEXITCODE -ne 0) { throw "Control submission failed (exit $LASTEXITCODE). Inspect its receipt before retrying." }
