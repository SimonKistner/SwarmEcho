# Continue the nine completed runs in the six G03 screenshot groups.
[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$pythonCommand = Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1
if ($pythonCommand.Source -match '[/\\](\.venv|venv)[/\\]') {
    throw 'Select a system Python with -Python; do not use the project virtual environment.'
}
$submitArgs = @((Join-Path $PSScriptRoot 'tools/resume_g03.py'))
if ($DryRun) { $submitArgs += '--dry-run' }
& $pythonCommand.Source @submitArgs
if ($LASTEXITCODE -ne 0) { throw "G03 continuation preparation/submission failed (exit $LASTEXITCODE)." }
