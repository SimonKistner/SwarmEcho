[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Offline,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
if ($Offline -and -not $DryRun) { throw '-Offline requires -DryRun.' }
$pythonCommand = Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1
if ($pythonCommand.Source -match '[/\\](\.venv|venv)[/\\]') {
    throw 'Select a system Python with -Python; do not use the project virtual environment.'
}
$branchArgs = @((Join-Path $PSScriptRoot 'tools/branch_g03_efficiency.py'))
if ($DryRun) { $branchArgs += '--dry-run' } else { $branchArgs += '--replace' }
if ($Offline) { $branchArgs += '--offline' }
& $pythonCommand.Source @branchArgs
if ($LASTEXITCODE -ne 0) { throw "G03 branch replacement failed (exit $LASTEXITCODE). Inspect the batch receipt before retrying." }
