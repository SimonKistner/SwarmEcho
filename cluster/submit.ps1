# Only standard-library preparation runs locally. Training runs on the cluster.
[CmdletBinding()]
param(
    [string]$Run,
    [switch]$DryRun,
    [switch]$ImageInfo,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$pythonCommand = Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1
if ($pythonCommand.Source -match '[/\\](\.venv|venv)[/\\]') {
    throw 'Select a system Python with -Python; do not use the project virtual environment.'
}
$submitArgs = @((Join-Path $PSScriptRoot 'submit.py'))
if ($ImageInfo) {
    if ($Run -or $DryRun) { throw '-ImageInfo cannot be combined with -Run or -DryRun.' }
    $submitArgs += '--image-info'
} else {
    if (-not $Run) { throw 'Specify -Run g03-smoke or -Run g03-full.' }
    $submitArgs += @('--run', $Run)
    if ($DryRun) { $submitArgs += '--dry-run' }
}
& $pythonCommand.Source @submitArgs
if ($LASTEXITCODE -ne 0) { throw "Cluster preparation/submission failed (exit $LASTEXITCODE)." }
