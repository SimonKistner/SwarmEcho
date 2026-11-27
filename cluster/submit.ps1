# Only standard-library preparation runs locally. Training runs on the cluster.
[CmdletBinding()]
param(
    [string]$Run,
    [string]$Config,
    [int[]]$Trials,
    [switch]$DryRun,
    [switch]$ImageInfo,
    [switch]$OverrideExisting,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$pythonCommand = Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1
if ($pythonCommand.Source -match '[/\\](\.venv|venv)[/\\]') {
    throw 'Select a system Python with -Python; do not use the project virtual environment.'
}
$submitArgs = @((Join-Path $PSScriptRoot 'tools/submit.py'))
if ($ImageInfo) {
    if ($Run -or $Config -or $PSBoundParameters.ContainsKey('Trials') -or $DryRun -or $OverrideExisting) { throw '-ImageInfo cannot be combined with submission options.' }
    $submitArgs += '--image-info'
} else {
    if ($Run -and $Config) { throw 'Use -Config or the historical -Run selector, not both.' }
    if ($Run) { $submitArgs += @('--run', $Run) }
    elseif ($Config) { $submitArgs += @('--config', $Config) }
    if ($PSBoundParameters.ContainsKey('Trials')) {
        if (-not $Config -or -not $Trials.Count) { throw '-Trials requires -Config and at least one trial number.' }
        $submitArgs += '--trials'
        $submitArgs += $Trials
    }
    if ($DryRun) { $submitArgs += '--dry-run' }
    if ($OverrideExisting) { $submitArgs += '--override-existing' }
}
& $pythonCommand.Source @submitArgs
if ($LASTEXITCODE -ne 0) { throw "Cluster preparation/submission failed (exit $LASTEXITCODE)." }
