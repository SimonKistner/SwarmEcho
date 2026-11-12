[CmdletBinding()]
param(
    [switch]$All,
    [switch]$Choose,
    [string]$Run,
    [string]$Checkpoint,
    [int]$EvalMaps = 3,
    [switch]$SelectStages,
    [switch]$Status,
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$pythonCommand = Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1
if ($pythonCommand.Source -match '[/\\](\.venv|venv)[/\\]') {
    throw 'Select a system Python with -Python; do not use the project virtual environment.'
}
$fetchArgs = @((Join-Path $PSScriptRoot 'tools/fetch.py'))
if ($All) { $fetchArgs += '--all' }
if ($Choose) { $fetchArgs += '--choose' }
if ($Run) { $fetchArgs += @('--run', $Run) }
if ($Checkpoint) { $fetchArgs += @('--checkpoint', $Checkpoint, '--eval-maps', "$EvalMaps") }
elseif ($PSBoundParameters.ContainsKey('EvalMaps')) { throw '-EvalMaps requires -Checkpoint.' }
if ($SelectStages) { $fetchArgs += '--select-stages' }
if ($Status) { $fetchArgs += '--status' }
& $pythonCommand.Source @fetchArgs
if ($LASTEXITCODE -ne 0) { throw "Cluster retrieval failed (exit $LASTEXITCODE)." }
