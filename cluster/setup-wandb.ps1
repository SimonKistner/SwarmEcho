# Read an existing local key and send it over SSH stdin; never print the key.
[CmdletBinding()]
param(
    [switch]$Replace,
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$pythonCommand = Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1
if ($pythonCommand.Source -match '[/\\](\.venv|venv)[/\\]') {
    throw 'Select a system Python with -Python; do not use the project virtual environment.'
}
$setupArgs = @((Join-Path $PSScriptRoot 'tools/setup_wandb.py'))
if ($Replace) { $setupArgs += '--replace' }
& $pythonCommand.Source @setupArgs
if ($LASTEXITCODE -ne 0) { throw "Cluster W&B setup failed (exit $LASTEXITCODE)." }
