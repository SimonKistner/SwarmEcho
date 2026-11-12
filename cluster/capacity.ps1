[CmdletBinding()]
param(
    [switch]$Json,
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$pythonCommand = Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1
$capacityArgs = @((Join-Path $PSScriptRoot 'tools/capacity.py'))
if ($Json) { $capacityArgs += '--json' }
& $pythonCommand.Source @capacityArgs
if ($LASTEXITCODE -ne 0) { throw "Workspace capacity check failed (exit $LASTEXITCODE)." }
