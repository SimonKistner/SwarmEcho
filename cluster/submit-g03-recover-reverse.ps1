# Resume only canceled experiment 55446, to 24 cumulative training hours.
[CmdletBinding()]
param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$pythonCommand = Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1
if ($pythonCommand.Source -match '[/\\](\.venv|venv)[/\\]') {
    throw 'Select a system Python with -Python; do not use the project virtual environment.'
}
& $pythonCommand.Source -X utf8 (Join-Path $PSScriptRoot 'tools/recover_g03_reverse.py')
if ($LASTEXITCODE -ne 0) { throw "G03 recovery preparation/submission failed (exit $LASTEXITCODE)." }
