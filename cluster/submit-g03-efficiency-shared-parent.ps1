# Six continuation-seed replacements sharing the retained jobs' seed-456789 parents.
[CmdletBinding()]
param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$pythonCommand = Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1
if ($pythonCommand.Source -match '[/\\](\.venv|venv)[/\\]') {
    throw 'Select a system Python with -Python; do not use the project virtual environment.'
}
$previousEncoding = $env:PYTHONIOENCODING
$previousUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8 = '1'
    & $pythonCommand.Source -X utf8 (Join-Path $PSScriptRoot 'tools/submit_g03_shared_parent.py')
    if ($LASTEXITCODE -ne 0) { throw "G03 shared-parent preparation/submission failed (exit $LASTEXITCODE)." }
} finally {
    $env:PYTHONIOENCODING = $previousEncoding
    $env:PYTHONUTF8 = $previousUtf8
}
