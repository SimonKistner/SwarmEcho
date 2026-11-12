$ErrorActionPreference = 'Stop'
foreach ($runName in @(
    'clu-g03-g01-rank1'
    'clu-g03-g01-rank2'
    'clu-g03-g01-rank3'
    'clu-g03-g01-rank4'
    'clu-g03-g01-rank5'
    'clu-g03-diversity-controlled'
    'clu-g03-wide-fresh'
    'clu-g03-short-moderate-reuse'
    'clu-g03-long-context-fresh'
    'clu-g03-balanced-diversity'
)) {
    & (Join-Path $PSScriptRoot '../submit.ps1') -Run $runName
}
