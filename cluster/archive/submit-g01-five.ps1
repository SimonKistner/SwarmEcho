$ErrorActionPreference = 'Stop'
foreach ($runName in @(
    'clu-g01-30m-wide-minibatch'
    'clu-g01-30m-fresh-data'
    'clu-g01-30m-rapid-feedback'
    'clu-g01-30m-long-context'
    'clu-g01-30m-intensive-reuse'
)) {
    & (Join-Path $PSScriptRoot '../submit.ps1') -Run $runName
}
