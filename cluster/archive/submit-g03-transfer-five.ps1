$ErrorActionPreference = 'Stop'
foreach ($runName in @(
    'clu-g03-transfer-30m-small-reuse'
    'clu-g03-transfer-30m-wide-reuse'
    'clu-g03-transfer-30m-fresh-balanced'
    'clu-g03-transfer-30m-wide-minibatch'
    'clu-g03-transfer-30m-fast-sampling'
)) {
    & (Join-Path $PSScriptRoot '../submit.ps1') -Run $runName
}
