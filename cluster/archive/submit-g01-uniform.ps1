$ErrorActionPreference = 'Stop'
foreach ($runName in @(
    'clu-g01-uniform-30m-repeat-small-reuse'
    'clu-g01-uniform-30m-repeat-wide-minibatch'
    'clu-g01-uniform-30m-repeat-fast-sampling'
    'clu-g01-uniform-30m-wide-reuse'
    'clu-g01-uniform-30m-fresh-reuse'
    'clu-g01-uniform-30m-longer-context'
    'clu-g01-uniform-30m-more-reuse'
    'clu-g01-uniform-30m-small-heavy-reuse'
)) {
    & (Join-Path $PSScriptRoot '../submit.ps1') -Run $runName
}
