# Requires 17 free one-GPU slots in the shared 24-slot workspace; this is a submission loop, not a capacity-aware queue.
$ErrorActionPreference = 'Stop'
foreach ($runName in @(
    'clu-g03-refine-wave3-leader-seed456790'
    'clu-g03-refine-wave3-envs10k-seed456789'
    'clu-g03-refine-wave3-envs10k-seed456790'
    'clu-g03-refine-wave3-horizon25-seed456789'
    'clu-g03-refine-wave3-horizon25-seed456790'
    'clu-g03-refine-wave3-horizon100-seed456789'
    'clu-g03-refine-wave3-horizon100-seed456790'
    'clu-g03-refine-wave3-epochs2-seed456789'
    'clu-g03-refine-wave3-epochs2-seed456790'
    'clu-g03-refine-wave3-epochs4-seed456789'
    'clu-g03-refine-wave3-epochs4-seed456790'
    'clu-g03-refine-wave3-minibatches4-seed456789'
    'clu-g03-refine-wave3-minibatches4-seed456790'
    'clu-g03-refine-wave3-minibatches8-seed456789'
    'clu-g03-refine-wave3-minibatches8-seed456790'
    'clu-g03-refine-wave3-horizon75-seed456789'
    'clu-g03-refine-wave3-horizon75-seed456790'
)) {
    & (Join-Path $PSScriptRoot '../submit.ps1') -Run $runName
}
