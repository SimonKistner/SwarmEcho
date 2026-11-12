# Requires six free GPU slots in workspace 129; this is not a capacity-aware queue.
$ErrorActionPreference = 'Stop'
foreach ($runName in @(
    'clu-g03-refine-wave2-envs4k-seed456789'
    'clu-g03-refine-wave2-envs8k-seed456789'
    'clu-g03-refine-wave2-horizon100-seed456789'
    'clu-g03-refine-wave2-epochs2-seed456789'
    'clu-g03-refine-wave2-minibatches4-seed456789'
    'clu-g03-refine-wave2-minibatches6-seed456789'
)) {
    & (Join-Path $PSScriptRoot '../submit.ps1') -Run $runName
}
