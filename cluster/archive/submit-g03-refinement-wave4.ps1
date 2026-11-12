# Requires 16 free one-GPU slots within the shared 24-slot workspace; submissions are sequential, training runs independently.
$ErrorActionPreference = 'Stop'
foreach ($runName in @(
    'clu-g03-refine-wave4-n4k-t25-e3-m5-seed456789'
    'clu-g03-refine-wave4-n4k-t75-e3-m5-seed456789'
    'clu-g03-refine-wave4-n4k-t100-e3-m5-seed456789'
    'clu-g03-refine-wave4-n6k-t75-e3-m5-seed456789'
    'clu-g03-refine-wave4-n10k-t25-e3-m5-seed456789'
    'clu-g03-refine-wave4-n6k-t75-e2-m5-seed456789'
    'clu-g03-refine-wave4-n6k-t75-e4-m5-seed456789'
    'clu-g03-refine-wave4-n10k-t25-e2-m5-seed456789'
    'clu-g03-refine-wave4-n10k-t25-e4-m5-seed456789'
    'clu-g03-refine-wave4-n4k-t100-e2-m5-seed456789'
    'clu-g03-refine-wave4-n6k-t75-e3-m6-seed456789'
    'clu-g03-refine-wave4-n6k-t75-e3-m10-seed456789'
    'clu-g03-refine-wave4-n4k-t100-e3-m10-seed456789'
    'clu-g03-refine-wave4-n6k-t100-e3-m10-seed456789'
    'clu-g03-refine-wave4-n8k-t100-e3-m10-seed456789'
    'clu-g03-refine-wave4-n6k-t50-e3-m6-seed456790'
)) {
    & (Join-Path $PSScriptRoot '../submit.ps1') -Run $runName
}
