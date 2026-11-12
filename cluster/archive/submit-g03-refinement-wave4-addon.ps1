# Add-on only: five missing seed-B runs. Requires five free GPU slots within the shared 24-slot limit; does not resubmit the original 16.
$ErrorActionPreference = 'Stop'
foreach ($runName in @(
    'clu-g03-refine-wave4-n4k-t25-e3-m5-seed456790'
    'clu-g03-refine-wave4-n4k-t75-e3-m5-seed456790'
    'clu-g03-refine-wave4-n4k-t100-e3-m5-seed456790'
    'clu-g03-refine-wave4-n6k-t75-e3-m5-seed456790'
    'clu-g03-refine-wave4-n10k-t25-e3-m5-seed456790'
)) {
    & (Join-Path $PSScriptRoot '../submit.ps1') -Run $runName
}
