# Static split: submit all twelve, the first one, or the other eleven.
[CmdletBinding()]
param(
    [ValidateSet('all12', 'remaining11', 'test1')]
    [string]$Batch = 'all12',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$configs = @(
    'forward_aux_l0p3_eff25_seed456789.json'
    'forward_aux_l0p3_eff25_seed456790.json'
    'forward_aux_l0p3_eff50_seed456789.json'
    'forward_aux_l0p3_eff50_seed456790.json'
    'forward_aux_l0p3_eff100_seed456789.json'
    'forward_aux_l0p3_eff100_seed456790.json'
    'reverse_aux_l0p3_eff25_seed456789.json'
    'reverse_aux_l0p3_eff25_seed456790.json'
    'reverse_aux_l0p3_eff50_seed456789.json'
    'reverse_aux_l0p3_eff50_seed456790.json'
    'reverse_aux_l0p3_eff100_seed456789.json'
    'reverse_aux_l0p3_eff100_seed456790.json'
)
$selected = switch ($Batch) {
    'all12' { $configs }
    'remaining11' { $configs[1..11] }
    'test1' { $configs[0] }
}
# Check the whole selected set before any submission. Static membership stays
# unchanged; confirm every conflict before the first child submission starts.
$repoRoot = Split-Path -Parent $PSScriptRoot
$receiptRoot = Join-Path $PSScriptRoot '.submissions'
$receipts = @()
if (Test-Path -LiteralPath $receiptRoot) {
    $receipts = @(Get-ChildItem -LiteralPath $receiptRoot -Directory)
}
$conflicts = @()
$overrideConfigs = @{}
foreach ($config in $selected) {
    $plan = Get-Content -LiteralPath (Join-Path $PSScriptRoot "g03_efficiency_branch_retry/$config") -Raw | ConvertFrom-Json
    $runName = 'CLU_' + $plan.name + '_seed' + $plan.seeds[0] + '_' + $plan.run_postfix
    $trackingPath = Join-Path $repoRoot "outputs/$runName/cluster_run.json"
    if (Test-Path -LiteralPath $trackingPath) {
        $experimentId = (Get-Content -LiteralPath $trackingPath -Raw | ConvertFrom-Json).experiment_id
        $conflicts += "$runName (experiment $experimentId; local tracking)"
        $overrideConfigs[$config] = $true
    }
    foreach ($receipt in $receipts) {
        if ($receipt.Name -ne $runName -and -not $receipt.Name.StartsWith($runName + '-receipt-')) { continue }
        $resultPath = Join-Path $receipt.FullName 'result.json'
        if ((Test-Path -LiteralPath $resultPath) -and (Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json).experiment_id) {
            $experimentId = (Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json).experiment_id
            $conflicts += "$runName (experiment $experimentId; $($receipt.Name))"
            $overrideConfigs[$config] = $true
        }
    }
}
if ($conflicts.Count -gt 0) {
    Write-Host 'Local cluster folders belong to previous experiments:'
    $conflicts | ForEach-Object { Write-Host "  $_" }
    Write-Host 'Override replaces local tracking after submission and archives old records.'
    Write-Host 'PFSS training history is retained; use a new run_postfix for fresh output directories.'
    if ([Console]::IsInputRedirected) {
        throw 'Override requires interactive confirmation; no jobs were submitted.'
    }
    do {
        $response = Read-Host 'Press Enter to override, or Ctrl+C to abort'
    } while ($response -ne '')
}
foreach ($config in $selected) {
    & (Join-Path $PSScriptRoot 'submit.ps1') -Config (Join-Path $PSScriptRoot "g03_efficiency_branch_retry/$config") -Python $Python -OverrideExisting:($overrideConfigs.ContainsKey($config))
}
