# Execute the actual platform-only entry point. Only wsl.exe is stubbed; no
# distribution, feature or Windows permission is changed by this test.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$installer = Join-Path $root 'packaging/runtime/wsl/Install-AgentFEM.ps1'
$statusScript = Join-Path $root 'packaging/runtime/wsl/Get-AgentFEMInstallStatus.ps1'
$originalLocal = $env:LOCALAPPDATA
$originalArch = $env:PROCESSOR_ARCHITECTURE
$originalWow = $env:PROCESSOR_ARCHITEW6432
$temp = Join-Path ([IO.Path]::GetTempPath()) ('AgentFEM platform ' + [guid]::NewGuid().ToString())
function Assert($Condition, $Message) { if (-not $Condition) { throw $Message } }
$global:AgentFEMTestVersionProbes = 0
function global:wsl.exe {
    if (($args -join ' ') -ne '--version') { throw "Unexpected WSL mutation: $args" }
    $global:AgentFEMTestVersionProbes++
    $global:LASTEXITCODE = 0
    return 'WSL version: 2.6.1.0'
}
try {
    $env:LOCALAPPDATA = $temp
    $env:PROCESSOR_ARCHITECTURE = 'AMD64'
    $env:PROCESSOR_ARCHITEW6432 = 'AMD64'
    # Raw source template: no rendered checksum, runtime image or verifier exists.
    & $installer -PrepareWslOnly -NonInteractive -WslBootstrapSource None
    Assert ($LASTEXITCODE -eq 0) 'Source-only platform entry point failed'
    $stateRoot = Join-Path $temp 'AgentFEM\Installations'
    $report = (& $statusScript -StateRoot $stateRoot | Out-String) | ConvertFrom-Json
    Assert ($report.installations.Count -eq 1) 'Platform state not addressable'
    Assert ($report.installations[0].stage -eq 'platform_ready') 'Platform was falsely reported as installed AgentFEM'
    $resume = $report.installations[0].resume
    Assert ($resume.arguments -contains '-PrepareWslOnly') 'Resume lost the mirror platform route'
    Assert ($resume.arguments -contains 'None') 'Resume lost the selected WSL source'
    Assert ($resume.arguments -contains $installer) 'Resume script path was not preserved as one argument'
    Assert (-not $report.installer_busy) 'Installer lock leaked after completion'
    & $installer -PrepareWslOnly -NonInteractive -WslBootstrapSource None
    Assert ($LASTEXITCODE -eq 0 -and $global:AgentFEMTestVersionProbes -eq 2) 'Platform preparation not resumable'
    & $installer -PrepareWslOnly -Upgrade -NonInteractive -WslBootstrapSource None
    Assert ($LASTEXITCODE -ne 0) 'Platform-only silently accepted destructive upgrade options'
    Assert ($global:AgentFEMTestVersionProbes -eq 2) 'Conflicting options reached WSL'
    function global:wsl.exe { $global:LASTEXITCODE = -2147024891; return '' }
    & $installer -PrepareWslOnly -NonInteractive -WslBootstrapSource None
    Assert ($LASTEXITCODE -ne 0) 'Denied version probe was accepted'
    $report = (& $statusScript -StateRoot $stateRoot | Out-String) | ConvertFrom-Json
    Assert ($report.installations[0].error_code -eq 'AFM-WIN-COMMAND-BLOCKED') 'Denied probe was confused with missing WSL'
    Assert ($report.installations[0].stage -eq 'awaiting_authorization') 'Denied probe did not request scoped approval'
    Write-Host 'Platform entry point passed (WSL mocked, no real host preparation).'
} finally {
    Remove-Item Function:\wsl.exe -ErrorAction SilentlyContinue
    Remove-Variable AgentFEMTestVersionProbes -Scope Global -ErrorAction SilentlyContinue
    $env:LOCALAPPDATA = $originalLocal
    $env:PROCESSOR_ARCHITECTURE = $originalArch
    $env:PROCESSOR_ARCHITEW6432 = $originalWow
    Remove-Item -Recurse -Force $temp -ErrorAction SilentlyContinue
}
# Expected failing installer calls above set LASTEXITCODE. Do not let the CI
# shell mistake that asserted failure for a failure of the test harness itself.
exit 0
