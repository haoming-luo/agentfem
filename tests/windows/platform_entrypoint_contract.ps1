# Execute the actual platform-only entry point. Only wsl.exe is a fixture; no
# distribution, feature or Windows permission is changed by this test.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$installer = Join-Path $root 'packaging/runtime/wsl/Install-AgentFEM.ps1'
$statusScript = Join-Path $root 'packaging/runtime/wsl/Get-AgentFEMInstallStatus.ps1'
$originalLocal = $env:LOCALAPPDATA
$originalArch = $env:PROCESSOR_ARCHITECTURE
$originalWow = $env:PROCESSOR_ARCHITEW6432
$originalPath = $env:PATH
$originalMode = $env:AGENTFEM_WSL_FIXTURE_MODE
$originalLog = $env:AGENTFEM_WSL_FIXTURE_LOG
$temp = Join-Path ([IO.Path]::GetTempPath()) ('AgentFEM [platform] ' + [char]0x6d4b + ' ' + [guid]::NewGuid().ToString())
function Assert($Condition, $Message) { if (-not $Condition) { throw $Message } }
try {
    New-Item -ItemType Directory -Path $temp | Out-Null
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    $exe = Join-Path $temp 'wsl.exe'
    & $compiler /nologo /target:exe "/out:$exe" (Join-Path $PSScriptRoot 'WslProbeFixture.cs')
    Assert ($LASTEXITCODE -eq 0) 'WSL fixture did not compile'
    $env:PATH = $temp + ';' + $originalPath
    $env:AGENTFEM_WSL_FIXTURE_MODE = 'platform'
    $env:AGENTFEM_WSL_FIXTURE_LOG = Join-Path $temp 'calls.txt'
    Assert ((Get-Command wsl.exe -CommandType Application).Source -eq $exe) 'Real Windows WSL would be invoked by this test'
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
    Assert ($LASTEXITCODE -eq 0) 'Platform preparation not resumable'
    $calls = @(Get-Content -LiteralPath $env:AGENTFEM_WSL_FIXTURE_LOG)
    Assert (($calls -join ',') -eq '--version,--status,--version,--status') 'Repeated platform entry launched an unexpected WSL operation'
    & $installer -PrepareWslOnly -Upgrade -NonInteractive -WslBootstrapSource None
    Assert ($LASTEXITCODE -ne 0) 'Platform-only silently accepted destructive upgrade options'
    Assert (@(Get-Content -LiteralPath $env:AGENTFEM_WSL_FIXTURE_LOG).Count -eq 4) 'Conflicting options reached WSL'
    $env:AGENTFEM_WSL_FIXTURE_MODE = 'denied'
    & $installer -PrepareWslOnly -NonInteractive -WslBootstrapSource None
    Assert ($LASTEXITCODE -ne 0) 'Denied version probe was accepted'
    $report = (& $statusScript -StateRoot $stateRoot | Out-String) | ConvertFrom-Json
    Assert ($report.installations[0].error_code -eq 'AFM-WIN-COMMAND-BLOCKED') 'Denied probe was confused with missing WSL'
    Assert ($report.installations[0].stage -eq 'awaiting_authorization') 'Denied probe did not request scoped approval'
    $env:AGENTFEM_WSL_FIXTURE_MODE = 'timeout'
    $timer = [Diagnostics.Stopwatch]::StartNew()
    & $installer -PrepareWslOnly -NonInteractive -WslBootstrapSource None -ProbeTimeoutSeconds 1
    $timer.Stop()
    Assert ($LASTEXITCODE -ne 0 -and $timer.Elapsed.TotalSeconds -lt 8) 'Installer query was not bounded'
    $report = (& $statusScript -StateRoot $stateRoot | Out-String) | ConvertFrom-Json
    Assert ($report.installations[0].error_code -eq 'AFM-WIN-PROBE-TIMEOUT') 'Timeout lost its stable failure code'
    Assert (-not $report.installer_busy) 'Timed-out probe leaked the installation lock'
    Assert ($report.installations[0].resume.arguments -contains '-ProbeTimeoutSeconds') 'Resume lost explicit timeout'
    Assert (@(Get-Content -LiteralPath $env:AGENTFEM_WSL_FIXTURE_LOG).Count -eq 6) 'Failure triggered an unexpected repair/retry'
    Write-Host 'Platform entry point passed (inert native WSL fixture, no real host preparation).'
} finally {
    $env:PATH = $originalPath
    $env:AGENTFEM_WSL_FIXTURE_MODE = $originalMode
    $env:AGENTFEM_WSL_FIXTURE_LOG = $originalLog
    $env:LOCALAPPDATA = $originalLocal
    $env:PROCESSOR_ARCHITECTURE = $originalArch
    $env:PROCESSOR_ARCHITEW6432 = $originalWow
    Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
}
# Expected failing installer calls above set LASTEXITCODE. Do not let the CI
# shell mistake that asserted failure for a failure of the test harness itself.
exit 0
