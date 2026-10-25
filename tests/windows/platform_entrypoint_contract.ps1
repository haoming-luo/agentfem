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

    # Exercise full-package validation before the inventory query. This is an
    # inert local fixture, NOT a built/accepted runtime or a real WSL image.
    $bundle = Join-Path $temp 'bundle [test]'
    New-Item -ItemType Directory -Path $bundle | Out-Null
    foreach ($helper in @('Installer-State.ps1', 'Windows-HostProbe.ps1', 'Installer-Presentation.ps1')) {
        Copy-Item -LiteralPath (Join-Path $root ('packaging/runtime/wsl/' + $helper)) -Destination $bundle
    }
    $image = Join-Path $bundle 'image [test].wsl'
    Set-Content -LiteralPath $image -Value 'inert image fixture'
    $digest = (Get-FileHash -LiteralPath $image -Algorithm SHA256).Hash
    Set-Content -LiteralPath (Join-Path $bundle 'verify_installed_runtime.py') -Value '# never executed'
    $fullInstaller = Join-Path $bundle 'Install-AgentFEM.ps1'
    $source = Get-Content -LiteralPath $installer -Raw
    $source.Replace('@VERSION@', 'fixture').Replace('@RUNTIME_RECORD_SHA256@', ('a' * 64)) |
        Set-Content -LiteralPath $fullInstaller -Encoding UTF8
    $env:LOCALAPPDATA = Join-Path $temp 'full-state'
    $env:AGENTFEM_WSL_FIXTURE_MODE = 'inventory-timeout'
    $timer.Restart()
    & $fullInstaller -Image $image -ExpectedSha256 $digest -NonInteractive -WslBootstrapSource None `
        -ProbeTimeoutSeconds 1 -ProjectDirectory (Join-Path $temp 'projects [test]')
    $timer.Stop()
    Assert ($LASTEXITCODE -ne 0 -and $timer.Elapsed.TotalSeconds -lt 8) 'Full installer inventory query was not bounded'
    $report = (& $statusScript -StateRoot (Join-Path $env:LOCALAPPDATA 'AgentFEM\Installations') | Out-String) | ConvertFrom-Json
    Assert ($report.installations[0].error_code -eq 'AFM-WIN-PROBE-TIMEOUT') 'Full package failed before its inventory timeout'
    Assert (-not $report.installer_busy -and -not $report.installations[0].distribution) 'Failed inventory acquired a distribution or leaked its lock'
    $calls = @(Get-Content -LiteralPath $env:AGENTFEM_WSL_FIXTURE_LOG)
    Assert ($calls.Count -eq 9 -and ($calls[6..8] -join ',') -eq '--version,--status,--list --quiet') 'Inventory failure triggered a mutation or retry'
    & $fullInstaller -Image $image -ExpectedSha256 ('0' * 64) -NonInteractive -WslBootstrapSource None
    Assert ($LASTEXITCODE -ne 0) 'Invalid image checksum was accepted'
    $report = (& $statusScript -StateRoot (Join-Path $env:LOCALAPPDATA 'AgentFEM\Installations') | Out-String) | ConvertFrom-Json
    Assert (@($report.installations | Where-Object { $_.error_code -eq 'AFM-WIN-PACKAGE-CHECKSUM' }).Count -eq 1) 'Wrong checksum lost its named failure'
    Assert (@(Get-Content -LiteralPath $env:AGENTFEM_WSL_FIXTURE_LOG).Count -eq 9) 'Bad image reached WSL before checksum rejection'
    # Inject a storage fault only into the private fixture helper. Recording a
    # failure may fail too, but the original error and nonzero exit must survive.
    $helperPath = Join-Path $bundle 'Installer-State.ps1'
    $helperSource = Get-Content -LiteralPath $helperPath -Raw
    $fault = @'

$script:SaveStateWithoutFault = ${function:Save-InstallState}
function Save-InstallState {
    param([string]$Stage, [string]$Message = '')
    if ($Stage -eq 'failed') { throw 'injected state-storage failure' }
    & $script:SaveStateWithoutFault -Stage $Stage -Message $Message
}
'@
    Set-Content -LiteralPath $helperPath -Value ($helperSource + $fault) -Encoding UTF8
    $failureOutput = (& $fullInstaller -Image $image -ExpectedSha256 ('0' * 64) `
        -NonInteractive -WslBootstrapSource None 3>&1 6>&1 | Out-String)
    Assert ($LASTEXITCODE -ne 0) 'Failure-record storage fault hid unsuccessful installation'
    Assert ($failureOutput -match 'AFM-WIN-STATE-WRITE' -and $failureOutput -match 'AFM-WIN-PACKAGE-CHECKSUM') 'Failure-record storage fault swallowed the original error'
    $report = (& $statusScript -StateRoot (Join-Path $env:LOCALAPPDATA 'AgentFEM\Installations') | Out-String) | ConvertFrom-Json
    Assert (-not $report.installer_busy) 'Failure-record storage fault leaked the installation lock'
    Assert (@(Get-Content -LiteralPath $env:AGENTFEM_WSL_FIXTURE_LOG).Count -eq 9) 'Failure-record storage fault launched WSL'
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
