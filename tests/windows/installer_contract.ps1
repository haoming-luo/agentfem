# These tests exercise policy/state behaviour, not a real WSL installation.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $root 'packaging/runtime/wsl/Installer-State.ps1')
$originalLocal = $env:LOCALAPPDATA
$temp = Join-Path ([IO.Path]::GetTempPath()) ('AgentFEM ' + [char]0x6d4b + [char]0x8bd5 + ' ' + [guid]::NewGuid().ToString())
$env:LOCALAPPDATA = $temp
function Assert($Condition, $Message) { if (-not $Condition) { throw $Message } }
try {
    Initialize-InstallState -Identity @{version='test'; image_sha256='abc'}
    $script:InstallState.distribution = 'AgentFEM-test'
    Save-InstallState -Stage 'importing'
    $state = Get-Content -Raw (Join-Path $script:InstallStateDirectory 'status.json') | ConvertFrom-Json
    Assert ($state.stage -eq 'importing') 'State stage not persisted'
    $busy = $false
    try { Initialize-InstallState -Identity @{version='test'; image_sha256='abc'} }
    catch { $busy = $_.Exception.Message -match 'AFM-WIN-INSTALL-BUSY' }
    Assert $busy 'Duplicate install was not rejected'
    $busy = $false
    try { Initialize-InstallState -Identity @{version='other'; image_sha256='def'} }
    catch { $busy = $_.Exception.Message -match 'AFM-WIN-INSTALL-BUSY' }
    Assert $busy 'Different packages can race over the same WSL distribution'
    $statusScript = Join-Path $root 'packaging/runtime/wsl/Get-AgentFEMInstallStatus.ps1'
    $stateRoot = Split-Path $script:InstallStateDirectory -Parent
    $report = (& $statusScript -StateRoot $stateRoot | Out-String) | ConvertFrom-Json
    Assert ($report.installer_busy -and -not $report.runtime_probed) 'Read-only status lost busy state'
    $script:InstallLock.Dispose()
    Initialize-InstallState -Identity @{version='test'; image_sha256='abc'}
    Assert ($script:InstallState.distribution -eq 'AgentFEM-test') 'Resume lost imported distribution'
    Save-InstallState -Stage 'restart_required'
    Save-InstallState -Stage 'completed'
    $state = Get-Content -Raw (Join-Path $script:InstallStateDirectory 'status.json') | ConvertFrom-Json
    Assert ($state.stage -eq 'completed') 'Atomic replacement failed'
    Assert ($state.next_action -eq 'use_verified_runtime') 'Agent next action absent'
    $script:InstallLock.Dispose()
    # A syntactically valid but foreign record must not authorise reuse either.
    $statePath = Join-Path $script:InstallStateDirectory 'status.json'
    $state.identity.version = 'tampered'
    $state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
    $identityMismatch = $false
    try { Initialize-InstallState -Identity @{version='test'; image_sha256='abc'} }
    catch { $identityMismatch = $_.Exception.Message -match 'STATE-IDENTITY' }
    Assert $identityMismatch 'Saved request identity was not validated'
    Assert (-not $script:InstallStateOwned) 'Invalid state could be overwritten as owned'
    $script:InstallLock.Dispose()
    # Corrupt state is retained instead of authorising reuse or silently resetting.
    $statePath = Join-Path $script:InstallStateDirectory 'status.json'
    Set-Content -LiteralPath $statePath -Value '{bad json' -Encoding UTF8
    $corrupt = $false
    try { Initialize-InstallState -Identity @{version='test'; image_sha256='abc'} }
    catch { $corrupt = $true }
    Assert $corrupt 'Corrupt state was silently accepted'
    Assert ((Get-Content -Raw -LiteralPath $statePath) -match 'bad json') 'Corrupt evidence was overwritten'
    $script:InstallLock.Dispose()
    $report = (& $statusScript -StateRoot $stateRoot | Out-String) | ConvertFrom-Json
    Assert ($LASTEXITCODE -eq 1 -and $report.errors.Count -eq 1) 'Status hid a damaged record'
    $report = (& $statusScript -StateRoot (Join-Path $temp 'absent') | Out-String) | ConvertFrom-Json
    Assert ($report.status -eq 'not_started') 'Empty status is not explicit'
    Assert (-not (Test-Path (Join-Path $temp 'absent'))) 'Status query created state'
    $tokens = $null; $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile(
        (Join-Path $root 'packaging/runtime/wsl/Install-AgentFEM.ps1'), [ref]$tokens, [ref]$errors)
    Assert ($errors.Count -eq 0) 'Installer syntax errors'
    # Load from a real file so PowerShell preserves $PSScriptRoot in functions.
    # Evaluating their text loses that automatic variable and does not reproduce
    # the packaged installer execution context.
    $definitions = $ast.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst]}, $false) |
        ForEach-Object { $_.Extent.Text }
    $functionFile = Join-Path $temp 'installer-functions.ps1'
    Set-Content -LiteralPath $functionFile -Value ($definitions -join "`n") -Encoding UTF8
    . $functionFile
    $completeImplementation = ${function:Complete-NewInstallation}
    $identityImplementation = ${function:Test-AgentFEMBuildIdentity}
    $script:InstallStateOwned = $false
    function Test-AdministratorSession { return $false }
    $NonInteractive = $true
    $script:launchCount = 0
    function Start-Process { $script:launchCount++; throw 'should not prompt' }
    $needsApproval = $false
    try { Install-WslPlatform -Source 'Auto' }
    catch { $needsApproval = $_.Exception.Message -match 'ELEVATION-REQUIRED' }
    Assert ($needsApproval -and $script:launchCount -eq 0) 'Agent mode opened a hidden UAC prompt'
    $NonInteractive = $false
    $script:launchCount = 0
    function Start-Process { $script:launchCount++; throw 'sandbox policy rejected launch' }
    $blocked = $false
    try { Install-WslPlatform -Source 'Auto' } catch { $blocked = $_.Exception.Message -match 'ELEVATION-BLOCKED' }
    Assert ($blocked -and $script:launchCount -eq 1) 'Policy rejection retried elevation'
    function Start-Process { $script:launchCount++; return [pscustomobject]@{ExitCode=3010} }
    $script:launchCount = 0
    $restart = $false
    try { Install-WslPlatform -Source 'Auto' } catch { $restart = $_.Exception.Message -match 'RESTART-REQUIRED' }
    Assert ($restart -and $script:launchCount -eq 1) 'Restart retried elevation'
    function Start-Process { $script:launchCount++; return [pscustomobject]@{ExitCode=740} }
    $script:launchCount = 0
    $denied = $false
    try { Install-WslPlatform -Source 'Auto' } catch { $denied = $_.Exception.Message -match 'ELEVATION-BLOCKED' }
    Assert ($denied -and $script:launchCount -eq 1) 'Elevation-required exit retried another source'
    function Start-Process { $script:launchCount++; return [pscustomobject]@{ExitCode=-2147024891} }
    $script:launchCount = 0
    $denied = $false
    try { Install-WslPlatform -Source 'Auto' } catch { $denied = $_.Exception.Message -match 'ELEVATION-BLOCKED' }
    Assert ($denied -and $script:launchCount -eq 1) 'HRESULT access denial retried another source'
    function Start-Process {
        $script:launchCount++
        return [pscustomobject]@{ExitCode=$(if ($script:launchCount -eq 1) {1} else {0})}
    }
    $script:launchCount = 0
    Install-WslPlatform -Source 'Auto'
    Assert ($script:launchCount -eq 2) 'Documented fallback did not run once'
    # Already elevated: use the current authorised process, no extra UAC launcher.
    function Test-AdministratorSession { return $true }
    function wsl.exe { $global:LASTEXITCODE = 0; $script:directCalls++ }
    $script:launchCount = 0; $script:directCalls = 0
    Install-WslPlatform -Source 'Auto'
    Assert ($script:directCalls -eq 1 -and $script:launchCount -eq 0) 'Admin session launched unnecessary UAC'
    function Test-AdministratorSession { return $false }
    function wsl.exe { $global:LASTEXITCODE = 0; return 'WSL version: 2.3.26.0' }
    $platformVersion = Require-WslPlatform -Source 'None' -PlatformOnly
    Assert ($platformVersion -eq [version]'2.3.26') 'Mirror route forced a packaging-only update'
    $imageVersionRejected = $false
    try { Require-WslPlatform -Source 'None' | Out-Null }
    catch { $imageVersionRejected = $_.Exception.Message -match 'WSL-TOO-OLD' }
    Assert $imageVersionRejected 'Full .wsl package lost its minimum-version gate'
    # A recorded import must be reused and verified, not imported twice.
    $script:InstallState = @{distribution='AgentFEM-test'}
    $RuntimeVersion = 'test'
    $script:imports = 0; $script:completed = 0
    function Get-WslDistributions { return @('AgentFEM-test') }
    function Get-AgentFEMRuntimeVersion { return 'test' }
    function Test-AgentFEMBuildIdentity { }
    function Install-AgentFEMDistribution { $script:imports++ }
    function Complete-NewInstallation { $script:completed++ }
    Install-SideBySide -RequestedName 'AgentFEM' -Projects 'C:\Projects'
    Assert ($script:imports -eq 0 -and $script:completed -eq 1) 'Resume duplicated an import'
    function Get-AgentFEMRuntimeVersion { return 'different' }
    $mismatch = $false
    try { Install-SideBySide -RequestedName 'AgentFEM' -Projects 'C:\Projects' }
    catch { $mismatch = $_.Exception.Message -match 'RESUME-IDENTITY' }
    Assert ($mismatch -and $script:completed -eq 1) 'Resume accepted a different runtime'
    # A same-version different wheel/lock is also not this package.
    $ExpectedRuntimeRecordSha256 = 'a' * 64
    function wsl.exe { $global:LASTEXITCODE = 0; return (('b' * 64) + '  runtime-release.json') }
    $mismatch = $false
    try { & $identityImplementation -Name 'AgentFEM-test' }
    catch { $mismatch = $_.Exception.Message -match 'RESUME-IDENTITY' }
    Assert $mismatch 'Same version / different build was accepted'
    function wsl.exe { $global:LASTEXITCODE = 0; return (('a' * 64) + '  runtime-release.json') }
    & $identityImplementation -Name 'AgentFEM-test'
    # First-install completion requires fresh acceptance, not only exit code zero.
    function Set-ProjectWorkspaceEnvironment { }
    function Restore-ProjectWorkspaceEnvironment { }
    function Get-DefaultLinuxUser { return 'researcher' }
    function Convert-ToWslPath { param($Name, $WindowsPath) return $WindowsPath }
    function Test-AgentFEMDistribution { }
    function Save-InstallState { param($Stage, $Message) $script:lastStage = $Stage }
    $script:writeAcceptance = $true
    function Invoke-WslChecked {
        param($Arguments, $FailureMessage)
        if ($Arguments -contains '--output' -and $script:writeAcceptance) {
            @{schema='agentfem.runtime-acceptance';status='accepted'} | ConvertTo-Json |
                Set-Content -LiteralPath (Join-Path $script:InstallStateDirectory 'acceptance.json') -Encoding UTF8
        }
    }
    $NonInteractive = $true
    function wsl.exe { $global:LASTEXITCODE = 0; return 'researcher L 2026-01-01 0 99999 7 -1' }
    $needsAccount = $false
    try { & $completeImplementation -Name 'AgentFEM-test' -Projects 'C:\Projects' }
    catch { $needsAccount = $_.Exception.Message -match 'ACCOUNT-INPUT-REQUIRED' }
    Assert $needsAccount 'Agent mode did not hand off a locked account'
    function wsl.exe { $global:LASTEXITCODE = 0; return 'researcher P 2026-01-01 0 99999 7 -1' }
    & $completeImplementation -Name 'AgentFEM-test' -Projects 'C:\Projects'
    Assert ($script:lastStage -eq 'completed') 'Valid acceptance did not finish'
    $script:writeAcceptance = $false
    $noReport = $false
    try { & $completeImplementation -Name 'AgentFEM-test' -Projects 'C:\Projects' }
    catch { $noReport = $true }
    Assert ($noReport -and $script:lastStage -ne 'completed') 'Missing fresh report reused stale success'
    Write-Host 'Installer contract passed (mocked, not real WSL).'
} finally {
    if ($script:InstallLock) { $script:InstallLock.Dispose() }
    $env:LOCALAPPDATA = $originalLocal
    Remove-Item -Recurse -Force $temp -ErrorAction SilentlyContinue
}
