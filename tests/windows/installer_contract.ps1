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
    $script:InstallLock.Dispose()
    Initialize-InstallState -Identity @{version='test'; image_sha256='abc'}
    Assert ($script:InstallState.distribution -eq 'AgentFEM-test') 'Resume lost imported distribution'
    Save-InstallState -Stage 'restart_required'
    Save-InstallState -Stage 'completed'
    $state = Get-Content -Raw (Join-Path $script:InstallStateDirectory 'status.json') | ConvertFrom-Json
    Assert ($state.stage -eq 'completed') 'Atomic replacement failed'
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
    $script:InstallStateOwned = $false
    function Test-AdministratorSession { return $false }
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
    # A recorded import must be reused and verified, not imported twice.
    $script:InstallState = @{distribution='AgentFEM-test'}
    $RuntimeVersion = 'test'
    $script:imports = 0; $script:completed = 0
    function Get-WslDistributions { return @('AgentFEM-test') }
    function Get-AgentFEMRuntimeVersion { return 'test' }
    function Install-AgentFEMDistribution { $script:imports++ }
    function Complete-NewInstallation { $script:completed++ }
    Install-SideBySide -RequestedName 'AgentFEM' -Projects 'C:\Projects'
    Assert ($script:imports -eq 0 -and $script:completed -eq 1) 'Resume duplicated an import'
    function Get-AgentFEMRuntimeVersion { return 'different' }
    $mismatch = $false
    try { Install-SideBySide -RequestedName 'AgentFEM' -Projects 'C:\Projects' }
    catch { $mismatch = $_.Exception.Message -match 'RESUME-IDENTITY' }
    Assert ($mismatch -and $script:completed -eq 1) 'Resume accepted a different runtime'
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
