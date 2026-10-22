# These tests exercise policy/state behaviour, not a real WSL installation.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $root 'packaging/runtime/wsl/Installer-State.ps1')
$originalLocal = $env:LOCALAPPDATA
$temp = Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
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
    $tokens = $null; $errors = $null
    $ast = [Management.Automation.Language.Parser]::ParseFile(
        (Join-Path $root 'packaging/runtime/wsl/Install-AgentFEM.ps1'), [ref]$tokens, [ref]$errors)
    Assert ($errors.Count -eq 0) 'Installer syntax errors'
    foreach ($fn in $ast.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst]}, $false)) {
        Invoke-Expression $fn.Extent.Text
    }
    $script:InstallStateOwned = $false
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
    Write-Host 'Installer contract passed (mocked, not real WSL).'
} finally {
    if ($script:InstallLock) { $script:InstallLock.Dispose() }
    $env:LOCALAPPDATA = $originalLocal
    Remove-Item -Recurse -Force $temp -ErrorAction SilentlyContinue
}
