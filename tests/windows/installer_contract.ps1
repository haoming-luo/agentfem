# These tests exercise policy/state behaviour, not a real WSL installation.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $root 'packaging/runtime/wsl/Installer-State.ps1')
. (Join-Path $root 'packaging/runtime/wsl/Windows-HostProbe.ps1')
$originalLocal = $env:LOCALAPPDATA
$temp = Join-Path ([IO.Path]::GetTempPath()) ('AgentFEM ' + [char]0x6d4b + [char]0x8bd5 + ' ' + [guid]::NewGuid().ToString())
$env:LOCALAPPDATA = $temp
function Assert($Condition, $Message) { if (-not $Condition) { throw $Message } }
try {
    New-Item -ItemType Directory -Path $temp -Force | Out-Null
    Push-Location $temp
    try {
        $paths = @{Image='.\source file.wsl'; ProjectDirectory='.\project files'; WslBootstrapSource='Store'}
        Resolve-InstallPathParameters -Parameters $paths
        Assert ($paths.Image -eq (Join-Path $temp 'source file.wsl')) 'Image path not anchored to original PowerShell location'
        Assert ($paths.ProjectDirectory -eq (Join-Path $temp 'project files')) 'Project destination not anchored'
        Assert ($paths.WslBootstrapSource -eq 'Store') 'Non-path option was changed'
        Assert (-not (Test-Path $paths.ProjectDirectory)) 'Resolving paths created user directories'
        $firstPath = $paths.ProjectDirectory
    } finally { Pop-Location }
    Resolve-InstallPathParameters -Parameters $paths
    Assert ($paths.ProjectDirectory -eq $firstPath) 'Resume from another directory changed target'
    function Resolve-BoundPath {
        param([string]$ProjectDirectory)
        Resolve-InstallPathParameters -Parameters $PSBoundParameters
        return $PSBoundParameters.ProjectDirectory
    }
    Assert ((Resolve-BoundPath -ProjectDirectory $firstPath) -eq $firstPath) 'Actual bound-parameter dictionary not supported'
    $invalidProvider = $false
    try { Resolve-InstallPathParameters -Parameters @{ProjectDirectory='Env:\TEMP'} }
    catch { $invalidProvider = $_.Exception.Message -match 'AFM-WIN-PATH-INVALID' }
    Assert $invalidProvider 'Non-filesystem path was accepted'
    $firstDefault = Join-Path $temp 'original [Documents]'
    $laterDefault = Join-Path $temp 'redirected [Documents]'
    function Get-DefaultInstallProjectDirectory { return $firstDefault }
    Initialize-InstallState -Identity @{version='path-test'; project_directory=''}
    Assert ((Resolve-InstallProjectDirectory) -eq $firstDefault) 'Default project directory was not resolved'
    Save-InstallState -Stage 'restart_required'
    $script:InstallLock.Dispose()
    function Get-DefaultInstallProjectDirectory { return $laterDefault }
    Initialize-InstallState -Identity @{version='path-test'; project_directory=''}
    Assert ((Resolve-InstallProjectDirectory) -eq $firstDefault) 'Resume followed a changed Documents folder instead of its recorded destination'
    Assert (-not (Test-Path -LiteralPath $firstDefault) -and -not (Test-Path -LiteralPath $laterDefault)) 'Path selection created workspace directories before approval'
    $badPathRejected = $false
    try { Resolve-InstallProjectDirectory -RequestedPath $laterDefault | Out-Null }
    catch { $badPathRejected = $_.Exception.Message -match 'AFM-WIN-STATE-PATH' }
    Assert $badPathRejected 'A conflicting recorded workspace was silently selected'
    $script:InstallState.effective_project_directory = '.\relative'
    $badPathRejected = $false
    try { Resolve-InstallProjectDirectory | Out-Null }
    catch { $badPathRejected = $_.Exception.Message -match 'AFM-WIN-STATE-PATH' }
    Assert $badPathRejected 'Relative saved workspace could change across resume'
    $pathTestDirectory = $script:InstallStateDirectory
    $script:InstallLock.Dispose()
    # Isolate the following inventory assertions from this additional record.
    Remove-Item -LiteralPath $pathTestDirectory -Recurse -Force
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
    $statusTokens = $null; $statusErrors = $null
    $statusAst = [Management.Automation.Language.Parser]::ParseFile($statusScript, [ref]$statusTokens, [ref]$statusErrors)
    Assert ($statusErrors.Count -eq 0) 'Status script syntax errors'
    $snapshotFunction = $statusAst.Find({param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Open-InstallStatusSnapshot'
    }, $false)
    . ([scriptblock]::Create($snapshotFunction.Extent.Text))
    $statePath = Join-Path $script:InstallStateDirectory 'status.json'
    $snapshot = Open-InstallStatusSnapshot -Path $statePath
    try {
        # Keep a real reader open across two actual atomic replacements. A
        # reader without delete sharing causes Windows sharing-violation here.
        Save-InstallState -Stage 'verifying_runtime'
        Save-InstallState -Stage 'importing' -Message 'new snapshot after concurrent publication'
        $reader = New-Object IO.StreamReader($snapshot, [Text.Encoding]::UTF8, $true)
        try { $oldSnapshot = $reader.ReadToEnd() | ConvertFrom-Json }
        finally { $reader.Dispose() }
        Assert ($oldSnapshot.stage -eq 'importing' -and $oldSnapshot.message -eq '') 'Opened snapshot changed while the installer published another revision'
    } finally { $snapshot.Dispose() }
    $report = (& $statusScript -StateRoot $stateRoot | Out-String) | ConvertFrom-Json
    Assert ($report.installations[0].message -eq 'new snapshot after concurrent publication') 'New status reader did not see the published snapshot'
    Assert ($report.installer_busy -and -not $report.runtime_probed) 'Read-only status lost busy state'
    $script:InstallLock.Dispose()
    $statePath = Join-Path $script:InstallStateDirectory 'status.json'
    $beforeStatus = (Get-FileHash -LiteralPath $statePath -Algorithm SHA256).Hash
    $report = (& $statusScript -StateRoot $stateRoot | Out-String) | ConvertFrom-Json
    Assert ($report.installations[0].activity -eq 'interrupted') 'Unlocked import was presented as still running'
    Assert ($report.installations[0].next_action -eq 'inspect_log_then_resume_same_installer') 'Interrupted agent would wait forever'
    Assert ($report.installations[0].recorded_next_action -eq 'wait_for_current_installer') 'Historical next action was lost'
    Assert ((Get-FileHash -LiteralPath $statePath -Algorithm SHA256).Hash -eq $beforeStatus) 'Read-only inference rewrote saved evidence'
    Initialize-InstallState -Identity @{version='test'; image_sha256='abc'}
    Assert ($script:InstallState.distribution -eq 'AgentFEM-test') 'Resume lost imported distribution'
    Save-InstallState -Stage 'checking_package'
    $script:InstallLock.Dispose()
    $report = (& $statusScript -StateRoot $stateRoot | Out-String) | ConvertFrom-Json
    Assert ($report.installations[0].activity -eq 'interrupted') 'Interrupted package hashing would be reported as running'
    Initialize-InstallState -Identity @{version='test'; image_sha256='abc'}
    Save-InstallState -Stage 'restart_required'
    Save-InstallState -Stage 'completed'
    $state = Get-Content -Raw (Join-Path $script:InstallStateDirectory 'status.json') | ConvertFrom-Json
    Assert ($state.stage -eq 'completed') 'Atomic replacement failed'
    Assert ($state.next_action -eq 'use_verified_runtime') 'Agent next action absent'
    $script:InstallLock.Dispose()
    $report = (& $statusScript -StateRoot $stateRoot | Out-String) | ConvertFrom-Json
    Assert ($report.installations[0].activity -eq 'completed_record' -and -not $report.runtime_probed) 'Historical completion was presented as fresh verification'
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
    # Policy unit tests use an in-process WSL stub. The actual entrypoint test
    # separately exercises native child probes and timeout against an inert EXE.
    function Invoke-InstallerPlatformProbe {
        param($Arguments)
        $output = (& wsl.exe $Arguments | Out-String)
        return @{ok=($LASTEXITCODE -eq 0); exit_code=$LASTEXITCODE;
            code=(Get-WslProbeFailureCode -ExitCode $LASTEXITCODE -Output $output);
            stdout=$output; stderr=''}
    }
    $completeImplementation = ${function:Complete-NewInstallation}
    $identityImplementation = ${function:Test-AgentFEMBuildIdentity}
    $invokeCheckedImplementation = ${function:Invoke-WslChecked}
    $Image = Join-Path $temp 'image [trial].wsl'
    Set-Content -LiteralPath $Image -Value 'inert test image'
    function Invoke-WslChecked {
        param($Arguments, $FailureMessage)
        Assert ($Arguments[2] -eq (Resolve-Path -LiteralPath $Image).Path) 'Import path was globbed or split'
        $script:importProbeCalls++
    }
    $script:importProbeCalls = 0
    Install-AgentFEMDistribution -Name 'AgentFEM-test'
    Assert ($script:importProbeCalls -eq 1) 'Literal bracketed image did not reach import adapter'
    $occupied = Join-Path $temp '[occupied]'
    New-Item -ItemType Directory -Path $occupied | Out-Null
    Set-Content -LiteralPath (Join-Path $occupied 'precious.txt') -Value 'user data'
    $nonemptyRejected = $false
    try { Install-AgentFEMDistribution -Name 'AgentFEM-test' -Location $occupied }
    catch { $nonemptyRejected = $_.Exception.Message -match 'must be empty' }
    Assert ($nonemptyRejected -and $script:importProbeCalls -eq 1) 'Bracketed occupied directory was not rejected before import'
    Assert ((Get-Content -LiteralPath (Join-Path $occupied 'precious.txt')) -eq 'user data') 'Occupied path was changed'
    ${function:Invoke-WslChecked} = $invokeCheckedImplementation
    function wsl.exe { $global:LASTEXITCODE = 0; return @('Ubuntu', ('Project ' + [char]0x6d4b)) }
    $inventory = @(Get-WslDistributions)
    Assert ($inventory.Count -eq 2 -and $inventory[1] -eq ('Project ' + [char]0x6d4b)) 'Inventory changed a spaced/localised distro name'
    function wsl.exe { $global:LASTEXITCODE = 0 }
    Assert (@(Get-WslDistributions).Count -eq 0) 'Successful empty inventory was not empty'
    function wsl.exe { $global:LASTEXITCODE = 5 }
    $deniedInventory = $false
    try { Get-WslDistributions | Out-Null }
    catch { $deniedInventory = $_.Exception.Message -match 'COMMAND-BLOCKED' }
    Assert $deniedInventory 'Denied inventory was treated as no distributions'
    function wsl.exe { $global:LASTEXITCODE = 0; return 'Kernel version: 5.15.0' }
    $wrongVersion = $false
    try { Get-InstalledWslVersion | Out-Null }
    catch { $wrongVersion = $_.Exception.Message -match 'VERSION-UNRECOGNIZED' }
    Assert $wrongVersion 'Installer used kernel version as WSL tooling version'
    function wsl.exe { $global:LASTEXITCODE = 1 }
    $unknownPlatform = $false
    try { Get-InstalledWslVersion | Out-Null }
    catch { $unknownPlatform = $_.Exception.Message -match 'WSL-PROBE-FAILED' }
    Assert $unknownPlatform 'Unknown probe error authorised platform reinstallation'
    Assert ((Get-AgentFEMRuntimeVersion -Name 'missing') -eq 'unknown') 'Blank runtime probe caused a null exception'
    $unknownUser = $false
    try { Get-DefaultLinuxUser -Name 'missing' | Out-Null }
    catch { $unknownUser = $_.Exception.Message -match 'non-root default user' }
    Assert $unknownUser 'Blank account probe lost its actionable failure'
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
    Assert ($platformVersion -eq [version]'2.3.26.0') 'Mirror route changed the tooling version or forced an image-only update'
    function wsl.exe {
        $global:LASTEXITCODE = 0
        if (($args -join ' ') -eq '--status') {
            $global:LASTEXITCODE = -1
            return 'Wsl/WSL_E_VIRTUAL_MACHINE_PLATFORM_REQUIRED'
        }
        return 'WSL version: 2.3.26.0'
    }
    $missingComponent = $false
    try { Require-WslPlatform -Source 'None' -PlatformOnly | Out-Null }
    catch { $missingComponent = $_.Exception.Message -match 'WSL-NOT-READY' }
    Assert $missingComponent 'Tooling version masked a missing Windows component'
    $platformInstallImplementation = ${function:Install-WslPlatform}
    $script:componentRepairs = 0
    function Install-WslPlatform { param($Source) $script:componentRepairs++ }
    function wsl.exe {
        $global:LASTEXITCODE = 0
        if (($args -join ' ') -eq '--status' -and $script:componentRepairs -eq 0) {
            $global:LASTEXITCODE = -2147220691
            return 'component required'
        }
        return 'WSL version: 2.3.26.0'
    }
    Require-WslPlatform -Source 'Store' -PlatformOnly | Out-Null
    Assert ($script:componentRepairs -eq 1) 'Known missing component did not use one preparation operation'
    function wsl.exe {
        $global:LASTEXITCODE = 0
        if (($args -join ' ') -eq '--status') { $global:LASTEXITCODE = 5 }
        return 'WSL version: 2.3.26.0'
    }
    $blockedStatus = $false
    try { Require-WslPlatform -Source 'Store' -PlatformOnly | Out-Null }
    catch { $blockedStatus = $_.Exception.Message -match 'COMMAND-BLOCKED' }
    Assert ($blockedStatus -and $script:componentRepairs -eq 1) 'Denied status triggered a repair'
    ${function:Install-WslPlatform} = $platformInstallImplementation
    function wsl.exe { $global:LASTEXITCODE = 0; return 'WSL version: 2.3.26.0' }
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
