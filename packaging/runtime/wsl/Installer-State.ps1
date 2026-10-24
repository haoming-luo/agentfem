# Installation state belongs to the invoking Windows user, not the elevated helper.
function ConvertTo-InstallIdentityJson {
    param([Parameter(Mandatory = $true)]$Identity)
    $ordered = [ordered]@{}
    if ($Identity -is [System.Collections.IDictionary]) {
        foreach ($key in ($Identity.Keys | Sort-Object)) { $ordered[$key] = $Identity[$key] }
    } else {
        foreach ($property in ($Identity.PSObject.Properties | Sort-Object Name)) {
            $ordered[$property.Name] = $property.Value
        }
    }
    return ($ordered | ConvertTo-Json -Compress -Depth 6)
}

function Initialize-InstallState {
    param([Parameter(Mandatory = $true)][hashtable]$Identity)
    $canonical = ($Identity.GetEnumerator() | Sort-Object Key | ForEach-Object {
        "$($_.Key)=$($_.Value)"
    }) -join "`n"
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $hash = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical)) }
    finally { $sha.Dispose() }
    $key = ([BitConverter]::ToString($hash)).Replace('-', '').ToLowerInvariant()
    $stateDirectory = Join-Path $env:LOCALAPPDATA "AgentFEM\Installations\$key"
    New-Item -ItemType Directory -Force -Path $stateDirectory | Out-Null
    # Different images/options can still select the SAME distribution. Serialize
    # installer mutations per Windows user, not merely per package identity.
    $lockPath = Join-Path (Split-Path $stateDirectory -Parent) 'install.lock'
    try {
        $lock = [IO.File]::Open(
            $lockPath,
            [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    } catch {
        $exception = $_.Exception.GetBaseException()
        if ($exception -is [IO.IOException] -and
            (([int64]$exception.HResult -band 65535) -in @(32, 33))) {
            throw '[AFM-WIN-INSTALL-BUSY] Another installer is using this installation. Do not start a duplicate.'
        }
        throw "[AFM-WIN-STATE-ACCESS] Cannot acquire installation-state access: $($exception.Message). Inspect permissions/storage; this is not evidence of a running installer."
    }
    $script:InstallLock = $lock
    $script:InstallStateDirectory = $stateDirectory
    $script:InstallStateOwned = $false
    $path = Join-Path $script:InstallStateDirectory 'status.json'
    $script:InstallState = @{
        schema = 'agentfem.windows-install'; schema_version = 1
        identity = $Identity; stage = 'created'; distribution = ''; linux_user = ''
        message = ''; updated_at = ''; installer = (Join-Path $PSScriptRoot 'Install-AgentFEM.ps1')
    }
    if (Test-Path -LiteralPath $path) {
        # Invalid state must not silently authorise reuse of an existing distribution.
        $old = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
        if ($old.schema -ne 'agentfem.windows-install' -or $old.schema_version -ne 1) {
            throw '[AFM-WIN-STATE-INVALID] Unsupported installation record. Preserve it for diagnosis.'
        }
        if ($null -eq $old.identity -or
            (ConvertTo-InstallIdentityJson $old.identity) -cne (ConvertTo-InstallIdentityJson $Identity)) {
            throw '[AFM-WIN-STATE-IDENTITY] Installation record does not match this request. Preserve it; no distribution was reused.'
        }
        $script:InstallState.distribution = $old.distribution
        $script:InstallState.linux_user = $old.linux_user
        $script:InstallState.effective_project_directory = $old.effective_project_directory
    }
    $script:InstallStateOwned = $true
}

function Get-DefaultInstallProjectDirectory {
    $documents = [Environment]::GetFolderPath('MyDocuments')
    if (-not $documents) { $documents = Join-Path $env:USERPROFILE 'Documents' }
    return (Join-Path $documents 'AgentFEMProjects')
}

function Resolve-InstallProjectDirectory {
    param([string]$RequestedPath = '')
    # A redirected Documents folder may change between a restart and resume.
    # Keep the original effective default in state without changing request
    # identity or rewriting the user's saved argv to a different request.
    $saved = $script:InstallState.effective_project_directory
    if ($null -ne $saved -and $saved -cne '') {
        if (-not ($saved -is [string]) -or -not [IO.Path]::IsPathRooted($saved) -or
            [IO.Path]::GetFullPath($saved) -cne $saved) {
            throw '[AFM-WIN-STATE-PATH] The recorded project directory is not an absolute canonical path. Preserve the record for inspection.'
        }
        if ($RequestedPath -and [IO.Path]::GetFullPath($RequestedPath) -ine $saved) {
            throw '[AFM-WIN-STATE-PATH] The recorded project directory conflicts with this installation request. No workspace was moved.'
        }
        return $saved
    }
    $selected = $RequestedPath
    if (-not $selected) { $selected = Get-DefaultInstallProjectDirectory }
    $selected = [IO.Path]::GetFullPath($selected)
    $script:InstallState.effective_project_directory = $selected
    return $selected
}

function Resolve-InstallPathParameters {
    param([Parameter(Mandatory = $true)][System.Collections.IDictionary]$Parameters)
    # Relative paths depend on the invoking PowerShell location, which may
    # change after reboot or a new agent tool call. Canonicalize before BOTH
    # identity construction and saving argv, not just before filesystem writes.
    foreach ($name in @('Image', 'InstallLocation', 'ProjectDirectory', 'BackupDirectory')) {
        # PSBoundParametersDictionary exposes a generic Contains overload,
        # unlike Hashtable.Contains(key). Inspect keys for both representations.
        if ($Parameters.Keys -contains $name -and -not [string]::IsNullOrWhiteSpace([string]$Parameters[$name])) {
            $provider = $null; $drive = $null
            $resolved = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
                [string]$Parameters[$name], [ref]$provider, [ref]$drive)
            if ($provider.Name -ne 'FileSystem') {
                throw "[AFM-WIN-PATH-INVALID] $name must identify a filesystem path."
            }
            $Parameters[$name] = $resolved
        }
    }
}

function Save-InstallState {
    param([Parameter(Mandatory = $true)][string]$Stage, [string]$Message = '')
    if (-not $script:InstallStateOwned) { return }
    if ($Stage -in @('failed', 'restart_required', 'awaiting_authorization')) {
        $script:InstallState.interrupted_stage = $script:InstallState.stage
    }
    $script:InstallState.stage = $Stage
    $script:InstallState.message = $Message
    $script:InstallState.error_code = ''
    if ($Message -match '\[(AFM-WIN-[A-Z-]+)\]') { $script:InstallState.error_code = $Matches[1] }
    $script:InstallState.next_action = switch ($Stage) {
        'restart_required' { 'save_work_restart_then_rerun_same_installer' }
        'awaiting_authorization' { 'request_host_approval_or_user_launch' }
        'account_setup' { 'user_enters_linux_credentials_in_local_terminal' }
        'completed' { 'use_verified_runtime' }
        'platform_ready' { 'reuse_or_install_named_distribution_then_mirror_environment' }
        'failed' { 'inspect_error_and_log_before_retry' }
        default { 'wait_for_current_installer' }
    }
    $script:InstallState.updated_at = (Get-Date).ToUniversalTime().ToString('o')
    $path = Join-Path $script:InstallStateDirectory 'status.json'
    $temporary = "$path.tmp"
    $script:InstallState | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath $temporary
    if (Test-Path -LiteralPath $path) {
        # A fixed .previous name races with a reader holding an older snapshot
        # across multiple publications. Use the API's no-backup atomic replace;
        # NullString passes an actual null string even on Windows PowerShell 5.1.
        # Existing .previous files are left untouched; the transcript retains
        # the diagnostic history, not a repeatedly overwritten backup inode.
        [IO.File]::Replace($temporary, $path, [System.Management.Automation.Language.NullString]::Value)
    } else { [IO.File]::Move($temporary, $path) }
}

function Set-InstallResumeInvocation {
    param([string]$InstallerPath, [System.Collections.IDictionary]$Parameters)
    # Store argv as data, never a command string for Invoke-Expression. There
    # are no password parameters. Callers still review the saved operation.
    $arguments = @('-NoProfile', '-File', $InstallerPath)
    foreach ($key in ($Parameters.Keys | Sort-Object)) {
        $value = $Parameters[$key]
        if ($value -is [Management.Automation.SwitchParameter]) {
            if ($value.IsPresent) { $arguments += "-$key" }
        } else { $arguments += @("-$key", [string]$value) }
    }
    $script:InstallState.resume = @{executable='powershell.exe'; arguments=$arguments}
}
