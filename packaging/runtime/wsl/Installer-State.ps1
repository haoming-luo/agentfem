# Installation state belongs to the invoking Windows user, not the elevated helper.
function Initialize-InstallState {
    param([Parameter(Mandatory = $true)][hashtable]$Identity)
    $canonical = ($Identity.GetEnumerator() | Sort-Object Key | ForEach-Object {
        "$($_.Key)=$($_.Value)"
    }) -join "`n"
    $sha = [Security.Cryptography.SHA256]::Create()
    try { $hash = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical)) }
    finally { $sha.Dispose() }
    $key = ([BitConverter]::ToString($hash)).Replace('-', '').ToLowerInvariant()
    $script:InstallStateDirectory = Join-Path $env:LOCALAPPDATA "AgentFEM\Installations\$key"
    New-Item -ItemType Directory -Force -Path $script:InstallStateDirectory | Out-Null
    try {
        $script:InstallLock = [IO.File]::Open(
            (Join-Path $script:InstallStateDirectory 'install.lock'),
            [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    } catch {
        throw '[AFM-WIN-INSTALL-BUSY] Another installer is using this installation. Do not start a duplicate.'
    }
    $path = Join-Path $script:InstallStateDirectory 'status.json'
    $script:InstallState = @{
        schema = 'agentfem.windows-install'; schema_version = 1
        identity = $Identity; stage = 'created'; distribution = ''; linux_user = ''
        message = ''; updated_at = ''; installer = (Join-Path $PSScriptRoot 'Install-AgentFEM.ps1')
    }
    if (Test-Path $path) {
        # Invalid state must not silently authorise reuse of an existing distribution.
        $old = Get-Content -Raw $path | ConvertFrom-Json
        if ($old.schema -ne 'agentfem.windows-install' -or $old.schema_version -ne 1) {
            throw '[AFM-WIN-STATE-INVALID] Unsupported installation record. Preserve it for diagnosis.'
        }
        $script:InstallState.distribution = $old.distribution
        $script:InstallState.linux_user = $old.linux_user
    }
    $script:InstallStateOwned = $true
}

function Save-InstallState {
    param([Parameter(Mandatory = $true)][string]$Stage, [string]$Message = '')
    if (-not $script:InstallStateOwned) { return }
    $script:InstallState.stage = $Stage
    $script:InstallState.message = $Message
    $script:InstallState.updated_at = (Get-Date).ToUniversalTime().ToString('o')
    $path = Join-Path $script:InstallStateDirectory 'status.json'
    $temporary = "$path.tmp"
    $script:InstallState | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $temporary
    if (Test-Path $path) { [IO.File]::Replace($temporary, $path, $null) }
    else { [IO.File]::Move($temporary, $path) }
}
