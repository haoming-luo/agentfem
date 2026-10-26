# Read-only agent entry point: no WSL launch, downloads, elevation or state writes.
[CmdletBinding()]
param(
    [string]$StateRoot = (Join-Path $env:LOCALAPPDATA 'AgentFEM\Installations'),
    [ValidateSet('Json', 'Text')][string]$Format = 'Json'
)
$ErrorActionPreference = 'Stop'

function Open-InstallStatusSnapshot {
    param([Parameter(Mandatory = $true)][string]$Path)
    # The installer publishes complete snapshots by File.Replace. Ordinary
    # Get-Content denies delete sharing on Windows and can make that publish
    # fail just because an agent is reading progress at the same instant.
    # A shared-delete reader keeps the opened revision intact through rename.
    return [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read,
        ([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
}

$records = @()
$errors = @()
$busy = $false
try {
    $lockPath = Join-Path $StateRoot 'install.lock'
    if (Test-Path -LiteralPath $lockPath) {
        try {
            $probe = [IO.File]::Open($lockPath, [IO.FileMode]::Open,
                [IO.FileAccess]::Read, [IO.FileShare]::None)
            $probe.Dispose()
        } catch [IO.IOException] {
            $exception = $_.Exception.GetBaseException()
            if (([int64]$exception.HResult -band 65535) -in @(32, 33)) {
                $busy = $true
            } else {
                # Missing/corrupt/unreadable storage is not a live installer.
                # Preserve the failure through the structured outer handler.
                throw
            }
        }
    }
    if (Test-Path -LiteralPath $StateRoot) {
        foreach ($directory in (Get-ChildItem -LiteralPath $StateRoot -Directory)) {
            $path = Join-Path $directory.FullName 'status.json'
            if (-not (Test-Path -LiteralPath $path)) { continue }
            try {
                $stream = Open-InstallStatusSnapshot -Path $path
                try {
                    $reader = New-Object IO.StreamReader($stream, [Text.Encoding]::UTF8, $true)
                    try { $record = $reader.ReadToEnd() | ConvertFrom-Json }
                    finally { $reader.Dispose() }
                } finally { $stream.Dispose() }
                if ($record.schema -ne 'agentfem.windows-install' -or $record.schema_version -ne 1 -or
                    $null -eq $record.identity -or -not ($record.stage -is [string])) {
                    throw 'Unsupported installation state schema.'
                }
                $activity = 'recorded'
                $nextAction = $record.next_action
                if ($busy) {
                    # The lock is per user, not per record: do not pretend all
                    # saved packages belong to the currently running installer.
                    $activity = 'not_inferred_while_installer_busy'
                } elseif ($record.stage -in @('completed', 'platform_ready')) {
                    $activity = 'completed_record'
                } elseif (-not $record.error_code -and $record.stage -in @(
                    'created', 'checking_package', 'preparing_wsl', 'requesting_elevation', 'importing',
                    'account_setup', 'protecting_workspace', 'verifying_runtime')) {
                    $activity = 'interrupted'
                    $nextAction = 'inspect_log_then_resume_same_installer'
                }
                $records += [pscustomobject]@{
                    state_file = $path; stage = $record.stage
                    activity = $activity
                    identity = $record.identity; distribution = $record.distribution
                    effective_project_directory = $record.effective_project_directory
                    updated_at = $record.updated_at; error_code = $record.error_code
                    message = $record.message; next_action = $nextAction
                    recorded_next_action = $record.next_action
                    interrupted_stage = $record.interrupted_stage
                    resume = $record.resume
                    installer = $record.installer; acceptance_report = $record.acceptance_report
                }
            } catch {
                $errors += @{code='AFM-WIN-STATE-INVALID'; path=$path; message=$_.Exception.Message}
            }
        }
    }
} catch { $errors += @{code='AFM-WIN-STATUS-READ'; message=$_.Exception.Message} }
$report = @{
    schema = 'agentfem.windows-install-status'; schema_version = 1
    status = $(if ($errors.Count) {'error'} elseif ($busy) {'busy'} elseif ($records.Count) {'recorded'} else {'not_started'})
    installer_busy = $busy
    # A past completion is evidence, NOT a fresh health check or permission grant.
    runtime_probed = $false
    installations = @($records | Sort-Object updated_at -Descending)
    errors = $errors
}
if ($Format -eq 'Text') {
    . (Join-Path $PSScriptRoot 'Installer-Presentation.ps1')
    Format-AgentFEMInstallStatus -Report $report
} else { $report | ConvertTo-Json -Depth 8 }
if ($errors.Count) { exit 1 }
exit 0
