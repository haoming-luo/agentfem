# Read-only agent entry point: no WSL launch, downloads, elevation or state writes.
[CmdletBinding()]
param([string]$StateRoot = (Join-Path $env:LOCALAPPDATA 'AgentFEM\Installations'))
$ErrorActionPreference = 'Stop'
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
        } catch [IO.IOException] { $busy = $true }
    }
    if (Test-Path -LiteralPath $StateRoot) {
        foreach ($directory in (Get-ChildItem -LiteralPath $StateRoot -Directory)) {
            $path = Join-Path $directory.FullName 'status.json'
            if (-not (Test-Path -LiteralPath $path)) { continue }
            try {
                $record = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
                if ($record.schema -ne 'agentfem.windows-install' -or $record.schema_version -ne 1 -or
                    $null -eq $record.identity -or -not ($record.stage -is [string])) {
                    throw 'Unsupported installation state schema.'
                }
                $records += [pscustomobject]@{
                    state_file = $path; stage = $record.stage
                    identity = $record.identity; distribution = $record.distribution
                    updated_at = $record.updated_at; error_code = $record.error_code
                    message = $record.message; next_action = $record.next_action
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
@{
    schema = 'agentfem.windows-install-status'; schema_version = 1
    status = $(if ($errors.Count) {'error'} elseif ($busy) {'busy'} elseif ($records.Count) {'recorded'} else {'not_started'})
    installer_busy = $busy
    # A past completion is evidence, NOT a fresh health check or permission grant.
    runtime_probed = $false
    installations = @($records | Sort-Object updated_at -Descending)
    errors = $errors
} | ConvertTo-Json -Depth 8
if ($errors.Count) { exit 1 }
exit 0
