# Source-usable, read-only preflight for agent hosts. This is not an installer.
[CmdletBinding()]
param(
    [ValidateSet('UbuntuMirror', 'CompleteRuntime')][string]$Route = 'UbuntuMirror',
    [ValidateRange(1, 120)][int]$TimeoutSeconds = 15
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Windows-HostProbe.ps1')
try { $report = Get-AgentFEMWindowsHostReport -Route $Route -TimeoutSeconds $TimeoutSeconds }
catch {
    $report = @{schema='agentfem.windows-host'; schema_version=1; status='probe_failed';
        next_action='inspect_probe_error'; changes_made=$false; runtime_verified=$false;
        errors=@(@{code='AFM-WIN-PROBE-FAILED'; message=$_.Exception.Message})}
}
$report | ConvertTo-Json -Depth 8
if ($report.errors.Count) { exit 1 }
exit 0
