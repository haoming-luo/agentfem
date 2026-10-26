# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
# Pure presentation: no WSL launch, download, UAC or user-state mutation.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $root 'packaging/runtime/wsl/Installer-Presentation.ps1')
function Assert($Condition, $Message) { if (-not $Condition) { throw $Message } }
$record = [pscustomobject]@{
    stage='restart_required'; activity='recorded'; distribution='Ubuntu [test]'
    error_code='AFM-WIN-RESTART-REQUIRED'; state_file='C:\User [test]\status.json'
    effective_project_directory='C:\User [test]\Projects'; acceptance_report=''
    message=''; next_action='save_work_restart_then_rerun_same_installer'
}
$report = @{installer_busy=$false; installations=@($record); errors=@()}
$text = Format-AgentFEMInstallStatus -Report $report
Assert ($text -match 'save your work' -and $text -match 'same installer') 'Restart handoff is not actionable'
Assert ($text.Contains($record.state_file)) 'Literal diagnostic path lost'
$record.stage = 'importing'; $record.activity = 'interrupted'; $record.error_code = ''
$text = Format-AgentFEMInstallStatus -Report $report
Assert ($text -match 'No installer holds the lock' -and $text -match 'Inspect the log') 'Interrupted operation shown as live progress'
$record.stage = 'platform_ready'; $record.activity = 'completed_record'
$text = Format-AgentFEMInstallStatus -Report $report
Assert ($text -match 'AgentFEM is not installed yet') 'Platform readiness confused with usable solver'
$record.stage = 'completed'
$text = Format-AgentFEMInstallStatus -Report $report
Assert ($text -match 'not a fresh health check' -and $text.Contains($record.effective_project_directory)) 'Recorded acceptance overclaimed or workspace omitted'
$report.installer_busy = $true
$record.stage = 'restart_required'
$text = Format-AgentFEMInstallStatus -Report $report
Assert ($text -match 'Do not open a second copy') 'Busy installer suggests duplicate work'
Assert ($text -notmatch 'Next: save your work' -and $text -match 'wait for the active installer') 'Old restart record advises reboot during active installation'
$report = @{installer_busy=$false; installations=@(); errors=@()}
Assert ((Format-AgentFEMInstallStatus -Report $report) -match 'No saved installation') 'Empty state has no first action'

foreach ($route in @('UbuntuMirror', 'CompleteRuntime')) {
    $hostReport = @{route=$route; status='no_distribution'; distributions=@(); errors=@(); wsl_version='2.5.0'}
    $text = Format-AgentFEMWindowsHost -Report $hostReport
    if ($route -eq 'UbuntuMirror') { Assert ($text -match 'named Ubuntu') 'Mirror route lost its distribution handoff' }
    else { Assert ($text -match 'does not need a separate Ubuntu') 'Bundle route downloads an unnecessary distribution' }
    $hostReport.status='blocked'
    Assert ((Format-AgentFEMWindowsHost -Report $hostReport) -match 'Do not switch launchers') 'Blocked host suggests policy bypass'
    $hostReport.status='inventory_available'
    $localName = 'Ubuntu ' + [char]0x4e2d + [char]0x6587
    $hostReport.distributions=@(@{name=$localName; wsl_version=2})
    $text = Format-AgentFEMWindowsHost -Report $hostReport
    Assert ($text.Contains($localName + ' (WSL 2)')) 'Localised distribution identity lost'
    Assert ($text -match 'not an AgentFEM first-solve check') 'Inventory presented as runtime acceptance'
}
Write-Host 'Installer presentation passed (no Windows changes).'
exit 0
