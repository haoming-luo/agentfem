# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
# Presentation only: no process launches, downloads, writes or permission changes.
function Get-AgentFEMInstallStageText {
    param([string]$Stage)
    switch ($Stage) {
        'created' { 'Preparing the installation record' }
        'checking_package' { 'Checking package integrity' }
        'preparing_wsl' { 'Checking the Windows Linux platform' }
        'requesting_elevation' { 'Waiting for Windows administrator approval' }
        'importing' { 'Importing the offline runtime; large images can take several minutes' }
        'account_setup' { 'Setting up your private Linux account' }
        'protecting_workspace' { 'Protecting your project storage' }
        'verifying_runtime' { 'Running the first finite-element check; initial compilation can take several minutes' }
        'restart_required' { 'Paused: Windows restart required' }
        'awaiting_authorization' { 'Paused: permission required' }
        'platform_ready' { 'WSL platform prepared; AgentFEM is not installed yet' }
        'completed' { 'Installation and first-solve checks completed' }
        'failed' { 'Installation did not complete; your diagnostic record is retained' }
        default { "Recorded stage: $Stage" }
    }
}

function Format-AgentFEMInstallStatus {
    param([Parameter(Mandatory = $true)]$Report)
    $lines = @('AgentFEM installation status')
    if ($Report.installer_busy) {
        $lines += 'An installer is active. Do not open a second copy.'
    } elseif (-not $Report.installations.Count -and -not $Report.errors.Count) {
        $lines += 'No saved installation. Open the trusted installer, or follow the Ubuntu + mirror guide.'
    }
    foreach ($record in $Report.installations) {
        $lines += ''
        $lines += (Get-AgentFEMInstallStageText -Stage $record.stage)
        if ($record.distribution) { $lines += "Distribution: $($record.distribution)" }
        if ($Report.installer_busy) {
            $lines += 'Saved record; wait for the active installer before taking recovery action.'
        } elseif ($record.activity -eq 'interrupted') {
            $lines += 'No installer holds the lock. Inspect the log, then reopen the same installer with the same options.'
        } elseif ($record.stage -eq 'restart_required') {
            $lines += 'Next: save your work, restart Windows, then reopen the same installer. Do not reinstall or remove a distribution.'
        } elseif ($record.stage -eq 'awaiting_authorization') {
            $lines += 'Next: approve this specific operation in your agent host, or open the trusted installer locally. Do not disable security controls.'
        } elseif ($record.stage -eq 'account_setup' -and $record.error_code) {
            $lines += 'Next: reopen the same installer locally without -NonInteractive. Enter credentials only in its private terminal, never in chat.'
        } elseif ($record.stage -eq 'platform_ready') {
            $lines += 'Next: reuse an appropriate WSL2 Ubuntu, then prepare and verify its AgentFEM environment.'
        } elseif ($record.stage -eq 'completed') {
            $lines += 'This is saved acceptance, not a fresh health check.'
            if ($record.effective_project_directory) { $lines += "Projects: $($record.effective_project_directory)" }
            if ($record.acceptance_report) { $lines += "Acceptance: $($record.acceptance_report)" }
        } elseif ($record.stage -eq 'failed') {
            $lines += 'Next: inspect the named error and log before retrying. Do not delete saved state or project files.'
        }
        if ($record.error_code) { $lines += "Reason: $($record.error_code)" }
        if ($record.message) { $lines += $record.message }
        $lines += "Record: $($record.state_file)"
    }
    foreach ($errorRecord in $Report.errors) {
        $lines += "Error: $($errorRecord.code): $($errorRecord.message)"
    }
    $lines += ''
    $lines += 'This status view does not launch WSL or change your computer.'
    return ($lines -join [Environment]::NewLine)
}

function Format-AgentFEMWindowsHost {
    param([Parameter(Mandatory = $true)]$Report)
    $next = switch ($Report.status) {
        'inventory_available' { 'Select an existing WSL2 distribution and verify its environment. No new distribution is needed automatically.' }
        'no_distribution' {
            if ($Report.route -eq 'UbuntuMirror') { 'Install the named Ubuntu distribution with approval, then complete its private account prompt.' }
            else { 'Open the verified Complete Runtime installer; this route does not need a separate Ubuntu download.' }
        }
        'wsl1_only' { 'Keep existing WSL1 data. Review a WSL2 route; do not convert or remove a distribution automatically.' }
        'blocked' { 'Request approval for this specific command in your agent host, or use a local terminal. Do not switch launchers to bypass policy.' }
        'restart_required' { 'Save your work and restart Windows, then rerun this check.' }
        'component_required' { 'Use the trusted platform installer with user consent; it will identify any required restart.' }
        'platform_update_required' { 'The Complete Runtime image needs WSL 2.4.4 or newer. Approve an update, or use the documented Ubuntu environment route.' }
        'command_unavailable' { 'Check the Windows component and agent-host command permission. Missing command visibility is not proof that WSL is absent.' }
        default { 'Inspect the error below. Do not reinstall based on an unknown probe failure.' }
    }
    $lines = @('AgentFEM Windows check', "Route: $($Report.route)", "Status: $($Report.status)")
    if ($Report.wsl_version) { $lines += "WSL tooling: $($Report.wsl_version)" }
    foreach ($distribution in $Report.distributions) {
        $lines += "Distribution: $($distribution.name) (WSL $($distribution.wsl_version))"
    }
    $lines += "Next: $next"
    foreach ($errorRecord in $Report.errors) { $lines += "Reason: $($errorRecord.code): $($errorRecord.message)" }
    $lines += 'No changes made. Platform inventory is not an AgentFEM first-solve check.'
    return ($lines -join [Environment]::NewLine)
}
