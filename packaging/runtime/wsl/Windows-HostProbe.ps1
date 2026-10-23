# Read-only Windows/WSL probes. No Linux process, elevation, download or repair.
# Keep these functions separate from the CLI for deterministic contract tests.
function ConvertFrom-WslProbeBytes {
    param([byte[]]$Bytes)
    if (-not $Bytes -or $Bytes.Length -eq 0) { return '' }
    # Redirected wsl.exe output may be UTF-16LE rather than the shell encoding.
    $unicode = $Bytes.Length -ge 2 -and $Bytes[0] -eq 255 -and $Bytes[1] -eq 254
    if (-not $unicode -and $Bytes.Length -ge 4) {
        $oddZeros = 0
        $pairs = [Math]::Min([int]($Bytes.Length / 2), 128)
        for ($i = 0; $i -lt $pairs; $i++) {
            if ($Bytes[2 * $i + 1] -eq 0) { $oddZeros++ }
        }
        $unicode = $oddZeros -gt ($pairs / 4)
        # A long all-Chinese distro name can have very few ASCII code units;
        # the UTF-16 CRLF suffix still identifies redirected WSL line output.
        $n = $Bytes.Length
        if ($n -ge 4 -and $Bytes[$n-4] -eq 13 -and $Bytes[$n-3] -eq 0 -and
            $Bytes[$n-2] -eq 10 -and $Bytes[$n-1] -eq 0) { $unicode = $true }
    }
    if ($unicode) { $text = [Text.Encoding]::Unicode.GetString($Bytes) }
    else { $text = [Text.Encoding]::UTF8.GetString($Bytes) }
    return $text.TrimStart([char]0xfeff).TrimEnd([char]0)
}

function Get-WslProbeFailureCode {
    param([long]$ExitCode, [string]$Output = '')
    if ($ExitCode -eq 0) { return '' }
    $win32Code = ($ExitCode -band 65535)
    if ($win32Code -in @(5, 740, 1223)) { return 'AFM-WIN-COMMAND-BLOCKED' }
    if ($win32Code -in @(3010, 1641)) { return 'AFM-WIN-RESTART-REQUIRED' }
    # Microsoft WSL HRESULTs; some clients return -1 but print the stable
    # symbolic code. Do not infer missing features from localised prose.
    $unsignedCode = ($ExitCode -band 4294967295)
    if ($unsignedCode -in @(2147746593, 2147746605) -or
        $Output -match '\bWSL_E_(WSL_OPTIONAL_COMPONENT|VIRTUAL_MACHINE_PLATFORM)_REQUIRED\b') {
        return 'AFM-WIN-COMPONENT-REQUIRED'
    }
    return 'AFM-WIN-PROBE-FAILED'
}

function Invoke-WslHostProbe {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)]
        [ValidateSet('--version', '--status', '--list --quiet', '--list --verbose')][string]$Arguments,
        [ValidateRange(1, 120)][int]$TimeoutSeconds = 15
    )
    $process = New-Object Diagnostics.Process
    $stdout = New-Object IO.MemoryStream
    $stderr = New-Object IO.MemoryStream
    try {
        $process.StartInfo.FileName = $Executable
        # Arguments are a fixed allowlist, never distro names or shell input.
        $process.StartInfo.Arguments = $Arguments
        $process.StartInfo.UseShellExecute = $false
        $process.StartInfo.CreateNoWindow = $true
        $process.StartInfo.RedirectStandardOutput = $true
        $process.StartInfo.RedirectStandardError = $true
        if (-not $process.Start()) { throw 'Read-only WSL probe could not start.' }
        # Drain both pipes concurrently; waiting before reading can deadlock.
        $outTask = $process.StandardOutput.BaseStream.CopyToAsync($stdout)
        $errTask = $process.StandardError.BaseStream.CopyToAsync($stderr)
        if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
            try { $process.Kill(); $null = $process.WaitForExit(1000) } catch { }
            return @{ok=$false; code='AFM-WIN-PROBE-TIMEOUT'; exit_code=$null;
                stdout=''; stderr='Read-only WSL probe timed out; no installation attempted.'}
        }
        if (-not [Threading.Tasks.Task]::WaitAll(
            [Threading.Tasks.Task[]]@($outTask, $errTask), 1000)) {
            throw 'WSL probe output did not close after exit.'
        }
        $outText = ConvertFrom-WslProbeBytes $stdout.ToArray()
        $errText = ConvertFrom-WslProbeBytes $stderr.ToArray()
        $code = Get-WslProbeFailureCode -ExitCode $process.ExitCode -Output ($outText + "`n" + $errText)
        return @{ok=($process.ExitCode -eq 0); code=$code; exit_code=$process.ExitCode;
            stdout=$outText; stderr=$errText}
    } catch {
        $exception = $_.Exception.GetBaseException()
        $code = 'AFM-WIN-PROBE-FAILED'
        if ($exception -is [ComponentModel.Win32Exception] -and
            $exception.NativeErrorCode -in @(5, 740, 1223)) { $code = 'AFM-WIN-COMMAND-BLOCKED' }
        return @{ok=$false; code=$code; exit_code=$null; stdout=''; stderr=$exception.Message}
    } finally {
        $process.Dispose(); $stdout.Dispose(); $stderr.Dispose()
    }
}

function Get-AgentFEMWindowsHostReport {
    param(
        [ValidateSet('UbuntuMirror', 'CompleteRuntime')][string]$Route = 'UbuntuMirror',
        [ValidateRange(1, 120)][int]$TimeoutSeconds = 15
    )
    $report = [ordered]@{
        schema='agentfem.windows-host'; schema_version=1
        observed_at=(Get-Date).ToUniversalTime().ToString('o')
        route=$Route; status='unknown'; next_action='inspect_probe_error'
        architecture=$env:PROCESSOR_ARCHITEW6432
        wsl_command=$null; wsl_version=$null; distributions=@(); probes=@(); errors=@()
        distribution_started=$false; runtime_verified=$false; changes_made=$false
    }
    if (-not $report.architecture) { $report.architecture = $env:PROCESSOR_ARCHITECTURE }
    if ($env:OS -ne 'Windows_NT') {
        $report.status = 'unsupported_host'
        $report.errors += @{code='AFM-WIN-HOST-UNSUPPORTED'; message='Run this probe on Windows.'}
        return $report
    }
    $command = Get-Command wsl.exe -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $command) {
        $report.status = 'command_unavailable'
        $report.next_action = 'inspect_windows_component_and_host_permission'
        $report.errors += @{code='AFM-WIN-WSL-COMMAND-MISSING'; message='wsl.exe is not visible to this tool session; installation state is unknown.'}
        return $report
    }
    $report.wsl_command = $command.Source
    foreach ($arguments in @('--version', '--status', '--list --quiet', '--list --verbose')) {
        $probe = Invoke-WslHostProbe -Executable $command.Source -Arguments $arguments -TimeoutSeconds $TimeoutSeconds
        $report.probes += @{arguments=$arguments; exit_code=$probe.exit_code; code=$probe.code}
        if (-not $probe.ok) {
            $report.status = switch ($probe.code) {
                'AFM-WIN-COMMAND-BLOCKED' { 'blocked' }
                'AFM-WIN-RESTART-REQUIRED' { 'restart_required' }
                'AFM-WIN-PROBE-TIMEOUT' { 'probe_timeout' }
                'AFM-WIN-COMPONENT-REQUIRED' { 'component_required' }
                default { 'probe_failed' }
            }
            $report.next_action = switch ($report.status) {
                'blocked' { 'request_scoped_host_approval' }
                'restart_required' { 'save_work_restart_then_probe_again' }
                'component_required' { 'prepare_windows_components_with_consent' }
                default { 'inspect_probe_error_without_reinstalling' }
            }
            $detail = ($probe.stderr + "`n" + $probe.stdout).Trim()
            if ($detail.Length -gt 8192) { $detail = $detail.Substring(0, 8192) }
            $report.errors += @{code=$probe.code; arguments=$arguments; message=$detail}
            return $report
        }
        switch ($arguments) {
            '--version' {
                $match = [regex]::Match($probe.stdout, '(?m)^\s*WSL[^\r\n\d]*(\d+\.\d+\.\d+(?:\.\d+)?)')
                if (-not $match.Success) {
                    $report.status = 'probe_failed'
                    $report.errors += @{code='AFM-WIN-VERSION-UNRECOGNIZED'; message='Cannot identify the WSL tooling version; do not substitute the kernel or Windows version.'}
                    return $report
                }
                $report.wsl_version = $match.Groups[1].Value
            }
            '--list --quiet' { $names = @($probe.stdout -split '\r?\n' | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
            '--list --verbose' { $verboseLines = @($probe.stdout -split '\r?\n') }
        }
    }
    foreach ($name in $names) {
        # Do not parse English state/header labels or infer the distro's version
        # from the WSL tooling version. Names come from the separate quiet list.
        $pattern = '^\s*(\*\s+)?' + [regex]::Escape($name) + '\s+(.+?)\s+([12])\s*$'
        $rows = @($verboseLines | Where-Object { $_ -match $pattern })
        if ($rows.Count -ne 1) {
            $report.status = 'probe_failed'; $report.next_action = 'inspect_inconsistent_distribution_inventory'
            $report.errors += @{code='AFM-WIN-DISTRO-INVENTORY'; message="Cannot identify one WSL version row for '$name'. Retry the read-only probe; do not reinstall."}
            return $report
        }
        $row = [regex]::Match($rows[0], $pattern)
        $report.distributions += @{name=$name; wsl_version=[int]$row.Groups[3].Value;
            state_label=$row.Groups[2].Value; is_default=$row.Groups[1].Success}
    }
    $wsl2 = @($report.distributions | Where-Object { $_.wsl_version -eq 2 })
    if ($Route -eq 'CompleteRuntime' -and [version]$report.wsl_version -lt [version]'2.4.4') {
        $report.status='platform_update_required'; $report.next_action='request_wsl_update_for_image_format'
    } elseif ($report.distributions.Count -eq 0) {
        $report.status='no_distribution'
        $report.next_action = $(if ($Route -eq 'UbuntuMirror') {'install_named_ubuntu_with_consent'} else {'install_verified_runtime_bundle'})
    } elseif ($wsl2.Count -eq 0) {
        $report.status='wsl1_only'; $report.next_action='preserve_existing_distro_and_review_wsl2_route'
    } else {
        $report.status='inventory_available'
        $report.next_action = $(if ($Route -eq 'UbuntuMirror') {'select_compatible_wsl2_distro_then_verify_environment'} else {'review_existing_runtime_before_installing'})
    }
    return $report
}
