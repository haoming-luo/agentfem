# Read-only preflight policy and encoding contracts. WSL is mocked; no distro
# is started/installed and no platform feature or permission is changed.
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
. (Join-Path $root 'packaging/runtime/wsl/Windows-HostProbe.ps1')
function Assert($Condition, $Message) { if (-not $Condition) { throw $Message } }
$nativeProbe = ${function:Invoke-WslHostProbe}
$originalOS = $env:OS
$temp = Join-Path ([IO.Path]::GetTempPath()) ('AgentFEM probe ' + [guid]::NewGuid().ToString())
try {
    # UTF-8, UTF-16LE, Chinese names and BOMs must not become broken identities.
    $chinese = ([string][char]0x6d4b + [char]0x8bd5) * 20
    foreach ($value in @('Ubuntu 24.04', $chinese)) {
        foreach ($encoding in @([Text.Encoding]::UTF8, [Text.Encoding]::Unicode)) {
            $bytes = $encoding.GetBytes($value + "`r`n")
            Assert ((ConvertFrom-WslProbeBytes $bytes).Trim() -ceq $value) 'Probe output encoding corrupted distro name'
            $bytes = [byte[]]@($encoding.GetPreamble() + $bytes)
            Assert ((ConvertFrom-WslProbeBytes $bytes).Trim() -ceq $value) 'Probe BOM was not removed'
        }
    }
    $script:responses = @{}
    $script:calls = @()
    function Get-Command { return [pscustomobject]@{Source='C:\Windows\System32\wsl.exe'} }
    function Invoke-WslHostProbe {
        param($Executable, $Arguments, $TimeoutSeconds)
        $script:calls += $Arguments
        if (-not $script:responses.ContainsKey($Arguments)) { throw "Unexpected probe $Arguments" }
        return $script:responses[$Arguments]
    }
    function Ok($Output) { return @{ok=$true; stdout=$Output; stderr=''; code=''; exit_code=0} }
    function Set-Inventory($Names, $Rows) {
        $script:calls = @()
        $script:responses = @{
            '--version' = (Ok "WSL version: 2.3.26.0`nKernel version: 5.15.0")
            '--list --quiet' = (Ok $Names)
            '--list --verbose' = (Ok $Rows)
        }
    }
    $env:OS = 'Windows_NT'
    Set-Inventory "Ubuntu 24.04`n$chinese" "NAME STATE VERSION`n* Ubuntu 24.04 Running 2`n  $chinese Stopped 1"
    $report = Get-AgentFEMWindowsHostReport
    Assert ($report.status -eq 'inventory_available') 'Available WSL2 not detected'
    Assert ($report.distributions.Count -eq 2 -and $report.distributions[0].is_default) 'Inventory/default marker lost'
    Assert ($report.distributions[1].name -ceq $chinese) 'Chinese name changed'
    Assert ($report.distributions[1].wsl_version -eq 1) 'WSL1 was assumed to be WSL2'
    Assert (-not $report.runtime_verified -and -not $report.distribution_started -and -not $report.changes_made) 'Inventory falsely certifies environment'
    $report = Get-AgentFEMWindowsHostReport -Route CompleteRuntime
    Assert ($report.status -eq 'platform_update_required') 'Image minimum version not distinguished from mirror'
    Set-Inventory 'Ubuntu' "NAME STATE VERSION`n Ubuntu Stopped 1"
    $report = Get-AgentFEMWindowsHostReport
    Assert ($report.status -eq 'wsl1_only') 'Existing WSL1 not preserved for user decision'
    Set-Inventory '' 'NAME STATE VERSION'
    $report = Get-AgentFEMWindowsHostReport
    Assert ($report.status -eq 'no_distribution') 'Empty successful inventory misclassified'
    Set-Inventory 'Ubuntu' 'NAME STATE VERSION'
    $report = Get-AgentFEMWindowsHostReport
    Assert ($report.errors[0].code -eq 'AFM-WIN-DISTRO-INVENTORY') 'Inconsistent inventory accepted'
    foreach ($code in @('AFM-WIN-COMMAND-BLOCKED', 'AFM-WIN-RESTART-REQUIRED', 'AFM-WIN-PROBE-TIMEOUT', 'AFM-WIN-PROBE-FAILED')) {
        Set-Inventory '' ''
        $script:responses['--version'] = @{ok=$false; code=$code; exit_code=-1; stdout=''; stderr='failure'}
        $report = Get-AgentFEMWindowsHostReport
        Assert ($script:calls.Count -eq 1 -and $report.errors[0].code -eq $code) 'Failed probe triggered another command or lost cause'
    }
    Set-Inventory '' ''
    $script:responses['--version'] = Ok 'Kernel version: 5.15.0'
    $report = Get-AgentFEMWindowsHostReport
    Assert ($report.errors[0].code -eq 'AFM-WIN-VERSION-UNRECOGNIZED') 'Kernel version mistaken for WSL version'
    function Get-Command { return $null }
    $script:calls = @()
    $report = Get-AgentFEMWindowsHostReport
    Assert ($report.status -eq 'command_unavailable' -and $script:calls.Count -eq 0) 'Missing command led to mutation/probe'
    $env:OS = 'Other'
    $report = Get-AgentFEMWindowsHostReport
    Assert ($report.status -eq 'unsupported_host') 'Non-Windows host accepted'
    # Exercise real process-start failure too, without any alternate launcher.
    $report = & $nativeProbe -Executable 'C:\AgentFEM-no-such-probe.exe' -Arguments '--version' -TimeoutSeconds 1
    Assert (-not $report.ok -and $report.code -eq 'AFM-WIN-PROBE-FAILED') 'Failed process launch not structured'
    # Windows runners have .NET Framework for PowerShell 5.1. Compile an inert
    # fixture to verify actual concurrent pipe draining and bounded timeout.
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    Assert (Test-Path -LiteralPath $compiler) 'Missing native probe fixture compiler'
    New-Item -ItemType Directory -Path $temp | Out-Null
    $exe = Join-Path $temp 'read-only-probe.exe'
    & $compiler /nologo /target:exe "/out:$exe" (Join-Path $PSScriptRoot 'WslProbeFixture.cs')
    Assert ($LASTEXITCODE -eq 0) 'Probe fixture compilation failed'
    $report = & $nativeProbe -Executable $exe -Arguments '--version'
    Assert ($report.ok -and $report.stdout.Trim() -eq 'WSL version: 2.6.1.0') 'Actual redirected UTF-16 output not decoded'
    $report = & $nativeProbe -Executable $exe -Arguments '--list --quiet'
    Assert ($report.ok -and $report.stdout.Length -eq 131072 -and $report.stderr.Length -eq 131072) 'Concurrent pipe drain truncated or deadlocked'
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $report = & $nativeProbe -Executable $exe -Arguments '--list --verbose' -TimeoutSeconds 1
    $timer.Stop()
    Assert ($report.code -eq 'AFM-WIN-PROBE-TIMEOUT' -and $timer.Elapsed.TotalSeconds -lt 5) 'Timeout was not bounded'
    Write-Host 'Windows host probe contracts passed (WSL mocked).'
} finally {
    $env:OS = $originalOS
    if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Recurse -Force }
}
exit 0
