[CmdletBinding()]
param(
    [string]$Image = (Join-Path $PSScriptRoot "@IMAGE_FILENAME@"),
    [string]$ExpectedSha256 = "@IMAGE_SHA256@",
    [string]$DistributionName = "AgentFEM",
    [string]$InstallLocation = "",
    [switch]$Upgrade,
    [string]$BackupDirectory = "",
    [switch]$RemoveBackupAfterSuccess
)

$ErrorActionPreference = "Stop"
$RuntimeVersion = "@VERSION@"
Write-Host "AgentFEM Runtime for WSL2" -ForegroundColor Blue

function Invoke-WslChecked {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$FailureMessage
    )

    & wsl.exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

function Get-WslDistributions {
    $reported = @(& wsl.exe --list --quiet)
    if ($LASTEXITCODE -ne 0) {
        throw "WSL could not list its installed distributions. Run 'wsl --status' and retry."
    }
    return @(
        $reported |
            ForEach-Object { ($_ -replace "`0", "").Trim() } |
            Where-Object { $_ }
    )
}

function Install-AgentFEMDistribution {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string]$Location = ""
    )

    $arguments = @(
        "--install",
        "--from-file", (Resolve-Path $Image).Path,
        "--name", $Name,
        "--no-launch"
    )
    if ($Location) {
        $resolvedLocation = [System.IO.Path]::GetFullPath($Location)
        if ((Test-Path $resolvedLocation) -and (Get-ChildItem -Force $resolvedLocation | Select-Object -First 1)) {
            throw "InstallLocation must be empty: $resolvedLocation"
        }
        New-Item -ItemType Directory -Force -Path $resolvedLocation | Out-Null
        $arguments += @("--location", $resolvedLocation)
    }
    Invoke-WslChecked -Arguments $arguments -FailureMessage (
        "WSL could not import the AgentFEM image. Run 'wsl --status' and " +
        "'wsl --version'; then retry after 'wsl --update --web-download'."
    )
}

function Test-AgentFEMDistribution {
    param([Parameter(Mandatory = $true)][string]$Name)

    Write-Host "Checking candidate runtime '$Name'..."
    Invoke-WslChecked -Arguments @(
        "--distribution", $Name,
        "--user", "root",
        "--exec", "/opt/conda/bin/agentfem", "doctor", "--json"
    ) -FailureMessage "The candidate runtime failed 'agentfem doctor'."
    $reportedVersion = (
        & wsl.exe --distribution $Name --user root --exec `
            sh -c "cat /usr/lib/agentfem/runtime-version"
    ).Trim()
    if ($LASTEXITCODE -ne 0 -or $reportedVersion -ne $RuntimeVersion) {
        throw "Candidate identity mismatch. Expected AgentFEM $RuntimeVersion, received '$reportedVersion'."
    }
    Write-Host "Candidate AgentFEM $reportedVersion passed its health check." -ForegroundColor Green
}

function Get-AgentFEMRuntimeVersion {
    param([Parameter(Mandatory = $true)][string]$Name)

    $reportedVersion = (
        & wsl.exe --distribution $Name --user root --exec `
            sh -c "cat /usr/lib/agentfem/runtime-version 2>/dev/null || true"
    ).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $reportedVersion) {
        return "unknown"
    }
    return $reportedVersion
}

function Get-DefaultLinuxUser {
    param([Parameter(Mandatory = $true)][string]$Name)

    $username = (
        & wsl.exe --distribution $Name --exec sh -c "id -un"
    ).Trim()
    if ($LASTEXITCODE -ne 0 -or $username -notmatch '^[a-z_][a-z0-9_-]*$' -or $username -eq "root") {
        throw "Could not identify a non-root default user in '$Name'."
    }
    return $username
}

function Convert-ToWslPath {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$WindowsPath
    )

    $linuxPath = (
        & wsl.exe --distribution $Name --exec wslpath -a -u $WindowsPath
    ).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $linuxPath) {
        throw "Could not map the backup path into WSL: $WindowsPath"
    }
    return $linuxPath
}

function Set-UpgradeUserEnvironment {
    param([Parameter(Mandatory = $true)][string]$Username)

    $script:PreviousUpgradeUser = $env:AGENTFEM_UPGRADE_USER
    $script:PreviousWslEnv = $env:WSLENV
    $env:AGENTFEM_UPGRADE_USER = $Username
    $entries = @()
    if ($env:WSLENV) {
        $entries = @($env:WSLENV -split ':')
    }
    if ($entries -notcontains "AGENTFEM_UPGRADE_USER") {
        $entries += "AGENTFEM_UPGRADE_USER"
    }
    $env:WSLENV = $entries -join ':'
}

function Restore-UpgradeUserEnvironment {
    if ($null -eq $script:PreviousUpgradeUser) {
        Remove-Item Env:AGENTFEM_UPGRADE_USER -ErrorAction SilentlyContinue
    } else {
        $env:AGENTFEM_UPGRADE_USER = $script:PreviousUpgradeUser
    }
    if ($null -eq $script:PreviousWslEnv) {
        Remove-Item Env:WSLENV -ErrorAction SilentlyContinue
    } else {
        $env:WSLENV = $script:PreviousWslEnv
    }
}

function Set-ImportedDefaultUser {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Username
    )

    $command = "printf '\n[user]\ndefault=$Username\n' >> /etc/wsl.conf"
    Invoke-WslChecked -Arguments @(
        "--distribution", $Name,
        "--user", "root",
        "--exec", "sh", "-c", $command
    ) -FailureMessage "The rollback image was restored, but its default Linux user could not be selected."
    Invoke-WslChecked -Arguments @(
        "--terminate", $Name
    ) -FailureMessage "The restored distribution could not be restarted cleanly."
}

function Install-SideBySide {
    param([Parameter(Mandatory = $true)][string]$RequestedName)

    $existing = Get-WslDistributions
    $selectedName = $RequestedName
    if ($existing -contains $selectedName) {
        if ($selectedName -ne "AgentFEM") {
            throw "A WSL distribution named '$selectedName' already exists. Choose a different -DistributionName; no existing distribution was changed."
        }
        $selectedName = "AgentFEM-$RuntimeVersion"
        if ($existing -contains $selectedName) {
            throw "AgentFEM $RuntimeVersion is already registered as '$selectedName'. No existing distribution was changed."
        }
        Write-Host "Keeping the existing AgentFEM distribution and installing this runtime side by side as '$selectedName'." -ForegroundColor Yellow
        Write-Host "Your existing projects and environment will not be modified." -ForegroundColor Yellow
    }

    Write-Host "Installing the offline AgentFEM distribution as '$selectedName'..."
    Install-AgentFEMDistribution -Name $selectedName -Location $InstallLocation
    Write-Host "Starting AgentFEM for first-time user setup..." -ForegroundColor Green
    Write-Host "When setup is complete, run 'agentfem doctor'." -ForegroundColor Green
    & wsl.exe --distribution $selectedName
}

function Upgrade-AgentFEMDistribution {
    param([Parameter(Mandatory = $true)][string]$Name)

    if ($Name -ne "AgentFEM") {
        throw "Transactional replacement currently supports only the stable 'AgentFEM' distribution name."
    }
    $existing = Get-WslDistributions
    if ($existing -notcontains $Name) {
        throw "No existing '$Name' distribution was found. Run the installer without -Upgrade for a first installation."
    }

    $candidateName = "AgentFEM-$RuntimeVersion-candidate"
    if ($existing -contains $candidateName) {
        throw "A previous candidate named '$candidateName' exists. Inspect it before retrying; nothing was changed."
    }

    if (-not $BackupDirectory) {
        $BackupDirectory = Join-Path $env:LOCALAPPDATA "AgentFEM\Backups"
    }
    $stamp = Get-Date -Format "yyyyMMddTHHmmss"
    $transactionDirectory = Join-Path (
        [System.IO.Path]::GetFullPath($BackupDirectory)
    ) "$stamp-before-$RuntimeVersion"
    New-Item -ItemType Directory -Force -Path $transactionDirectory | Out-Null
    $fullBackup = Join-Path $transactionDirectory "AgentFEM-previous-full.tar"
    $homeBackup = Join-Path $transactionDirectory "AgentFEM-user-home.tar.gz"
    $rollbackLocation = Join-Path $transactionDirectory "rollback-runtime"
    $transactionRecord = Join-Path $transactionDirectory "upgrade.json"
    $oldRemoved = $false
    $oldUser = ""
    $oldVersion = Get-AgentFEMRuntimeVersion -Name $Name
    if ($oldVersion -eq $RuntimeVersion) {
        throw "AgentFEM $RuntimeVersion is already installed as '$Name'."
    }
    Write-Host "Preparing AgentFEM replacement upgrade: $oldVersion -> $RuntimeVersion"
    Write-Host "A complete recovery snapshot may require several gigabytes of free disk space." -ForegroundColor Yellow
    @{
        schema = "agentfem.runtime-upgrade"
        schema_version = 1
        status = "preparing"
        distribution = $Name
        from_version = $oldVersion
        to_version = $RuntimeVersion
        created_at = (Get-Date).ToUniversalTime().ToString("o")
        full_backup = $fullBackup
        home_backup = $homeBackup
    } | ConvertTo-Json | Set-Content -Encoding UTF8 $transactionRecord

    try {
        # The new scientific runtime must work before the old installation is touched.
        Install-AgentFEMDistribution -Name $candidateName
        Test-AgentFEMDistribution -Name $candidateName

        $oldUser = Get-DefaultLinuxUser -Name $Name
        Write-Host "Backing up the complete existing runtime and '$oldUser' home directory..."
        $homePath = Convert-ToWslPath -Name $Name -WindowsPath $homeBackup
        Invoke-WslChecked -Arguments @(
            "--distribution", $Name,
            "--user", "root",
            "--exec", "tar", "-C", "/", "-czf", $homePath, "home/$oldUser"
        ) -FailureMessage "The user project backup failed. The existing AgentFEM distribution was not removed."
        Invoke-WslChecked -Arguments @(
            "--terminate", $Name
        ) -FailureMessage "The existing AgentFEM distribution could not be stopped for its complete backup."
        Invoke-WslChecked -Arguments @(
            "--export", $Name, $fullBackup
        ) -FailureMessage "The existing AgentFEM distribution could not be exported. It was not removed."

        Write-Host "Backups completed. Switching the stable AgentFEM runtime..." -ForegroundColor Green
        Invoke-WslChecked -Arguments @(
            "--unregister", $Name
        ) -FailureMessage "The old runtime could not be unregistered; its backups remain at '$transactionDirectory'."
        $oldRemoved = $true

        Install-AgentFEMDistribution -Name $Name -Location $InstallLocation

        # Reuse the previous Linux username. The OOBE still asks the user to
        # choose a password, but the command exits automatically afterwards.
        Set-UpgradeUserEnvironment -Username $oldUser
        try {
            Write-Host "Recreating Linux account '$oldUser'. Choose its sudo password when prompted..."
            Invoke-WslChecked -Arguments @(
                "--distribution", $Name,
                "--exec", "true"
            ) -FailureMessage "The upgraded runtime could not complete first-use account setup."
        } finally {
            Restore-UpgradeUserEnvironment
        }

        $newUser = Get-DefaultLinuxUser -Name $Name
        if ($newUser -ne $oldUser) {
            throw "The upgraded runtime created user '$newUser', but '$oldUser' is required to restore the previous home directory."
        }
        $newHomePath = Convert-ToWslPath -Name $Name -WindowsPath $homeBackup
        Invoke-WslChecked -Arguments @(
            "--distribution", $Name,
            "--user", "root",
            "--exec", "tar", "-C", "/", "-xzf", $newHomePath
        ) -FailureMessage "The new runtime was installed, but the previous user files could not be restored."

        Test-AgentFEMDistribution -Name $Name
        Invoke-WslChecked -Arguments @(
            "--distribution", $Name,
            "--exec", "/opt/conda/bin/agentfem", "doctor"
        ) -FailureMessage "The upgraded user environment failed its final health check."

        @{
            schema = "agentfem.runtime-upgrade"
            schema_version = 1
            status = "completed"
            distribution = $Name
            from_version = $oldVersion
            to_version = $RuntimeVersion
            completed_at = (Get-Date).ToUniversalTime().ToString("o")
            full_backup = $fullBackup
            home_backup = $homeBackup
            doctor = "passed"
            projects_restored = $true
        } | ConvertTo-Json | Set-Content -Encoding UTF8 $transactionRecord

        Write-Host "AgentFEM was upgraded in place to $RuntimeVersion." -ForegroundColor Green
        Write-Host "Projects and user settings were restored for '$oldUser'." -ForegroundColor Green
        & wsl.exe --unregister $candidateName | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Upgrade succeeded, but temporary candidate '$candidateName' could not be removed. Remove only that candidate after inspection."
        }
        if ($RemoveBackupAfterSuccess) {
            try {
                Remove-Item -Force $fullBackup, $homeBackup
                Write-Host "Successful-upgrade backups were removed as requested."
            } catch {
                Write-Warning "Upgrade succeeded, but recovery files could not be removed: $($_.Exception.Message)"
            }
        } else {
            Write-Host "Recovery backup retained at: $transactionDirectory" -ForegroundColor Yellow
        }
        Write-Host "Start it with: wsl -d AgentFEM" -ForegroundColor Green
    } catch {
        $failure = $_
        Write-Warning "Upgrade did not complete: $($failure.Exception.Message)"
        if ($oldRemoved -and (Test-Path $fullBackup)) {
            Write-Warning "Restoring the previous AgentFEM distribution from its complete backup..."
            $registered = Get-WslDistributions
            if ($registered -contains $Name) {
                Invoke-WslChecked -Arguments @(
                    "--unregister", $Name
                ) -FailureMessage "Automatic rollback could not remove the incomplete new runtime. Keep '$fullBackup' for manual recovery."
            }
            New-Item -ItemType Directory -Force -Path $rollbackLocation | Out-Null
            Invoke-WslChecked -Arguments @(
                "--import", $Name, $rollbackLocation, $fullBackup, "--version", "2"
            ) -FailureMessage "Automatic rollback failed. Keep '$fullBackup' and restore it manually with 'wsl --import'."
            Set-ImportedDefaultUser -Name $Name -Username $oldUser
            @{
                schema = "agentfem.runtime-upgrade"
                schema_version = 1
                status = "rolled_back"
                distribution = $Name
                from_version = $oldVersion
                to_version = $RuntimeVersion
                rolled_back_at = (Get-Date).ToUniversalTime().ToString("o")
                full_backup = $fullBackup
                home_backup = $homeBackup
                error = $failure.Exception.Message
            } | ConvertTo-Json | Set-Content -Encoding UTF8 $transactionRecord
            Write-Warning "The previous AgentFEM runtime was restored. Backup retained at '$transactionDirectory'."
        }
        $registered = Get-WslDistributions
        if ($registered -contains $candidateName) {
            & wsl.exe --unregister $candidateName | Out-Null
        }
        if (-not $oldRemoved) {
            @{
                schema = "agentfem.runtime-upgrade"
                schema_version = 1
                status = "failed_before_switch"
                distribution = $Name
                from_version = $oldVersion
                to_version = $RuntimeVersion
                failed_at = (Get-Date).ToUniversalTime().ToString("o")
                full_backup = $fullBackup
                home_backup = $homeBackup
                error = $failure.Exception.Message
            } | ConvertTo-Json | Set-Content -Encoding UTF8 $transactionRecord
        }
        throw $failure
    }
}

if (-not (Test-Path $Image)) {
    throw "Runtime image not found: $Image"
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL is not installed. In Administrator PowerShell run 'wsl --install --no-distribution --web-download', restart Windows, then run this installer again."
}

$versionText = (& wsl.exe --version 2>&1 | Out-String)
$versionMatch = [regex]::Match($versionText, '(\d+\.\d+\.\d+)')
if ($LASTEXITCODE -ne 0 -or -not $versionMatch.Success) {
    throw "A current Store-delivered WSL is required. Run 'wsl --update --web-download', restart Windows if requested, and retry. Use 'wsl --status' to diagnose an incomplete Windows feature installation."
}
$installedWslVersion = [version]$versionMatch.Groups[1].Value
if ($installedWslVersion -lt [version]'2.4.4') {
    throw "WSL $installedWslVersion is too old for .wsl runtime packages. Run 'wsl --update --web-download' and retry; AgentFEM requires WSL 2.4.4 or newer."
}
Write-Host "WSL $installedWslVersion is ready." -ForegroundColor Green

if ($ExpectedSha256) {
    $actual = (Get-FileHash -Algorithm SHA256 $Image).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedSha256.ToLowerInvariant()) {
        throw "SHA-256 mismatch. Expected $ExpectedSha256, received $actual."
    }
    Write-Host "Image integrity verified." -ForegroundColor Green
}

$requestedName = $DistributionName.Trim()
if (-not $requestedName) {
    throw "DistributionName must not be empty."
}
if ($Upgrade) {
    Upgrade-AgentFEMDistribution -Name $requestedName
} else {
    Install-SideBySide -RequestedName $requestedName
}
