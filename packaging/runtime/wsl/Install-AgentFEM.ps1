[CmdletBinding()]
param(
    [string]$Image = (Join-Path $PSScriptRoot "@IMAGE_FILENAME@"),
    [string]$ExpectedSha256 = "@IMAGE_SHA256@",
    [string]$DistributionName = "AgentFEM",
    [string]$InstallLocation = "",
    [string]$ProjectDirectory = "",
    [switch]$Upgrade,
    [string]$BackupDirectory = "",
    [switch]$RemoveBackupAfterSuccess,
    [switch]$NonInteractive,
    [switch]$PrepareWslOnly,
    [ValidateSet("Auto", "Web", "Store", "None")]
    [string]$WslBootstrapSource = "Auto"
)

$ErrorActionPreference = "Stop"
$RuntimeVersion = "@VERSION@"
$ExpectedRuntimeRecordSha256 = "@RUNTIME_RECORD_SHA256@"
. (Join-Path $PSScriptRoot "Installer-State.ps1")
Write-Host "AgentFEM Runtime for WSL2" -ForegroundColor Blue

function Get-InstalledWslVersion {
    $versionText = (& wsl.exe --version 2>&1 | Out-String)
    $versionMatch = [regex]::Match($versionText, '(\d+\.\d+\.\d+)')
    if ($LASTEXITCODE -ne 0 -or -not $versionMatch.Success) {
        return $null
    }
    return [version]$versionMatch.Groups[1].Value
}

function Test-AdministratorSession {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-ElevatedWslCommand {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Action
    )

    Write-Host "$Action requires Windows administrator approval." -ForegroundColor Yellow
    if ($NonInteractive -and -not (Test-AdministratorSession)) {
        throw '[AFM-WIN-ELEVATION-REQUIRED] An administrator operation needs user approval. Open this same installer in a local terminal without -NonInteractive; do not bypass host policy.'
    }
    Save-InstallState -Stage "requesting_elevation" -Message $Action
    try {
        if (Test-AdministratorSession) {
            # Already-authorised sessions do not need another elevated child process.
            & wsl.exe @Arguments
            $commandExitCode = $LASTEXITCODE
        } else {
            $process = Start-Process -FilePath "wsl.exe" -ArgumentList $Arguments `
                -Verb RunAs -Wait -PassThru
            $commandExitCode = $process.ExitCode
        }
    } catch {
        $nativeCode = $_.Exception.NativeErrorCode
        if ($nativeCode -eq 1223) {
            throw "[AFM-WIN-ELEVATION-CANCELLED] Administrator approval was cancelled. No alternate launch will be attempted."
        }
        throw "[AFM-WIN-ELEVATION-BLOCKED] Cannot launch the approved administrator operation: $($_.Exception.Message). If WorkBuddy blocks it, request its supported approval or launch the official installer yourself; do not switch shells to bypass the policy."
    }
    if ($commandExitCode -in @(3010, 1641)) {
        throw "[AFM-WIN-RESTART-REQUIRED] Windows requires a restart. Save your work, restart when ready, then reopen Install-AgentFEM.cmd."
    }
    if ($commandExitCode -in @(5, 740, 1223)) {
        throw "[AFM-WIN-ELEVATION-BLOCKED] Windows rejected or requires administrator approval (exit $commandExitCode). No alternate source will be attempted."
    }
    if ($commandExitCode -ne 0) {
        throw "[AFM-WIN-WSL-COMMAND-FAILED] $Action failed with exit code $commandExitCode."
    }
}

function Install-WslPlatform {
    param([Parameter(Mandatory = $true)][string]$Source)

    if ($Source -eq "None") {
        throw (
            "[AFM-WIN-WSL-NOT-READY] WSL is not ready and automatic platform " +
            "preparation was disabled. Install or update WSL, restart Windows, " +
            "then run this same installer again."
        )
    }

    $arguments = @("--install", "--no-distribution")
    if ($Source -in @("Auto", "Web")) {
        $arguments += "--web-download"
    }
    try {
        Invoke-ElevatedWslCommand -Arguments $arguments -Action (
            "Preparing WSL without downloading an unnecessary Linux distribution"
        )
    } catch {
        if ($Source -ne "Auto" -or $_.Exception.Message -notmatch "\[AFM-WIN-WSL-COMMAND-FAILED\]") {
            throw
        }
        Write-Warning "The direct web route failed; trying the Microsoft Store route once."
        Invoke-ElevatedWslCommand -Arguments @(
            "--install", "--no-distribution"
        ) -Action "Preparing WSL through the Microsoft Store route"
    }
}

function Update-WslPlatform {
    param([Parameter(Mandatory = $true)][string]$Source)

    if ($Source -eq "None") {
        throw (
            "[AFM-WIN-WSL-TOO-OLD] WSL must be updated to 2.4.4 or newer. " +
            "Automatic platform preparation was disabled."
        )
    }
    $arguments = @("--update")
    if ($Source -in @("Auto", "Web")) {
        $arguments += "--web-download"
    }
    try {
        Invoke-ElevatedWslCommand -Arguments $arguments -Action "Updating WSL"
    } catch {
        if ($Source -ne "Auto" -or $_.Exception.Message -notmatch "\[AFM-WIN-WSL-COMMAND-FAILED\]") {
            throw
        }
        Write-Warning "The direct web update failed; trying the Microsoft Store route once."
        Invoke-ElevatedWslCommand -Arguments @("--update") -Action (
            "Updating WSL through the Microsoft Store route"
        )
    }
}

function Require-WslPlatform {
    param([Parameter(Mandatory = $true)][string]$Source)

    $nativeArchitecture = $env:PROCESSOR_ARCHITEW6432
    if (-not $nativeArchitecture) {
        $nativeArchitecture = $env:PROCESSOR_ARCHITECTURE
    }
    if ($nativeArchitecture -ne "AMD64") {
        throw (
            "[AFM-WIN-ARCH-UNSUPPORTED] This AgentFEM WSL image is x86-64, " +
            "but Windows reports '$nativeArchitecture'. Download a runtime " +
            "matching the host architecture."
        )
    }

    $wslCommand = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (-not $wslCommand) {
        throw (
            "[AFM-WIN-WSL-COMMAND-MISSING] This session cannot resolve wsl.exe. " +
            "Check the Windows WSL component and the host's command permissions. " +
            "Do not infer an unsupported Windows version from this alone. " +
            "Use Microsoft's supported WSL installation route, then retry."
        )
    }

    $installedVersion = Get-InstalledWslVersion
    if ($null -eq $installedVersion) {
        Write-Host "WSL is not ready; preparing the Windows platform first." -ForegroundColor Yellow
        Install-WslPlatform -Source $Source
        $installedVersion = Get-InstalledWslVersion
        if ($null -eq $installedVersion) {
            throw (
                "[AFM-WIN-RESTART-REQUIRED] Windows has prepared WSL but has not " +
                "activated it yet. Restart Windows, then run this same " +
                "Install-AgentFEM.ps1 command again."
            )
        }
    }

    if ($installedVersion -lt [version]'2.4.4') {
        Write-Host "WSL $installedVersion must be updated for .wsl packages." -ForegroundColor Yellow
        Update-WslPlatform -Source $Source
        $installedVersion = Get-InstalledWslVersion
        if ($null -eq $installedVersion -or $installedVersion -lt [version]'2.4.4') {
            throw (
                "[AFM-WIN-RESTART-OR-UPDATE] WSL 2.4.4 or newer is required. " +
                "Restart Windows after the update and run this same installer again."
            )
        }
    }
    Write-Host "WSL $installedVersion is ready." -ForegroundColor Green
    return $installedVersion
}

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
    Test-AgentFEMBuildIdentity -Name $Name
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

function Test-AgentFEMBuildIdentity {
    param([Parameter(Mandatory = $true)][string]$Name)
    # This file is embedded before the image is hashed. Comparing its digest
    # distinguishes same-version development builds without a circular image hash.
    $output = (& wsl.exe --distribution $Name --user root --exec sha256sum `
        /opt/conda/share/agentfem/runtime-release.json | Out-String).Trim()
    $digest = ($output -split '\s+')[0]
    if ($LASTEXITCODE -ne 0 -or $digest -notmatch '^[a-fA-F0-9]{64}$' -or
        $digest -ne $ExpectedRuntimeRecordSha256) {
        throw '[AFM-WIN-RESUME-IDENTITY] Installed build does not match this package. No account, workspace or distribution will be replaced.'
    }
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

function Set-ProjectWorkspaceEnvironment {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)

    $script:PreviousProjectsHome = $env:AGENTFEM_PROJECTS_HOME
    $script:PreviousProjectsWslEnv = $env:WSLENV
    $env:AGENTFEM_PROJECTS_HOME = $WindowsPath
    $entries = @()
    if ($env:WSLENV) {
        $entries = @($env:WSLENV -split ':')
    }
    if ($entries -notcontains "AGENTFEM_PROJECTS_HOME/p") {
        $entries += "AGENTFEM_PROJECTS_HOME/p"
    }
    $env:WSLENV = $entries -join ':'
}

function Restore-ProjectWorkspaceEnvironment {
    if ($null -eq $script:PreviousProjectsHome) {
        Remove-Item Env:AGENTFEM_PROJECTS_HOME -ErrorAction SilentlyContinue
    } else {
        $env:AGENTFEM_PROJECTS_HOME = $script:PreviousProjectsHome
    }
    if ($null -eq $script:PreviousProjectsWslEnv) {
        Remove-Item Env:WSLENV -ErrorAction SilentlyContinue
    } else {
        $env:WSLENV = $script:PreviousProjectsWslEnv
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
    param(
        [Parameter(Mandatory = $true)][string]$RequestedName,
        [Parameter(Mandatory = $true)][string]$Projects
    )

    $existing = Get-WslDistributions
    $selectedName = $RequestedName
    $resumeName = $script:InstallState.distribution
    if ($resumeName -and $existing -contains $resumeName) {
        if ((Get-AgentFEMRuntimeVersion -Name $resumeName) -ne $RuntimeVersion) {
            throw "[AFM-WIN-RESUME-IDENTITY] Recorded distribution has a different runtime identity. Nothing was replaced."
        }
        Test-AgentFEMBuildIdentity -Name $resumeName
        Complete-NewInstallation -Name $resumeName -Projects $Projects
        return
    }
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

    $script:InstallState.distribution = $selectedName
    Save-InstallState -Stage "importing"
    Write-Host "Installing the offline AgentFEM distribution as '$selectedName'..."
    Install-AgentFEMDistribution -Name $selectedName -Location $InstallLocation
    Test-AgentFEMBuildIdentity -Name $selectedName
    Complete-NewInstallation -Name $selectedName -Projects $Projects
}

function Upgrade-AgentFEMDistribution {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Projects
    )

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
                "--user", "root", "--exec", "/usr/lib/agentfem/oobe.sh"
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

        $newProjectsPath = Convert-ToWslPath -Name $Name -WindowsPath $Projects
        Invoke-WslChecked -Arguments @(
            "--distribution", $Name,
            "--exec", "/opt/conda/bin/agentfem", "workspace",
            "--protect", "--path", $newProjectsPath, "--json"
        ) -FailureMessage (
            "The upgraded runtime was restored, but its projects could not be " +
            "verified and moved to persistent Windows storage."
        )

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
            persistent_projects = $Projects
        } | ConvertTo-Json | Set-Content -Encoding UTF8 $transactionRecord

        Write-Host "AgentFEM was upgraded in place to $RuntimeVersion." -ForegroundColor Green
        Write-Host "Projects and user settings were restored for '$oldUser'." -ForegroundColor Green
        Write-Host "Persistent projects: $Projects" -ForegroundColor Green
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

function Complete-NewInstallation {
    param([string]$Name, [string]$Projects)
    Save-InstallState -Stage "account_setup"
    if ($NonInteractive) {
        # Explicit root is limited to this read-only readiness probe, never used
        # as the simulation account or to set a password on the user's behalf.
        $account = (& wsl.exe --distribution $Name --user root --exec sh -c `
            'u=$(getent passwd 1000 | cut -d: -f1); test -n "$u" && passwd -S "$u"' | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $account -notmatch '^[a-z_][a-z0-9_-]*\s+P\s') {
            throw '[AFM-WIN-ACCOUNT-INPUT-REQUIRED] A Linux username/password must be entered privately in a local terminal. Reopen this installer without -NonInteractive; never send the password in chat.'
        }
    }
    Set-ProjectWorkspaceEnvironment -WindowsPath $Projects
    try {
        # WSL documents OOBE for the first interactive shell. An exec of `true`
        # is not an account-setup contract: invoke our idempotent setup explicitly.
        Invoke-WslChecked -Arguments @("--distribution", $Name, "--user", "root",
            "--exec", "/usr/lib/agentfem/oobe.sh") `
            -FailureMessage "[AFM-WIN-ACCOUNT-SETUP] Linux account setup is incomplete. Reopen the installer to continue."
    } finally {
        Restore-ProjectWorkspaceEnvironment
    }
    $user = Get-DefaultLinuxUser -Name $Name
    Save-InstallState -Stage "protecting_workspace"
    $linuxProjects = Convert-ToWslPath -Name $Name -WindowsPath $Projects
    Invoke-WslChecked -Arguments @("--distribution", $Name, "--exec", "/opt/conda/bin/agentfem",
        "workspace", "--protect", "--path", $linuxProjects, "--json") `
        -FailureMessage "[AFM-WIN-WORKSPACE] Project protection did not complete."
    Save-InstallState -Stage "verifying_runtime"
    Test-AgentFEMDistribution -Name $Name
    # Run as the ordinary user with a fresh JIT cache, not merely an import check.
    $verifier = Convert-ToWslPath -Name $Name -WindowsPath (Join-Path $PSScriptRoot "verify_installed_runtime.py")
    $acceptancePath = Join-Path $script:InstallStateDirectory "acceptance.json"
    if (Test-Path -LiteralPath $acceptancePath) {
        Move-Item -LiteralPath $acceptancePath -Destination (Join-Path $script:InstallStateDirectory ("acceptance-" + [guid]::NewGuid().ToString() + ".json"))
    }
    $report = Convert-ToWslPath -Name $Name -WindowsPath $acceptancePath
    Invoke-WslChecked -Arguments @("--distribution", $Name, "--exec", "/opt/conda/bin/python",
        $verifier, "--prefix", "/opt/conda", "--profile", "core", "--skip-mpi", "--output", $report) `
        -FailureMessage "[AFM-WIN-FIRST-SOLVE] Runtime acceptance failed. Keep the log and reopen the installer after correcting the reported issue."
    $acceptancePath = Join-Path $script:InstallStateDirectory "acceptance.json"
    $acceptance = Get-Content -LiteralPath $acceptancePath -Raw | ConvertFrom-Json
    if ($acceptance.schema -ne "agentfem.runtime-acceptance" -or $acceptance.status -ne "accepted") {
        throw "[AFM-WIN-FIRST-SOLVE] The acceptance record is missing or did not accept this runtime."
    }
    $script:InstallState.acceptance_report = $acceptancePath
    $script:InstallState.linux_user = $user
    Save-InstallState -Stage "completed"
    Write-Host "AgentFEM $RuntimeVersion passed its first-solve checks. Persistent projects: $Projects" -ForegroundColor Green
}

$exitCode = 0
try {
    if ($PrepareWslOnly) {
        # Also usable from a source checkout for Ubuntu + mirror users. No
        # unpublished Complete Runtime image or scientific Python is required.
        Initialize-InstallState -Identity @{operation='prepare_wsl_platform'; architecture='x86_64'}
    } else {
        Initialize-InstallState -Identity @{
            image_sha256 = $ExpectedSha256
            runtime_record_sha256 = $ExpectedRuntimeRecordSha256
            version = $RuntimeVersion
            distribution = $DistributionName
            install_location = $InstallLocation
            project_directory = $ProjectDirectory
            upgrade = [bool]$Upgrade
        }
    }
    Start-Transcript -Path (Join-Path $script:InstallStateDirectory "install.log") -Append | Out-Null
    $script:TranscriptStarted = $true
    $script:InstallState.bootstrap_source = $WslBootstrapSource
    if ($PrepareWslOnly) {
        if ($Upgrade -or $InstallLocation -or $ProjectDirectory) {
            throw '[AFM-WIN-OPTION-CONFLICT] Platform-only preparation does not install or upgrade an AgentFEM distribution. Remove distribution-install options.'
        }
        Save-InstallState -Stage 'preparing_wsl'
        $platformVersion = Require-WslPlatform -Source $WslBootstrapSource
        $script:InstallState.wsl_version = $platformVersion.ToString()
        Save-InstallState -Stage 'platform_ready'
        Write-Host 'WSL platform prepared. AgentFEM is not installed by this operation.' -ForegroundColor Green
        Write-Host 'Reuse an existing WSL2 Ubuntu or install the named Ubuntu distribution, then follow INSTALL.md for the mirror environment.'
        exit 0
    }
    if ($NonInteractive -and $Upgrade) {
        throw '[AFM-WIN-UPGRADE-INTERACTIVE] Replacement upgrades require an attended terminal. No distribution was changed.'
    }
    Save-InstallState -Stage 'checking_package'

    foreach ($requiredFile in @($Image, (Join-Path $PSScriptRoot "verify_installed_runtime.py"))) {
        if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
            throw "[AFM-WIN-PACKAGE-INCOMPLETE] Missing package file: $requiredFile. Extract the entire official ZIP before installing."
        }
    }
    if ($ExpectedSha256 -notmatch '^[a-fA-F0-9]{64}$' -or
        $ExpectedRuntimeRecordSha256 -notmatch '^[a-fA-F0-9]{64}$') {
        throw "[AFM-WIN-PACKAGE-IDENTITY] A rendered release image checksum is required. Use the official complete package."
    }


    if ($ExpectedSha256) {
        $actual = (Get-FileHash -Algorithm SHA256 $Image).Hash.ToLowerInvariant()
        if ($actual -ne $ExpectedSha256.ToLowerInvariant()) {
            throw "SHA-256 mismatch. Expected $ExpectedSha256, received $actual."
        }
        Write-Host "Image integrity verified." -ForegroundColor Green
    }

    Save-InstallState -Stage "preparing_wsl"
    $installedWslVersion = Require-WslPlatform -Source $WslBootstrapSource

    $requestedName = $DistributionName.Trim()
    if (-not $requestedName) {
        throw "DistributionName must not be empty."
    }
    if (-not $ProjectDirectory) {
        $documents = [Environment]::GetFolderPath("MyDocuments")
        if (-not $documents) {
            $documents = Join-Path $env:USERPROFILE "Documents"
        }
        $ProjectDirectory = Join-Path $documents "AgentFEMProjects"
    }
    $persistentProjects = [System.IO.Path]::GetFullPath($ProjectDirectory)
    New-Item -ItemType Directory -Force -Path $persistentProjects | Out-Null
    Write-Host "User projects will persist outside WSL at: $persistentProjects" -ForegroundColor Green
    if ($Upgrade) {
        Upgrade-AgentFEMDistribution -Name $requestedName -Projects $persistentProjects
    } else {
        Install-SideBySide -RequestedName $requestedName -Projects $persistentProjects
    }
    Save-InstallState -Stage "completed"
} catch {
    $message = $_.Exception.Message
    $stage = "failed"
    $exitCode = 1
    if ($message -match "AFM-WIN-RESTART") { $stage = "restart_required"; $exitCode = 3010 }
    elseif ($message -match "AFM-WIN-ELEVATION") { $stage = "awaiting_authorization" }
    elseif ($message -match "AFM-WIN-ACCOUNT-INPUT-REQUIRED") { $stage = "account_setup" }
    if ($script:InstallStateOwned) { Save-InstallState -Stage $stage -Message $message }
    Write-Host $message -ForegroundColor Red
    if ($script:InstallStateDirectory) { Write-Host "Installation records: $script:InstallStateDirectory" }
} finally {
    if ($script:TranscriptStarted) { Stop-Transcript | Out-Null }
    if ($script:InstallLock) { $script:InstallLock.Dispose() }
}
exit $exitCode
