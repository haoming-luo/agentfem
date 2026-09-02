[CmdletBinding()]
param(
    [string]$DistributionName = "AgentFEM",
    [string]$ProjectDirectory = "",
    [string]$BackupDirectory = "",
    [switch]$SkipRuntimeBackup
)

$ErrorActionPreference = "Stop"
Write-Host "AgentFEM Runtime safe removal" -ForegroundColor Blue

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL is not installed."
}

$reported = @(& wsl.exe --list --quiet) | ForEach-Object {
    ($_ -replace "`0", "").Trim()
} | Where-Object { $_ }
if ($LASTEXITCODE -ne 0 -or $reported -notcontains $DistributionName) {
    throw "No WSL distribution named '$DistributionName' was found."
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
$linuxProjects = (
    & wsl.exe --distribution $DistributionName --exec `
        wslpath -a -u $persistentProjects
).Trim()
if ($LASTEXITCODE -ne 0 -or -not $linuxProjects) {
    throw "Could not map the persistent Windows project directory into WSL. Nothing was removed."
}

Write-Host "Protecting projects outside the WSL distribution..."
& wsl.exe --distribution $DistributionName --exec `
    /opt/conda/bin/agentfem workspace --protect --path $linuxProjects --json
if ($LASTEXITCODE -ne 0) {
    throw "Project protection did not pass. Nothing was removed."
}

$backup = $null
if (-not $SkipRuntimeBackup) {
    if (-not $BackupDirectory) {
        $BackupDirectory = Join-Path $env:LOCALAPPDATA "AgentFEM\Backups"
    }
    $stamp = Get-Date -Format "yyyyMMddTHHmmss"
    $transactionDirectory = Join-Path (
        [System.IO.Path]::GetFullPath($BackupDirectory)
    ) "$stamp-before-uninstall"
    New-Item -ItemType Directory -Force -Path $transactionDirectory | Out-Null
    $backup = Join-Path $transactionDirectory "$DistributionName-full.tar"
    Write-Host "Exporting a complete recovery snapshot..."
    & wsl.exe --terminate $DistributionName
    if ($LASTEXITCODE -ne 0) {
        throw "The distribution could not be stopped. Nothing was removed."
    }
    & wsl.exe --export $DistributionName $backup
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $backup)) {
        throw "The recovery snapshot failed. Nothing was removed."
    }
}

Write-Host "Removing only the AgentFEM runtime; projects remain on Windows..."
& wsl.exe --unregister $DistributionName
if ($LASTEXITCODE -ne 0) {
    throw "WSL could not unregister '$DistributionName'. The project directory and any recovery snapshot remain safe."
}

Write-Host "AgentFEM runtime removed." -ForegroundColor Green
Write-Host "Projects retained at: $persistentProjects" -ForegroundColor Green
if ($backup) {
    Write-Host "Recovery snapshot retained at: $backup" -ForegroundColor Yellow
}
