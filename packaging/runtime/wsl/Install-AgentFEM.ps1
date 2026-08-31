[CmdletBinding()]
param(
    [string]$Image = (Join-Path $PSScriptRoot "@IMAGE_FILENAME@"),
    [string]$ExpectedSha256 = "@IMAGE_SHA256@"
)

$ErrorActionPreference = "Stop"
Write-Host "AgentFEM Runtime for WSL2" -ForegroundColor Blue

if (-not (Test-Path $Image)) {
    throw "Runtime image not found: $Image"
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL is not installed. Run 'wsl --install --no-distribution' as Administrator, reboot, then run this installer again."
}

if ($ExpectedSha256) {
    $actual = (Get-FileHash -Algorithm SHA256 $Image).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedSha256.ToLowerInvariant()) {
        throw "SHA-256 mismatch. Expected $ExpectedSha256, received $actual."
    }
    Write-Host "Image integrity verified." -ForegroundColor Green
}

$existing = @(wsl.exe --list --quiet) -replace "`0", ""
if ($existing -contains "AgentFEM") {
    throw "A WSL distribution named AgentFEM already exists. This installer will not overwrite it."
}

Write-Host "Installing the offline AgentFEM distribution..."
wsl.exe --install --from-file (Resolve-Path $Image)
if ($LASTEXITCODE -ne 0) {
    throw "WSL could not import the image. Update WSL to 2.4.4 or newer and retry."
}

Write-Host "Starting AgentFEM for first-time user setup..." -ForegroundColor Green
wsl.exe --distribution AgentFEM
