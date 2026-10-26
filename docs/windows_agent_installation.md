---
title: "Install AgentFEM on Windows with an AI assistant"
description: "Choose a documented WSL2 installation route, reuse existing environments, and verify a first finite-element solve without replacing user projects."
---

# Windows: WSL → Ubuntu → AgentFEM

For an AI assistant: complete the permitted work yourself. Show the user only
the next action that needs their approval. **No GitHub clone, Windows Python,
Docker or AgentFEM installer scripts are required.** Already have working
Ubuntu/WSL2? Skip platform installation and reuse it.

国内用户：先准备 WSL，再从清华镜像导入 Ubuntu，最后镜像安装 AgentFEM。
不用先克隆 GitHub，不用默认打开“Windows 功能”逐项勾选；让智能体完成允许的操作。

## One prompt, one visible next step

The [web prompt](get_started/index.md#let-your-agent-install-it) and
`installation.onboarding` in [agentfem.json](agentfem.json) share this flow.
Detect the OS, not the user's location. Use an explicitly requested/recorded
mirror route; do not infer it from Chinese text, IP or VPN. Try each permitted
stage once. If it fails, stop that stage and give one manual action immediately;
do not cycle through commands or wait silently. A new attempt needs changed
readiness/source or user direction. Successful earlier stages stay completed.
If the host cannot execute commands, give the same manual path directly.

Show a short status such as `WSL: needs approval — open Terminal (Administrator)`
or `Environment: installing — no action needed`. Do not dump raw diagnostics
unless asked; keep the exact error for troubleshooting. This is agent-assisted
installation, not an unattended executable that bypasses Windows approval.

## 1. Prepare WSL only if needed

Inspect `wsl --status` and `wsl --list --verbose` in PowerShell. A blocked
command is a permission issue, not proof that WSL is missing. A Store Ubuntu
App is not yet a registered distribution: check the WSL inventory.

If WSL is missing, use approved elevation. If the agent cannot obtain it,
give just this handoff: **“Open Terminal (Administrator), run the following
command, and restart only if Windows asks.”**

```powershell
wsl --install --no-distribution --web-download
```

This deliberately installs the platform without downloading Ubuntu.
`--web-download` is an alternative to the Store, not a domestic mirror or a
speed guarantee. If its download fails, an approved Store route is
`wsl --install --no-distribution`. Do not switch routes after denied consent
or a restart request. Do not start with the Windows Features GUI; reserve
manual component/firmware repair for a specific diagnosed failure.

## 2. Add Ubuntu from the mirror, only if absent

For a requested mainland/mirror route, use the verified Ubuntu image import
below directly. Do not first loop through Store, winget and
`wsl --install -d Ubuntu` after a distribution-metadata timeout. Importing a
local image avoids that metadata download; it does not bypass permissions.

### Copy-safe x64 mirror recipe (PowerShell)

For a new distribution only. Reuse an existing compatible one instead of
running this block. It uses a user-owned directory, checks every native
command, and verifies the exact image against the mirror's published manifest.
It does not overwrite a registered distribution or an existing destination.

```powershell
$ErrorActionPreference = 'Stop'
$afmDistro = 'Ubuntu-24.04'
$afmTarget = Join-Path $env:LOCALAPPDATA 'AgentFEM\WSL\Ubuntu-24.04'
$afmArch = $env:PROCESSOR_ARCHITEW6432
if (-not $afmArch) { $afmArch = $env:PROCESSOR_ARCHITECTURE }
if ($afmArch -ne 'AMD64') { throw 'This recipe is x64 only; select a verified image for your architecture.' }
$afmExisting = @(wsl.exe --list --quiet)
if ($LASTEXITCODE -ne 0) { throw 'Check WSL readiness/permission before importing.' }
$afmExisting = @($afmExisting | ForEach-Object { ($_ -replace "`0", '').Trim() })
if (($afmExisting -contains $afmDistro) -or (Test-Path -LiteralPath $afmTarget)) {
    throw 'Existing distribution or directory: inspect and reuse; do not overwrite.'
}
$afmDownload = Join-Path $env:TEMP ('agentfem-ubuntu-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $afmDownload | Out-Null
$afmBase = 'https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/24.04'
$afmSums = Join-Path $afmDownload 'SHA256SUMS'
curl.exe -fL --retry 2 --connect-timeout 20 -o $afmSums "$afmBase/SHA256SUMS"
if ($LASTEXITCODE -ne 0) { throw 'Ubuntu checksum download failed.' }
$afmEntries = @(Get-Content -LiteralPath $afmSums | Select-String '^([0-9a-fA-F]{64})\s+\*?(ubuntu-24\.04(?:\.\d+)?-wsl-amd64\.wsl)$')
if ($afmEntries.Count -ne 1) { throw 'Expected one Ubuntu 24.04 x64 WSL image; inspect the published manifest.' }
$afmMatch = $afmEntries[0].Matches[0]
$afmFile = $afmMatch.Groups[2].Value
$afmImage = Join-Path $afmDownload $afmFile
curl.exe -fL --retry 2 --connect-timeout 20 -o $afmImage "$afmBase/$afmFile"
if ($LASTEXITCODE -ne 0) { throw 'Ubuntu image download failed; do not import it.' }
if ((Get-FileHash -LiteralPath $afmImage -Algorithm SHA256).Hash -ne $afmMatch.Groups[1].Value) {
    throw 'Ubuntu checksum mismatch; do not import.'
}
New-Item -ItemType Directory -Force -Path (Split-Path $afmTarget) | Out-Null
wsl.exe --import $afmDistro $afmTarget $afmImage --version 2
if ($LASTEXITCODE -ne 0) { throw 'Import failed; retain the image and error for diagnosis.' }
wsl.exe --list --verbose
if ($LASTEXITCODE -ne 0) { throw 'Cannot verify distribution inventory.' }
```

Retain the downloaded manifest, filename and digest. This checks integrity
against the selected HTTPS mirror, not independent signature authentication.
If policy requires upstream authenticity, also verify Ubuntu's signed
`SHA256SUMS.gpg` against a trusted Ubuntu signing key before import. Never
skip a failed checksum. The 24.04 directory tracks its maintenance release;
do not guess a filename or use a desktop ISO. See the
[TUNA image directory](https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/24.04/).

### Account: automate setup, not a shared password

Imported distributions normally start as root. This is useful for initial
setup, **not** a reason to run the solver as root. With permission to prepare
this new distribution, the agent can create a normal account without asking
the user to choose a password:

```powershell
wsl.exe -d Ubuntu-24.04 -u root -- useradd --create-home --shell /bin/bash afm
if ($LASTEXITCODE -ne 0) { throw 'Account creation failed; inspect existing users before retrying.' }
wsl.exe -d Ubuntu-24.04 -u afm --cd ~
```

Use this only for the newly imported distribution; reuse existing users.
The new account has password login locked, not an empty/shared password, and
is not granted passwordless sudo. Use explicit `-u afm` for subsequent agent
calls (or preserve other settings when configuring `[user] default=afm` in
`/etc/wsl.conf`). System administration remains separately authorised.
Imported distributions do not use `ubuntu.exe config --default-user`.
Store/bundle OOBE is a different route: if it asks for a password, the user
enters it privately, never in chat.

## 3. Install the environment and show the first result

Inside Ubuntu as the ordinary user, reuse working Linux Miniforge. Otherwise
use its [TUNA release mirror](https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/)
or [official distribution](https://github.com/conda-forge/miniforge#download).
Select a versioned **Linux** installer matching the architecture and its
matching `.sha256` file; verify before executing. The mirror's moving
`LatestRelease` directory may change: missing files/checksum mean stop and
reselect the published pair, not execute an unchecked download.

Example verified-release pair available on 2026-10-26, for a new x64 install.
Run as one Bash block; no root, shell-profile edits or existing-prefix overwrite:

```bash
(
  set -eu
  test "$(id -u)" -ne 0
  test "$(uname -m)" = x86_64
  test ! -e "$HOME/miniforge3"
  afm_download=$(mktemp -d)
  cd "$afm_download"
  afm_base=https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease
  afm_file=Miniforge3-26.9.4-0-Linux-x86_64.sh
  curl -fL --retry 2 --connect-timeout 20 -O "$afm_base/$afm_file"
  curl -fL --retry 2 --connect-timeout 20 -O "$afm_base/$afm_file.sha256"
  sha256sum --check "$afm_file.sha256"
  bash "$afm_file" -b -p "$HOME/miniforge3"
)
```

Then use [INSTALL.md's mirror command and installed smoke test](https://github.com/haoming-luo/agentfem/blob/main/INSTALL.md#mirror-channel--镜像通道).
Use `~/miniforge3/bin/mamba` if it is not on PATH. In agent tools use
`mamba run -n agentfem-env ...` for each command; activation does not persist
across tool calls. Query the selected mirror for a required version: an older
package passing a basic solve does not satisfy a project's newer API needs.

Finish with doctor → workspace protection → init/check/run/verify. Report
**installed version, first-solve outcome, result path and one useful numerical
check with units where applicable**. Provenance verification is not a general
scientific certification. Do not append unrequested benchmarks or a long essay.

## Agent handoffs: one sentence, only when needed

| Actual blocker | Tell the user |
| --- | --- |
| Host blocks WSL | “Please approve this WSL operation in your agent, or run the displayed command in your own terminal.” |
| Elevation unavailable | “Open Terminal (Administrator) and run the platform command above.” |
| Windows requests restart | “Save your work and restart; I will continue from the installed state.” |
| Store/bundle asks for credentials | “Enter your Linux account details in the local window, not here.” |

Do not ask users to disable security controls, retry blocked tools through
another shell, reinstall healthy WSL, or enable every Windows feature.
After interruption inspect what already exists; do not duplicate downloads,
imports or package transactions. To avoid PowerShell/Bash quoting errors,
send complex Linux operations as an inspected LF `.sh` file, rather than
nested interpolated command strings.

## Keep the chosen mirror on later visits

Record the user's selected channel, environment path and installed version in
the local project/agent handoff note; do not record credentials. Read it on
later visits and keep using that channel for optional packages unless the user
changes it. Use explicit per-command conda/pip sources; do not rewrite global
configuration or mix compiled package channels. No new mirror CLI is needed.
[Optional package examples](https://github.com/haoming-luo/agentfem/blob/main/INSTALL.md#optional-tools).

## Other routes and evidence

If the user already has a compatible published Complete Runtime ZIP, use its
installer instead; do not also install Ubuntu/Miniforge. Source/bundle helpers
and detailed recovery records are [optional tooling](windows_installer_reference.md),
not prerequisites. No repository clone is needed for any step above.

The [2026-10-26 field feedback](https://github.com/haoming-luo/agentfem/discussions/3)
reports Ubuntu mirror download around one minute, environment creation around
2.5 minutes, AgentFEM 0.4.1 and a completed static-solid solve. These are
user-reported measurements, not timing guarantees or a validation of the
current release. It used root; our recommended non-root preparation above is
a deliberate safety improvement, not a claim that the reporter tested it.

Sources: [Microsoft installation](https://learn.microsoft.com/en-us/windows/wsl/install),
[WSL commands](https://learn.microsoft.com/en-us/windows/wsl/basic-commands),
[imported distributions](https://learn.microsoft.com/en-us/windows/wsl/use-custom-distro).
