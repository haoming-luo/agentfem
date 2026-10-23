# Windows installation with an AI assistant

The goal is a working first finite-element solve, with the user participating
only in required platform consent, restart and account credentials. Never claim
an unattended installation when the host disallows WSL/elevation.

## Route selection

Read INSTALL.md and the published release assets first. New x64 Windows users
may use a compatible published Complete Runtime Preview. Check its version,
SHA-256 and published acceptance status; a development branch is not a release.
This route imports the bundled environment, without installing Ubuntu/Miniforge
and resolving the numerical stack again. If the user requests the documented
mirror route, follow that route without mixing it with Complete Runtime.
Reuse working existing environments and preserve their projects.

### Mainland network: keep the platform and package routes separate

An AgentFEM/conda mirror does **not** mirror the Windows WSL platform. With an
existing healthy Ubuntu WSL2 distribution, go directly to the single-mirror
command in INSTALL.md. Do not install a second distro or replace a working
Miniforge installation. Use `mamba run -n agentfem-env ...` in agent tools;
an interactive `mamba activate` in one tool invocation may not persist in the
next invocation. Query the mirror for the requested version before installing.

For a new machine, choose one route and retain it after reboot:

| Route | Windows preparation | Linux environment |
| --- | --- | --- |
| Published Complete Runtime bundle | Bundled installer prepares WSL **without** a separate Ubuntu download | Imports the verified local `.wsl` image; no conda solve |
| Ubuntu + documented mirror | Microsoft `wsl --install -d Ubuntu`; user approves platform changes and required restart | Reuse/install Miniforge from its verified distribution, then use the one-mirror command in INSTALL.md |

For the Ubuntu/mirror route, an agent may prepare just the WSL platform through
the same tested elevation/restart policy (no runtime image or Python needed):

```powershell
powershell.exe -NoProfile -File .\packaging\runtime\wsl\Install-AgentFEM.ps1 -PrepareWslOnly -WslBootstrapSource Store
```

Use that path from a verified source checkout; in an extracted bundle use
`.\Install-AgentFEM.ps1` instead. Keep `Installer-State.ps1` beside the installer.
Use `-NonInteractive` if the host has no interactive console. This operation
does **not** install Ubuntu, AgentFEM or change existing distro contents. Its
successful state is `platform_ready`, not `completed`. After any required reboot,
rerun it with the same options. The agent then inspects `wsl --list --verbose`,
reuses an appropriate WSL2 distro, or installs the named Ubuntu only if absent.
Opening Ubuntu for its initial username/password is a one-time user step;
afterwards the agent continues with the documented mirror commands. Do not
convert/remove an existing WSL1 or unrelated distro automatically.

The warning against bare `wsl --install` applies to the Complete Runtime route;
it is **not** a claim that Microsoft's default Ubuntu is unrelated to the
Ubuntu/mirror route. Do not invent a candidate package or a mirror download URL
if the required artifact has not been published. Obtain a published compatible
artifact or use the documented source-environment route.

On restricted networks select the accessible Microsoft source explicitly:
`-WslBootstrapSource Store`, `Web`, or `None` (platform already prepared).
`Web` is not a domestic mirror and may still be unreachable. For an offline
platform bootstrap use Microsoft's documented signed MSI route, not a random
repacked executable. No VPN, global proxy changes, disabled TLS verification,
or unrestricted agent permissions are prerequisites imposed by AgentFEM.

## Agent procedure

1. Inspect Windows architecture, WSL availability and existing distributions with
   the host's supported tools. In WorkBuddy use its PowerShell tool directly.
2. Download and extract the official compatible bundle; verify its digest.
   Prepare the installer before requesting any user action.
3. Invoke Install-AgentFEM.ps1 in the allowed PowerShell tool. The installer
   requests administrator approval only for WSL preparation/update. Never ask
   for manual commands merely because the current shell is non-admin.
   For a tool without an interactive console, add `-NonInteractive`. It performs
   permitted setup but returns a named handoff instead of opening UAC or waiting
   for a password. Use the normal installer in a local terminal for that handoff;
   the same identity/state is reused. This mode rejects replacement upgrades.
4. If WorkBuddy blocks wsl.exe, request its authorised allowlist/approval. If
   elevation is explicitly blocked, explain that exact blocker and ask the user
   to open the extracted Install-AgentFEM.cmd. Do not change shells or APIs to
   bypass policy. The CMD launcher is for human use, not a sandbox escape.
5. If restart is required, preserve the package and installation record; ask the
   user to save their work and restart. On session recovery read status.json,
   then rerun the same installer/options. A completed import is reused only
   when the recorded distribution and runtime identity match.
6. Complete the ordinary account prompt with the user entering their password.
   Never request passwords in chat or recommend routine root use.
7. Require project protection, runtime identity/doctor and the installed
   fresh-cache solve check. Report the actual distribution/version and results.

## State and failure handling

Records are under LocalAppData/AgentFEM/Installations/<identity>/ with status.json,
install.log and acceptance.json. The identity includes image checksum, version,
requested distribution, locations and upgrade mode. An exclusive per-user lock
also excludes **different** packages that could target the same distribution.
Never launch a second copy while one is progressing. Before resuming, the saved
request identity must match and the installed runtime-release record must match
the package's SHA-256. The same version string alone is not sufficient.

Agents can read progress without launching WSL or modifying Windows:

```powershell
powershell.exe -NoProfile -File .\Get-AgentFEMInstallStatus.ps1
```

This prints JSON with `installer_busy`, recorded stages, `error_code`,
`next_action`, and evidence paths. It does not prove the current environment is
healthy (`runtime_probed: false`) and does not authorise installation. A stale
`preparing_wsl` record with no lock is interrupted, not necessarily still running.
Inspect the log and resume the same verified installer/options; do not delete
the record or distribution to make the status look clean. Keep corrupt records
for diagnosis; the query exits nonzero if any record cannot be read.

| Recorded state / code | Agent action |
| --- | --- |
| Busy lock | Wait for the existing installer; do not duplicate it |
| `AFM-WIN-ELEVATION-REQUIRED` / `awaiting_authorization` | One specific approval/local-terminal request; no alternate-shell retries |
| `restart_required` | Ask the user to save work and restart; keep installer and options |
| `AFM-WIN-ACCOUNT-INPUT-REQUIRED` | User enters credentials privately; never in chat |
| `AFM-WIN-RESUME-IDENTITY` / `AFM-WIN-STATE-IDENTITY` | Preserve both installations and evidence; do not auto-repair/reuse |
| `completed` | Read the acceptance report; use ordinary non-root account |

Account setup invokes the bundled idempotent OOBE explicitly and returns.
It does not rely on `wsl --exec true` triggering an interactive first launch.

A restart request (exit 3010), declined UAC, and blocked process launch must not
cause a Store/Web fallback. Only a WSL command that actually ran and returned
an ordinary error may use the documented one-time fallback. Identify whether
slowness occurs during download, WSL setup, checksum, import or first JIT solve;
do not reset environments or invent alternative package sources.

The launcher keeps failures visible. Automatic login continuation is not yet
implemented: reopen it after reboot. Existing transactional replacement upgrades
remain explicit and retain their backup/rollback procedure; the resumable
first-install workflow is not a crash-safe replacement-upgrade transaction.

## Windows acceptance required before promotion

Test the exact published bundle on clean Windows 11 x64 with WorkBuddy:
- WSL missing, standard user session, approved UAC;
- blocked wsl.exe/elevation: one actionable request, no bypass or retry loop;
- cancelled UAC and restart: correct state, reopen and continue;
- interrupted first account setup/import: preserve distribution and recheck;
- existing unrelated distributions and existing AgentFEM: preserve them;
- paths with spaces/Chinese characters; relocated extraction folder;
- corrupt/incomplete package, failed first JIT solve, duplicate installer;
- same-version/different-build runtime; foreign saved state; two different
  packages racing; non-interactive password/approval handoff;
- healthy completion with non-root user, Windows project protection and solve.

PowerShell contract CI mocks process exits and tests state handling on Windows
5.1 and 7. It does not validate UAC, WorkBuddy, firmware or real WSL installation.

## Platform sources

- [Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install)
  and [command options](https://learn.microsoft.com/en-us/windows/wsl/basic-commands).
- [Custom distro/OOBE contract](https://learn.microsoft.com/en-us/windows/wsl/build-custom-distro).
- [WorkBuddy permissions](https://www.codebuddy.cn/docs/workbuddy/From-Beginner-to-Expert-Guide/Function-Description/Permission-Modes):
  request scoped approval; do not require disabling its sandbox or switching on
  global full access. Exact UI labels may change; follow the installed client's
  permission request, not an invented blacklist menu.

These contracts are model-neutral: Doubao or another model using an agent host
needs actual authorised terminal tools. A chat-only model cannot install WSL.

## Suggested initial user prompt

> Bring AgentFEM to life using https://github.com/haoming-luo/agentfem.
> Read INSTALL.md and AGENT_GUIDE.md; on Windows also follow
> docs/windows_agent_installation.md. Reuse a healthy existing environment;
> otherwise choose the documented compatible installation route. If I request
> the mirror channel, use its verified route. Complete every permitted step
> yourself. Ask me only for required consent, restart or account credentials,
> explaining the exact action. Respect the host's security controls; do not
> retry blocked actions through another shell. Preserve installation progress
> and continue after interruption. Verify the installed version, doctor and
> first solve before reporting success. Reply briefly with the environment,
> AgentFEM version and actual verification result.
