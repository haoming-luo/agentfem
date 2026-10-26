# Windows installation with an AI assistant

The goal is a working first finite-element solve, with the user participating
only in required platform consent, restart and account credentials. Never claim
an unattended installation when the host disallows WSL/elevation.

## The short onboarding path

1. **Inspect, do not reinstall.** Reuse the source-usable Windows host check
   below. Add `-Format Text` for a person; keep the default JSON for an agent.
2. **Choose one route.** A published Complete Runtime supplies its own Linux
   image. Ubuntu + mirror uses the existing Ubuntu/Miniforge environment.
   Do not make the user download both or search for an unpublished package.
3. **Do the permitted work.** Ask the user only for the specific approval,
   restart or private account input that is actually needed. A blacklisted
   command is not a missing WSL installation.
4. **Verify the first solve.** For Ubuntu + mirror, from the trusted checkout
   inside Ubuntu, run `python3 packaging/runtime/setup_environment.py --channel tuna`,
   inspect the plan, then repeat with `--apply --json`. The entry point
   discovers Linux Mamba/Conda, or accepts `--manager` and `--prefix` explicitly.
   It creates a missing environment through one channel, but never changes an
   existing environment's packages implicitly. It verifies installed identity,
   doctor, workspace protection, an installed-template solve and result provenance.
5. **Hand off one useful summary.** State the distribution/environment, installed
   version, first-solve outcome, project location and any remaining action.
   Do not paste a page of platform commands or claim optional accelerators,
   Windows firmware, another material model or a full scientific benchmark were
   validated by this small solve.

`setup_environment.py` is a Linux source-usable installer entry, not a new
solver or an executable downloaded from a package index. It requires a trusted
existing Miniforge and Python 3; it never bootstraps an unsigned/unverified
installer or changes the user's shell startup files. With `--apply`, package
download/solve logs and structured stage evidence stay in
`~/.local/state/agentfem/setup/`. A requested version unavailable on the selected
mirror fails before environment creation. No alternate channel is selected.
The normal `--apply` route protects the conventional workspace; advanced
`--no-protect-workspace` inspects only and still rejects unprotected WSL storage.
Its first-solve check uses Linux-local files and a fresh JIT cache. Runtime and
case-specific performance must still be measured separately.

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
After a reviewed environment-create plan, agents may add `--yes` to avoid a
hidden package confirmation prompt. If Mamba is not on PATH after a batch
Miniforge install, use its actual absolute executable path; do not rewrite the
user's shell startup files just to make a transient agent tool work.
Finish with INSTALL.md's installed `init/check/run/verify` smoke test, which
does not require cloning the repository from GitHub. Preserve its small project
and result as evidence instead of accepting imports alone.

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
`.\Install-AgentFEM.ps1` instead. Keep `Installer-State.ps1`,
`Installer-Presentation.ps1` and `Windows-HostProbe.ps1` beside the installer.
Use `-NonInteractive` if the host has no interactive console. This operation
does **not** install Ubuntu, AgentFEM or change existing distro contents. Its
successful state is `platform_ready`, not `completed`. This checks WSL tooling
and the read-only platform status;
it does not certify a working WSL2 VM or installed scientific environment. The
2.4.4 minimum for `.wsl` packages is not imposed on this Ubuntu/mirror route.
After any required reboot,
rerun it with the same options. The agent then inspects `wsl --list --verbose`,
reuses an appropriate WSL2 distro, or installs the named Ubuntu only if absent.
Opening Ubuntu for its initial username/password is a one-time user step;
afterwards the agent continues with the documented mirror commands. Do not
convert/remove an existing WSL1 or unrelated distro automatically.

The installer's read-only version/status/inventory queries use the same bounded probe as
the preflight (15 seconds per query; `-ProbeTimeoutSeconds 1..120` when needed).
A timeout records `AFM-WIN-PROBE-TIMEOUT`, releases the installer lock and stops;
it does not terminate an installation, distribution, service or simulation.
Downloads, imports and first solves are not given this short probe timeout.

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

Before installing, the source-usable read-only preflight is:

```powershell
powershell.exe -NoProfile -File .\packaging\runtime\wsl\Test-AgentFEMWindowsHost.ps1 -Route UbuntuMirror
```

In an extracted bundle use `Test-AgentFEMWindowsHost.ps1` and keep its companion
`Windows-HostProbe.ps1` beside it. Select `CompleteRuntime` only for that route.
The JSON distinguishes tooling version from each distro's WSL1/WSL2 version,
blocked/missing commands, failed probes, and required image-format updates.
It also runs `wsl --status`: a successful version query alone does not show
that the Windows components are ready. Known missing-component HRESULTs or
stable WSL error names report `component_required`; generic failures remain
unknown rather than triggering reinstallation. Platform preparation uses its
existing consent/restart path for those known component failures, then checks
status again. It does not alter firmware, convert distributions or reboot.
The raw `platform_status_text` is retained because WSL may print a virtualisation
warning while returning success. `virtual_machine_verified` remains false:
neither status nor inventory substitutes for launching the selected WSL2 distro.
An empty successful quiet inventory skips the verbose query (which WSL rejects
when there is no default distribution); it is not treated as a broken install.
It never launches a distribution or changes Windows. Each read-only command
has a 15-second timeout (configurable with `-TimeoutSeconds`); a failed command
stops the probe sequence, not triggers another launcher. It reports
`runtime_verified: false` even when a WSL2 inventory is available. Inventory is
not evidence that Linux, a mirror or AgentFEM is healthy. Preserve a failed
probe and inspect its code instead of inferring that WSL needs reinstallation.

1. Inspect Windows architecture, WSL availability and existing distributions with
   the host's supported tools. In WorkBuddy use its PowerShell tool directly.
2. Retain the selected route. For Complete Runtime, obtain the official
   compatible bundle and verify its digest. For Ubuntu/mirror, reuse the
   existing environment or prepare WSL as described above; no runtime bundle
   is required. Prepare the relevant installer before requesting user action.
3. When using Install-AgentFEM.ps1, invoke it in the allowed PowerShell tool
   with the selected route options (`-PrepareWslOnly` for platform-only work).
   The installer
   requests administrator approval only for WSL preparation/update. Never ask
   for manual commands merely because the current shell is non-admin.
   For a tool without an interactive console, add `-NonInteractive`. It performs
   permitted setup but returns a named handoff instead of opening UAC or waiting
   for a password. Use the normal installer in a local terminal for that handoff;
   the same identity/state is reused. This mode rejects replacement upgrades.
4. If WorkBuddy blocks wsl.exe, request its authorised allowlist/approval. If
   elevation is explicitly blocked, explain that exact blocker and ask the user
   to run the same trusted installer/options in their local terminal. The
   extracted Install-AgentFEM.cmd is for the Complete Runtime route only.
   Do not change shells or APIs to
   bypass policy. The CMD launcher is for human use, not a sandbox escape.
5. If restart is required, preserve the package and installation record; ask the
   user to save their work and restart. On session recovery read status.json,
   then rerun the same installer/options. A completed import is reused only
   when the recorded distribution and runtime identity match.
6. Complete the ordinary account prompt with the user entering their password.
   Never request passwords in chat or recommend routine root use.
7. Require project protection, runtime identity/doctor and the installed
   first-solve check. The bundle runs its fresh-cache acceptance check;
   Ubuntu/mirror users use INSTALL.md's installed-template check. Report the
   actual distribution/version and results, not just platform readiness.

## State and failure handling

Records are under LocalAppData/AgentFEM/Installations/<identity>/ with status.json,
install.log and acceptance.json. The identity includes image checksum, version,
requested distribution, locations and upgrade mode. An exclusive per-user lock
also excludes **different** packages that could target the same distribution.
Never launch a second copy while one is progressing. Before resuming, the saved
request identity must match and the installed runtime-release record must match
the package's SHA-256. The same version string alone is not sufficient.
For full-runtime installs the resolved project directory is also saved before
platform preparation can request a restart. Resume reuses that destination even
if Windows redirects Documents in the meantime. Status exposes it as
`effective_project_directory`; explicit destinations must agree with it. Older
records without that field resolve and save the destination on their next run;
they cannot reconstruct a former default that was never recorded.

Agents can read progress without launching WSL or modifying Windows:

```powershell
powershell.exe -NoProfile -File .\Get-AgentFEMInstallStatus.ps1
```

For a user-readable view, add `-Format Text`. Both formats read the same
snapshot and preserve the same nonzero error exit. They perform no installation
or health-check operation. The human view names the next step and local record,
instead of asking the user to interpret the JSON.

This prints JSON with `installer_busy`, recorded stages, `error_code`,
`next_action`, the saved `resume` argument array, and evidence paths. Read the
arguments as data: review the route, target and privileges before invoking a
trusted installer; never feed them to `Invoke-Expression`. Keep the recorded
`-PrepareWslOnly`/source options after reboot. For the private user handoff,
omit only `-NonInteractive`; if the package moved, verify its identity before
substituting the new installer path. The query does not prove the environment is
healthy (`runtime_probed: false`) and does not authorise installation. A stale
`preparing_wsl` record with no lock is interrupted, not necessarily still running.
The status query reports `activity: interrupted` and an inspect/resume action
for an unfinished stage with no busy lock or recorded handoff. It preserves
`recorded_next_action` and never rewrites the saved evidence. When the shared
lock is busy, it does not guess which saved record owns that installer.
Status readers allow atomic replacement of the state file while they hold a
complete prior snapshot, so polling progress does not lock out state publication.
Inspect the log and resume the same verified installer/options; do not delete
the record or distribution to make the status look clean. Keep corrupt records
for diagnosis; the query exits nonzero if any record cannot be read.
If a full disk or permission error prevents writing a failure record, the
installer still reports the original error and a nonzero exit, plus
`AFM-WIN-STATE-WRITE`. Keep the terminal/transcript: the prior JSON record may
then be stale and must not be taken as current success.

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
- [Microsoft WSL error definitions](https://github.com/microsoft/WSL/blob/master/src/windows/service/inc/wslservice.idl):
  optional-component and virtual-machine-platform requirements are distinct
  from version discovery, denied access and unknown probe failure.
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
