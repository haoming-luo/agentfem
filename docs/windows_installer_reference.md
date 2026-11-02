# Windows installer reference

This is optional tooling documentation, not an installation prerequisite.
Start with the [short Windows onboarding](windows_agent_installation.md).
No repository checkout, AgentFEM PowerShell helper or offline runtime bundle
is required for the Ubuntu + mirror route.

## Optional source / bundle tools

Only when a trusted checkout or bundle is already available, use its read-only
host probe or `Install-AgentFEM.ps1 -PrepareWslOnly`. Keep the companion files
with the installer. `-WslBootstrapSource Store|Web|None` selects the platform
source; `Web` is not a domestic mirror. `-NonInteractive` returns a handoff
instead of opening an approval/password prompt. Platform preparation does not
install Ubuntu or AgentFEM. Do not fetch the repository just to run this helper.

The optional Linux helper `python3 packaging/runtime/setup_environment.py
--channel tuna` plans an environment; `--apply --json` creates a missing one or
verifies an existing one and stops after readiness checks. Add `--first-solve`
only when the user requested an example (or for explicit developer acceptance).
It requires trusted Linux
Miniforge, never silently upgrades an existing environment, and keeps stage
logs in `~/.local/state/agentfem/setup/`. This helper does not prepare Windows
or download Miniforge. Use direct installed commands when no checkout exists.

## Agent procedure

When using optional source/bundle tooling, the read-only preflight is:

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
6. For a bundle/Store account prompt, let the user enter credentials privately.
   For a newly imported mirror distribution, use the short guide's non-root
   account setup instead; do not force a nonexistent OOBE prompt.
7. For Ubuntu/mirror, finish with runtime identity, doctor and workspace
   protection. Report readiness and offer an optional example. The current
   Complete Runtime source installer uses readiness-only checks too. Older
   bundles may include a packaged solve; do not add another automatically.

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
