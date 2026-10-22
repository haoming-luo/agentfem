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

## Agent procedure

1. Inspect Windows architecture, WSL availability and existing distributions with
   the host's supported tools. In WorkBuddy use its PowerShell tool directly.
2. Download and extract the official compatible bundle; verify its digest.
   Prepare the installer before requesting any user action.
3. Invoke Install-AgentFEM.ps1 in the allowed PowerShell tool. The installer
   requests administrator approval only for WSL preparation/update. Never ask
   for manual commands merely because the current shell is non-admin.
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
requested distribution, locations and upgrade mode. Identical installations
hold an exclusive lock. Never launch a second copy while one is progressing.

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
- healthy completion with non-root user, Windows project protection and solve.

PowerShell contract CI mocks process exits and tests state handling on Windows
5.1 and 7. It does not validate UAC, WorkBuddy, firmware or real WSL installation.
