# Windows first-install recovery development

2026-10-23. Branch: feat/windows-install-recovery. Base: 36f4c1b0.

## Implemented

- Human launcher keeps its window and exit status; no improvised customer script.
- Structured per-installation state and transcript in LocalAppData. Exclusive lock
  prevents duplicate installs with the same identity. State survives reboot.
- Reopening the same installer resumes a recorded import after runtime identity
  verification; no automatic reboot/login task and no silent unregister.
- Declined/blocked elevation and reboot exit codes do not trigger source fallback.
- Complete package/image checksum checked before changing WSL.
- First-use account setup returns to installer instead of waiting in an open shell.
- Interrupted password setup resumes; established account credentials are preserved.
- First install explicitly protects projects, checks version/doctor and executes
  existing fresh-cache serial runtime acceptance as the ordinary Linux user.
- Updated WorkBuddy policy and route-selection documentation.

## Evidence

31 local pytest tests passed (runtime packaging and stubbed OOBE behaviour).
Bash syntax and Python lint/diff checks passed. These are not Windows acceptance.
Added Windows PowerShell 5.1/7 CI tests for state persistence, duplicate locks,
elevation refusal, restart classification, one-time source fallback, recorded
import reuse and identity mismatch. CI has NOT been executed in this session.

## Required before customer distribution

Run the exact built ZIP on clean Windows 11 with WorkBuddy, including reboot,
UAC rejection, blacklist, paths with spaces/Chinese text, interrupted OOBE,
existing distributions, project protection and first cold-cache solve.
Follow docs/windows_agent_installation.md. Preserve logs/status/acceptance JSON.

This is source development, not a published installer. No Windows VM/WorkBuddy
execution is available here. Attempts to fetch current main via GitHub SSH and
HTTPS failed/timed out; portable PowerShell retrieval also timed out. Development
therefore starts from the last known local main commit 36f4c1b0, not a claimed
current remote head. Reconcile latest main and run Windows CI before promotion.
No release or merge is performed.

## Remaining product work

- Signed installer/bootstrap executable and native UI; current CMD wrapper is
  a visible, reopenable launcher, not a new trusted elevation mechanism.
- Optional consent-based login continuation (currently manual reopen after restart).
- Verify distribution mirror/download-resume behaviour against actual release assets.
- WorkBuddy's approved WSL/elevation channel remains outside AgentFEM's control.
- Crash-safe recovery of replacement upgrades is separate from first-install resume;
  existing explicit backup/rollback upgrade logic is retained.
