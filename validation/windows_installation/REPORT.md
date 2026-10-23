<!-- SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Windows first-install recovery development

2026-10-23. Branch: feat/windows-install-recovery. Base: a87cc121 (rebased onto remote main).

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
import reuse and identity mismatch. CI has now executed on both Windows shells.
It exposed a PowerShell 5.1 atomic state-file replacement incompatibility, which
was fixed by supplying a real backup filename. Subsequent runs reached the
first-solve test but exposed a harness issue: dynamically evaluated functions
lost their script directory. The harness now loads definitions from a real file.
Fresh acceptance reports, stale-report rejection, Chinese/spaced paths and
corrupt state preservation are also covered. Both Windows PowerShell 5.1 and
PowerShell 7 passed on commit adf8bae9:
https://github.com/haoming-luo/agentfem/actions/runs/39826091662
Documentation CI also passed. The broader numerical CI was still running when
this installer-specific evidence was recorded; no full-suite result is claimed.

## Required before customer distribution

Run the exact built ZIP on clean Windows 11 with WorkBuddy, including reboot,
UAC rejection, blacklist, paths with spaces/Chinese text, interrupted OOBE,
existing distributions, project protection and first cold-cache solve.
Follow docs/windows_agent_installation.md. Preserve logs/status/acceptance JSON.

This is source development, not a published installer. No clean Windows
VM/WorkBuddy installation is available here. Windows CI uses real PowerShell
but mocked WSL/elevation operations; it is not end-to-end customer acceptance.
Development is rebased onto remote main a87cc121. Draft PR:
https://github.com/haoming-luo/agentfem/pull/119
No release or merge is performed.

## Remaining product work

- Signed installer/bootstrap executable and native UI; current CMD wrapper is
  a visible, reopenable launcher, not a new trusted elevation mechanism.
- Optional consent-based login continuation (currently manual reopen after restart).
- Verify distribution mirror/download-resume behaviour against actual release assets.
- WorkBuddy's approved WSL/elevation channel remains outside AgentFEM's control.
- Crash-safe recovery of replacement upgrades is separate from first-install resume;
  existing explicit backup/rollback upgrade logic is retained.

## Agent onboarding follow-up (2026-10-23)

Isolated follow-up branch `feat/wsl-agent-onboarding`, based on c9bcdedc;
[PR #120](https://github.com/haoming-luo/agentfem/pull/120) targets #119's branch.
The original contributor branch and simulation checkout were not changed.

- Added `-PrepareWslOnly` for Ubuntu/mirror users. It reuses the platform
  preparation policy without a Complete Runtime image, verifier, or Python.
  It reports `platform_ready`, never an installed scientific environment.
- `-NonInteractive` hands off elevation and account credentials explicitly;
  no hidden password/UAC wait and no automatic replacement upgrade.
- `Get-AgentFEMInstallStatus.ps1` returns read-only structured progress without
  launching WSL. Saved invocation arguments preserve route/source after reboot;
  they are data, not executable shell expressions or new authorisation.
- A per-user lock excludes competing packages targeting the same distro.
  Saved request identity and embedded runtime-release SHA-256 must match before
  reuse. Same-version development builds are no longer interchangeable.
- OOBE is invoked explicitly rather than assuming `--exec true` starts an
  interactive account wizard. Signed HRESULT access-denied results do not
  trigger another download/elevation attempt.
- Public guidance separates Windows-platform downloads from conda mirrors,
  respects an explicit mirror request, uses non-activating environment commands,
  and provides an installed-template first solve without cloning GitHub examples.

Evidence: 31 packaging/OOBE Python tests, installation-documentation regression,
Bash syntax, Python lint and diff checks pass locally. Windows PowerShell 5.1
and 7 contract/real-entrypoint harnesses passed at 75e2f8fc:
https://github.com/haoming-luo/agentfem/actions/runs/39889656101
The first platform-only harness kept an expected nonzero installer exit code;
the harness now exits zero only after all assertions pass. WSL and elevation
remain mocked. Latest-head checks are attached to PR #120; this earlier run is
not evidence for later changes by itself.

The documented `init/check/run/verify` sequence was also exercised in a fresh
temporary project on the existing Mac `fenicsx-env` (installed 0.4.2.dev0),
with a fresh JIT cache. Project check passed, the solve completed, and provenance
verification passed. That verifies command usability, not a newly installed
Windows build or broad scientific validation. No installation-speed claim is
made, and no published runtime bundle has been rebuilt.

The remaining customer gate is still a clean Windows/WorkBuddy run with real
UAC, reboot and account setup, plus network-route availability. Source changes
and mocked Windows tests do not remove that gate.
