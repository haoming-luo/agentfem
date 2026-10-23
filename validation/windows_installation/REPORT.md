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

Final implementation checkpoint `96e2525e`: both Windows PowerShell contract
jobs passed, including source-only platform entry, preserved resume arguments,
and denied WSL probe handling:
https://github.com/haoming-luo/agentfem/actions/runs/39890900292
Documentation CI also passed:
https://github.com/haoming-luo/agentfem/actions/runs/39890900164
A local strict documentation build passed into a temporary directory; existing
`site/` output in the simulation checkout was not touched. The broader Test
workflow was still running when this evidence was recorded; no full-suite pass
is claimed.

Route audit follow-up `1069eb45` also passed both Windows shells and
documentation CI. The Ubuntu/mirror platform entry accepts existing WSL tooling
without the `.wsl` image's 2.4.4 upgrade floor; the full image route still enforces
that floor. Both behaviours are covered by the Windows contracts:
https://github.com/haoming-luo/agentfem/actions/runs/39891849926
https://github.com/haoming-luo/agentfem/actions/runs/39891849915
The final local rerun again passed 31 packaging/OOBE tests, one targeted
installation-documentation test, and strict documentation construction.

The remaining customer gate is still a clean Windows/WorkBuddy run with real
UAC, reboot and account setup, plus network-route availability. Source changes
and mocked Windows tests do not remove that gate.

## Host diagnosis and recent-change review (2026-10-23)

This follow-up keeps the Ubuntu/mirror route primary for the user's request.
It does not publish a new runtime bundle, change security permissions, replace
an existing environment or claim an installation-speed improvement.

Implemented on `feat/wsl-agent-onboarding`:

- Read-only `Test-AgentFEMWindowsHost.ps1`: bounded native command probes,
  UTF-8/UTF-16 output, localised distro names, separate tooling and distro
  versions, explicit permission/restart/component/unknown failure.
- Probe both output streams concurrently and terminate only the timed-out
  read-only probe child. Windows CI compiles an inert subprocess fixture to
  test encoding, dual-pipe drain and timeout, not merely mocked return values.
- Preserve relative-path identity across restart/tool working-directory
  changes. Detect inaccessible state separately from an active lock.
- Preserve the fourth WSL tooling version component; do not parse a kernel
  version as WSL. An unknown failed query does not authorise reinstallation.
- Check platform status independently. Only known Microsoft component codes
  use the existing consent-based preparation path. Preserve status warnings
  even on exit zero; a working VM remains unverified until actual launch.
- Skip verbose inventory for an empty successful quiet list, matching the
  Microsoft client implementation rather than interpreting its deliberate
  no-default-distro error as corruption.
- Avoid recursive ownership changes to existing user data and duplicate,
  ignored health checks during account-setup recovery.

Windows PowerShell 5.1 and 7 passed at `7003c974`:
https://github.com/haoming-luo/agentfem/actions/runs/39941540010
Documentation passed at that revision:
https://github.com/haoming-luo/agentfem/actions/runs/39941539957
Later empty-inventory/status-warning changes (`c16956e3`) require their own
head checks; the earlier passing run is not substituted for them.
Local packaging/OOBE tests: **32 passed**. Failures found and fixed during CI
include a three-vs-four-component version expectation and PowerShell's
`PSBoundParametersDictionary.Contains` overload; both now have regressions.

The scoped simulation review is on the separate existing PR #118, not this
installer branch. Commits `1a60d6a8` and `f12d6dee` address accepted-series close:
retry only final metadata publication without rewriting closed heavy data;
keep storage-close failures explicit; close HDF5 even if metadata is invalid.
Both output layouts are read by an independent meshio reader after injected
failures. Source-runtime tests: **86 output tests passed**, **56 increment/
linear-correction tests passed**, **27 retained-material tests passed**.
One MPI-admission test was skipped in this serial run; no MPI success is
claimed. Subprocess tests used an explicit source PYTHONPATH to avoid loading
the older installed package. This is not an installed-wheel or full-run test.

Remaining gates, in order:

1. Exact-head Windows contract and documentation checks.
2. Review/integrate #119 and its dependent #120 in order; do not merge #120
   into the wrong base or claim a public release from a draft branch.
3. A real clean Windows/WorkBuddy mirror-route trial: scoped consent, reboot,
   private credentials, normal-user environment and installed first solve.
   Network reachability, UAC and firmware cannot be certified by these mocks.
4. Only then promote the documented onboarding claim. The simulation review
   is risk-based (state reuse, increment control, output failure), not a claim
   that all recent source changes or contact capability gates were revalidated.

Primary references: Microsoft WSL command documentation, WslClient.cpp's
Status/ListDistributionsHelper implementations, WSL error definitions and
the WorkBuddy permission documentation linked in the user guide. No locale-
specific error prose or speculative alternate security channel is relied on.
