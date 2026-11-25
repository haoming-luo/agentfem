# 0.5.0 release candidate

Status: preparation, not released. PR #199 remains unmerged pending approval.
Scope is a reliable installed workflow, not new P92 parameter identification.

| Task | Required evidence | Status |
| --- | --- | --- |
| Output contract | Installed scalar/vector/tensor/custom-state round trips retain association, component order, units and time; unknown order is not guessed | Partial: component tests pass; installed matrix pending |
| Viewer workflow | Real project/run and standalone result; play/pause/scrub, recent paths, saved-view remove/undo; narrow window without blocked controls | Partial: targeted UI evidence; final frozen pass pending |
| Media | PNG and 1080p/4K video, cancellation/retry, complete XYZ and readable legend; fit framing contains model; current zoom deliberately preserves camera | Pending final batch |
| Launch and lifetime | Maintained shortcut uses installed Python, survives terminal closure, actionable logs; multi-tab close, idle exit and relaunch verified | In progress |
| Installed engineering | Supported environment: check/run/inspect/verify; representative material history and restart remain equivalent within existing tolerances | Pending frozen install |
| Release integration | Review changes since 0.4.4, reconcile capability/docs/version, latest impact-selected CI passes, no unreviewed branch imports | Pending |

## Decision rules

Follow-up batch: a real subprocess regression confirms a newly selected viewer
survives shutdown of the old event loop. Its output no longer depends on the old
session's pipe; failed startup remains reported. Native offscreen PNG, process
video, smooth displacement video and actual export cancellation passed (35 tests
in that batch; seven picker tests after adding the lifecycle regression).

An isolated wheel installation, sharing the existing FEniCSx dependencies but
not the source checkout, completed `init`, `doctor`, `check`, `run` and `verify`
for the standard static-solid beam. Maximum displacement: 0.000427424 m;
strain energy: 42.7303; relative force balance error: 1.90512e-12;
energy balance error: 2.55156e-11. The result seal verified without issues.
This is macOS installed-use evidence, not a clean dependency installation,
Windows acceptance, full recovery regression or final 4K/browser acceptance.
The wheel keeps the development version until the release gates are complete.

2026-11-25 implementation batch: added opt-in installed-interpreter desktop
entries (macOS/native Linux). Ten shortcut/chooser tests passed. Latest published
PR #199 CI exposed a real meshio incompatibility with redundant XDMF Attribute
Information children; removed those children while retaining standard layout
labels. Local output/inspection/viewer regression: 140 passed, one offscreen
render skipped. Lint passed. Desktop lifetime and frozen installation remain
unverified; this is not final release acceptance.

- Unit tests do not replace actual installed execution or browser checks.
- Any broken default opening, playback, cancellation, or corrupted result blocks
  release. Unsupported scientific combinations remain explicitly bounded.
- P92 cross-protocol research (#197) is excluded; no engineering lifetime
  certification or new whole-simulation speedup is claimed.
- Keep source results unchanged; visual interpolation and averaging are display
  operations only. Retain raw-cell viewing.
- Batch a coherent tested change before pushing. Freeze one candidate for final
  regression rather than repeatedly restarting full CI during experiments.
- Merge and publication require authorization; a candidate is not a release.
