# 0.5.0 release candidate

Status: preparation, not released. PR #199 remains unmerged pending approval.
Scope is a reliable installed workflow, not new P92 parameter identification.

| Task | Required evidence | Status |
| --- | --- | --- |
| Output contract | Installed scalar/vector/tensor/custom-state round trips retain association, component order, units and time; unknown order is not guessed | Installed output/viewer regression passed; unknown layouts remain explicit |
| Viewer workflow | Real project/run and standalone result; play/pause/scrub, recent paths, saved-view remove/undo; narrow window without blocked controls | Real composite playback and 500×800 window checked; fit now uses actual viewport aspect |
| Media | PNG and 1080p/4K video, cancellation/retry, complete XYZ and readable legend; fit framing contains model; current zoom deliberately preserves camera | Synthetic encoding/cancellation and representative composite media passed |
| Launch and lifetime | Maintained shortcut uses installed Python, survives terminal closure, actionable logs; multi-tab close, idle exit and relaunch verified | Installed macOS shortcut opened a real result after launcher exit; close/idle exit verified; Linux desktop interaction not locally tested |
| Installed engineering | Supported environment: check/run/inspect/verify; representative material history and restart remain equivalent within existing tolerances | macOS installed beam and 127 history/recovery regressions passed; other platform CI pending |
| Release integration | Review changes since 0.4.4, reconcile capability/docs/version, latest impact-selected CI passes, no unreviewed branch imports | Pending |

## Decision rules

Latest local closeout: 170 installed output/viewer regressions passed in 49.11 s
with the optional viewer dependencies installed and source-path injection
disabled. The real 25-frame composite (296,074 points, 255,041 cells) produced a
4K damage PNG, a 4K displacement-motion video and a 1080p orbit video. These are
short visual acceptance exports, not a new simulation performance claim.
Actual in-app playback showed intermediate deformations, not just the final
frame. A narrow-window check exposed and fixed aspect-dependent fit cropping.

Desktop acceptance exposed a process-group lifetime defect missed by startup
checks. The installed macOS shortcut now launches a detached service and waits
for readiness, retaining a diagnostic log. Eight shortcut tests include a real
process-group termination regression. Chrome opened the installed beam result
after the launcher exited; after closing the tab, the service released its
listening port on idle timeout.

Cloud run 45586435975 on f5a8360b completed with 7,376 passed, 463 skipped and
one attribution failure: the new shortcut module lacked its SPDX header. That
header and the new launcher header are now present; all three local attribution
tests pass. The final follow-up revision must pass CI before this gate closes.
The dependent macOS job did not run on that failed revision.

Resolution/lifecycle closeout: added real 1920×1080 and 3840×2160 encoded
smooth-video regressions, checking frame dimensions, changing geometry, legend,
orientation annotations and unchanged source hashes. Both passed; inspected the
4K decoded final frame. This synthetic two-frame test does not certify every
camera/mesh combination or sustained playback performance.

In the actual in-app browser, two tabs connected to one test session. Closing
the first left the second able to play (frame 1/2, Pause, then frame 2/2, Play).
Session stop showed confirmation and a closed-state message; the listening port
was released. The earlier no-client test exited normally on its idle timeout.
This does not yet test every browser or desktop shortcut failure scenario.

The isolated installed wheel also passed 127 material-history/output-cadence,
checkpoint-policy, temperature-history and plastic-history regressions in
57.77 seconds, with source-path injection disabled. The launcher/picker suite
passed 12 tests. That checkpoint preceded the cloud result recorded above.

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
This is macOS installed-use evidence, not a clean dependency installation or
Windows desktop acceptance. Later recovery and media evidence is recorded above.
The wheel keeps the development version until the release gates are complete.

2026-11-25 implementation batch: added opt-in installed-interpreter desktop
entries (macOS/native Linux). Ten shortcut/chooser tests passed. Latest published
PR #199 CI exposed a real meshio incompatibility with redundant XDMF Attribute
Information children; removed those children while retaining standard layout
labels. Local output/inspection/viewer regression: 140 passed, one offscreen
 render skipped. Lint passed. This earlier checkpoint is superseded by the
installed and desktop acceptance above; release CI is still a separate gate.

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
