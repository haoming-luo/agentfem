# 0.5.0 release candidate

Status: PR #199 merged; not released. Exit-lifecycle follow-up acceptance is
required before promotion. The tested package identity is still `0.4.5.dev0`.
Scope is a reliable installed workflow, not new P92 parameter identification.

## Exit-lifecycle release blocker (2026-11-25 follow-up)

PR #199 was merged after its CI passed. A subsequent local process audit found
ten validation processes still consuming CPU after printing their results.
Sampled stacks were inside PETSc/MPICH `MPI_Finalize`, including `doctor`.
An MPI-only import reproduced the hang with macOS MPICH 5.0.1/ch4:ofi; importing
AgentFEM alone exited normally. This supersedes any assumption that a printed
pytest summary proves complete local process acceptance.

Release 0.5.0 additionally requires:

- CLI doctor observes actual worker exit before publishing readiness; a bounded
  timeout cleans its process group and returns `AFM-RUNTIME-EXIT-TIMEOUT`.
- Fresh subprocess tests cover printed success followed by a hang, normal exit,
  nonzero exit and structured failure. Test launchers must also observe exit.
- The affected machine's environment repair passes serial and two-rank solves
  with unchanged observables and no remaining worker. It must not become an
  unconditional transport override on Linux, WSL or clusters.
- Final installed acceptance checks shutdown as well as numerical output.

Locally, `FI_PROVIDER=tcp` passed three MPI import/exit repeats, PETSc exit,
doctor, and serial/two-rank static-solid solves. Maximum displacement remained
0.000427424 m and strain energy 42.7303. Without it, the bounded doctor returned
124 after 30.08 s and cleaned its worker; with it, doctor returned 0 with observed
exit. The workaround is scoped to the affected `fenicsx-env` activation, not
package defaults. It is not a general transport-performance recommendation.

Follow-up local acceptance: 57 related CLI, platform, MPI and supervisor tests
passed against source (6.19 s) and an isolated installed wheel (6.84 s). The
installed console command, without source injection, returned 124/false on the
unrepaired environment and 0/true with observed exit on the repaired environment.
All ten identified stale validation processes were terminated and confirmed
absent; viewer and unrelated user processes were preserved. CI for this follow-up
must still pass before it can satisfy the 0.5.0 promotion gate.

| Task | Required evidence | Status |
| --- | --- | --- |
| Output contract | Installed scalar/vector/tensor/custom-state round trips retain association, component order, units and time; unknown order is not guessed | Installed output/viewer regression passed; unknown layouts remain explicit |
| Viewer workflow | Real project/run and standalone result; play/pause/scrub, recent paths, saved-view remove/undo; narrow window without blocked controls | Real composite playback and 500×800 window checked; fit now uses actual viewport aspect |
| Media | PNG and 1080p/4K video, cancellation/retry, complete XYZ and readable legend; fit framing contains model; current zoom deliberately preserves camera | Synthetic encoding/cancellation and representative composite media passed |
| Launch and lifetime | Maintained shortcut uses installed Python, survives terminal closure, actionable logs; multi-tab close, idle exit and relaunch verified | Installed macOS shortcut opened a real result after launcher exit; close/idle exit verified; Linux desktop interaction not locally tested |
| Installed engineering | Supported environment: check/run/inspect/verify; representative material history and restart remain equivalent within existing tolerances | Local installed beam and 127 history/recovery regressions passed; Linux/MPI and macOS installed-wheel CI passed |
| Release integration | Review changes since 0.4.4, reconcile capability/docs/version, latest impact-selected CI passes, no unreviewed branch imports | Candidate CI passed; version promotion, merge and tag remain separate steps |

## Frozen candidate evidence

Commit `d3239fe72ce52a980143c5c79acfaf585bfc6322` passed
[Test run 45595771421](https://github.com/haoming-luo/agentfem/actions/runs/45595771421).
The full unit/interface suite reported **7,381 passed, 465 skipped** in 2,398.28 s.
Linux installed workflows, MPI/restart checks, release examples, native-kernel
checks, macOS installed-wheel acceptance, lint, documentation, portable Linux
distribution and the optional PyTorch bridge passed. The targeted-only job was
skipped because the full gate ran; Pages deployment is not performed on a PR.
Skipped tests are not additional passed capabilities.

The final local installed output/viewer suite passed **171 tests in 56.77 s**.
Its rendering-enabled checks complement cloud CI; they are not Windows GUI
certification. PR #200 is already in the tested main baseline
`75a0750b87647432bf014a315a5ece6737275dc0`; PR #197 remains excluded.

## Promotion checklist

1. Merge the accepted PR only after authorization and a fresh HEAD/base check.
2. Prepare one version-promotion batch: package version, `__version__`, matching
   immutable release contract, changelog, citation and published-install metadata.
   Preserve capability maturity; do not promote experimental physics by renaming
   a package version.
3. Build the final versioned artifacts and run the existing distribution and
   installed-use release gates. Evidence for a development wheel does not certify
   a different artifact solely because its source differs only in metadata.
4. Publish an immutable tag only after approval and the required release gates.
   The publisher verifies its exact artifacts before upload; no local ad-hoc
   package or desktop shortcut is a substitute for the release workflow.

## Decision rules

Earlier local closeout: 170 installed output/viewer regressions passed in 49.11 s
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
tests pass. The frozen candidate above subsequently passed the complete gate,
including macOS; the failed revision's dependent macOS job did not run.

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

## Release communication and agent guidance

- The bundled AgentFEM skill routes interactive-result tasks to a short viewer
  reference. Installation, viewing and numerical execution stay separate tasks.
- The performance skill requires observed process exit and bounded validation;
  a written result is not evidence that MPI/PETSc shut down successfully.
- Release visuals must come from actual saved results. Captions distinguish
  computed states, averaged colors, displacement interpolation and camera motion.
  Demonstration models are not presented as experimental validation.
- Before tagging 0.5.0, merge the accepted exit-readiness fix, freeze the final
  commit, synchronize package/manifest/version references, and pass the final
  installed-distribution gate. Publish only after explicit release approval.
