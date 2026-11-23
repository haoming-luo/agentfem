# Product workflow closure — 2026-11-23

P92 mechanism fitting is paused. This batch owns product workflow changes,
not another material-law research trial. Do not merge or release PR #197.

## Audit before implementation

- Generic finite-strain local history materials already use the framework's
  nonlinear solve, accepted/trial state, physical time, output and checkpoint
  lifecycle. The existing `history_material` example swaps viscoelastic and
  damage updates without a project solver. Do not implement another adapter.
- `thermal_cyclic_creep` already demonstrates one-way heat transfer to
  creep/plasticity on a common mesh. This is not a new two-way heat-feedback
  qualification or a calibrated material.
- Common first/second-order mesh routes and a bounded linear prism route
  already have import/patch/output tests. A stale elements README calling this
  a reserved namespace is corrected; no new element formula is claimed.
- The existing optional viewer already reads, warps, plays and refreshes saved
  frames. The missing user path was project → run → result selection.

## Implemented viewer slice

Read project metadata with `ProjectConfig`, honor configured output folders,
and browse immediate projects/runs without executing their cases. Retain the
existing scoped file reader and animation controller. Duplicate project names
show relative paths. A single result is preselected, not automatically opened.
The displayed source remains visible while browsing another project, avoiding
confusion between a selected run and the result actually on screen.

Real browser testing found the initial selection callback ran before the
selected value reached the server. State-change callbacks now own selection;
the controller regression exercises the same state changes. Play/pause and
open have familiar icons, retaining text labels and keyboard-accessible controls.

## Recorded acceptance

- 219 targeted material, temperature-history, creep/plasticity, mesh archive
  and element-contract tests passed in 135.37 s.
- 58 viewer, mesh-format and mesh-quality tests passed in 1.51 s, with the
  optional viewing dependencies installed; no viewer tests were skipped.
- A separate temporary environment installed this checkout in editable mode;
  this is installed-entrypoint testing, not a clean release-wheel certification.
  Standard `agentfem run` completed 16 increments for each of viscoelastic and
  damage materials, and for the thermal-creep example. Synthetic parameters
  remain synthetic; `computed` is not experimental verification.
- The installed viscoelastic case paused at 8 increments, resumed to 16, and
  its final U, F, both states, P, S, dP/dF and stored-energy arrays were exactly
  equal to its uninterrupted run on this machine.
- Codex in-app browser selected project/run/XDMF and played a real 16-frame
  material result from frame 1 to frame 15. A separate VTU navigation check
  confirmed actual geometry rendering. This is not Windows browser acceptance.
- After moving this product-only batch onto main `1e36a98b`, the combined
  regression passed **277 tests in 262.04 s**. A subsequent deterministic
  playback/pause regression brought the focused viewer suite to **15 passed
  in 1.13 s**; these counts overlap and must not be summed. Browser checking
  also exercised reference-geometry warp and KAPPA history coloring on the
  independently run damage example. No solver speedup is claimed.

Local installed acceptance artifacts:
`/private/tmp/agentfem-product-acceptance-6b8qKr` (temporary, not distributable).
The repository tests are the reusable evidence; temporary paths are not runtime
dependencies. Existing numerical tolerances and material laws are unchanged.

## Remaining bounded work

Validate installed wheel and Windows viewing at an integration milestone;
add broader browser automation coverage for animation, deformation and run
switching. Continue from existing thermal and element routes only where an
actual user workflow is missing. Clip/probe/part controls are subsequent viewer
slices, not prerequisites for this project/run/playback improvement.

## Viewer lifecycle and publication follow-up

The follow-up adds independent automatically allocated local ports, an explicit
close action and configurable no-client shutdown. Appearance presets are
Scientific, Publication and Presentation. Field values and cell association
are preserved. Exports run in an isolated cancellable process and publish only
completed PNG/MP4 plus a plain JSON view recipe. Browser cameras are captured
explicitly; saved views preflight source fields/deformation before replacing the
current scene. Image and video dependencies are part of the viewer extra.

Local acceptance includes actual transparent PNG and MP4 encoding, fixed color
bar pixels across frames, saved-view roundtrip without source mutation, worker
failure/cancellation cleanup, and installed download assets. Set
`AGENTFEM_VIEWER_RENDER_TESTS=1` to include offscreen rendering tests on a host
with a usable VTK rendering backend. Other hosts run the contract/controller
tests, not an assumed graphics pass.

The final targeted run passed 39 tests with actual offscreen rendering enabled;
Ruff, JavaScript syntax checking and generated documentation checking passed.
Two concurrent installed CLI processes selected different automatic ports and
both exited normally after their no-client timeouts. Counts here overlap earlier
viewer runs and must not be summed into a scientific verification count.

The real Codex embedded browser exercised presets, camera/view restoration,
image export/download, process and turntable exports, and explicit closure.
Testing found and corrected Vue 3 reference lookup and direct-link download
navigation; the small packaged download helper keeps the websocket document
in place. No-client shutdown was observed after 2.78 s for a 2 s test timeout.

A private existing 25-frame result (296,074 points / 255,041 cells) exported a
1920×1080 H.264 process video: 250 encoded frames, 24 fps, 10.42 s, 831,301 bytes.
These are 25 saved states held for 10 video frames each, not 250 computed states.
A 120-frame camera-only turntable exported at 1920×1080, 1,595,582 bytes. The
publication illustration is 2400×1600, cividis, actual displacement scale 1,
with declared mm/s labels. Private result files/media are not added to Git.

This is a local macOS rendering qualification, not Windows graphics-driver
certification or a new scientific validation of the underlying simulation.
The download helper and Python renderer remain optional, outside the numerical
solver path. No solver timing benefit is claimed.
