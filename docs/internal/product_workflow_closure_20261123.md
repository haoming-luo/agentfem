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

Local installed acceptance artifacts:
`/private/tmp/agentfem-product-acceptance-6b8qKr` (temporary, not distributable).
The repository tests are the reusable evidence; temporary paths are not runtime
dependencies. Existing numerical tolerances and material laws are unchanged.

## Remaining bounded work

Validate installed wheel and Windows viewing at an integration milestone;
add full browser automation coverage for animation, deformation and run
switching. Continue from existing thermal and element routes only where an
actual user workflow is missing. Clip/probe/part controls are subsequent viewer
slices, not prerequisites for this project/run/playback improvement.
