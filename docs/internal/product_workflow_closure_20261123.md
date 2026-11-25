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

### Visual refinement and bounded next steps

User review rejected the yellow/dark default and moving inline annotations.
The default light styles now use Ocean (ice/teal/blue), with Iris
(lavender/indigo) as an alternative. Both use monotonically decreasing relative
luminance; this is tested, but is not a claim of perceptual uniformity or a
Nature-endorsed palette. Existing named palettes/saved views remain compatible.
Time and deformation labels have independent fixed anchors. Association is
retained in the field selector and saved recipe, not printed as "Nodal values"
on every figure. Softer diffuse lighting reduces misleading specular highlights.

Design references: Apple's current [color guidance](https://developer.apple.com/design/human-interface-guidelines/color)
and [content-first interface description](https://www.apple.com/newsroom/2025/06/apple-introduces-a-delightful-and-elegant-new-software-design/),
[Nature's figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/),
and [Loro Piana's material presentation](https://www.loropiana.com/interiors/collection-2025/matter/index_EN.php).
These motivate hierarchy, restraint and legibility, not copying branding or
substituting decorative gradients for scientific scalar maps. Figure labels,
units and ranges remain legible; a pretty image is not scientific acceptance.

Temporal interpolation is a separate, not-yet-implemented display option.
[ParaView's tutorial](https://docs.paraview.org/en/v5.13.0/Tutorials/SelfDirectedTutorial/basicUsage.html)
explicitly distinguishes saved frames from optional temporal interpolation and
warns about artifacts. Proposed acceptance: unchanged connectivity and stable
point/cell identity, finite strictly increasing saved times, endpoint-exact
linear interpolation, original extrema/color limits preserved, and a recipe
recording the two source frames and interpolation weight. Sample uniformly in
physical time, not just frame index. No optical-flow/AI interpolation of rendered
pixels: this can invent crack motion and blur legends. Start with explicitly
continuous displacement/temperature fields; do not interpolate category IDs,
contact status, deletion flags or fracture events. Unknown semantics stay in
saved-frame playback. The smooth view must be labeled as interpolated, cannot
be queried as a computed state, and must not overwrite source H5. Only two source
states should be cached. Camera turntable animation is already continuous and
does not require interpolating the solution. Current process exports still
hold saved states; the new sample does not claim otherwise.

Material-linked rendering is feasibility only, with no new implementation:

- Keep the ordinary scalar-field mode as default. Add an optional appearance
  switch between scalar field, component identity and material appearance,
  with distinct scalar versus categorical legends.
- Use stable exported cell/component IDs plus a declarative material-name
  mapping, not geometric proximity, node merging, guessing from modulus, or
  executing case.py. Existing Abaqus lowering carries material assignments,
  but the current viewer does not consume a general component/appearance map.
  Verify that identity survives output conversion, continuation and all frames.
- Generic neutral colors require no extra metadata. Optional explicit material
  families can suggest a visual preset; names alone do not establish surface
  finish. Roughness, metallic response and opacity are visual overrides separate
  from constitutive properties. Composite weave requires declared orientation
  and a deliberately illustrative texture, not a fabricated microstructure.
- Reuse [VTK/PyVista PBR](https://docs.pyvista.org/examples/02-plot/pbr.html),
  with a simpler fallback where the browser renderer differs. Cache surfaces
  per component and reuse data, not an actor per cell. Verify picking, legend,
  units, original values and deformation are identical when appearance changes.
- Do not silently mix metal highlights/material color with quantitative scalar
  coloring. Missing labels fall back to plain geometry; no data migration is
  required for legacy files. Source results stay unchanged.

### Motion and automatic deformation sizing — subsequent implementation

The bounded displacement-only part of the preceding interpolation proposal is
now implemented in the live viewer and video renderer. A shared small motion
module owns sizing, nonuniform-coordinate sampling and reference-mesh checks;
it does not add solver concepts. Damage/other fields are not enabled. Source
coordinates/connectivity must match exactly, and interpolation weights and source
pairs are written in the video recipe. No unrecorded mechanics is inferred.
Intermediate still-view saving is rejected explicitly; selecting an original
frame restores that workflow. Full continuous-field interpolation and semantic
restart-gap detection are not claimed.

Auto scale explicitly scans published frames, uses representative displacement
with a maximum-excursion guard, and holds the factor during animation. Raw
displacements remain unchanged. Pure translations, zero displacement and already
large motions keep scale 1. The algorithm and limitations are documented in the
viewer guide. Export annotations are stacked at fixed anchors; UI status is a
compact block with frame identity, coordinate/time and scale. Mesh/path details
are disclosed separately. Physical units are not inferred from XDMF timestamps.

The local real-render suite passed 45 tests, including readable-scale rounding.
Browser acceptance exercised the 25-state full composite
result: Auto scale selected 1, interpolation advanced with explicit source-frame
labels, and playback reached the last actual state. The same source rendered a
six-second smooth video using 144 display samples with a complete interpolation
recipe. This is post-processing qualification, not new simulation evidence or
a browser frame-rate guarantee. Quantile sizing is only performed on request,
not repeated for interpolated display frames.

UX references: Apple's [motion guidance](https://developer.apple.com/design/human-interface-guidelines/motion)
supports purposeful, optional animation; the [ParaView tutorial](https://docs.paraview.org/en/v5.13.0/Tutorials/SelfDirectedTutorial/basicUsage.html)
documents explicit temporal interpolation and its topology/artifact limits.
No decorative transitions were added to numerical state changes.
