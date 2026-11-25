# Local result viewer (experimental)

View saved results in a local browser without executing a case. This optional
installed viewer uses PyVista/VTK and Trame; ParaView remains available for
full post-processing. The solver and its output formats are unchanged.

## Open a result

In your viewing environment, install the optional dependencies once:

```sh
python -m pip install 'agentfem[viewer]'
```

Open a result file or a directory containing your results:

```sh
agentfem-view /path/to/project/outputs
```

The command opens your browser and automatically chooses a free local port.
There is no backend setup or port selection to perform. Keep the command running
while viewing; Ctrl+C closes that viewer. `--no-browser` prints the local link
without opening a browser, useful for an agent's embedded browser.

Each invocation is an independent viewer process, with its own selected result,
field and playback state. Launch it again for an independent comparison or
another agent conversation; both can read the same saved files without modifying
them. **Opening the same URL twice connects to the same session**, not an
independent workspace: field/frame/playback controls may affect both tabs.
Do not reuse another conversation's URL when independent controls are wanted.
An explicit `--port` remains available for integrations that require a fixed port.
**Session → Stop viewer service** closes this viewer after confirmation, cancels its active
export and releases its process. Other independently launched viewers continue.
After the last browser disconnects, the process exits after two minutes; a brief
refresh does not lose the session. `--idle-timeout 0` keeps it running explicitly.
Completed exports are retained; closing the viewer never deletes simulation results.

For project/run navigation, open a project itself or the folder containing your
projects: `agentfem-view /path/to/projects`. Select **Project**, then **Run**,
then the desired XDMF (or other supported result) and **Show result** in **Open**. Project selection
uses `agentfem.toml`, including a custom `run.output_directory`; it never imports
or executes `case.py`. Discovery is limited to the opened folder and immediate
child projects, not a recursive disk search. Ordinary folder browsing remains
available for nested projects or standalone results. Opening a file directly
uses its nearest containing `agentfem.toml` project as the browsing root, so
other runs are reachable; standalone files use their parent folder.
**Session → Refresh saved frames** also
refreshes the available projects and runs. Select **Play** to animate saved
frames; the adjacent icon changes to **Pause** during playback.

Use the recommended visualization artifact reported by the run. Supported
reader inputs are XDMF, VTU, legacy VTK and PVD. The viewer accepts one dataset
per saved time; it does not merge distinct physical blocks or coincident nodes.
Use trusted local files. The service listens only on `127.0.0.1`; Ctrl+C stops
it. It does not upload results or execute a case. No AI is needed to use it.
Click a folder in the file list to enter it, or click a result to display it;
the up-arrow returns to the parent (tooltip: **Parent folder**). Folder icons
identify directories without repeating “Folder” in their names. Result rows say
**Simulation result · click to view**, rather than using an external-link icon.
The dialog keeps a fixed height when projects/runs change; its list scrolls.
The search field filters the current folder. **Paste a path instead** is optional.

The folder-search icon opens an in-page folder browser. Select a folder or paste
a path on the computer running the viewer; no H5 data is uploaded or copied.
Selecting a different root automatically opens an independent local session
without a second confirmation. Existing view/export records remain scoped to
their original root. On a remote host this browses the server's filesystem,
not the browser computer's files.
Standard DOLFINx same-document mesh pointers are normalized into a temporary
single-grid-per-time descriptor. Numerical arrays and original files are unchanged;
external XML includes and incompatible time grids are still reported explicitly.
Deformation recognizes names such as `Displacement_m` and exposes the vector
selector; frames without a nodal vector remain viewable without deformation.
Export starts outside the button request, so cancellation remains responsive and
terminates the renderer/encoder process group on POSIX. Cancelled staging files
are not published as completed downloads.
The selected root limits browsing and XML companion references. For continuation
indexes referencing sibling runs, open their common parent directory.

## Controls

**View** offers Isometric, XY, XZ and YZ camera directions. A small rose/green/blue
axis marker shows orientation. Errors have an actual close button; the severity
symbol is not used as a false dismissal affordance. Export failures retain the
full traceback in `viewer-exports/last-export-error.log`, not in the main dialog.

For a moving-coordinate archive, the viewer checks whether the first and last
states have the same connectivity and matching `coordinates - displacement`.
It reports detected deformed coordinates explicitly; Settings lets the user
override the interpretation. Magnification and smooth export reconstruct the
reference coordinates first, avoiding double deformation. Each interpolated pair
still checks reference geometry and connectivity, allowing only storage-precision
roundoff. This does not support remeshing or create new computed states.

Spatial cell-field smoothing is enabled by default and can be switched off. VTK's
[cell-to-point filter](https://vtk.org/doc/nightly/html/classvtkCellDataToPointData.html)
averages incident cells; it is not constitutive recovery or a higher-accuracy
stress solution. **Smooth colors** averages the selected
scalar for display, retaining the original color limits and source arrays.
Choose an integer cell-region field to keep material/interface regions separate;
nodes are not welded across regions. Without that field, **One continuous region**
explicitly assumes continuity. Keep raw cells for damage/contact discontinuities.
The legend identifies averaged display and exported view settings record it.
Interactive rendering and image/video export use the same averaging operation.
This is separate from temporal displacement interpolation and does not interpolate
damage histories or claim improved numerical accuracy.

**Show deformation** has the same meaning for supported reference-coordinate
and moving-coordinate archives: off shows the reference shape; on applies the
selected multiplier. Moving coordinates are first normalized using the selected
displacement. The control cannot reconstruct an unknown reference configuration
without a valid displacement field and coordinate convention.

Negative/positive colors default to Azure / Rose, followed by Lagoon / Violet and
Blue / Red. The legend overlays the model view transparently in both the browser
and exports. The browser legend is accessible HTML, avoiding vtk.js automatic text
rescaling in small panes. Export titles are optional, never inferred from generic
filenames such as `fields`. Without a declared time unit, exports identify the
frame rather than calling a load parameter seconds. Information buttons provide
the longer display-only explanations without crowding the controls.

Scene updates explicitly advance the render-window version before sending them
to vtk.js. Timeline advancement alone is not playback acceptance: verify that
the geometry actually changes between early and late frames, including smooth
interpolation within the same pair of saved frames.

**Open** selects projects, runs and files in a dedicated dialog. **Display**
toggles a compact, docked field/palette/deformation sidebar. It occupies its own
space and never covers playback. Play/Pause is centered below the canvas and
saved-frame slider, with **Animation** settings alongside it. On narrow embedded
panes, collapse **Display** to give the model more space.
Refresh and stop-service actions are under **Session**, not unlabeled toolbar
icons. Closing a tab and stopping the service are different actions; neither
deletes results. Nothing here changes the stored result.

- **Settings → Background** offers Scientific (soft gray), Publication (white),
  and an optional Dark background. Background changes preserve the chosen palette.
  Exact limits remain directly editable under **Set limits manually**, without
  numeric spinner arrows. Deformation also has a drag slider, Auto scale and ×1.
  Units live in Settings, not in the export workflow.
  Ocean uses pale ice/teal/deep blue; Iris offers a lavender/indigo alternative.
  Garnet uses pale rose/burgundy and Jade uses pale celadon/forest green; these
  also work for localized damage on a quiet, low-value background.
  These sequential palettes have monotonically ordered relative luminance;
  they are not claimed to be perceptually uniform or journal-endorsed palettes.
  Established viridis/cividis maps remain selectable, including in saved views.

- **Color mapping → Negative / Positive (±)** offers Azure / Rose, Lagoon / Violet and Blue / Red.
  **Range** offers Ocean, viridis, cividis and the same three diverging palettes.
  In Range mode a diverging palette's center represents the range midpoint,
  not necessarily zero. Older saved palettes remain readable, without crowding
  the everyday menu. Symmetric limits `[-A, A]`
  put equal positive/negative magnitudes at equal distances from the center.
  **Limit ±** adjusts both ends; **Auto range · all frames** finds
  the largest absolute value across the saved series. The fixed range remains
  unchanged during playback and is retained in exports and saved views.
  Choose Negative / Positive for signed stress/strain components when appropriate.
  Field names and the presence of negative values alone do not establish physics.
  Manual selection is retained when changing fields and components.
  Each mode remembers its selected palette. The two new palettes have monotonic
  relative luminance on each side of their light center. Screening includes
  severity-1 protan/deutan/tritan simulations using the
  [Machado model](https://doi.org/10.1109/TVCG.2009.113), with coefficients checked
  against [Colour's tables](https://github.com/colour-science/colour/blob/develop/colour/blindness/datasets/machado2010.py).
  Our tests require endpoint CIE76 distance above 10 in each simulation and
  normal-vision endpoint luminance difference below 0.03; these are internal
  screening criteria, not accessibility certification or perceptual uniformity.
  Lagoon's cyan end was adjusted after the initial green-end prototype failed
  the deutan endpoint screen. Near-zero values and equal-magnitude signs can
  still be hard to distinguish, especially in grayscale: retain labeled color
  bars and use quantitative values for interpretation. These are original,
  Paris-inspired palettes, not official Olympic or journal palettes.

- **Field** selects the saved quantity, with **Elements** or **Nodes** indicating
  where values are stored. **Component** selects its component separately;
  three-component vectors offer **Magnitude**, **X**, **Y**, **Z**. Magnitude
  is computed only for display. Declared tensor matrices use XX/XY/etc.; unknown
  compact tensor layouts retain numbered ordering rather than guessing a Voigt
  convention. No stress invariant is invented.
- **Saved frame** selects an actual output station, not an interpolated time.
- **Play** advances through saved frames; **Pause** stops. It does not imply a
  uniform physical sampling interval.
- **Smooth displacement** is optional and only enabled for declared reference
  geometry with displacement coloring or solid color. It interpolates the
  displacement vectors between identical reference meshes, then computes the
  displayed magnitude/component. It does not interpolate damage, stress,
  deletion flags or material IDs. Changed connectivity/reference coordinates
  stop playback with an explanation. Only two adjacent source frames are cached.
  Playback samples uniformly in the saved coordinate, including unequal output
  intervals. It assumes a continuous displacement path between saved stations;
  do not use it across impacts or discontinuous restart gaps. Interpolation
  cannot recover dynamics absent from the output.
- **Seconds per animation** is target presentation duration, not simulation
  time. Each scene must load before advancing; large models may take longer.
  Pause keeps the interpolated view; choosing a saved frame returns to an actual
  stored state. Resume starts at that interpolated coordinate. Switching to a
  non-displacement field stops smooth playback and returns to a saved state.
  Still-image/view saving requires an actual saved frame.
- **Keep colors consistent** is on by default. Selecting a new field/component
  initializes its limits from the displayed frame, then holds them during
  playback. **Fit all saved frames** reads the published frames one at a time
  to find the full range; this explicit operation may take time for large
  results, but does not keep every frame in memory. Minimum/maximum remain
  available for optional custom limits. Turning the checkbox off rescales each
  frame independently. Refresh does not silently enlarge a previously fixed range.
- **Refresh frames** reopens the published index, preserving the selected
  station if it still exists. New frames become selectable; old H5 files are
  neither duplicated nor edited. Refresh an atomically published frame-file
  series; an actively written single H5 file may be locked by its writer.
- **Show mesh edges** and **Reset view** control the display. Drag to rotate and scroll
  to zoom using VTK's browser interaction.

To enable deformation, select **Show deformation**, choose the nodal
displacement vector, and set **Deformation scale** (default 1 shows actual
saved displacement). A sole vector named `Displacement`, `displacement` or `U`
is selected automatically; other names require explicit selection.
For known reference-coordinate output, the declaration can also be supplied
when launching:

```sh
agentfem-view /path/to/results.xdmf --reference-geometry --displacement displacement
```

Warp starts from the saved frame every time, so changing the scale does not
accumulate deformation. Recognized moving-coordinate output is normalized to
reference geometry first. Opening another file clears the declaration. Displacement must be a
three-component nodal field. Units and component
conventions come from the result. The viewer does not infer missing units.

**Auto scale** scans the currently published frames once, without retaining the
whole series, and suggests a fixed multiplier. It uses the largest reference
extent and the largest per-frame 95th-percentile displacement relative to the
componentwise median translation. This translation subtraction is only a sizing
statistic; it does not modify the plotted displacement. The target is 10% of the
extent, capped so the maximum absolute displacement remains below 20% after
magnification. Motion already reaching 5% of the extent, zero distortion and
pure translation keep ×1. The suggested factor is bounded at 1e6, rounded down
to three significant digits, and does not adapt during playback. **Actual ×1**
and manual entry remain available. New live frames do not silently change it;
run Auto scale again explicitly. The percentile is node-sampled, not a
mesh-independent physical statistic, and multi-body or strongly nonuniform
meshes may need a manual choice. This is a viewing aid, not a strain estimate.

The compact information block distinguishes saved/interpolated frames, the
saved coordinate and the deformation multiplier. A coordinate is labeled as
physical time only when its unit is supplied from the result contract. Do not
label a static load factor as seconds. Presentation seconds and video fps do not
establish the physical units of an XDMF time coordinate.

## Save a figure, video or view

**Export** follows three steps: choose output, choose quality, generate and
download. Resolution choices are HD (1280 × 720), Full HD (1920 × 1080, default),
and 4K (3840 × 2160). These apply to both PNG and MP4. A completed export offers
one media download, not several competing save/download actions. New export
starts another flow; failed/cancelled exports never show an earlier download as new.

- **Image · current frame** renders a PNG with the current camera, fixed color
  limits, title, saved time coordinate and deformation scale. Transparent
  backgrounds are optional. Specify field/time units from the result contract;
  the viewer does not guess them. Custom units are display labels, not conversions.
  Time and deformation scale occupy stacked, fixed positions, so changing time
  digits cannot move the scale label. Node/element association stays in the
  field selector and view record, rather than decorating the exported figure.
- **Animation · simulation history** exports actual saved states as MP4 at 24 fps, holding
  each state for 10 video frames. This is an output-station animation, not a
  claim of real-time playback or interpolated physical evolution. Physical
  time coordinates remain visible. It supports at most 500 saved states.
  With **Smooth displacement** enabled it instead samples the selected playback
  duration at 24 fps, including both endpoints. The video is marked
  **Interpolated display**; `view.json` records each source-frame pair, weight,
  coordinate and fixed scale. These video frames are not additional solved states.
- **Turntable · rotate current frame** rotates the camera around the current saved state
  for five seconds. It does not advance simulation time.
- Export runs in a separate bounded process; **Cancel export** terminates it.
  Only a fully completed image/video and its `view.json` are published. Failure
  does not replace earlier exports or touch source results.
- **Download image/video** saves through the browser. Completed bundles also
  remain in `viewer-exports/export-…` under the opened browsing root. The viewer
  needs write access there. They contain derived media and view settings, not
  duplicated meshes or H5 histories. Export dependencies are included in the
  optional `viewer` extra; rendering requires a working VTK offscreen backend.
- **Views → Remember this view** saves settings without rendering, independently
  of Export. Select a saved view and **Restore** to reopen its source and camera, field/component, range,
  time index, preset and deformation declaration. The source must remain under
  the same browsing root. Settings are plain validated JSON, not executable code.
  **Remove** hides the recipe from the list; **Undo remove** restores the last
  removed recipe in this session. Removed recipes are retained as hidden JSON
  beside their exports, so no source data or media is deleted.

The docked layout and concise control groups follow the principles of Apple's
[sidebar](https://developer.apple.com/design/human-interface-guidelines/sidebars)
and [layout](https://developer.apple.com/design/human-interface-guidelines/layout)
guidance, adapted to a scientific inspector rather than copied as a native UI.

Export uses server-side VTK while interactive rendering uses vtk.js; lighting
can differ slightly. Camera and numerical selections are shared. Both paths
honor optional display averaging; neither edits mesh coordinates in the source
or invents tensor invariants.
Anti-aliasing improves image edges, not the numerical discretization. A view
record is a reproducible presentation recipe, not a scientific validation seal.

## Continuation results

Use the existing continuation index, or create one with the established API:

```python
from agentfem import results

results.join_xdmf_segments(
    ["first/results.xdmf", "resumed/results.xdmf"],
    "combined/results.xdmf",
)
```

Open the common parent directory and select `combined/results.xdmf`. Joining checks compatible meshes and fields and
references original H5 data rather than copying it. Conflicting overlap is
rejected. Missing output stations are not reconstructed. Incomplete and
uncertified-coverage markers are displayed when present in the index. A joined
index is a snapshot: regenerate it through the joining API if its source
segments grow; Refresh does not regenerate joins.

## Animation and publication exports

**Animation → Smooth motion** interpolates displacement for display. **Seconds
per animation** is the target viewing time, not physical simulation time. The
viewer waits for each browser scene to load before advancing; large models may
take longer rather than skipping to their final state. Stress and damage remain
saved-state displays, not interpolated predictions.

Exports default to **Current zoom**, preserving intentional close-ups. Choose
**Fit whole model** to enclose all exported states with fixed conservative
framing, independently of browsing zoom. Text and orientation
markers scale with output resolution. Videos use higher-quality encoding;
PNG remains the lossless choice for figures.

**Showcase · 45° arc** follows a ten-second closed oblique path spanning 45°
around the selected view, with only ±6° elevation and constant distance.
Arc-length resampling keeps camera travel speed nearly constant, without zoom
pulses or a full sweep around the underside. Export legends share the full
model viewport on a transparent annotation layer rather than a separate panel.
**Turntable** retains level rotation. Both preserve the data and record their camera recipe in
`view.json`. Target tracking keeps the subject in view, following the principle in
the [Blender tracking manual](https://docs.blender.org/manual/en/4.4/animation/constraints/tracking/track_to.html).
Click X, Y or Z in the lower-left marker to select the corresponding orthogonal view.

Component labels follow declared field layout, not field-name guesses. XDMF
Vector fields expose X/Y/Z; Tensor matrices expose row-major XX/XY/XZ/YX/YY/YZ/
ZX/ZY/ZZ (or XX/XY/YX/YY in 2D). An Attribute-level Information element named
`component_names`, containing a JSON string list, can declare custom state or
six-component tensor names. Undeclared Tensor6 ordering retains numbered labels.
VTK component names are also honored. The same labels appear in the component
selector and exported legend; no invariants or unit conversions are inferred.
The native array writer declares standard vector/tensor layouts; the viewer
derives their component names without adding reader-incompatible XML children.

Export defaults to Current zoom and level Turntable motion. Fit whole model
remains available when the whole moving model must stay visible. Next, Generate
and Download occupy one fixed action position. Removing a saved view selects
the next list entry (or the previous one at the end); Undo remove remains available.

**Smooth colors** uses fast native nodal averaging of the selected cell scalar.
Explicit region IDs keep material regions separate. Original arrays and color
limits are unchanged; turn smoothing off to inspect cell discontinuities.
The slower experimental Edge-aware and Preserve jumps modes have been removed;
older saved views use Average instead. Color mapping (Range or
Negative / Positive) stays selected when changing fields and components.
The browser legend is a transparent overlay inside the model view, not a
separate opaque panel. It does not crop the model or intercept camera gestures.

**Auto range · all frames** reports saved-frame progress and can be cancelled.
The existing limits stay in effect until the complete scan succeeds. Selecting
another result or field discards the old scan rather than applying stale limits.
For manual limits, enter decimal or scientific notation (for example `1e-10`)
and press Enter or **Apply limits**. Typing does not update the view midway;
Scroll over a limit to adjust it relative to the current range; hold Shift for
ten-times finer adjustment. Edits remain drafts until Apply limits or Enter.
Folder selection opens inside the current dialog, with Parent and Home shortcuts,
instead of a separate operating-system window.

Paste a local folder or result path, including one outside the currently open
project. External locations open a new local viewer scope in the same tab;
file validation remains scoped to that selected location. Recent locations
are stored locally in `~/.cache/agentfem/viewer-recent.json`, limited to eight
deduplicated existing paths. No paths are uploaded. Projects require an
`agentfem.toml`; runs come from its configured output directory. Standalone
XDMF/VTU results remain directly browseable without a project or run record.
Display settings start open; smooth-motion duration defaults to three seconds.
Dragging the timeline temporarily disables model camera interaction.
Lagoon / Violet is a custom teal–purple display palette, not an official
Olympic palette; old Lagoon / Rose view recipes remain supported.

The lower-left orientation marker uses screen-facing X/Y/Z labels, with
contrasting colors for light and dark backgrounds. **View** contains standard
camera directions and saved views. On narrow screens, **Display** opens a
temporary panel instead of consuming the rendering width.

## Mesh and element checks

Before conversion, use `agentfem inspect-mesh` and the existing
`mesh.compatibility_matrix()`. Inventory checks finite coordinates, integer
connectivity, declared node counts, index bounds and repeated nodes per cell.
It does not repair a mesh or certify its Jacobians. Named sets and source cell
types stay visible; quality checks and the selected Step determine admissibility.

First- and second-order triangle, quadrilateral, tetrahedron and hexahedron
result reading/warping are covered by numerical-output tests. Displaying a VTK
cell does not qualify a shell, reduced-integration or other element formulation.
A local-browser check additionally exercised 25 saved frames of a 255,041-cell
hexahedral result, cell damage coloring and nodal displacement warping. This is
an installed optional-viewer check in the Codex in-app browser, including saved-
frame playback, a fixed 0–1 damage range, rotation and collapsible controls;
it is not a native Codex widget or a separate AI-dependent rendering path. It is
not a general browser memory limit. Windows browser behavior and additional
high-order topologies remain separate acceptance work.
## Desktop entry

After installing the viewer, macOS users can create a double-click entry once:

```sh
agentfem-view /path/to/results --create-shortcut "$HOME/Desktop/AgentFEM Viewer.command"
```

This entry uses the installed environment, not a development checkout or a
temporary viewer environment. Keep that environment installed. Each launch
opens an independent local session. Closing the last browser connection allows
the service to exit after its two-minute grace period; closing a terminal does
not stop the macOS shortcut's service. Startup logs are retained in the system
temporary directory under `agentfem-viewer`. Existing shortcuts are never
overwritten. Native Linux accepts a `.desktop` destination (the desktop may
require marking it trusted); this is not a Windows/WSL desktop installer.
