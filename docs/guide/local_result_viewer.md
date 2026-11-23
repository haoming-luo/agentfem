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
The power button closes this viewer after confirmation, cancels its active
export and releases its process. Other independently launched viewers continue.
After the last browser disconnects, the process exits after two minutes; a brief
refresh does not lose the session. `--idle-timeout 0` keeps it running explicitly.
Completed exports are retained; closing the viewer never deletes simulation results.

For project/run navigation, open a project itself or the folder containing your
projects: `agentfem-view /path/to/projects`. Select **Project**, then **Run**,
then the desired XDMF (or other supported result) and **Show result**. Project selection
uses `agentfem.toml`, including a custom `run.output_directory`; it never imports
or executes `case.py`. Discovery is limited to the opened folder and immediate
child projects, not a recursive disk search. Ordinary folder browsing remains
available for nested projects or standalone results. **Refresh frames** also
refreshes the available projects and runs. Select **Play** to animate saved
frames; the adjacent icon changes to **Pause** during playback.

Use the recommended visualization artifact reported by the run. Supported
reader inputs are XDMF, VTU, legacy VTK and PVD. The viewer accepts one dataset
per saved time; it does not merge distinct physical blocks or coincident nodes.
Use trusted local files. The service listens only on `127.0.0.1`; Ctrl+C stops
it. It does not upload results or execute a case. No AI is needed to use it.
Choose **Result or folder**, then **Show result** (or **Browse folder** for a
directory); **Parent folder** returns to the parent. **Location** shows the
opened browsing root.
The selected root limits browsing and XML companion references. For continuation
indexes referencing sibling runs, open their common parent directory.

## Controls

The left panel groups file, field and deformation controls. Its top-left menu
button hides the panel when you want more space for the model. The saved-frame
slider stays above the canvas. Nothing here changes the stored result.

- **Style** offers Scientific (neutral, Ocean), Publication (white, Ocean), and
  Presentation (dark, magma). Presets change appearance, not values. Color and
  custom label settings are disclosed under **Custom colors and labels**.
  Ocean uses pale ice/teal/deep blue; Iris offers a lavender/indigo alternative.
  Both sequential palettes have monotonically ordered relative luminance;
  they are not claimed to be perceptually uniform or journal-endorsed palettes.
  Established viridis/cividis maps remain selectable, including in saved views.

- **Field** selects the saved quantity, with **Elements** or **Nodes** indicating
  where values are stored. **Component** selects its component separately;
  three-component vectors offer **Magnitude**, **X**, **Y**, **Z**. Magnitude
  is computed only for display. Tensor components retain numbered ordering
  rather than guessing an undocumented Voigt convention. No nodal smoothing
  or stress invariant is invented.
- **Saved frame** selects an actual output station, not an interpolated time.
- **Play** advances through saved frames; **Pause** stops. It does not imply a
  uniform physical sampling interval.
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

To enable deformation, select **Deform reference mesh**, choose the nodal
displacement vector, and set **Deformation scale** (default 1 shows actual
saved displacement). A sole vector named `Displacement`, `displacement` or `U`
is selected automatically; other names require explicit selection.
Only enable this when the output contract says the coordinates are on reference
geometry. The same declaration can be supplied when launching:

```sh
agentfem-view /path/to/results.xdmf --reference-geometry --displacement displacement
```

Warp starts from the saved frame every time, so changing the scale does not
accumulate deformation. Do not enable it for geometry already deformed in the
output. Opening another file clears this declaration. Displacement must be a
three-component nodal field. Units and component
conventions come from the result. The viewer does not infer missing units.

## Save a figure, video or view

**Export** opens a compact panel rather than adding controls around the model.

- **Save image** renders a 1920 × 1080 PNG with the current camera, fixed color
  limits, title, saved time coordinate and deformation scale. Transparent
  backgrounds are optional. Specify field/time units from the result contract;
  the viewer does not guess them. Custom units are display labels, not conversions.
  Time and deformation scale occupy separate fixed positions, so changing time
  digits cannot move the scale label. Node/element association stays in the
  field selector and view record, rather than decorating the exported figure.
- **Save process video** exports actual saved states as MP4 at 24 fps, holding
  each state for 10 video frames. This is an output-station animation, not a
  claim of real-time playback or interpolated physical evolution. Physical
  time coordinates remain visible. It supports at most 500 saved states.
- **Save turntable video** rotates the camera around the current saved state
  for five seconds. It does not advance simulation time.
- Export runs in a separate bounded process; **Cancel export** terminates it.
  Only a fully completed image/video and its `view.json` are published. Failure
  does not replace earlier exports or touch source results.
- **Download image/video** saves through the browser. Completed bundles also
  remain in `viewer-exports/export-…` under the opened browsing root. The viewer
  needs write access there. They contain derived media and view settings, not
  duplicated meshes or H5 histories. Export dependencies are included in the
  optional `viewer` extra; rendering requires a working VTK offscreen backend.
- **Remember this view** saves settings without rendering. Select a saved view
  and **Restore view** to reopen its source and camera, field/component, range,
  time index, preset and deformation declaration. The source must remain under
  the same browsing root. Settings are plain validated JSON, not executable code.

Export uses server-side VTK while interactive rendering uses vtk.js; lighting
can differ slightly. Camera and numerical selections are shared. Neither path
averages element values, smooths mesh coordinates, or invents tensor invariants.
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
