# Mesh and lightweight result inspection

Independent branch: `feat/mesh-result-inspection`, based on merged main
`8b629b29`. No new solve, state store, mesh format or material semantics.

## Decisions

1. Reuse Gmsh for meshing, meshio for neutral format inventory/conversion,
   Basix/DOLFINx for coordinate elements and numerical formulations. Existing
   `mesh.compatibility_matrix()` and Step contracts remain authoritative.
   Import, geometric quality, element formulation and output interpolation
   are four distinct acceptance questions. Do not advertise solver support
   merely because VTK displays a cell.
2. Read saved output through VTK/PyVista, one selected frame at a time. Preserve
   node/cell association, independent coincident nodes and high-order cells.
   No automatic cell-to-point averaging, connectivity rewriting, surface-only
   scientific replacement or interpolation of unsaved times.
3. Keep the browser UI an optional consumer. First implement and test the
   data/warp boundary independently of a UI; then connect a small local Trame
   view. No server, network upload or browser launch during installation/solve.
   Default local-only service, one dataset, no filesystem browser or Python
   execution endpoint. ParaView remains the full-featured interchange route.
4. A standalone HTML scene is useful for sharing a selected view, but does not
   by itself provide dynamic time/field selection. Do not label a camera-only
   HTML export as a complete postprocessor.

## First slice

- Local XDMF/VTU/PVD reading, exact saved-time selection and a single uniform
  dataset per frame. Multiple physical blocks remain explicit and are not
  blindly merged to make a field appear complete.
- Explicit field association; node-vector warp always starts from the original
  frame. Existing geometry metadata must decide whether a stored displacement
  can be applied; no automatic double deformation.
- Test triangle/quad/tetra/hex first/second-order output, nodal displacement and
  cell stress on the same warped dataset, and unchanged source bytes/arrays.
- Verify indexed reading for published/restarted XDMF and selected-frame memory
  behavior. Mesh/field errors precede plotting.

## Next slices

- Mesh preflight: named-region coverage, geometry degree/order, Jacobian quality
  and the actual Step's admitted element policy, using existing checks.
- Optional browser: rotate/zoom/reset, scalar field/component, saved time,
  explicit deformation scale, edges and a restrained light/dark theme. Retain
  cell discontinuities. Slice/probe and offline time bundles follow evidence,
  not the first delivery.
- High-order/import expansion is driven by independent patch tests and VTK
  interpolation tests, not a longer list of mesh suffixes.

## Run the experimental viewer

From this source checkout, in an environment with AgentFEM and PyVista:

```sh
python -m pip install trame trame-vtk trame-vuetify
python -m tools.result_viewer /path/to/results.xdmf
```

The optional dependencies belong to the viewing environment, not the solver
installation. The service binds only to `127.0.0.1`; stop it with Ctrl+C.
Use only trusted local result files. It does not run `case.py` or upload data.
For an artifact explicitly stored on reference geometry:

```sh
python -m tools.result_viewer /path/to/results.xdmf --reference-geometry --displacement displacement
```

Select a saved frame, field/component and deformation scale. Cell fields stay
cell-associated. Scalar ranges are recalculated per frame; changing colors
across frames therefore does not alone establish damage evolution. Units and
tensor conventions remain those of the source result. No unsaved frames are
reconstructed. There is no live-file refresh, slicing, probing, browser-only
offline bundle, or installed `agentfem` viewer command in this first slice.

Local acceptance: 15 reader/warp tests passed, including real DOLFINx output
for triangle, quadrilateral, tetrahedron and hexahedron at degrees 1 and 2.
The browser was exercised with a synthetic display fixture: selection of cell
damage and reference displacement warp worked together. This is UI evidence,
not a material or structural validation result. Cross-platform browser and
large-result memory measurements remain pending.

Subsequent acceptance note: the standalone field-choice test passed (1 test).
Repeated combined FEniCSx/reader reruns terminated with exit 143 after partial
progress, with no assertion traceback. The earlier 15-test result is retained
as historical evidence, not substituted for this incomplete rerun. Broader
writer/reader regression and investigation of termination remain release gates.

## Primary references consulted

- [Gmsh manual](https://gmsh.info/doc/texinfo/): physical groups, element
  properties/order, local mesh sizing and sampled Jacobian quality.
- [PyVista XdmfReader](https://docs.pyvista.org/api/readers/_autosummary/pyvista.xdmfreader):
  time-aware VTK reading.
- [PyVista HTML export](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.export_html):
  interactive scene export and its optional Trame dependency.
- [Trame guide](https://kitware.github.io/trame/guide/intro/getting_started.html):
  Python/VTK-backed interactive views without replacing scientific objects.
- [Kitware VTK.js discussion](https://discourse.vtk.org/t/reading-a-vtk-data-using-vtk-js-locally/13837):
  do not assume a browser PolyData reader is a volume/high-order mesh reader.

These references inform the architecture; local reader/field tests determine
the admitted formats and cells. No new visualization dependency is mandatory
for solving or reading result manifests.
