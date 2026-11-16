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

Use the recommended visualization artifact reported by the run. Supported
reader inputs are XDMF, VTU, legacy VTK and PVD. The viewer accepts one dataset
per saved time; it does not merge distinct physical blocks or coincident nodes.
Use trusted local files. The service listens only on `127.0.0.1`; Ctrl+C stops
it. It does not upload results or execute a case. No AI is needed to use it.
Choose **Local result or folder**, then **Open**; **Up** returns to the parent.
The selected root limits browsing and XML companion references. For continuation
indexes referencing sibling runs, open their common parent directory.

## Controls

- **Field · association** selects a scalar or stored component, keeping cell
  and node fields distinct. No cell-to-node smoothing or tensor invariant is
  invented.
- **Saved frame** selects an actual output station, not an interpolated time.
- **Play** advances through saved frames; **Pause** stops. It does not imply a
  uniform physical sampling interval.
- **Fixed color range** keeps the same minimum and maximum across frames.
  Without it, each frame has its own range; colors alone cannot compare damage.
- **Refresh frames** reopens the published index, preserving the selected
  station if it still exists. New frames become selectable; old H5 files are
  neither duplicated nor edited. Refresh an atomically published frame-file
  series; an actively written single H5 file may be locked by its writer.
- **Edges** and **Reset view** control the display. Drag to rotate and scroll
  to zoom using VTK's browser interaction.

To enable deformation, the source must explicitly be on reference geometry:

```sh
agentfem-view /path/to/results.xdmf --reference-geometry --displacement displacement
```

Warp starts from the saved frame every time, so changing the scale does not
accumulate deformation. Do not enable it for geometry already deformed in the
output. Displacement must be a three-component nodal field. Units and component
conventions come from the result. The viewer does not infer missing units.

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
Large-result browser memory, Windows browser behavior and additional high-order
topologies are subsequent acceptance work, not implied by these tests.
