# Checked array meshes and part exteriors

Use existing Gmsh, XDMF and external-format import for ordinary mesh files.
For a reviewed numeric NPZ already produced by a conversion pipeline,
`mesh.read_array_mesh` replaces application-owned decoding and connectivity
admission. This does not translate an engineering input deck or repair geometry.

```python
from agentfem import mesh, spaces

geometry = mesh.read_array_mesh("inputs/mesh.npz")
domain = geometry.domain
surfaces = mesh.exterior_regions(
    domain, input_cell_tags=geometry.arrays["part_id"],
    names={"plate": 1, "insert": 2},
)
all_exterior = mesh.exterior_boundary(domain)
# After creating a continuous P1/Q1 field:
dofs = spaces.locate_input_nodes(u.value.function_space,
                                geometry.arrays["selected_input_nodes"])
```

These are ordinary `BoundaryRegion`/`RegionSet` objects usable by the existing
load, constraint and contact APIs. Region names and which surfaces are coupled
remain engineering decisions in `case.py`. For 3D Q1 traces use the existing
`interfaces.reference_trace_from_boundary(..., geometry="bilinear")`; do not
triangulate warped quadrilaterals in the case.

## Archive contract

`mesh.npz` requires numeric arrays `cells` and `coordinates`. Connectivity uses
zero-based input coordinate indices, with the declared Basix/DOLFINx order.
Extra numeric arrays retain application-defined meaning. No pickle or executable
object is admitted. Arrays are immutable snapshots, not backend state aliases.

The adjacent `mesh.json` (or explicit `manifest=` path) declares:

```json
{
  "schema": "agentfem.array-mesh.v1",
  "sha256": "SHA256_OF_THE_NPZ_BYTES",
  "cell_type": "hexahedron",
  "cell_order": "dolfinx_basix",
  "geometry_degree": 1,
  "length_unit": "mm",
  "source": {"description": "reviewed mesh conversion"}
}
```

The reader hashes and decodes the same file snapshot. It rejects wrong hashes,
unknown order, unsupported topology/degree, nonnumeric or nonfinite arrays,
invalid indices and repeated nodes within a cell before backend construction.
It does not infer order from a plausible shape or convert units. Metadata is
returned detached through `geometry.metadata`. Integrity is not source trust,
positive-Jacobian quality, or scientific verification; use the existing mesh
quality audit for the selected element and geometry.

Initial scope: one serial, homogeneous degree-one interval, triangle,
quadrilateral, tetrahedral or hexahedral mesh. Embedded geometry is allowed when
the geometric dimension is at least the topology dimension. Mixed topologies,
higher-order archives and distributed input are rejected rather than guessed.
Other existing importers retain their own broader capabilities.

## Cell tags, exterior and identity

`input_cell_tags` contains exactly one integer per original input cell. The
backend's `original_cell_index` maps reordered cells back to this table; array
length alone is not a valid mapping. Missing named exterior groups fail early.
The region extractor currently accepts serial 2D/3D domains only.

An exterior is the topological exterior of the entire mesh. A shared-node
material interface is **not** an exterior and is not split automatically.
Coincident independently meshed parts retain independent nodes and surfaces.
This distinction is essential for contact and bonding. Extracting a part's
internal shared interface requires a separate explicit interface operation.

Input node indices, source file node labels and finite-element DOFs are distinct.
Reuse `spaces.locate_input_nodes` for the supported nodal spaces; never match
coordinates to collapse coincident nodes or assume original indices are DOFs.
