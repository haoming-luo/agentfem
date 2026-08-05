# External Mesh Interoperability

## Scope

AgentFEM uses the optional `meshio` dependency to inspect and convert common
external mesh formats. The official
[meshio format list](https://github.com/nschloe/meshio) includes Abaqus
`.inp`, ANSYS `.msh`, Nastran `.bdf/.fem/.nas`, Exodus, MED, Gmsh, VTK/VTU,
and XDMF among others.

This is mesh interoperability, not full solver-deck import. Material cards,
contacts, element formulations, steps, amplitudes, coordinate systems, and
solver controls require format-specific semantic adapters.

Gmsh is a separate optional route. Direct in-memory Gmsh models and
`mesh.read_gmsh_mesh(...)` require `agentfem[gmsh]`; structured DOLFINx meshes,
XDMF, and the meshio conversion described on this page do not. This keeps both
the runtime and the separately licensed Gmsh package outside the AgentFEM core.

## Inspect Before Converting

```python
from agentfem import mesh

summary = mesh.inspect_external_mesh("model.inp")
print(summary.as_dict())
```

The inventory exposes point count, element blocks, cell sets, point sets, and
data arrays. Choosing a volume topology without this inspection can silently
discard boundary elements or mixed element families.

## Preserve Volume and Boundary Sets

```python
conversion = mesh.convert_external_mesh_to_xdmf(
    "model.inp",
    "model.xdmf",
    cell_type="triangle",
    facet_type="line",
    prune_z=True,
)
converted_mesh = mesh.read_converted_xdmf(conversion)
```

The main XDMF contains `agentfem_region` tags for named volume/cell sets. The
separate facet XDMF contains `agentfem_boundary` tags for named boundary sets.
A JSON manifest records source blocks, numeric tag mapping, complete
memberships, selected topologies, and warnings.
`read_converted_xdmf(...)` also handles the meshio-XDMF distinction between
grid names and tag attribute names.

One DOLFINx `MeshTags` object stores one integer per entity. If source sets
overlap, one deterministic tag is written and the complete overlapping
membership remains in the manifest.

## Imported Boundaries Have One Source of Truth

Physical surface tags should define imported engineering boundaries:

```python
support = mesh.tagged_boundary_region(
    domain, facet_tags, tag=101, name="bolt_holes"
)
pressure = mesh.tagged_boundary_region(
    domain, facet_tags, tag=102, name="pressure_surface"
)

model.clamp(U, on=support)
model.pressure(16.0e6, on=pressure)
evidence = model.audit_boundaries(strict=True)
```

Strong constraints use the tagged facets through topological dof location and
weak terms use the same `ds(tag)`. An optional geometric marker creates a
`hybrid` region: it is independent audit evidence, not an alternative hidden
selection rule. The audit reports tagged/marker set differences, facet count,
measure, midpoint bounds, and integrated normal.

## Abaqus Labels, Custom Extensions, and Equations

Generic conversion is not enough when constraints refer to Abaqus node
labels. `mesh.read_abaqus_mesh(...)` explicitly selects Abaqus syntax, so a
keyword mesh may use `.dat` without relying on extension guessing. It returns
the DOLFINx mesh together with the preserved source node table and conversion
evidence.

`mesh.abaqus.read_equations(...)` parses homogeneous linear `*EQUATION`
constraints, including continued term lines. For periodic finite-deformation
cells, `constraints.abaqus_periodic_cell(...)` maps source labels to
displacement dofs and constructs exact affine elimination. See
[Abaqus C3D10 Periodic Cell](abaqus_periodic_cell.md).

## Current Limits

- point/node sets are inventoried but not yet converted to a DOLFINx point
  region;
- surface definitions derived from element faces are not reconstructed unless
  explicit lower-dimensional cells exist;
- mixed top-dimensional element families need an explicit selection;
- high-order compatibility remains format-specific; Abaqus `C3D10` /
  meshio `tetra10` / DOLFINx quadratic tetrahedral geometry is now covered by
  a real import and nonlinear example;
- ANSYS CDB support depends on the installed reader and is not claimed merely
  from the file extension.

The next release gate is a corpus of small legal meshes from each target
format, with named-set golden manifests and a real DOLFINx read/solve check.
