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

When several topologies must be retained, convert an explicit bundle:

```python
bundle = mesh.convert_external_mesh_bundle(
    "assembly.inp", "output/mesh",
    cell_types=("tetra10", "hexahedron"),
)
```

Each topology receives its own XDMF/HDF5 domain and manifest, plus one bundle
manifest. AgentFEM does not merge unlike cells into an opaque solve mesh while
mixed-topology support in DOLFINx remains incomplete.

## Source mesh, converted artifact, and runtime mesh

`mesh.read_abaqus_mesh(source, converted_path, ...)` does not remesh the
geometry. Rank zero reads the Abaqus source, converts its selected topology to
XDMF/HDF5 at the caller-supplied `converted_path`, writes an adjacent
`.mesh.json` evidence manifest, and then all ranks read that converted mesh
into DOLFINx. The finite-element solve therefore uses the in-memory DOLFINx
mesh reconstructed from XDMF/HDF5; it does not repeatedly solve from the
original keyword file.

The adjacent manifest stores a SHA-256 source fingerprint and the topology,
facet, dimension-pruning, and reader choices.  `read_abaqus_mesh(...)` reuses
the conversion by default only when that complete identity still matches and
the XDMF/HDF5 pair is present. Editing the source mesh or changing a conversion
choice invalidates the cache and triggers conversion on rank zero.

The output location is not forced to a global `mesh/` directory. A project may
keep derived conversion artifacts under `output/mesh/`, while retaining the
original `.inp`/`.dat` as the authoritative source and reconversion input. The
manifest links both identities and records topology selection and omissions.

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

The Abaqus adapter also preserves inline and explicit `NSET`/`ELSET`
definitions, expands `GENERATE` ranges, and records node- and element-based
`SURFACE` entries. `imported.surface_faces(name)` expands an element-based
surface to source `(element_label, face_identifier)` pairs. This is durable
engineering evidence even when a selected neutral topology cannot yet become
one DOLFINx facet-tag field.

### C3D10 and C3D10H are not the same solver formulation

Both Abaqus `C3D10` and `C3D10H` have ten-node quadratic tetrahedral geometry,
so meshio maps both to `tetra10`. AgentFEM now preserves the source keyword
identity beside the neutral mesh. For `C3D10H` the conversion manifest records
the hybrid constant-pressure formulation and its one additional element
pressure variable, and emits a warning that XDMF contains topology rather than
that pressure formulation.

AgentFEM now provides an explicit mixed route:

```python
unknown = model.field(fields.displacement_pressure(domain))  # P2 / DG0
material = model.material(
    constitutive.mixed_neo_hookean(young=1.0e6, poisson=0.499)
)
model.fix(unknown.displacement, on=support)
step = model.step(target=unknown, material=material)
```

The monolithic solution has quadratic displacement and one independent
constant pressure value per cell. The provider consumes known `C3D10H`
constant-pressure source semantics; the ordinary displacement-only provider
continues to reject them. This is an AgentFEM mixed variational analogue, not
a claim that neutral conversion reproduces Abaqus internal element code.

## Current Limits

- node sets and element-face surfaces are preserved and queryable, but node
  sets are not yet first-class DOLFINx point regions and source face pairs are
  not yet reconstructed as facet tags unless lower-dimensional cells exist;
- mixed top-dimensional element families need an explicit selection;
- high-order topology compatibility remains format-specific; Abaqus `C3D10` /
  meshio `tetra10` / DOLFINx quadratic tetrahedral geometry is now covered by
  a real import and nonlinear example;
- the C3D10H constant-pressure route is solved explicitly; other hybrid and
  enhanced suffixes remain source evidence until a matching provider exists;
- ANSYS CDB support depends on the installed reader and is not claimed merely
  from the file extension.

The next release gate is a corpus of small legal meshes from each target
format, with named-set golden manifests and a real DOLFINx read/solve check.
