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

For Abaqus input decks, run the semantic inventory before generic mesh
conversion:

```bash
agentfem inspect-abaqus model.inp --json --write migration.json
```

The inspector resolves nested `*INCLUDE` declarations into a
content-addressed source graph. Every source file and include edge is retained,
a change in any included file changes the graph fingerprint, and missing files
or recursive cycles receive stable diagnostic codes. This stage intentionally
does not concatenate files or pretend that Part/Instance scopes have already
been flattened; it establishes the provenance required by later scope-aware
migration and conversion caching.

The [Abaqus migration guide](abaqus_migration.md) separates declaration and
topology, neutral conversion, native formulation, and verification evidence.

Gmsh is a separate optional route. Direct in-memory Gmsh models and
`mesh.read_gmsh_mesh(...)` require `agentfem[gmsh]`; structured DOLFINx meshes,
XDMF, and the meshio conversion described on this page do not. This keeps both
the runtime and the separately licensed Gmsh package outside the AgentFEM core.

## Inspect Before Converting

Start with the CLI when the file comes from another mesher or solver:

```bash
agentfem inspect-mesh model.msh
agentfem inspect-mesh model.inp --json
```

The report lists every cell block, named set, and its declared compatibility.
`verified` means that the neutral geometry route has executable import and
solver evidence. `conditional` means that topology exists but an order,
format, or formulation acceptance gap remains. Unknown cells are `blocked`.
None of these states silently equates an Abaqus, ANSYS, or COMSOL element
formulation with a DOLFINx cell topology.

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
manifest. AgentFEM does not merge unlike cells into an opaque solve mesh until
its own mixed-topology assembly, regions, output, checkpoint, and MPI lifecycle
has release-level evidence.

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
[Abaqus C3D10H Periodic Cell](abaqus_c3d10h_periodic_cell.md).

The Abaqus adapter also preserves inline and explicit `NSET`/`ELSET`
definitions, expands `GENERATE` ranges, and records node- and element-based
`SURFACE` entries. For supported three-dimensional continuum families, these
source semantics now become ordinary AgentFEM regions:

```python
cell = mesh.read_abaqus_mesh("part.inp", "output/part.xdmf")
support_nodes = cell.node_set("FIXED")
loaded_surface = cell.boundary("LOAD_FACE")

model.fix(U, on=support_nodes)
model.pressure(12.0e6, on=loaded_surface)
```

`node_set(...)` preserves source-node identity as a coordinate-backed region
used by strong constraints and probes. It includes high-order geometry nodes,
so a C3D10 midside `NSET` can locate the corresponding P2 degree of freedom;
it deliberately has no boundary integration measure. `boundary(...)`
reconstructs exterior facets from official Abaqus face numbering and returns
the same tagged `BoundaryRegion` consumed by weak loads and output resultants.
Tetrahedral, hexahedral, and wedge solid families have source face semantics.
The broader catalog can retain common continuum, heat, interface, line, and
shell declarations without claiming that every Abaqus formulation has a
native solver equivalent. Missing nodes, internal faces, unknown face
identifiers, unsupported lowering, and ambiguous coincident source nodes fail
explicitly.

`imported.surface_faces(name)` remains available as source-level evidence and
for adapters that need the original `(element_label, face_identifier)` pairs.

### Named internal surfaces for cohesive fracture

Internal interface semantics can now be lowered directly to the neutral split-
interface mesh used by the cohesive force and checkpoint machinery. For an
Abaqus C3D4 deck, one `ELSET` selects the positive cell partition and an
explicit element-face `SURFACE` independently proves the exact triangular
interface:

```python
imported = mesh.read_abaqus_mesh("weak_interface.inp", "output/body.xdmf")
split = imported.cohesive_interface(
    positive_elset="UPPER_BODY",
    surface="WEAK_INTERFACE",
)
```

For Gmsh, named volume and surface physical groups carry the same contract:

```python
split = mesh.split_gmsh_physical_interface(
    "weak_interface.msh",
    positive_group="upper_body",
    interface_group="weak_interface",
)
```

Both adapters reject a named surface that differs from the boundary implied by
the cell partition. The resulting coincident faces have independent nodes and
can be passed to `fracture.mode_i_cohesive_force(...)` in serial or MPI. The
current 3D kernel integrates linear triangular faces with three quadrature
points. Abaqus C3D10 cohesive lowering remains intentionally blocked until a
matching quadratic-face cohesive kernel and verification suite exist.

## Mesh-quality preflight

Imported and generated domains can be audited before a solve:

```python
quality = mesh.audit_quality(cell.domain, threshold=0.1, strict=True)
print(quality.summary())
```

Triangles and tetrahedra use normalized simplex mean ratio, where one is an
equilateral simplex. High-order simplex geometry additionally receives a
sampled coordinate-map validity check. Quadrilaterals, hexahedra, prisms, and
pyramids use the minimum sampled scaled Jacobian of the *active coordinate
map*. Samples include quadrature points and reference vertices, so high-order
curvature is not silently reduced to corner connectivity. For the scaled-
Jacobian metric, one is locally orthogonal and zero is singular or folded. The
report records the metric,
coordinate-element degree, samples per cell, global minimum/mean/maximum, poor
cell count, and invalid cell count.

The threshold is a project acceptance choice, not a universal engineering
limit. AgentFEM rejects non-positive or folded geometry in strict mode but does
not pretend that one quality threshold is correct for every formulation or
physics.

## Unified discretization preflight

Mesh connectivity, coordinate geometry, a finite-element space, and a
numerical formulation are different objects. AgentFEM keeps them separate but
now reports them together before a trusted run:

```python
from agentfem import elements

audit = elements.audit(
    model,
    check_quality=True,
    quality_threshold=0.1,
    reject_poor_quality=False,
)
print(audit.summary())
audit.validation.raise_if_errors()
```

The report records the runtime topology and its maturity, the actual UFL/Basix
family, degree, value shape, Sobolev space and mixed sub-elements of every
registered field, mesh ownership, Study/field shape compatibility, and the
optional collective quality result. `Model.validate()` consumes only the
metadata-level part, so ordinary validation does not silently traverse every
cell. Long runs and release evidence should request the explicit quality path.

The geometry identity is reported separately from solution fields. It records
the coordinate-basis family and variant, degree, nodes per cell, mapping,
topological and geometric dimensions, and whether the mesh is high order or
embedded. This prevents a quadratic field on a linear mesh, a quadratic
coordinate map, and a vendor's quadratic formulation from being described as
the same thing.

For high-order simplex geometry, quality combines the corner mean-ratio with
the minimum sampled scaled Jacobian of the real coordinate map. A curved cell
that remains positive but approaches singularity therefore degrades
continuously; it is no longer reported as healthy merely because its corner
triangle or tetrahedron looks regular.

`acceptable=False` does not by itself claim that one universal quality limit
exists. Invalid or folded cells are errors. Cells below a positive project
threshold are warnings unless `reject_poor_quality=True`; the chosen threshold
therefore remains visible scientific policy instead of hidden solver logic.

This preflight still does not infer formulation from topology. A hexahedron is
not automatically C3D8R, a line is not automatically a beam, and a quadratic
tetrahedral mesh is not automatically a constant-pressure hybrid element. The
selected Step provider remains the owner of those numerical claims.

Runtime topology summaries also expose narrow evidence-bearing capabilities,
such as topology inspection, geometry-quality auditing, a conforming P1 patch,
and quadratic-geometry preflight. Prism and pyramid patch tests establish
basic H1 assembly and affine-gradient reproduction; they do not silently
promote those topologies to every solid, mixed, shell, or nonlinear procedure.

## Cell compatibility contract

The current release-level neutral-geometry matrix is deliberately explicit:

| Source cell | Solver topology | Status | Quality evidence |
| --- | --- | --- | --- |
| `triangle` | triangle P1 | verified | simplex mean ratio |
| `quad` | quadrilateral Q1 | verified | sampled scaled Jacobian |
| `tetra` | tetrahedron P1 | verified | simplex mean ratio |
| `tetra10` | tetrahedron P2 | verified | simplex mean ratio plus curved-map checks where applicable |
| `hexahedron` | hexahedron Q1 | verified | sampled scaled Jacobian |
| `triangle6`, `quad8`, `quad9` | high-order 2D | conditional | metric exists; import corpus is incomplete |
| `hexahedron20`, `hexahedron27` | high-order 3D | conditional | metric exists; import corpus is incomplete |
| `wedge`, `wedge15`, `pyramid` | prism/pyramid | conditional | sampled scaled Jacobian; release solver evidence pending |
| `line`, `line3` | interval | conditional | topology alone is not a beam, truss, or cable |

Use `agentfem capabilities meshes` for the installed machine-readable matrix.
Reduced integration, hybrid pressure, incompatible modes, hourglass control,
shell directors, beam sections, and cohesive kinematics require dedicated
AgentFEM formulations; importing the connectivity does not reproduce them.

### C3D10 and C3D10H are not the same solver formulation

Both Abaqus `C3D10` and `C3D10H` have ten-node quadratic tetrahedral geometry,
so meshio maps both to `tetra10`. AgentFEM now preserves the source keyword
identity beside the neutral mesh. For `C3D10H` the conversion manifest records
the hybrid constant-pressure formulation and its one additional element
pressure variable, and emits a warning that XDMF contains topology rather than
that pressure formulation. This identity follows the
[Abaqus three-dimensional solid element library](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEELMRefMap/simaelm-r-3delem.htm),
which distinguishes C3D10H from the linear-pressure C3D10HS and C3D10MH
families.

Direct C3D10H input uses the ordinary Abaqus entry point; no source rewrite is
needed:

```python
cell = mesh.read_abaqus_mesh(
    "periodic_cell_c3d10h.dat",
    "output/periodic_cell.xdmf",
    cell_type="tetra10",
)
cell.require_formulation("hybrid")
```

`cell.element_definitions` and the conversion manifest retain the original
`C3D10H`, hybrid, constant-pressure identity. AgentFEM then provides an
explicit mixed route:

```python
unknown = model.field(fields.displacement_pressure(domain))  # P2 / DG0
material = model.material(
    constitutive.mixed_neo_hookean(young=1.0e6, poisson=0.499)
)
model.fix(unknown.displacement, on=support)
step = model.step(target=unknown, material=material)
```

For controlled same-mesh studies, a large source file need not be copied and
edited by hand:

```python
evidence = mesh.abaqus.derive_element_formulation(
    "cell.dat",
    "output/mesh/cell_C3D10H.dat",
    source_type="C3D10",
    target_type="C3D10H",
)
```

The helper accepts only known equal-topology/equal-connectivity families,
changes matching `*ELEMENT, TYPE=` values only, and writes a validated
provenance sidecar consumed by the conversion manifest.

The monolithic solution has quadratic displacement and one independent
constant pressure value per cell. The provider consumes known `C3D10H`
constant-pressure source semantics; the ordinary displacement-only provider
continues to reject them. This is an AgentFEM mixed variational analogue, not
a claim that neutral conversion reproduces Abaqus internal element code.

## Current Limits

- NSET point regions and exterior element-face surfaces are reconstructed for
  supported 3D solid families; assembly/instance-scoped duplicate labels,
  automatically generated free surfaces, internal interfaces, and other
  element families still require dedicated adapters;
- mixed top-dimensional element families need an explicit selection;
- high-order topology compatibility remains format-specific; Abaqus `C3D10` /
  meshio `tetra10` / DOLFINx quadratic tetrahedral geometry is now covered by
  a real import and nonlinear example;
- the C3D10H constant-pressure route, including serial affine-periodic
  equations, is solved explicitly; its distributed mixed-MPC extension and
  other hybrid/enhanced suffixes remain separate provider obligations;
- ANSYS CDB support depends on the installed reader and is not claimed merely
  from the file extension.

The next release gate is a corpus of small legal meshes from each target
format, with named-set golden manifests and a real DOLFINx read/solve check.
