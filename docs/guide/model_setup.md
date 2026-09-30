# Meshes, loads, and constraints

Model setup is where a finite-element tool becomes practically useful. AgentFEM
keeps imported geometry semantics, named regions, loads, constraints, boundary
models, and step activation reusable rather than embedding them in one solver.

## Mesh routes

- structured meshes for reproducible studies and tests;
- XDMF/HDF5 as the direct DOLFINx solver representation;
- optional Gmsh model and `.msh` routes;
- optional meshio conversion for external formats;
- Abaqus element-semantic inventory, set, surface, and equation parsing;
- conversion fingerprints so stale cached meshes are not silently reused.

## Common engineering actions

- prescribed displacement and temperature;
- traction, pressure, body force, gravity, centrifugal and hydrostatic loading;
- elastic foundations and boundary models;
- periodic/equation constraints;
- distributed force and moment resultants;
- step-wise activation and deactivation.

## Elastic foundations

An elastic foundation is weak boundary physics, not a disguised fixed
displacement. Register it independently of the material and loading:

```python
support = mesh.boundary(domain, left, name="support")
model.elastic_foundation(on=support, stiffness=5.0e6, mode="isotropic")
result = model.step(target=displacement).solve_result()
```

For a linear-static solid, the foundation provider publishes a nodal reaction
field and MPI-global resultant through the same balance contract as other
constraints. Its spring energy is already part of the assembled stiffness and
system strain energy, so the result ledger records that ownership instead of
adding the energy a second time as prescribed-boundary work. `mode="normal"`
retains only the current reference-boundary normal component. A conservative
coupled support uses a symmetric positive-semidefinite matrix in the model
coordinate system:

```python
model.elastic_foundation(
    on=support,
    stiffness=((5.0e6, 0.5e6), (0.5e6, 2.0e6)),
    mode="matrix",
)
```

The matrix dimension must match the displacement field; an asymmetric or
indefinite numerical matrix is rejected before assembly. General
nonlinear, moving-normal, damping, and contact foundations remain separate
future providers.

## Add bounded contact with an analytical rigid plane

The first contact provider is deliberately narrow and inspectable: a
small-strain solid, one analytical plane, frictionless one-sided contact, and a
penalty selected by the user. The compatibility route below keeps the plane
fixed. Its normal points from the obstacle into the admissible half-space,
while a positive `initial_gap` means that the reference boundary is open:

```python
model.rigid_obstacle_contact(
    on=possible_contact,
    normal=(-1.0, 0.0),
    initial_gap=0.0,
    penalty=1.0e10,
)
result = model.step(target=displacement).solve_result()
```

AgentFEM lowers the contact potential, residual, and consistent tangent into
the ordinary incremental Newton Procedure. After convergence, the same
provider reports the nodal reaction distribution, MPI-global resultant,
penetration norm, active contact measure, and conservative contact energy.
The contact energy belongs to the system internal energy and is not counted a
second time as external work. For a fixed obstacle, proportional dead load,
and zero prescribed motion, the result also records the accepted natural-load
coordinate and closes natural plus provider-dual work against bulk strain
energy plus contact potential.

Define geometry and motion explicitly when the rigid tool moves:

```python
from agentfem import boundary_models

tool = boundary_models.rigid_plane(
    point=(1.0, 0.5),
    normal=(-1.0, 0.0),
)
tool_motion = boundary_models.prescribed_rigid_motion(
    translation=(-0.01, 0.0),
    rotation=0.0,
    reference_point=(1.0, 0.5),
)
model.rigid_obstacle_contact(
    on=possible_contact,
    surface=tool,
    motion=tool_motion,
    penalty=1.0e10,
)
```

The translation and rotation are end values scaled by the accepted nonlinear
load coordinate. At each accepted station the provider records the conjugate
resultant force and moment. Their trapezoidal path integral is the moving
tool's work and enters the same bulk-plus-contact energy ledger. Failed Newton
trials and cutbacks do not enter that ledger. Rotation uses radians: one
counter-clockwise angle in 2D and an axis-angle vector in 3D.

The surface can also be queried independently of a solve:

```python
projection = tool.project(
    [(0.99, 0.2), (1.01, 0.8)],
    motion=tool_motion,
    factor=0.5,
)
assert projection.all_valid
print(projection.signed_gaps)
```

The returned contract keeps the query points, closest points, admissible-side unit
normals, signed gaps, validity flags, projection method and optional entity
identities together with the geometry fingerprint that gives those identities
meaning. Positive gap is admissible and negative gap is
penetration. Analytical planes use exact orthogonal projection; future
tessellated search must use the same contract and report failed or ambiguous
queries explicitly.

For triangle surfaces, `projection.local_coordinates` contains barycentric
weights in connectivity order. Together with the facet ID and geometry
fingerprint, these weights provide the stable surface location needed by later
finite-sliding state and restart. Invalid queries carry NaN weights.

Common curved tools can use exact analytical projection without a search
backend:

```python
sphere = boundary_models.rigid_sphere(
    center=(0.0, 0.0, 0.0),
    radius=0.01,
    admissible_side="exterior",
)
roller = boundary_models.rigid_cylinder(
    axis_point=(0.0, 0.0, 0.0),
    axis_direction=(0.0, 1.0, 0.0),
    radius=0.02,
)
```

A two-coordinate sphere center creates a circle. The cylinder is explicitly
infinite: caps and rims require a reviewed compound or triangulated surface.
At a sphere center or on a cylinder axis, closest-point direction is not
unique and the projection returns `singular_projection`. These analytical
assets are currently projection/evidence objects; the bounded contact
Operator still accepts only a plane.

For reviewed tool meshes, create a projection-only triangle surface with
explicitly oriented connectivity and stable facet IDs:

```python
tool_mesh = boundary_models.triangulated_rigid_surface(
    vertices=vertices,
    triangles=triangles,
    facet_ids=facet_ids,
    name="reviewed_tool_mesh",
)
projection = tool_mesh.project(query_points, maximum_distance=search_radius)
```

Construction rejects duplicate/unreferenced vertices, degenerate or duplicate
facets, non-manifold edges and inconsistent shared-edge orientation. The result
preserves `no_candidate` and
`ambiguous_projection` instead of choosing a facet silently. The built-in
search checks every triangle and is the correctness reference for a future
DOLFINx BVH adapter; `rigid_obstacle_contact(...)` does not yet accept this
surface.

For repeated queries, build the deterministic broad phase once:

```python
search = boundary_models.triangle_surface_bvh(tool_mesh)
outcome = search.project_with_diagnostics(
    query_points,
    maximum_distance=search_radius,
)
projection = outcome.projection
print(outcome.diagnostics.summary())
```

The AABB/BVH tree prunes distant facets, but every surviving leaf uses the
same exact closest-triangle kernel as the exhaustive reference. Equal-distance
candidates remain visible to the ambiguity check. For a partition-independent
MPI correctness check, use:

```python
partition = boundary_models.partition_triangle_surface(tool_mesh, comm)
search = boundary_models.distributed_triangle_surface_bvh(partition, comm)
outcome = search.project_with_diagnostics(rank_local_query_points)
```

The distributed object retains only its rank's stable-ID facet shard and local
BVH. It gathers rank-local queries and nearest-candidate evidence, reports the
communication volume, and must match the exhaustive global oracle. This is a
correctness reference, not yet scalable neighborhood routing or a general
contact backend.

For sparse rank routing over the same partition, use:

```python
search = boundary_models.routed_distributed_triangle_surface_bvh(partition, comm)
outcome = search.project_with_diagnostics(rank_local_query_points)
print(outcome.diagnostics.summary())
```

The routed path queries the nearest rank bounding box first and then only boxes
that can still tie or improve the exact candidate. Its diagnostics expose the
queried ranks, avoided messages, and actual encoded payload bytes. Integer
identities and floating geometry travel in aligned packed `MPI_Alltoallv`
buffers, so 64-bit facet identities are not coerced into floating-point
packets. The all-gather implementation remains its partition-independent
correctness oracle; this search contract is not yet wired into the contact
residual or finite-sliding state machine.

For an existing first-order three-dimensional DOLFINx tetrahedral mesh, owned
exterior facets can enter the same contract without first replicating a global
surface object:

```python
partition = boundary_models.dolfinx_exterior_triangle_partition(domain)
search = boundary_models.routed_distributed_triangle_surface_bvh(partition, comm)
```

Imported physical groups and AgentFEM boundary regions retain their named
selection instead of requiring users to pass local facet indices manually:

```python
partition = boundary_models.dolfinx_tagged_exterior_triangle_partition(
    domain, facet_tags, tag=tool_tag, name="forming_tool"
)
# Or, when the named region already exists:
partition = boundary_models.dolfinx_boundary_region_triangle_partition(tool_region)
```

The adapter uses input-global vertex identities, orients every triangle away
from its adjacent volume cell, and permits a rank to own no selected facets.
It currently rejects non-tetrahedral, higher-order, interior, duplicate, or
non-integer facet selections instead of flattening them into an ambiguous
triangle soup. This is a geometry/search adapter, not yet a finite-sliding
contact pair or a STEP/STL repair pipeline.

The first finite-sliding State boundary can retain an accepted closest-point
map without confusing it with search or contact enforcement:

```python
projection_state = boundary_models.ContactProjectionState()
projection_state.begin(stable_contact_point_ids, outcome.projection)

# Accept only after the owning nonlinear increment is accepted.
projection_state.commit()
checkpoint_record = projection_state.snapshot()
```

Records are canonically ordered by stable 64-bit contact-point IDs. A later
trial may move to different stable surface-facet IDs, which is the expected
finite-sliding behavior, while silently changing the contact points, surface
identity, dimension, or coordinate convention is rejected. `rollback()`
discards only the trial; checkpoints are permitted only at an accepted
boundary. This is deliberately a solver-neutral State contract: it does not
yet assemble contact force, decide Newton search cadence, aggregate a global
MPI checkpoint, or claim a finite-sliding contact Procedure.

The solver-neutral lifecycle is the next ownership layer. It evaluates an
analytical surface or reviewed BVH on every nonlinear evaluation, retains
search diagnostics outside State, and accepts or rejects the projection with
the owning increment:

```python
lifecycle = boundary_models.ContactProjectionLifecycle(
    search,
    stable_contact_point_ids,
)
trial = lifecycle.evaluate(current_contact_point_coordinates)

if increment_converged:
    lifecycle.commit_increment()
else:
    lifecycle.rollback_increment()
```

Invalid projections fail closed by default. Callers may explicitly set
`require_all_valid=False` when `no_candidate` means an inactive point and the
future contact Operator is prepared to consume that status. Distributed search
backends reach a collective decision: one rank cannot reject a projection
while another continues into assembly. The lifecycle deliberately repeats the
exact projection at every evaluation. A search backend may later reuse a facet
candidate as a warm start, but signed gaps and normals are never reused as
stale physics.

`frictionless_penalty_contact_law(...)` evaluates the conservative local law on
that projection evidence. It works identically for analytical and reviewed
triangulated surfaces, reports both the deformable structural residual and the
conjugate prescribed-surface traction, and treats invalid points as errors
unless the caller explicitly selects the `inactive` policy. Its output is a
pointwise response, not an assembled finite-element contact element: a backend
must still own boundary trace interpolation, quadrature, assembly, and a
geometry-consistent linearization.

The next ownership boundary is `ContactTrace`: a backend supplies stable point
IDs, node interpolation, quadrature weights, and an explicit reference/current
measure declaration. `ContactTrace.evaluate(...)` produces the current slave
points; its assembly route then requires both the matching
`ContactProjectionRecord` and local penalty response before it will integrate
nodal residual, conservative potential, contact resultant, and optional tool
moment. Identity or coordinate mismatches fail before assembly. This reference
route still supplies no Newton linearization and is therefore not, by itself,
an implicit general-contact solver.

Pass `checkpoint=checkpointing.every(...)` to `model.step(...)` when a long
load path must be restartable. Only accepted load boundaries are published.
The portable checkpoint keeps the displacement, increment/cutback ledger,
next automatic increment, execution events, contact dual history, and accepted
work/energy history together;
restart rejects changed mesh/function-space identity, loads, constraints, time
inputs, or nonlinear controls. The same accepted state can be resumed with a
different compatible MPI partition or rank count.

This route has no surface search, finite sliding, friction, multiple contact
pairs, free rigid-body dynamics, or deformable-to-deformable coupling. The
moving surface remains one analytical plane. It rejects incompatible boundary
providers and constraint types before assembly. The penalty has units of
traction per length and must therefore be selected and checked by mesh
refinement for the problem at hand. General contact remains a separate future
provider.

## Declare the numerical unit contract

Finite-element kernels operate on consistent numbers. Record the convention
once so manifests, agents, datasets, and future interfaces do not infer units
from magnitude:

```python
model = models.create(
    study=studies.static_solid(dimension=3),
    mesh=domain,
    units=units.n_mm_mpa(),
)
```

`units.si()` and `units.n_mm_mpa()` are ready-to-use contracts;
`units.consistent(...)` records another coherent system. AgentFEM does not
silently convert material constants in this layer.

For imported simplex meshes, run `mesh.audit_quality(..., strict=True)` before
assembly. The report is collective under MPI and records the threshold and
number of poor/invalid owned cells.

## Go deeper

- [Materials and constitutive behaviors](materials.md)
- [Mesh interoperability](../mesh_interoperability.md)
- [Migrating Abaqus projects](../abaqus_migration.md)
- [Engineering loads, steps, and resultants](../engineering_workflows.md)
- [Abaqus C3D10H periodic cell](../abaqus_c3d10h_periodic_cell.md)
- [Example gallery](../examples/index.md)
