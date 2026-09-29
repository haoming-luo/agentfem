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
