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
assets share the same projection/evidence contract. Curved-normal geometry
currently requires a caller-reviewed explicit stability ceiling because the
automatic piecewise-planar bound does not include geometric stiffness.

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

An external triangle tool can enter the same route through the optional
`meshio` adapter:

```python
tool_mesh = boundary_models.triangulated_rigid_surface_from_mesh(
    "tools/punch.stl",
    coordinate_scale=1.0e-3,  # millimetres in the STL -> metres in the model
    name="punch",
)
```

The scale is mandatory because STL carries no dependable length unit. Import
records the source-file SHA-256 without making its machine-specific path part
of scientific identity. Exact duplicate STL vertices are welded and source
ordering is canonicalized so stable facet IDs do not depend on triangle-block
order. `flip_normals=True` is an explicit whole-surface orientation decision;
AgentFEM does not guess an admissible contact side.

The reviewed import boundary accepts triangle-only files. Mixed cells, volume
elements, non-manifold edges, inconsistent winding, degenerate facets and
duplicate facets fail before search or assembly. Approximate vertex welding,
hole filling and CAD repair remain explicit geometry-preparation steps rather
than silent solver behavior.

When this surface is attached to a `RigidBody` and no projector is supplied,
the explicit DOLFINx contact Operator selects the reviewed search backend:
a deterministic local BVH in serial and a partitioned, routed BVH under MPI.
The `RigidBody` therefore remains a Model asset containing geometry and motion;
search ownership stays in the Backend. An explicitly supplied compatible
projector still takes precedence for controlled studies.

Construction rejects duplicate/unreferenced vertices, degenerate or duplicate
facets, non-manifold edges and inconsistent shared-edge orientation. The result
preserves `no_candidate` and
`ambiguous_projection` instead of choosing a facet silently. The built-in
search checks every triangle and remains the correctness oracle for the
deterministic serial and distributed BVH routes. The explicit DOLFINx contact
Operator consumes the reviewed triangle/BVH contract; the nonlinear-static
`rigid_obstacle_contact(...)` route remains restricted to its analytical plane.

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
correctness oracle. The explicit DOLFINx contact Operator selects this routed
backend automatically for triangulated rigid tools under MPI.

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

The experimental serial finite-J2 Newton Step accepts independently named
frictionless rigid pairs on reference-area quadrature. It reprojects every
trial, sums the contact residual and tangent with the material Operator, and
owns accepted force--moment work, rollback, and checkpoint identity. A
normalized 0-to-1 prescribed motion shares the Step amplitude. Strong
prescribed displacement on the contact trace is lifted by the same Newton
correction as the material tangent. The serial affine-periodic J2 route
reduces the **total** material-plus-contact residual and tangent through one
MPC map; it also records each normalized moving tool's accepted work.

Open triangle patches still reject active free or noncoplanar edge branches.
The separate `convex_polyhedral_rigid_surface(...)` source candidate accepts
only a closed, outward-oriented, certified convex triangle tool. Its interior
gap follows one limiting face at a sharp feature; it does not smooth the
corner or admit arbitrary STL/CAD. Interior penetration checks every support
plane; inactive exterior points use an exact triangle-BVH closest-point search.
This is not a general nonconvex contact search. See
`examples/finite_j2_closed_convex_contact.py` and the
[closed-convex candidate record](../internal/closed_convex_implicit_contact_candidate_20261018.md).
Fixed, translating or rotating analytical-plane Coulomb pairs have a separate
narrow serial strong-BC route. The rotational friction tangent follows the
same master material point at fixed accepted State; its generally nonsymmetric
Newton derivative, force--moment work and interrupted restart have focused
tests. See `examples/finite_j2_moving_plane_friction.py` for the ordinary
`model.step()` path. A certified closed-convex tool also has a selected
piecewise-planar Coulomb branch under serial strong-BC finite-J2, including
one tested noncoplanar feature switch and interrupted restart. This is a
one-sided semismooth branch, not a smooth-normal law at a sharp edge.
A reviewed open triangle patch admits Coulomb friction only while every active
projection remains on its planar face-interior branch. Active free or
noncoplanar patch features, general nonconvex or curved-tool friction,
distributed implicit assembly, and independent industrial benchmarks remain
unsupported. The
candidate subset does not turn `ContactTrace` into a universal implicit
contact solver or promote released capability by association.

This is the current candidate's admission boundary, not an Abaqus-equivalence
matrix:

| Rigid geometry | Normal law | Coulomb friction | Affine-periodic J2 | MPI implicit J2 |
| --- | --- | --- | --- | --- |
| Analytical plane, fixed or prescribed motion | Serial supported | Serial strong-BC candidate | Frictionless only | Rejected |
| Reviewed triangle patch | Face-interior planar branch; active free/noncoplanar edges reject | Serial strong-BC face-interior planar branch only; active free/noncoplanar edges reject | Frictionless only | Rejected |
| Certified closed convex triangle solid | Selected one-sided sharp-feature branch | Serial strong-BC piecewise-planar candidate | Frictionless only | Rejected |
| Nonconvex/general mesh or deformable master | Rejected | Rejected | Rejected | Rejected |

All admitted implicit paths use finite-strain J2 on a three-dimensional solid.
Unsupported combinations fail during capability checking or before assembly;
they are not silently replaced with a plane, frictionless law, or serial solve.
Two region-assigned J2 parameter sets have been checked with a moving convex
frictional tool and interrupted restart. This does not admit arbitrary user
materials or certify general mixed-material interfaces.

After a serial implicit contact increment is accepted, the optional
`results.export_accepted_contact_points(step, directory, units=..., result=...)` writes
`accepted_contact_points.npz` and a JSON schema/identity record. Set
`vtk_view=True` to additionally write vertex-only `accepted_contact_points.vtp`
for ParaView at the **current slave positions**. The VTP carries the same raw
samples and pair indices; it does not make a continuous contact surface or
smooth pressure across facets. Arrays include
stable pair/point identity, current slave and closest master positions,
reference quadrature weights, normals, signed gaps, positive penalty pressure,
friction traction and accepted tangential state. The location is **reference
boundary quadrature**, with no cross-facet nodal smoothing; normal pressure is
force per reference area, not Cauchy pressure. The export refuses a modified
trial displacement, nonaccepted material coordinate, unsupported MPI path,
duplicate pair identity or an existing output. It does not advance State or
serve as a restart archive. Pair names and geometry fingerprints must accompany
point IDs when comparing archives. For example:

```python
from agentfem import results

paths = results.export_accepted_contact_points(
    step, "output/contact-points", units={"length": "m", "force": "N"},
    result=result, vtk_view=True,
)
```

Unknown units should be omitted, never inferred from the mesh coordinates.
Passing `result` attaches the archive and raw quadrature pressure/traction
field descriptors to that `SimulationResult`, plus the optional VTP artifact;
it does not create a smoothed ParaView nodal field.

For the reviewed DOLFINx route,
`dolfinx_boundary_region_contact_trace(region, V)` adapts a tagged exterior
boundary on a tetrahedral or hexahedral mesh and a continuous blocked vector
CG1 or CG2 space. It restricts the cell basis to positive triangle or
quadrilateral facet quadrature and evaluates the physical surface Jacobian
from complete Lagrange first- or second-order coordinate geometry. The trace uses
partition-independent facet/point identity and evaluates current positions
from a synchronized DOLFINx displacement field. Q2 hexahedra pass the complete
Explicit route. A standard P2 tetrahedral trace is available for interpolation
and geometry evidence, but its non-positive row-sum mass is rejected before
central difference; use a reviewed positive mass-lumped simplex element before
claiming high-order tetrahedral Explicit dynamics. Prism/pyramid mixed-facet,
scalar, discontinuous, and non-blocked spaces remain rejected collectively.

That trace now has one bounded Procedure consumer:
`dolfinx_explicit_contact_residual(...)`. It combines an ordinary bulk
residual with exact projection, the normal penalty law, optional
penalty--Coulomb friction, and trace integration before the existing
central-difference acceleration update. In
MPI, contact contributions are first assembled into a separate zeroed ghosted
vector, reverse-added to their owners, and only then added to the already
assembled bulk owned entries. This ordering prevents a second accumulation of
bulk ghost contributions.

Pass the Step's `LumpedMassOperator` as `lumped_mass=` to screen the additional
contact stability limit automatically. AgentFEM forms a conservative
mass-scaled absolute row-sum bound from trace interpolation, quadrature,
normal penalty, optional tangential penalty, and Coulomb pressure-cap coupling,
reverse-accumulates ghost rows, and records the selected limit. The reviewed
automatic path accepts fixed-normal and piecewise-planar projectors; curved
normal geometry requires an explicit ceiling until geometric stiffness is
included. A caller may instead provide
`maximum_stable_time_increment`. For automatic whole-system selection, also
pass the unsafed body/material/interface limit as
`noncontact_unsafed_stability_limit=`;
AgentFEM converts it back to a spectral upper bound, adds the contact
contribution, and only then applies the safety factor. Taking the smaller of
separate body and contact time limits is not conservative. If a caller ceiling
is also supplied, the stricter combined limit wins. Construction of an
explicit Step rejects a larger `dt`.
This is intentional: penalty contact adds stiffness and can reduce the
critical central-difference step below the body-wave estimate. A fixed surface
remains the default. For one proportional
prescribed rigid path, construct `prescribed_rigid_motion_schedule(...)` and
bind it with the scientific surface in `rigid_body(...)`. Pass that immutable
asset with the slave `BoundaryRegion` and scalar penalty in
`rigid_contact_pair(...)`, then pass the pair as `contact_pair`. The Backend
selects a serial or routed distributed BVH for triangulated tools unless an
explicit compatible projector is supplied. The schedule
states physical start/end times explicitly instead of treating solver time as
an unnamed load factor. The pair and rigid body's surface, motion, reference
point, local law, and stable scientific identities enter result and checkpoint
evidence. The earlier direct `projector` plus `motion_schedule` form remains
available as a lower-level route.

The schedule defaults to a linear factor. For a C2 loading transition, pass
`amplitude=amplitudes.quintic_step(start_time=..., end_time=...)` with the same
physical interval. It must run from factor zero to one; sampled factors outside
`[0,1]` are rejected. `amplitudes.smooth_step` retains its half-cosine definition.
The explicit amplitude summary participates in restart identity.

Moving contact records an MPI-global initial force/moment station and one
station at every accepted increment. Generalized tool work is integrated from
force--translation and moment--rotation pairs by the trapezoidal rule. Rejected
increments discard their trial station; checkpoints retain only the accepted
path and verify the schedule identity before restore. The same record appears
in the residual summary and therefore in the Step/SimulationResult evidence.
The ordinary throttled transient progress event also reports pair count,
active-point count, maximum penetration, contact-force norm, accepted tool
work, and latest interval power. These are scalar metrics in the bounded
execution-event recorder; they
do not retain per-point arrays at every increment or bypass `print_every`.
Several bounded contact pairs can be composed by using one contact
residual as the next pair's base residual. Time updates, commit/rollback,
per-pair evidence, summed contact potential/work, and nested checkpoint State
then follow the same residual chain; every pair should have a distinct name
and schedule identity. A serial acceptance combines an analytical tool and a
triangulated tool through this same route; no two-tool special solver is used.

The same explicit consumer also accepts the reviewed serial or routed
distributed triangle BVH as its projector. A moving triangulated rigid tool is
reprojected at every residual evaluation, so stable slave-point IDs may change
their closest surface-facet IDs during finite sliding. Serial and two-rank
tests cover that facet crossing, packed rank-AABB routing, rank-canonical
accepted work, and the absence of spurious normal work under purely tangential
tool translation. Projection remains evaluation-local and is recomputed after
restart from accepted displacement and schedule time. The contact-only
stability screen is conservative rather than an exact global eigenanalysis.
This route still does not claim arbitrary topology, multiple
distributed tools in one broad phase, free rigid-body dynamics, or an implicit
consistent tangent.

The friction extension starts with a solver-neutral scientific contract rather
than a hidden extension of that residual. `penalty_coulomb_friction_law(...)`
implements elastic stick and return to the Coulomb cap at already projected
points. `TangentialContactState` keeps stable point IDs, transported
tangential elastic slip, cumulative irreversible dissipation, and atomic
trial/commit/rollback state. Its response reports recoverable tangential
penalty energy, sliding dissipation, and penalty energy released at contact
opening as three different quantities. A `RigidContactPair` may declare this
law with `friction_coefficient=` and `tangential_penalty=`, so the Model and its
scientific identity are complete. The DOLFINx explicit residual consumes that
declaration. It computes relative slave-minus-master motion against the same
rigid material point for fixed, translating, and rotating tools; stages point
history with the time increment; and assembles tangential force and moment
without changing the Model API.

`ContactTraceEvaluation.assemble_friction(...)` is the first completed piece
of that Operator promotion. It integrates tangential residual traction and
rigid-body force/moment with the same interpolation and quadrature contract as
normal contact, while keeping recoverable tangential penalty energy,
irreversible sliding dissipation, and separation release in named channels.
The explicit Procedure now supplies relative slip and accepted-state
commit/rollback. The trace itself remains free of time integration. This follows
the penalty-friction distinction between elastic stick and the Coulomb cap in
the [Abaqus friction theory](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-coulombfric.htm)
and the separate contact-energy categories in its
[contact energy example](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEEXARefMap/simaexa-c-contactenergy.htm),
without copying Abaqus-specific solver ownership into AgentFEM.

The parallel vector ordering follows PETSc's finite-element contract:
[`ADD_VALUES` with reverse scatter](https://petsc.org/release/manual/vec/)
accumulates ghost contributions onto the owning rank. Penalty-contact effects
on explicit stability follow the central-difference spectral restriction
described in the
[Abaqus explicit-dynamics theory](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-expdynamic.htm)
and its warning that penalty enforcement may reduce the stable increment in
the
[explicit-contact documentation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-expcontactconstraints.htm).
The interface contribution is also discussed, for example, by
[Otto et al. (2020)](https://doi.org/10.1002/nme.6264).

Pass `checkpoint=checkpointing.every(...)` to `model.step(...)` when a long
load path must be restartable. Only accepted load boundaries are published.
The portable checkpoint keeps the displacement, increment/cutback ledger,
next automatic increment, execution events, contact dual history, and accepted
work/energy history together;
restart rejects changed mesh/function-space identity, loads, constraints, time
inputs, or nonlinear controls. The same accepted state can be resumed with a
different compatible MPI partition or rank count.

This route has no free rigid-body dynamics, deformable-to-deformable
coupling, implicit consistent linearization, or shared broad phase for several
distributed tools. Its reviewed moving geometries are one analytical surface
or one oriented triangulated rigid surface per pair. It rejects incompatible
boundary providers and constraint types before assembly. The penalty has units
of traction per length and must therefore be selected and checked by mesh
refinement for the problem at hand. For frictional pairs the automatic contact
screen covers both normal and tangential penalty stiffness, but penalty and
time-step sensitivity still require refinement evidence. General industrial
contact remains a separate scientific promotion until an external sliding
benchmark and forming-relevant validation pass.

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
