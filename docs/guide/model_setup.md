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

## Add bounded contact with a fixed rigid plane

The first contact provider is deliberately narrow and inspectable: a
small-strain solid, one fixed plane, frictionless one-sided contact, and a
penalty selected by the user. The normal points from the obstacle into the
admissible half-space, while a positive `initial_gap` means that the reference
boundary is open:

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
energy plus contact potential. A time-varying load or nonzero prescribed motion
is reported as unavailable until its missing work channel is supplied; no
partial energy ledger is promoted as complete.

Pass `checkpoint=checkpointing.every(...)` to `model.step(...)` when a long
load path must be restartable. Only accepted load boundaries are published.
The portable checkpoint keeps the displacement, increment/cutback ledger,
next automatic increment, execution events, contact dual history, and accepted
work/energy history together;
restart rejects changed mesh/function-space identity, loads, constraints, time
inputs, or nonlinear controls. The same accepted state can be resumed with a
different compatible MPI partition or rank count.

This route has no surface search, friction, moving obstacle, or
deformable-to-deformable coupling. It rejects incompatible boundary providers
and constraint types before assembly. The penalty has units of traction per
length and must therefore be selected and checked by mesh refinement for the
problem at hand. General contact remains a separate future provider.

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
