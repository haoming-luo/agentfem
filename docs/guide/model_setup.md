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
retains only the current reference-boundary normal component; general
nonlinear, moving-normal, damping, and contact foundations remain separate
future providers.

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
