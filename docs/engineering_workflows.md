# Engineering Loads, Steps, and Resultants

AgentFEM treats common CAE operations as reusable scientific assets rather
than case-local UFL fragments.

## Inspectable scientific formulas

Configuration files and AI agents often need to describe a source, boundary
value, or initial field as mathematics rather than executable Python:

```python
source = expressions.as_ufl(
    "2*pi**2*sin(pi*x)*sin(pi*y)",
    domain,
)
expressions.interpolate(
    temperature,
    "exp(-t)*sin(pi*x)*sin(pi*y)",
    parameters={"t": 0.25},
)
```

`ScientificExpression` is a deliberately small language. It accepts
`x/y/z/t`, declared parameters, arithmetic, and reviewed functions such as
`sin`, `cos`, `exp`, and `sqrt`. Arbitrary calls, attributes, indexing, and
Python statements are rejected before UFL compilation. Its summary retains
the original formula and language version, so an agent-readable input remains
inspectable instead of becoming an opaque callable.

The same validated formula has two deliberate execution routes. `as_ufl`
lowers symbolic physics into a variational form; `interpolate` evaluates known
loads, coefficients, initial values, and boundary data directly on NumPy
coordinate arrays without `eval` or a per-form C++ JIT. This keeps scientific
meaning identical while allowing many generated cases to reuse one compiled
finite-element operator.

This interface is for trusted mathematical structure, not for guessing a
missing equation. A formula that is inconsistent with its stated PDE remains
a model or dataset error.

## Sequential thermal--mechanical field handoff

Solve the thermal problem first on the same runtime mesh. Keep its accepted
temperature separate from the structural input field, then use the existing
engineering configuration and ordinary structural Step:

```python
thermal_result = thermal_model.step(target=T).solve_result()
T_solid = fields.temperature(domain, value=300.0)
solid_model.eigenstrain(eigenstrains.thermal(T_solid))
stage = solid_model.stage("thermal-to-solid")
stage.predefine(T_solid, T, method="interpolate")
with stage.field_transaction(displacement=u):
    result = solid_model.step(target=u, configuration=stage).solve_result()
```

The models, material, boundary conditions and `u` must be defined normally;
this fragment is a handoff, not another solver. `copy` (the default) requires
the identical function-space object. `interpolate` explicitly permits different
spaces on the same mesh with identical value shapes. Neither mode transfers
across meshes or silently smooths/project fields. Interpolation is not an
L2-conservative projection. Imported/reordered meshes need a separately
verified transfer method even when coefficient-array lengths match.

For a transient source, pass both `source_time=t` and `target_time=t` to
`predefine`. These are explicit caller-supplied physical times, checked for
finite equality; they do not infer a field's history or convert time units.
Omitted times are recorded as `unspecified`, appropriate for static snapshots.
Temperature values and reference temperatures must already use a consistent
unit convention. No unit conversion is guessed.

Assignments are staged before any target is changed and reject non-finite
values collectively. `model.step(configuration=...)` also restores the fields
if lowering fails. To cover downstream solve/result failure, the explicit
`field_transaction` above protects predefined targets and named unknowns; the
accepted thermal source is untouched. This is an in-memory nodal-field
transaction, not rollback of files, solver caches, or plastic material history.
All ranks must participate with the same field names, mesh, and declaration.
Backend failures within unfinished collectives need backend error handling.

Each new ordinary static structural Step assembles its operator for the supplied
temperature, including temperature-dependent material properties. Do not reuse
a manually prepared stiffness after changing temperature-dependent stiffness.
For existing transient Procedures, use the explicit time-input contract: an
eigenstrain-only update changes the RHS; material stiffness changes invalidate
the operator. `predefine` itself is a snapshot, not a live callback binding.

The result retains the construction-time configuration under
`metadata.engineering_step.field_transfers`, with source/target elements,
method, time declaration and mesh relation. This record is not a scientific
validation claim or a portable mesh fingerprint. Thermal restart uses the
thermal Procedure's existing checkpoint; after restoring, transfer the accepted
temperature at its accepted time and recompute the structural stage. No
combined multiphysics checkpoint or two-way coupled iteration is claimed.

Regression coverage: `tests/test_sequential_field_transfer.py` solves uniform
and linear steady temperature fields, checks free expansion and restrained
plane-strain stress, rejects mesh/component/time mismatch, verifies collective
rollback, and transfers a restored transient heat solution. The same tests run
with two MPI ranks. Multiple-material constitutive/eigenstrain semantics retain
their independent regression in `tests/test_thermoelastic_semantics.py`.
Two successive temperature-dependent-stiffness stages also check the stress
against the current elastic modulus, rather than only checking successful solve.

### Retry mechanics without repeating heat

The existing `examples/thermal_stress_wall_2d.py` now exposes a small restart
workflow using the same public Step API:

```bash
python examples/thermal_stress_wall_2d.py --smoke --output outputs/wall
python examples/thermal_stress_wall_2d.py --smoke --output outputs/wall --resume-heat
```

The first command saves the accepted thermal checkpoint and heat result. The
second restores that completed stage and solves elasticity again; it performs
zero thermal increments. Omit `--smoke` for the normal mesh/time horizon, but
do not mix smoke and normal checkpoints. Existing accepted heat is never
silently overwritten by a fresh run. The example checks its source-file hash,
AgentFEM version and smoke setting before restoration; a changed recipe needs
a new directory. This deliberately conservative example guard is not a general
semantic fingerprint for arbitrary scientific Python or external assets.

The structural result records the upstream checkpoint manifest SHA-256,
accepted time, recovery policy and thermal solve count. The checkpoint loader
validates its payload; the digest alone is not a scientific validation claim.
Existing portable checkpoint support is reused, not duplicated into a new
coupling archive. Both stages still need their own scientific acceptance.

If a structural trial is rejected inside `stage.field_transaction`, both the
nodal fields and the stage's prior transfer evidence are restored. Correct the
explicit cause and build a fresh structural Step; the accepted thermal source
remains unchanged. Do not reuse or publish a result object from the rejected
attempt. Files already written are not rolled back: use an attempt-specific
output path when preserving failed-attempt artifacts matters. This recovery
recipe applies to elasticity; history-dependent mechanics additionally needs
its own committed material checkpoint and is not covered by nodal rollback.

The regression now includes two thermal conductivities, two regional
temperature-dependent moduli and expansion coefficients, checking the exact
DG0 cell average of quadratic thermal stress. It also rejects a completed
structural trial on one MPI rank, retries without additional thermal solves,
and checks that accepted result metadata does not change with later transfers.

The one-way dependency is the same distinction used in the
[Abaqus analysis overview](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-solving.htm):
sequential thermal stress is appropriate when the thermal solution does not
require feedback from the mechanical response. See the
[bounded two-way design](coupling_design.md) before introducing feedback.

## Continuum loads

```python
model.elastic_foundation(on=base, stiffness=2.0e8, mode="normal")
model.centrifugal((0.0, 0.0, 120.0), center=(0.0, 0.0, 0.0))
model.hydrostatic_pressure(
    density=1000.0, gravity=(0.0, 0.0, -9.81),
    reference_point=(0.0, 0.0, 0.0), on=wetted_surface,
)
model.distributing_coupling(
    (0.0, 0.0, -50_000.0), moment=(2_000.0, 0.0, 0.0),
    reference_point=load_application_point, on=loaded_surface,
)
```

For ordinary self-weight, density belongs to the material and acceleration
belongs to the load:

```python
steel = model.material(
    elasticity.isotropic_elastic(
        young=210.0e9,
        poisson=0.30,
        density=7850.0,
    )
)
model.gravity((0.0, 0.0, -9.81))
```

`model.gravity(...)` creates the reference volume force density
`rho * acceleration`. With multiple registered materials it creates one load
per explicitly assigned material region, preserving each density. A raw force
density remains available as `model.body_force(...)`. Both routes enter the
same external-force operator, amplitude, Step activation, reaction, and work
contracts used by the selected procedure. Abaqus `*DLOAD, GRAV` migration can
lower one or more explicitly assigned three-dimensional solid-section regions;
parent or overlapping ELSET membership is never guessed.

The foundation contributes a boundary stiffness matrix. Centrifugal loading
uses registered material densities and regions. Hydrostatic pressure follows
`p = p_ref + rho g dot (x - x_ref)`. Distributing coupling constructs a
traction whose integrated force and moment equal the reference-point
resultants, avoiding a mesh-sensitive single solid-node force.

## Local coordinates and named reference points

```python
local = coordinates.cartesian(
    origin=(0.0, 0.0, 0.0),
    x=(0.0, 1.0, 0.0), y=(-1.0, 0.0, 0.0), z=(0.0, 0.0, 1.0),
    name="fixture",
)
rp = coordinates.reference_point((100.0, 20.0, 0.0), name="RP-1")

model.remote_force(
    (0.0, -50_000.0, 0.0), moment=(2_000.0, 0.0, 0.0),
    reference_point=rp, system=local, on=loaded_surface,
)
model.remote_displacement(
    U, reference_point=rp, translation=(0.0, 1.0, 0.0),
    rotation=(0.0, 0.0, 0.01), system=local, on=driven_surface,
)
```

Coordinate axes are validated as a right-handed orthonormal basis. Vector and
tensor transforms are public and inspectable. `remote_force` uses the existing
continuum distribution and preserves force and moment about the named point;
`remote_displacement` prescribes the corresponding rigid boundary motion and
participates in nonlinear load-factor ramping. It does not claim an unknown
reference-point degree of freedom or a general kinematic MPC.

## Step inheritance

```python
preload = model.stage("preload")
preload.activate_load(gravity)

service = model.stage("service", previous=preload)
service.activate_load(pressure)
service.deactivate_load("temporary_fixture_force")
service.deactivate_constraint("temporary_fixture")
service.predefine(temperature, initial_temperature)

step = model.step(target=U, configuration=service)
```

An `EngineeringStep` says which named loads and constraints are active. It is
separate from `steps.automatic(...)`, which controls increments, and from
`solvers.newton(...)`, which controls algebraic convergence. Only explicit
changes are recorded; other assets inherit from the preceding Step.

Lowering uses a shallow configured Model view. The source Model's load and
constraint registries are never temporarily replaced, while the executable
Step retains the exact active view used to produce its result. This makes
exception handling, concurrent campaign preparation, and provenance
inspection deterministic. Applying an explicitly declared predefined field
remains an intentional field-state operation.

## Engineering resultants

```python
section = results.section_resultant(S, on=cut, about=reference_point)
free_body = results.free_body_resultant(
    boundary_tractions=((traction, outer_boundary),),
    body_forces=((rho_g, volume),), about=reference_point,
)
path = results.sample_path(S, start=a, end=b)
```

Section force/moment, free-body force/moment, and path sampling are MPI-global
scientific quantities for verification, histories, campaigns, and learning
datasets. A fully kinematic reference-point MPC remains separate from the
implemented load-distribution contract.

## Repeated linear systems

For a transient or parameter loop whose left-hand side and constrained degree
set stay fixed, prepare the algebraic problem once:

```python
with solvers.prepare_linear_problem(a, L, T, bcs=bcs, options=options) as solve:
    for time_value in accepted_times:
        t.value = time_value
        solve.solve()  # refreshes the right-hand side; reuses A and the KSP
```

This is the lower-level counterpart of AgentFEM's managed transient Steps. It
keeps the mathematical forms public while avoiding repeated matrix assembly
and factorization in external protocol adapters.
