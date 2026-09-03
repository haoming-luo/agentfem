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
