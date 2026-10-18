# Solid mechanics

AgentFEM keeps the engineering model readable while allowing advanced
constitutive and weak-form work to descend into the reusable finite-element
layer.

## Current routes

| Route | Present maturity | Typical use |
| --- | --- | --- |
| Linear elasticity | Release workflow | Small-strain static solids and first models |
| Axisymmetric elasticity | Release workflow | Revolved solids using a readable 2D meridian and full 3D stress |
| Thermoelasticity | Engineering workflow | Temperature-driven stress after thermal analysis |
| Compressible Neo-Hookean | Engineering workflow | Finite-strain hyperelastic solids |
| Mooney--Rivlin | Experimental FEM workflow | Compressible 3D solids and incompressible plane-stress sheets |
| Mixed displacement-pressure Neo-Hookean | Experimental/engineering | Near-incompressible quadratic tetrahedral solids |
| Small-strain J2 plasticity | Engineering path | Stateful elastoplastic loading with consistent tangent |
| Finite-strain logarithmic J2 | Experimental public path | Strong-boundary solids with dead or current-configuration pressure loads, and regional 3D affine-periodic cells |
| Small-strain power-law creep | Engineering path | 3D or axisymmetric stateful creep with adaptive physical time |

## Modeling sequence

1. Select the physical study.
2. Create or import the mesh and name physical regions.
3. Declare the displacement field and material.
4. Apply constraints, tractions, body forces, pressure, foundations, or
   distributed resultants.
5. Select the solution step and incrementation policy.
6. Request standard fields and histories.
7. Accept the result only after the required verification policy passes.

## Exact periodic linear systems

Rectangular small-strain cells can use the same public Step grammar with an
exact distributed constraint:

```python
periodicity = constraints.rectangular_periodic_mpc(displacement)
result = model.step(
    target=displacement,
    constraints=periodicity,
).solve_result()
```

The linear provider separates ordinary Dirichlet data from exact MPC
elimination before assembly and supports the same route in serial and MPI.
Only one exact-MPC provider may own a linear system. The solved field and
constraint-construction diagnostics are retained in the Result. After a
converged solve, the provider recovers the owned slave multipliers from the
full residual, publishes a nodal `rectangular_periodic_mpc_reaction` field,
checks the constraint gap, and contributes its physical resultant and virtual
work to the common balance ledger. The current relation is homogeneous, so
its capability contract declares exact zero constraint work rather than
inventing an endpoint coordinate; affine macroscopic loading and nonlinear
path work use typed accepted-path evidence from the affine-periodic provider.
Solver convergence alone is not used to invent any missing dual quantity.

## Axisymmetric solids

Declare the formulation on the Study; do not manually insert cylindrical
weights into an ordinary planar model:

```python
study = studies.static_solid(dimension=2, assumption="axisymmetric")
model = models.create(study=study, mesh=meridian)
u = model.field(fields.displacement(meridian, degree=2))
model.material(constitutive.isotropic_elastic(young=E, poisson=nu))
model.pressure(p, on=inner_wall)
result = model.step(target=u).solve_result()
```

Coordinates and displacement are \((r,z)\); `S` and `E` are full
\((r,\theta,z)\) tensors. Pressure, traction, body loads, energy, and standard
projection automatically use \(2\pi r\). When calling a public result integral
directly, pass `study=study` to request the same physical measure. See the
[axisymmetric thick-cylinder source](https://github.com/haoming-luo/agentfem/blob/main/examples/axisymmetric_thick_cylinder.py).
If the meridian reaches `r=0`, register
`constraints.axisymmetric_axis(u, on=axis)`; model validation warns when that
regularity declaration is absent.

## Finite-strain J2 plasticity

The same material and `model.step(...)` vocabulary selects one compatible
equilibrium provider. Ordinary solids use strong Dirichlet or
remote-displacement constraints and may include body forces or natural loads
defined in the reference configuration:

```python
material = model.material(
    constitutive.finite_strain_j2_logarithmic(
        young=210e3,
        poisson=0.3,
        yield_stress=250.0,
        hardening_modulus=1e3,
    )
)
model.fix(u, on=clamp, value=(0.0, 0.0, 0.0))
model.body_force((0.0, 0.0, -body_force), target=u)
step = model.step(
    target=u,
    material=material,
    incrementation=steps.automatic(initial=0.1, maximum=0.2),
)
result = step.solve_result()
```

The ordinary finite-J2 Step also has a bounded experimental serial route for
registered, prescribed rigid contact pairs: analytical planes or spheres,
reviewed triangular surfaces, certified closed-convex triangular solids, and
explicitly reviewed closed nonconvex triangle solids in the later source
candidate. The latter requires an embedding attestation, rejects ambiguous
nearest features, and has a serial O(facets) winding-sign cost.
The contact residual and tangent enter the same Newton, line-search, accepted
State, work, and checkpoint lifecycle as the material. A certified convex
solid or a reviewed triangular tool on an active face-interior planar branch
can use the bounded penalty-Coulomb tangential law. For an open triangle
patch, active noncoplanar edges or vertices remain unsupported; the certified
closed-convex tool has a separate one-sided selected-feature branch. Curved
tool friction and distributed implicit friction remain unsupported. This is
experimental, not general frictional contact. Multiple named pairs and a
two-region J2 material map have serial regression evidence. Unsupported
geometry, material/contact composition, or MPI operation fails before solve;
do not infer general industrial-contact support from this route. See
`examples/finite_j2_closed_convex_contact.py` and
`examples/finite_j2_hertz_contact.py` for the supported shapes and the
remaining independent accuracy gap.

For a first- or second-order tetrahedral or hexahedral contact boundary, the
optional `contact_quadrature_degree=5` or `7` selects a positive Basix face
rule: 7 or 15 points per triangle, or 3×3 or 4×4 points per quadrilateral.
The CG1 defaults remain three-point triangles and 2×2 quadrilaterals; a CG2
trace retains its degree-four default. It is a convergence-study control, not an accuracy
preset: higher contact integration did not monotonically improve a finite-
block Hertz screen. The benchmark separately reports the analytic pressure
sampled on the actual trace and a nominal pressure difference, because a
close total force can hide an inaccurate pressure distribution. The chosen
rule likewise changes sampled physical-edge contact on a reviewed reentrant
tool; mesh and penalty sensitivity remain mandatory before engineering use.
The rule is part of the checkpoint identity and
is rejected for legacy plane contact and affine-periodic J2; no
unsupported combination silently falls back. Other weak boundary models and
formulations need their own reviewed lowering.

For the narrower experimental case of a frictionless, scalar-penalty analytical plane
against an affine CG1 tetrahedral slave boundary, the ordinary Step can select
`contact_integration="planar_active_partition"`. It partitions each
sign-crossing triangle at the *current trial* zero-gap contour and uses positive
reference-area quadrature on its active part. This avoids a documented blind
spot where a small contact strip falls between all ordinary face points.
The same trial partition supplies residual and tangent, and the choice is
checkpoint-identified. It is not an automatic fix for curved or faceted tools,
sharp features, friction, higher-order/quad faces, or MPI. Exported point IDs
in this mode identify one accepted snapshot, not a stable material point across
increments. Use independent mesh/penalty/increment checks before interpreting
local contact pressure or plasticity as engineering results.

The separate experimental `contact_integration="convex_feature_partition"`
option addresses a **certified closed-convex** rigid tool when an affine CG1
tetrahedral slave triangle has at most three positive-area dominant support
planes. It partitions both the active area and the compressed feature switch;
the latter contributes a line term to the Newton tangent. A moving/rotating
tool, force--moment work, total J2-plus-contact Jacobian, affine-periodic MPC
reduction, and interrupted restart have bounded regressions. The option
requires frictionless scalar-penalty contact, the default three-point
reference triangle rule, and a single MPI rank. More than three dominant planes
or an undefined boundary-aligned feature switch rejects the trial; no normal is
smoothed or third feature discarded. Accepted contact-point IDs are local to
that snapshot. This does **not** apply to concave STL geometry, nonlinear
nearest-edge/vertex distance branches, frictional edge transitions, or
general hard contact. Its successful small tests are not a penalty- or
mesh-convergence certificate.

The runnable, dimensionless
[feature-partition example](https://github.com/haoming-luo/agentfem/blob/main/examples/finite_j2_convex_feature_partition.py)
exercises the ordinary installed-package workflow; it is a regression fixture,
not an external forming benchmark.

Bulk and contact integration are separate choices. For an ordinary CG2
hexahedral finite-J2 displacement field, `model.step(...)` selects bulk
`quadrature_degree=4` by default; an explicit lower degree is rejected before
solving because it can leave displacement modes uncontrolled. This safeguard
does not establish mesh-independent contact accuracy or stability for every
material path.

### Affine-periodic cells

`constitutive.finite_strain_j2_logarithmic(...)` is lowered by the ordinary
`model.step(...)` interface when a three-dimensional nonlinear-static model
contains compatible explicitly partitioned material regions and exactly one
`AbaqusPeriodicConstraint`. This provider prescribes the macroscopic
deformation gradient; it does not accept
body or natural loads. Serial execution uses exact affine reduction and MPI
execution uses the reviewed `dolfinx_mpc` reduction.

A proportional endpoint may be passed as `deformation_gradient=F_end`. For
unloading, reloading, or a change of loading direction, declare the physical
matrix history explicitly:

```python
macro_path = constraints.deformation_gradient_path(
    (0.0, 0.35, 0.65, 1.0),
    (F_identity, F_loaded, F_unloaded, F_shear),
)
periodicity = constraints.abaqus_periodic_cell(
    displacement,
    nodes=nodes,
    equations=equations,
    anchor_node=anchor,
    reference_nodes=references,
    deformation_gradient_path=macro_path,
)
step = model.step(
    target=displacement,
    material=material,
    constraints=periodicity,
    incrementation=steps.automatic(initial=0.2, maximum=0.2),
)
```

The normalized step coordinate remains increasing while the physical
deformation-gradient components may reverse. The path must start at the
identity and preserve positive determinant throughout every linear segment.
Every path knot is a mandatory accepted boundary: fixed incrementation that
omits one fails before assembly, while automatic incrementation lands on it.
The complete path enters the constraint fingerprint and checkpoint identity.
Knots define the physical loading history; they are not, by themselves, a
constitutive-integration convergence study. Refine the accepted increments
between knots when plastic history accuracy matters.

Accepted increments retain provider-owned quadrature fields `F`, `P`, `S`,
`MISES`, `SENER`, `ELENER`, `HARDENER`, `PDENER`, `FP`, and `PEEQ`, with
separately named cell averages for visualization. For this material,
`SENER = ELENER + HARDENER`; it records recoverable elastic and hardening
storage, not accumulated plastic dissipation. The same accepted boundary can
be checkpointed and restored across a compatible MPI repartition. This route
remains experimental. The true spherical-void RVE has geometric pairing,
positive-J, Hill--Mandel, public-lifecycle, and two-rank execution evidence,
and now has a versioned fixed-stack Golden. That Golden freezes one
`h/L=0.25` first-order mesh, two accepted increments, the runtime stack, and
the portable mesh identity. It detects software drift in the macroscopic
first-Piola stress, physical-weighted PEEQ distribution, and solid fraction;
it is not a mesh-converged RVE reference solution. A separate opt-in check
compares two and four increments and two successive mesh levels. Passing that
check establishes only the declared successive-refinement stability, not
formal asymptotic convergence or a GCI uncertainty estimate. The independent
Zhang et al. periodic-composite gate still fails rather than being promoted.
Experimental serial 3D P2/DG0 and 2D plane-strain Q2/DPC1 mixed affine routes
now solve an independent tension-positive `MEAN_KIRCHHOFF_STRESS`, assemble
all four mixed Newton blocks, and keep the condensed `ELENER` representation
separate from `MIXED_POTENTIAL`. Their
thin-3D tetrahedral diagnostic remains distinct from the now executable 2D
Q2/DPC1 interpolation. The direct Zhang driver now evaluates the geometry,
stress, primal/condensed energy channels and a Schur-condensed current-state
effective tangent. Its homogeneous analytical tangent check has passed, but
the published Table 5 comparison, load-path and formulation convergence,
cell-replication checks, distributed mixed MPC/restart, and a production
mixed-formulation conditioning study remain promotion gates. The local
finite-strain J2 return now uses an analytical spectral `dP/dF` by default and
retains a central-difference oracle for verification.

## Go deeper

- [Nonlinear solid architecture](../nonlinear_solid_architecture.md)
- [Nonlinear materials](../nonlinear_materials.md)
- [Engineering loads and resultants](../engineering_workflows.md)
- [Scientific function reference](../reference/scientific_function_reference.md)
- [Static-elasticity example](../examples/index.md#2d-static-elasticity)
- [Axisymmetric thick-cylinder source](https://github.com/haoming-luo/agentfem/blob/main/examples/axisymmetric_thick_cylinder.py)
