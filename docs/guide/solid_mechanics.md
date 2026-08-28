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
| Finite-strain logarithmic J2 | Experimental public path | Regional-material 3D affine-periodic cells with MPI state, output, and restart |
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

## Finite-strain periodic plasticity

`constitutive.finite_strain_j2_logarithmic(...)` is lowered by the ordinary
`model.step(...)` interface when a three-dimensional nonlinear-static model
contains compatible explicitly partitioned material regions and exactly one
`AbaqusPeriodicConstraint`. The current
route prescribes the macroscopic deformation gradient; it does not yet accept
body or natural loads. Serial execution uses exact affine reduction and MPI
execution uses the reviewed `dolfinx_mpc` reduction.

Accepted increments retain provider-owned quadrature fields `F`, `P`, `S`,
`MISES`, `SENER`, `ELENER`, `HARDENER`, `FP`, and `PEEQ`, with separately
named cell averages for visualization. For this material,
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
Zhang et al. periodic-composite gate still fails rather than being promoted. A
mixed displacement--pressure route for near-incompressible plasticity and a
production analytical tangent are not yet complete.

## Go deeper

- [Nonlinear solid architecture](../nonlinear_solid_architecture.md)
- [Nonlinear materials](../nonlinear_materials.md)
- [Engineering loads and resultants](../engineering_workflows.md)
- [Scientific function reference](../reference/scientific_function_reference.md)
- [Static-elasticity example](../examples/index.md#2d-static-elasticity)
- [Axisymmetric thick-cylinder source](https://github.com/haoming-luo/agentfem/blob/main/examples/axisymmetric_thick_cylinder.py)
