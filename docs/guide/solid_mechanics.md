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

## Go deeper

- [Nonlinear solid architecture](../nonlinear_solid_architecture.md)
- [Nonlinear materials](../nonlinear_materials.md)
- [Engineering loads and resultants](../engineering_workflows.md)
- [Scientific function reference](../reference/scientific_function_reference.md)
- [Static-elasticity example](../examples/index.md#2d-static-elasticity)
- [Axisymmetric thick-cylinder source](https://github.com/haoming-luo/agentfem/blob/main/examples/axisymmetric_thick_cylinder.py)
