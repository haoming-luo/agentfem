# Solid mechanics

AgentFEM keeps the engineering model readable while allowing advanced
constitutive and weak-form work to descend into the reusable finite-element
layer.

## Current routes

| Route | Present maturity | Typical use |
| --- | --- | --- |
| Linear elasticity | Release workflow | Small-strain static solids and first models |
| Thermoelasticity | Engineering workflow | Temperature-driven stress after thermal analysis |
| Compressible Neo-Hookean | Engineering workflow | Finite-strain hyperelastic solids |
| Mooney--Rivlin | Experimental FEM workflow | Compressible 3D solids and incompressible plane-stress sheets |
| Mixed displacement-pressure Neo-Hookean | Experimental/engineering | Near-incompressible quadratic tetrahedral solids |
| Small-strain J2 plasticity | Engineering path | Stateful elastoplastic loading with consistent tangent |

## Modeling sequence

1. Select the physical study.
2. Create or import the mesh and name physical regions.
3. Declare the displacement field and material.
4. Apply constraints, tractions, body forces, pressure, foundations, or
   distributed resultants.
5. Select the solution step and incrementation policy.
6. Request standard fields and histories.
7. Accept the result only after the required verification policy passes.

## Go deeper

- [Nonlinear solid architecture](../nonlinear_solid_architecture.md)
- [Nonlinear materials](../nonlinear_materials.md)
- [Engineering loads and resultants](../engineering_workflows.md)
- [Scientific function reference](../reference/scientific_function_reference.md)
- [Static-elasticity example](../examples/index.md#2d-static-elasticity)
