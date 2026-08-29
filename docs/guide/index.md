# User guide

The user guide is organized by engineering task rather than by Python module.
Each physical guide identifies the governing problem, current solution route,
public workflow, output, and maturity boundary. Mathematical definitions are
linked to the [theory and reference](../reference/index.md) section.

## Public workflow

```text
Study → Model → Mesh/Regions → Fields → Materials → Loads/Constraints
      → Solution Procedure/Step → Output → Result/Verification
```

`Study` describes the physical analysis. `SolutionProcedure` describes how the
load or time coordinate is advanced. Materials, loads, constraints, boundary
models, output, and results remain separate model assets.

## Analysis procedures

| Guide | Present scope |
| --- | --- |
| [Solid mechanics](solid_mechanics.md) | Linear elasticity, thermoelasticity, finite strain, mixed hyperelasticity, small-strain J2, and experimental finite-strain J2 providers |
| [Heat transfer](heat_transfer.md) | Steady conduction, implicit transient heat transfer, thermal fields, and thermo-mechanical handoff |
| [Dynamics and waves](dynamics.md) | Newmark, generalized-\(\alpha\), explicit central difference, wave sources, and time-series output |
| [Creep and inelasticity](creep_and_inelasticity.md) | Global creep, material-point models, quadrature state, cutback, restart, and maturity distinctions |

## Model definition

- [Meshes, loads, and constraints](model_setup.md) covers geometry, imported
  regions, material assignment, essential and natural boundary conditions,
  amplitudes, reference points, and multi-step activation.
- [Engineering workflows](../engineering_workflows.md) covers model assembly,
  solver-facing problem definitions, and reusable operators.

## Results and data

- [Results and post-processing](results.md) covers standard fields, histories,
  probes, resultants, output files, and visualization.
- [Results and campaigns](../results_and_campaigns.md) covers the structured
  result contract, repeated simulations, datasets, and quality policies.
- [Simulation to learning](simulation_to_learning.md) covers NumPy/PyTorch
  handoff, built-in surrogate baselines, existing user models, applicability
  guards, and FEM fallback.

## Nonlinear and coupled workflows

- [Nonlinear materials](../nonlinear_materials.md)
- [Nonlinear solid architecture](../nonlinear_solid_architecture.md)
- [Thermal stress and creep procedures](../solution_procedures_and_thermal_creep.md)

These pages are more detailed than the first application guides. They explain
state transactions, tangents, incrementation, rollback, restart, and the
boundary between an implemented finite-element route and a material-point tool.
