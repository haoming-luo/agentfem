# Engineering guides

Use these guides by physical problem, not by internal module name. Each page
identifies the public workflow, the currently supported route, and the deeper
reference needed when the default path is not enough.

## The public workflow

```text
Study → Model → Mesh/Regions → Fields → Materials → Loads/Constraints
      → Solution Procedure/Step → Output → Result/Verification
```

`Study` describes the physical analysis. `SolutionProcedure` describes the
numerical route. Loads, constraints, boundary models, materials, output, and
results remain distinct so that a model can evolve without becoming a single
problem-specific solver function.

## Browse by physics

<div class="grid cards" markdown>

- **Solid mechanics** — linear elasticity, thermoelasticity, finite strain,
  mixed hyperelasticity, and stateful J2 plasticity.
  [:octicons-arrow-right-24: Open guide](solid_mechanics.md)
- **Heat transfer** — steady conduction, implicit transient heat transfer,
  thermal fields, and thermo-mechanical handoff.
  [:octicons-arrow-right-24: Open guide](heat_transfer.md)
- **Dynamics and waves** — implicit dynamics, explicit central difference,
  sources, boundary models, and time-series output.
  [:octicons-arrow-right-24: Open guide](dynamics.md)
- **Creep and inelasticity** — global creep, material-point models, state,
  cutback, restart, and maturity boundaries.
  [:octicons-arrow-right-24: Open guide](creep_and_inelasticity.md)
- **Model setup** — meshes, regions, loads, constraints, boundary models, and
  multi-step activation.
  [:octicons-arrow-right-24: Open guide](model_setup.md)
- **Results and learning** — fields, histories, checkpoints, campaigns,
  datasets, PyTorch handoff, and surrogate fallback.
  [:octicons-arrow-right-24: Open guide](results.md)

</div>
