# AgentFEM Examples

These examples are release-facing workflow references. They should show the
finite-element sequence clearly and avoid exposing low-level DOLFINx details
unless the example is intentionally advanced.

## Recommended Reading Order

1. `static_elasticity_2d.py`
   Beginner template. Demonstrates the preferred MVP path:
   `Study -> Model -> Field -> Material -> Regions -> Constraints/Loads ->
   model.linear_static_step(...) -> AF-IR record -> solve -> XDMF`.

2. `transient_heat_2d.py`
   Intermediate template. Demonstrates first-order transient heat conduction
   with visible `C`, `K`, history, source, time stepper, and output.

3. `wave_packet_plate_2d.py`
   Advanced explicit dynamics example. Demonstrates source amplitudes,
   central-difference integration, periodic projection, and absorbing boundary
   contributions.

4. `wave_packet_inclusion_2d.py`
   Advanced material-region example. Demonstrates a stiff circular inclusion,
   regional material assignment, explicit dynamics, periodic projection, and
   absorbing boundary handling.

5. `static_elasticity_surrogate_campaign.py`
   AI-native collection workflow. Demonstrates typed parameters, a
   reproducible design of experiments, fresh FEniCSx model construction per
   case, resumable execution records, a unit-aware scientific dataset, a
   transparent surrogate baseline, independent validation, and guarded
   high-fidelity fallback.

6. `campaign_from_json.py`
   Small declarative sweep. Parameters, sampling, outputs, and execution policy
   come from JSON; trusted scientific evaluation remains Python.

7. `material_models.py`
   Nonlinear-material scope example. Prints the maturity catalog and exercises
   a J2 uniaxial load path, power-law creep history, rainflow counting, and
   Miner damage.

8. `j2_plasticity_3d.py`
   Global path-dependent mechanics. Demonstrates quadrature-point state,
   analytical consistent tangent, automatic incrementation, rollback, and the
   standard progress stream.

9. `thermal_stress_wall_2d.py`
   Power-plant-oriented sequential coupling. One thermoelastic material feeds
   implicit heat transfer and the equivalent thermal-expansion load of a
   plane-strain stress solve.

10. `abaqus_c3d10_periodic_cell/agentfem_periodic_hyperelastic.py`
   Advanced interoperability and finite-deformation reference. Imports a real
   Abaqus `C3D10` mesh, preserves node labels, eliminates 4,212 periodic
   `*EQUATION` constraints, solves a 3D Neo-Hookean load path, and writes
   scale-one deformed VTU/PNG evidence. Read the folder README before treating
   it as an Abaqus comparison: the unavailable user material and user MPC are
   deliberately replaced by explicit AgentFEM semantics.

## Run

From the `agentfem` directory:

```bash
python examples/static_elasticity_2d.py
python examples/transient_heat_2d.py
python examples/static_elasticity_surrogate_campaign.py
python examples/campaign_from_json.py
python examples/material_models.py
python examples/j2_plasticity_3d.py
python examples/thermal_stress_wall_2d.py
```

From the parent development directory:

```bash
python agentfem/examples/static_elasticity_2d.py
python agentfem/examples/transient_heat_2d.py
```

Outputs are written to `examples_output/*.xdmf` and can be opened in ParaView.
The static example also writes `static_elasticity_2d.afir.json`, an
experimental versioned record of the supported scientific model structure
before execution.
The campaign example writes per-case evidence, a compressed dataset, and a
portable NumPy surrogate artifact beneath
`examples_output/static_elasticity_surrogate_campaign/`.

## Style Rules

- Beginner examples should use `model.tree()`, `model.check()`, and
  model-owned steps when possible.
- Intermediate examples may show `operators` and `problems` directly when the
  algebraic system is the teaching point.
- Advanced examples may expose time-integration formulas, projections, and
  boundary-model internals, but should keep reusable operations in AgentFEM
  modules.
