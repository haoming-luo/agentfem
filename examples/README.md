# AgentFEM Examples

These examples are release-facing workflow references. They should show the
finite-element sequence clearly and avoid exposing low-level DOLFINx details
unless the example is intentionally advanced.

## Recommended Reading Order

1. `static_elasticity_2d.py`
   Beginner template. Demonstrates the preferred MVP path:
   `Study -> Model -> Field -> Material -> Regions -> Constraints/Loads ->
   model.linear_static_step(...) -> solve -> XDMF`.

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

## Run

From the `agentfem` directory:

```bash
python examples/static_elasticity_2d.py
python examples/transient_heat_2d.py
```

From the parent development directory:

```bash
python agentfem/examples/static_elasticity_2d.py
python agentfem/examples/transient_heat_2d.py
```

Outputs are written to `examples_output/*.xdmf` and can be opened in ParaView.

## Style Rules

- Beginner examples should use `model.tree()`, `model.check()`, and
  model-owned steps when possible.
- Intermediate examples may show `operators` and `problems` directly when the
  algebraic system is the teaching point.
- Advanced examples may expose time-integration formulas, projections, and
  boundary-model internals, but should keep reusable operations in AgentFEM
  modules.
