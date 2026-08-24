# Dynamics and waves

AgentFEM separates the physical study from the solution procedure. Structural
dynamics can use an implicit Standard-like route or an explicit
central-difference route without changing the meaning of materials, regions,
loads, fields, and results.

## Current routes

| Procedure | Character | Typical use |
| --- | --- | --- |
| Newmark | Implicit | Structural response with controllable numerical parameters |
| Generalized-alpha | Implicit | Dynamics with high-frequency numerical dissipation |
| Central difference | Explicit | Wave propagation and short transient events |

The solution procedure and the constraint enforcement are checked together
before assembly. Projection periodicity is a serial, non-strict nodal
projection supported by central difference. Newmark and generalized-alpha do
not silently reinterpret it as a Dirichlet condition: model validation emits
`AFM-CONSTRAINT-PROCEDURE-001` and directs the user to an exact affine/MPC
backend. A serial-only constraint requested under MPI similarly emits
`AFM-CONSTRAINT-PARALLEL-001` before the solver starts.

`constraint.summary()` exposes the same capability contract to humans, agents,
and future GUIs. `PeriodicProjectionConstraint.diagnostics(field)` adds the
pair count, coordinate pairing error, unmatched count, and live field
mismatch. Exact rectangular matching-face construction is shared through
`constraints.rectangular_periodic_mpc`; consumers still select an MPC-aware
linear or nonlinear solver explicitly because ordinary DOLFINx and
`dolfinx_mpc` assembly are not interchangeable.

## Engineering questions to make explicit

- mass representation and density;
- damping model and whether it is physical or numerical;
- stable time increment for explicit analysis;
- source amplitude and time support;
- absorbing, periodic, or reflective boundary behavior;
- field sampling cadence versus integration cadence;
- kinetic, strain, external-work, and balance histories.

## Go deeper

- [Scientific operator contracts](../operator_contracts.md)
- [Stable steps and output](../step_and_output_architecture.md)
- [Wave packet with an inclusion](../examples/wave_packet_inclusion.md)
