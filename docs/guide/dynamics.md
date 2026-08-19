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
