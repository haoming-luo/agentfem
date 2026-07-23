# AgentFEM Examples

These examples are written as workflow references for both researchers and
agents. Each one should expose the finite-element sequence clearly:

1. Mesh
2. Function spaces and fields
3. Constitutive law
4. Constraints, loads, and boundary models
5. Forms
6. Assembly and solve or time stepping
7. Diagnostics and output

## Examples

- `static_elasticity_2d.py`: small linear-elastic cantilever-style solve.
- `wave_packet_plate_2d.py`: simplified plate wave-packet propagation with
  source displacement, top/bottom periodic projection, and right absorbing
  boundary model.
