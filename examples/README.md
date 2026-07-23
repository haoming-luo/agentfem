# AgentFEM Examples

These examples are written as workflow references for both researchers and
agents. Each one should expose the finite-element sequence clearly:

1. Mesh
2. Regions for boundary and domain locations
3. Application unknowns with `fields`
4. Material properties and constitutive relations
5. Constraints, loads, and boundary models applied to regions
6. Operators such as `K`, `M`, `C`, and `F`
7. Assembly and solve or time stepping
8. Diagnostics and output

Beginner application examples should prefer operator-level language such as
`K x = F` and `LinearSystemProblem` before introducing weak-form details.
Use `constraints.fixed(...)` for application-level fixed-value boundary
conditions. For vector fields it fixes all components by default; pass
`components=0` or `components=(0, 1)` to constrain selected components.

## Examples

- `static_elasticity_2d.py`: small linear-elastic cantilever-style solve.
- `transient_heat_2d.py`: implicit-Euler transient heat-conduction solve.
- `wave_packet_plate_2d.py`: simplified plate wave-packet propagation with
  source displacement, top/bottom periodic projection, and right absorbing
  boundary model.
