# AgentFEM 0.2.3

AgentFEM 0.2.3 adds a native small-strain axisymmetric solid route and
strengthens the scientific controls around implicit creep. A revolved solid is
now declared through the ordinary public workflow:

```python
study = studies.static_solid(dimension=2, assumption="axisymmetric")
model = models.FEMProblem("thick-cylinder", study=study)
```

The meridian unknown remains readable as `(u_r, u_z)`, while the formulation
retains full `(r, theta, z)` strain and stress components and lowers physical
integrals with the full-revolution `2*pi*r` measure. The same semantics apply
to stiffness, pressure and total-force loading, energy, result integrals,
standard field projection, J2 plasticity, and global power-law creep.

## Verification evidence

- The analytical Lamé thick-cylinder benchmark exercises the native
  axisymmetric kinematics, full tensor recovery, loading, and result measures.
  Four quadratic radial cells keep the sampled radial, hoop, and axial stress
  errors below `0.03%`.
- The NAFEMS R0027 Test 7 route now uses the native axisymmetric formulation.
  Its `0.5%` endpoint-integration contract is an AgentFEM acceptance criterion,
  not a tolerance prescribed by NAFEMS.
  The public Abaqus comparison reports differences rather than a pass/fail
  threshold: up to `1.84%` radial stress for CAX8R (`1.85%` for CCL24R) and
  `1.00%` hoop stress in its tabulated locations.
- The stateful thick-cylinder J2 path exercises first yield, quadrature state,
  nonlinear equilibrium, and serial/two-rank equivalence.
- The global implicit creep step now separates Newton convergence, accepted
  CEEQ increment control, and endpoint creep-rate time-integration accuracy.
  Every rejected attempt restores displacement, quadrature state, stress,
  tangent, load, temperature, and time atomically.

## Public modeling semantics

Models whose meridian reaches the axis register
`constraints.axisymmetric_axis(...)`, making radial regularity visible to
validation and agents. The long-cylinder specialization can declare
`constraints.axisymmetric_plane_strain(...)`. Negative radii are rejected, and
unsupported axisymmetric remote/distributing coupling semantics fail explicitly
instead of silently treating a revolved ring as a Cartesian face.

## Release boundary

This release establishes small-strain axisymmetric elasticity, J2 plasticity,
and power-law creep. It does not yet claim finite-strain axisymmetry,
axisymmetric contact, general ring/reference-point coupling, or a universal
acceptance tolerance for NAFEMS benchmarks.

The release candidate is accepted only after the complete serial suite,
distributed MPI tests, scientific-knowledge audit, strict documentation build,
distribution inspection, installed-wheel identity check, project-template
loop, and release workflow smoke tests pass.
