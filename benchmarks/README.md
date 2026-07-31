# Benchmarks

This package contains a machine-readable verification inventory through
`benchmarks.list_benchmarks()`.

Benchmarks are different from examples:

- Examples teach workflow usage.
- Benchmarks should have expected quantities, tolerances, and repeatable checks.

Add a benchmark here when it can validate a reusable AgentFEM capability rather
than demonstrate one application script.

Each registry entry must identify:

- a stable capability and benchmark ID;
- the verification level (material point, finite element, postprocess, or
  interface);
- a reference/formula/invariant;
- a numerical acceptance criterion;
- the automated test that supplies evidence;
- the current status.

The current nonlinear-material obligations cover Neo-Hookean energy and a
nonlinear patch, J2 radial return, power-law creep closed forms, and
rainflow/Miner fatigue. External-mesh named-set preservation is also registered
as an interface benchmark. Abaqus equation parsing/chained affine reduction is
automated; the full 40k-dof C3D10 periodic finite-deformation case is a
documented manual regression because it performs six sparse nonlinear solves
and produces visual evidence.
