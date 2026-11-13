# Benchmarks

This package contains a machine-readable verification inventory through
`benchmarks.list_benchmarks()`.
`benchmarks.audit_capability_evidence()` joins that inventory to the
constitutive maturity catalog and reports any unsupported declaration.

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
- explicit evidence dimensions when the benchmark establishes MPI, restart,
  failure-behavior, or other evidence not unambiguously expressed by its level.

The current nonlinear-material obligations cover Neo-Hookean energy and a
nonlinear patch, J2 radial return, power-law creep closed forms, and
rainflow/Miner fatigue. External-mesh named-set preservation is also registered
as an interface benchmark. Abaqus equation parsing/chained affine reduction is
automated; the full 40k-dof C3D10H periodic finite-deformation case is a
documented manual regression because it performs six sparse nonlinear solves
and produces visual evidence.

## Readable physical models

Ordinary benchmark providers use public mesh, material, load, operator and
`model.step(...)` APIs. Physical geometry, assumptions, loading, observables
and reference formulas remain visible. Common connectivity, integration-point
sampling and accepted-state bookkeeping belong in the platform.

See [benchmark architecture](../../../docs/reference/benchmark_architecture.md)
for the audited module map and the explicitly retained independent UFL
references. An analytical oracle must not be rewritten to call the same
numerical implementation it is meant to check.
