# AgentFEM Validation Guide

Every nontrivial change should be checked at three levels.

Model validation, solver convergence, and scientific verification are
different operations. `model.validate()` checks structural/model contracts;
`SimulationResult.status` describes execution;
`verification.VerificationReport` records reference, convergence, invariant,
cross-solver, or physical evidence. See
[Scientific Trust and Verification](scientific_verification.md).

Public model validation should use `model.validate()` when callers need a
complete structured report, or `model.check()` when execution must stop on
errors. New issues should define a stable `AFM-*` code, a scientific-object
path, a severity, and a repair hint where a local repair is meaningful.

Validation is capability-based rather than a registry-presence check. It must
find a target field compatible with the Study and verify that registered or
explicit materials implement the protocol consumed by the selected provider.
For example, steady heat requires a scalar temperature target and
conductivity; transient heat additionally requires volumetric heat capacity;
default structural dynamics requires elastic response and density. A complete
expert-supplied algebraic system may deliberately replace those model-owned
material operators.

## Structural Checks

- Imports use the current package layout.
- Public concepts appear in the correct module family.
- Application-specific code does not leak into the platform core.
- Mesh summaries and required tag checks are used when a workflow depends on
  material or boundary labels.
- Public dataclasses that represent reusable FEM assets provide inspectable
  summaries when practical.
- AF-IR exports are deterministic JSON, contain no non-finite values, and mark
  unresolved runtime objects as opaque.
- Backend capability claims are covered by tests and do not exceed the actual
  lowering implementation.

## Numerical Checks

- Run a small case with very few degrees of freedom or time steps.
- Check that output files are produced.
- Check that diagnostic values are finite.
- For MPI-aware code, run at least one small parallel smoke test when possible.
- For workflow examples, run the example and check that XDMF/HDF5 output is
  produced.
- For transient examples, verify that time-series fields are written at more
  than one time value.
- Never promote a fixed-mesh Golden into mesh-convergence evidence.
- Record an analytical reference's validity domain; mark an out-of-domain
  comparison inconclusive.

## Modeling Checks

- Units are explicit and consistent.
- Constraints, loads, and boundary models are not mixed.
- Required material and boundary tags are checked before weak forms are built.
- Time-step choices are connected to the relevant stability estimate.
- Mesh density is connected to wavelength, gradients, or geometry resolution.

## Agent Behavior

When an agent modifies AgentFEM, it should report:

- What concept was changed.
- Which files were edited.
- Which validation commands were run.
- What remains unverified.
- Which validation issue codes or AF-IR fields changed.

## Mesh Conversion Checks

For external mesh-format support:

- Import check: `from agentfem import mesh`.
- Optional dependency check: call `mesh.formats.require_meshio()` only when
  conversion is requested.
- Conversion smoke test: convert a tiny `.inp`, `.msh`, `.vtk`, or `.bdf` file
  to XDMF and then read it with `mesh.read_xdmf_mesh(...)`.
- Tag check: verify that expected cell or facet labels survived conversion.

## Campaign and Learning Checks

- Verify `SimulationResult -> declared QoIs -> ScientificDataset` without
  serializing live fields.
- Failed cases block dataset use by default.
- Use a named quality policy for release or learned-model datasets; a
  successful solve alone is not sufficient admission evidence. A raw minimum
  trust level remains available for a deliberately custom policy.
- Keep failed and inconclusive verification claims in provenance so a later
  training step cannot erase why data was rejected.
