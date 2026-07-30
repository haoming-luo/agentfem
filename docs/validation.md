# AgentFEM Validation Guide

Every nontrivial change should be checked at three levels.

Public model validation should use `model.validate()` when callers need a
complete structured report, or `model.check()` when execution must stop on
errors. New issues should define a stable `AFM-*` code, a scientific-object
path, a severity, and a repair hint where a local repair is meaningful.

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
