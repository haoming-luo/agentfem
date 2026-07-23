# AgentFEM Validation Guide

Every nontrivial change should be checked at three levels.

## Structural Checks

- Imports use the current package layout.
- Public concepts appear in the correct module family.
- Application-specific code does not leak into the platform core.

## Numerical Checks

- Run a small case with very few degrees of freedom or time steps.
- Check that output files are produced.
- Check that diagnostic values are finite.
- For MPI-aware code, run at least one small parallel smoke test when possible.

## Modeling Checks

- Units are explicit and consistent.
- Constraints, loads, and boundary models are not mixed.
- Time-step choices are connected to the relevant stability estimate.
- Mesh density is connected to wavelength, gradients, or geometry resolution.

## Agent Behavior

When an agent modifies AgentFEM, it should report:

- What concept was changed.
- Which files were edited.
- Which validation commands were run.
- What remains unverified.

## Mesh Conversion Checks

For external mesh-format support:

- Import check: `from agentfem import mesh_formats`.
- Optional dependency check: call `mesh_formats.require_meshio()` only when
  conversion is requested.
- Conversion smoke test: convert a tiny `.inp`, `.msh`, `.vtk`, or `.bdf` file
  to XDMF and then read it with `mesh.read_xdmf_mesh(...)`.
- Tag check: verify that expected cell or facet labels survived conversion.
