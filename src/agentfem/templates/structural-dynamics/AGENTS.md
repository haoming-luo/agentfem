# AgentFEM Case Instructions

- Export through `step.solve_result(output=...)`; for custom loops use
  `io.ResultWriter`. Do not hand-write connectivity, node permutations or XDMF
  blocks. Use the result manifest's recommended visualization artifact.
- Keep point/cell/quadrature fields distinct. Use explicit cell identities for
  external arrays, not equal-length assumptions or default interface smoothing.
- Apply Warp once to reference geometry, never again to already deformed output.

- Keep the physical dynamics Study separate from the Newmark,
  generalized-alpha, or central-difference procedure choice.
- For a model change: `agentfem check` → `agentfem run` → `agentfem show latest`.
- Use `agentfem doctor` for setup, environment changes or runtime failures.
  For a result-only request, view the saved result without starting a new run.
- Inspect accepted increments and result artifacts in the published manifest.
- Do not change the time increment without considering temporal resolution and
  stability or accuracy requirements.
