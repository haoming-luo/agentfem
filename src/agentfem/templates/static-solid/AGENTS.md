# AgentFEM Case Instructions

- Export through `step.solve_result(output=...)`; for custom loops use
  `io.ResultWriter`. Do not hand-write connectivity, node permutations or XDMF
  blocks. Use the result manifest's recommended visualization artifact.
- Keep point/cell/quadrature fields distinct. Use explicit cell identities for
  external arrays, not equal-length assumptions or default interface smoothing.
- Apply Warp once to reference geometry, never again to already deformed output.

- Keep the finite-element sequence visible in `case.py`.
- Use named mesh regions and existing AgentFEM materials, constraints, loads,
  steps, and result helpers before writing UFL directly.
- For a model change: `agentfem check` → `agentfem run` → `agentfem show latest`.
- Use `agentfem doctor` for setup, environment changes or runtime failures.
  For a result-only request, view the saved result without starting a new run.
- Read `outputs/latest.json` and the referenced result manifest
  before reporting success.
- Do not describe a completed computation as verified unless the manifest
  contains accepted verification evidence.
- Store generated fields, plots, datasets, and reports through the active
  `project.RunContext`; do not write into the installed package.
