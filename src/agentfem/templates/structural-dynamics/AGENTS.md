# AgentFEM Case Instructions

- Keep the physical dynamics Study separate from the Newmark,
  generalized-alpha, or central-difference procedure choice.
- Run `agentfem doctor`, `agentfem check`, and `agentfem run`.
- Inspect accepted increments and result artifacts in the published manifest.
- Do not change the time increment without considering temporal resolution and
  stability or accuracy requirements.
