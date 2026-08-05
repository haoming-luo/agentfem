# AgentFEM Case Instructions

- Keep temperature constraints, heat sources, heat fluxes, and convection as
  distinct engineering concepts.
- Run `agentfem doctor`, `agentfem check`, and `agentfem run` in that order.
- Inspect the result manifest referenced by
  `outputs/{{PROJECT_NAME}}/latest.json` before interpreting the field.
- Preserve units and sign conventions in every change.
