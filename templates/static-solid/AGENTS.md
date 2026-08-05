# AgentFEM Case Instructions

- Keep the finite-element sequence visible in `case.py`.
- Use named mesh regions and existing AgentFEM materials, constraints, loads,
  steps, and result helpers before writing UFL directly.
- Run `agentfem doctor`, then `agentfem check`, then `agentfem run`.
- Read `outputs/{{PROJECT_NAME}}/latest.json` and the referenced result manifest
  before reporting success.
- Do not describe a completed computation as verified unless the manifest
  contains accepted verification evidence.
- Store generated fields, plots, datasets, and reports through the active
  `project.RunContext`; do not write into the installed package.
