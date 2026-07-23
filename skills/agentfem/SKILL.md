# AgentFEM Skill

Use this skill when building, reviewing, or extending finite-element simulations
with AgentFEM.

## Required Reading Order

1. Read `references/workflow.md`.
2. Read `references/concepts.md`.
3. Read `references/module_map.md`.
4. Read focused references only as needed.

## Rules

- Keep the finite-element workflow visible.
- Inspect mesh summaries and required tags before building weak forms.
- Use `FEMProblem.summary()` or equivalent structured summaries when auditing a
  workflow.
- Use AgentFEM modules before writing ad hoc DOLFINx/PETSc boilerplate.
- Put local response relations under `constitutive/`.
- Put weak boundary physics under `boundary_models/`.
- Put Dirichlet, periodic, and MPC relations under `constraints`.
- Put Neumann, traction, flux, and body sources under `loads`.
- Validate small cases before reporting success.

## When Extending AgentFEM

Before adding a public helper, read `references/extension_rules.md`. If the new
helper is application-specific, keep it in the application package.
