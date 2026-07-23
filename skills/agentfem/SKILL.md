# AgentFEM Skill

Use this skill when building, reviewing, or extending finite-element simulations
with AgentFEM.

## Required Reading Order

When this file is installed as a Codex skill, the reference paths below are
relative to the skill directory `skills/agentfem/`. In the generated
documentation site, use the left navigation pages `Workflow`, `Concepts`, and
`Module Map` instead.

1. Read `references/workflow.md`.
2. Read `references/concepts.md`.
3. Read `references/module_map.md`.
4. Read focused references only as needed.

## Rules

- Keep the finite-element workflow visible.
- Identify the study context before selecting constitutive laws or operators.
- Inspect mesh summaries and required tags before building weak forms.
- Use lightweight models for registry, checks, summaries, and model-owned
  analysis steps; do not hide K/F systems or finite-element meaning.
- Prefer model registration helpers such as `model.field(...)`,
  `model.material(...)`, `model.fix(...)`, and `model.traction(...)` for
  application examples.
- Use `FEMProblem.summary()` or equivalent structured summaries when auditing a
  workflow.
- Use AgentFEM modules before writing ad hoc DOLFINx/PETSc boilerplate.
- Prefer the stable public workflow modules first: `studies`, `mesh`, `models`, `fields`,
  `materials`, `constitutive`, `constraints`, `loads`, `operators`,
  `problems`, `solvers`, `time`, and `io`.
- Prefer `step = model.linear_static_step(target=u)` when a model has
  registered materials, constraints, and loads. Use `model.stiffness(...)`,
  `model.external_force(...)`, and `operators.combine(...)` when an example
  must expose individual contributions.
- Keep operator notation such as `K = operators.stiffness(...)`,
  `F = operators.load_vector(...)`, and
  `step = problems.linear_static(K, F, study=..., ...)` available for
  transparent research/debugging examples.
- Prefer `step = problems.first_order_transient(...)` for first-order transient
  workflows instead of hand-combining effective matrices in tutorial code.
- Put local response relations under `constitutive/`.
- Put weak boundary physics under `boundary_models/`.
- Put Dirichlet, periodic, and MPC relations under `constraints`.
- Put Neumann, traction, flux, and body sources under `loads`.
- Validate small cases before reporting success.

## When Extending AgentFEM

Before adding a public helper, read `references/extension_rules.md`. If the new
helper is application-specific, keep it in the application package.
