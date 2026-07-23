# AgentFEM Agent Guide

Use this guide when an AI agent is asked to build, review, extend, or debug a
finite-element simulation with AgentFEM.

## First Steps

1. Read `WORKFLOW.md` to identify the standard finite-element sequence.
2. Read `CONCEPTS.md` to align terminology before changing code.
3. Identify or create the `studies.Study` before choosing constitutive laws or
   operators.
4. Inspect or create a mesh summary before reasoning about boundary tags,
   material regions, or output dimensions.
5. Inspect the target application only after mapping it to AgentFEM concepts.
6. Prefer existing `agentfem` APIs before adding new helpers.
7. Add reusable code only when it belongs to a standard FEM concept.

## Progressive Reading

- Mesh or boundary tagging: read `docs/module_map.md`, then use
  `mesh.summarize_mesh`, `mesh.require_cell_tags`, and
  `mesh.require_facet_tags`.
- Study setup: use `studies.linear_static`, `studies.first_order_transient`, or
  `studies.second_order_dynamics` before building operators.
- Model audit: use `model.tree()`, `model.manifest()`, `model.summary()`, and
  `model.check()` when multiple fields, regions, loads, constraints, or steps
  must be inspected.
- Model registration: use `model.field(...)`, `model.material(...)`,
  `model.fix(...)`, and `model.traction(...)` in application workflows when the
  assets should stay visible and auditable.
- Application unknowns: use `fields.py` before dropping to `spaces.py`.
- Function spaces: inspect `spaces.py`; only inspect `kernel/dofs.py` for
  implementation-level dof work.
- Essential boundary conditions: read `CONCEPTS.md`, then use `constraints/`.
- Natural loads: read `CONCEPTS.md`, then use `loads.body_load`,
  `loads.neumann`, or `loads.boundary_load`.
- Constitutive laws: read `docs/extension_rules.md`, then use `constitutive/`.
- Absorbing or Robin-like terms: use `boundary_models/`.
- Assembly or lumped operators: inspect `assembly.py`.
- Time stepping: inspect `time/` and `problems.py`.
- Solves: inspect `solvers.py`.
- Analysis steps: prefer `model.step(...)` or `model.linear_static_step(...)`
  when a model owns fields, materials, loads, and constraints. Use
  `problems.linear_static` or `problems.first_order_transient` when a workflow
  intentionally starts from explicit K/C/F operators without model ownership.
- Problem summaries: use `problems.FEMProblem` when a workflow needs a
  broader structured audit record.
- Results: inspect `diagnostics.py`, then use `io.CSVLogger`,
  `io.XDMFTimeSeries`, or `io.ResultWriter`.
- Example workflows: inspect `examples/` after reading `WORKFLOW.md`.

## Agent Rules

- Do not mix constraints, loads, and boundary models.
- Do not hide the finite-element workflow inside overly broad abstractions.
- Keep application-specific geometry, source definitions, and parameter choices
  outside the core platform.
- Validate imports, syntax, and a small runnable case after structural changes.
- When changing a public concept, update `CONCEPTS.md`, `WORKFLOW.md`, and the
  relevant skill references.

## Current Platform Focus

AgentFEM currently focuses on DOLFINx/PETSc workflows, linear elasticity,
explicit dynamics, weak loads, absorbing boundary models, diagnostics, and XDMF
output. The architecture should remain open to thermal, multiphysics,
viscoelastic, anisotropic, and coupled simulations.
