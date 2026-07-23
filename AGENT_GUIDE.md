# AgentFEM Agent Guide

Use this guide when an AI agent is asked to build, review, extend, or debug a
finite-element simulation with AgentFEM.

## First Steps

1. Read `WORKFLOW.md` to identify the standard finite-element sequence.
2. Read `CONCEPTS.md` to align terminology before changing code.
3. Inspect the target application only after mapping it to AgentFEM concepts.
4. Prefer existing `agentfem` APIs before adding new helpers.
5. Add reusable code only when it belongs to a standard FEM concept.

## Progressive Reading

- Mesh or boundary tagging: read `docs/module_map.md`, then inspect `mesh.py`.
- Function spaces or fields: inspect `spaces.py` and `dofs.py`.
- Essential boundary conditions: read `CONCEPTS.md` and use `constraints.py`.
- Natural loads: read `CONCEPTS.md` and use `loads.py`.
- Constitutive laws: read `docs/extension_rules.md`, then use `constitutive/`.
- Absorbing or Robin-like terms: use `boundary_models/`.
- Assembly or lumped operators: inspect `assembly.py`.
- Time stepping: inspect `time.py`, `runtime.py`, and `problems.py`.
- Solves: inspect `solvers.py`.
- Results: inspect `diagnostics.py` and `io.py`.

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
