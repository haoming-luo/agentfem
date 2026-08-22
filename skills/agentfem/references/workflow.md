# Workflow Reference

For an existing installed project, start with `agentfem doctor --json`,
`agentfem check --json`, and `agentfem upgrade --json`. Automatic migration is
limited to deterministic metadata. A finding with `semantic_review=true`
requires inspection and re-verification of the finite-element meaning.

AgentFEM workflow:

Public discovery is progressive. Begin with `agentfem.public_api("core")` and
`models.model_api("core")`. Methods reported by
`models.model_api("compatibility")` remain executable during 0.2.x but are not
the language for new cases. Built-in Step providers publish accepted and
required keyword names through `StepOptionContract`; inspect the same contracts
with `agentfem capabilities --json` before generating a Step call.
The CLI capability record and `/agentfem.json` documentation manifest are
generated views of the same dependency-free product contract; do not infer a
parallel workflow language from one example.

1. Study context: analysis type, physics, dimension, and assumptions
2. Solution procedure preference when more than one algorithm can solve the
   same physical equation
3. Mesh
4. Model registry when auditability is useful
5. Mesh summary and required tag checks
6. Function spaces
7. Fields and states
8. Constitutive laws
9. Amplitudes when prescribed data changes with time
10. Constraints
11. Loads
12. Boundary models
13. Operators or forms: model-first helpers for standard registered assets,
    operator-first constructors for explicit contributions
14. Analysis step through `model.step(...)` and an inspectable provider, or
    directly from visible operators for research/debugging
    - nonlinear subdivision belongs to `steps.automatic(...)` or the explicit
      compatibility mode `steps.fixed(...)`;
    - output intervals are a separate result request, not solver increments.
15. Structured validation and optional AF-IR record
16. Assembly, solve, or time integration
    - shared transient procedures may pause through `run(until_step=...)` and
      resume from `save_checkpoint(...)` / `load_checkpoint(...)`;
    - a resumed `solve_result(output=...)` produces an explicitly identified
      continuation segment, not reconstructed earlier frames.
17. `SimulationResult`, physical QoIs, diagnostics, and histories
18. `result.verify("exploratory" | "engineering" | "release")`, required
    outputs, and explicit scientific claims when the result is described as
    verified/validated
19. A declarative `results.output_plan(...)` for reusable field, history,
    diagnostic, and optional presentation requests
20. Compact unified XDMF/HDF5 artifacts attached to the result; use PVD/VTU
    only when specifically required
21. Optional runtime lock before a frozen or blind experiment

For related-case collections:

22. Typed parameter space and deterministic sampling plan, optionally from
    safe JSON
23. Fresh model construction per case
24. Scientific-input declaration and fingerprint-coverage review
25. Serial, spawned local-process, or externally sharded campaign execution
26. Optional multi-axis convergence certificate with explicit fixed coordinates
27. Optional response experiment lowered to baseline/perturbation Campaign cases
28. Named quality policy before dataset admission
29. Scientific dataset plus independent train/validation split
30. Surrogate/ROM validation, applicability guard, and explicit FEM fallback

Do not collapse these steps so aggressively that a reviewer cannot see the
finite-element model.

Use `examples/` as executable workflow references after identifying the target
problem type.
