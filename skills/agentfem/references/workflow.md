# Workflow Reference

AgentFEM workflow:

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
17. `SimulationResult`, physical QoIs, diagnostics, and histories
18. `result.verify("exploratory" | "engineering" | "release")`, required
    outputs, and explicit scientific claims when the result is described as
    verified/validated
19. A declarative `results.output_plan(...)` for reusable field, history,
    diagnostic, and optional presentation requests
20. Compact unified XDMF/HDF5 artifacts attached to the result; use PVD/VTU
    only when specifically required

For related-case collections:

21. Typed parameter space and deterministic sampling plan, optionally from
    safe JSON
22. Fresh model construction per case
23. Resumable campaign execution; evaluators may return `SimulationResult`
24. Named quality policy before dataset admission
25. Scientific dataset plus independent train/validation split
26. Surrogate/ROM validation, applicability guard, and explicit FEM fallback

Do not collapse these steps so aggressively that a reviewer cannot see the
finite-element model.

Use `examples/` as executable workflow references after identifying the target
problem type.
