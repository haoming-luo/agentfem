# Workflow Reference

AgentFEM workflow:

1. Study context: analysis type, physics, dimension, and assumptions
2. Mesh
3. Model registry when auditability is useful
4. Mesh summary and required tag checks
5. Function spaces
6. Fields and states
7. Constitutive laws
8. Amplitudes when prescribed data changes with time
9. Constraints
10. Loads
11. Boundary models
12. Operators or forms: model-first helpers for standard registered assets,
    operator-first constructors for explicit contributions
13. Analysis step through `model.step(...)` and an inspectable provider, or
    directly from visible operators for research/debugging
    - nonlinear subdivision belongs to `steps.automatic(...)` or the explicit
      compatibility mode `steps.fixed(...)`;
    - output intervals are a separate result request, not solver increments.
14. Structured validation and optional AF-IR record
15. Assembly, solve, or time integration
16. `SimulationResult`, physical QoIs, diagnostics, and histories
17. Compact unified XDMF/HDF5 visualization/output artifacts attached to the
    result; use PVD/VTU only when specifically required

For related-case collections:

18. Typed parameter space and deterministic sampling plan, optionally from
    safe JSON
19. Fresh model construction per case
20. Resumable campaign execution; evaluators may return `SimulationResult`
21. Scientific dataset plus independent train/validation split
22. Surrogate/ROM validation, applicability guard, and explicit FEM fallback

Do not collapse these steps so aggressively that a reviewer cannot see the
finite-element model.

Use `examples/` as executable workflow references after identifying the target
problem type.
