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
13. Analysis step from visible operators
14. Assembly, solve, or time integration
15. Diagnostics
16. Output

Do not collapse these steps so aggressively that a reviewer cannot see the
finite-element model.

Use `examples/` as executable workflow references after identifying the target
problem type.
