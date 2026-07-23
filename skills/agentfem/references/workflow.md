# Workflow Reference

AgentFEM workflow:

1. Mesh
2. Mesh summary and required tag checks
3. Function spaces
4. Fields and states
5. Constitutive laws
6. Problem summary when auditability is useful
7. Constraints
8. Loads
9. Boundary models
10. Forms
11. Assembly
12. Solve or time step
13. Diagnostics
14. Output

Do not collapse these steps so aggressively that a reviewer cannot see the
finite-element model.

Use `examples/` as executable workflow references after identifying the target
problem type.
