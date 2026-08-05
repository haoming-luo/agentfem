# {{PROJECT_NAME}}

This template uses an implicit Newmark structural-dynamics procedure. Change
the Study procedure to `generalized_alpha` or use the explicit-dynamics Study
when that numerical route is appropriate.

The compact starter publishes the accepted-increment history and final
displacement statistics. Use a dedicated transient `OutputPlan` when an
animation or additional displacement, velocity, acceleration, and energy
frames are required.

It is intentionally a small API and installation smoke case, not a spatial or
temporal convergence example.

```bash
agentfem check
agentfem run
```
