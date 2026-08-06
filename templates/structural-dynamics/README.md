# {{PROJECT_NAME}}

This template uses an implicit Newmark structural-dynamics procedure. Change
the Study procedure to `generalized_alpha` or use the explicit-dynamics Study
when that numerical route is appropriate.

The compact starter publishes built-in mechanical-energy histories, a
user-declared `tip_U2` probe, and final displacement statistics. Adjust
`save_every`, `fields=...`, and `history=...` on `solve_result()` when more
field frames or compact engineering histories are required.

It is intentionally a small API and installation smoke case, not a spatial or
temporal convergence example.

```bash
agentfem check
agentfem run
```
