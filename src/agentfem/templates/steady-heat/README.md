# {{PROJECT_NAME}}

This template demonstrates steady conduction with a prescribed temperature
and convection to an ambient environment.

```bash
agentfem check
agentfem run
```

Package the unchanged project for a PETSc/MPI server with
`agentfem pack --output {{PROJECT_NAME}}.afm`, then select the `cluster`
profile at execution time.
