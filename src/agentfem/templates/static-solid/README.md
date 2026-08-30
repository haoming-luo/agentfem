# {{PROJECT_NAME}}

Run this installed AgentFEM project with:

```bash
agentfem check
agentfem run
```

The same case remains ordinary Python and can also be executed with
`python case.py`. Results are written below `outputs/`.

The generated `agentfem.toml` keeps local and cluster execution profiles
separate from `case.py`:

```bash
agentfem pack --output {{PROJECT_NAME}}.afm
agentfem run --project {{PROJECT_NAME}}.afm --profile cluster --mpi 8
```
