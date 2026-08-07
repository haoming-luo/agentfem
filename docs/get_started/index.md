# Getting started

This section takes a new user from an installed numerical environment to one
inspectable finite-element result. You do not need to understand the internal
architecture before running the first model.

## Recommended sequence

1. Follow [Installation and first project](../getting_started.md) to create a
   compatible FEniCSx environment and install AgentFEM.
2. Run `agentfem doctor` to record the actual numerical stack.
3. Create a project with the `static-solid` template.
4. Read the generated `case.py`, then run `agentfem check` and `agentfem run`.
5. Inspect `result.json` and the field output instead of relying only on a
   successful process exit.
6. Continue to the [user guide](../guide/index.md) for the nearest physical
   problem and the [examples](../examples/index.md) for executable evidence.

## First project

```bash
mkdir beam
cd beam
agentfem init --template static-solid .
agentfem check
agentfem run
agentfem inspect
agentfem verify
```

The generated project remains ordinary Python:

```text
beam/
├── agentfem.toml
├── case.py
├── AGENTS.md
├── README.md
└── outputs/
```

`case.py` is the source of modeling truth. `agentfem.toml` contains operational
project information but does not duplicate the finite-element model.

## Files produced by a run

```text
outputs/<project>/<run-id>/
├── execution.json       process and failure record
├── result.json          quantities, fields, histories, trust, and artifacts
├── fields.xdmf          field metadata and time-series structure
├── fields.h5            numerical field arrays
└── logs/                captured execution logs
```

The exact artifacts depend on the analysis and output request. A completed run
is not automatically a scientifically verified result; convergence,
verification claims, and required artifacts remain separate evidence.

## Where to continue

| If you want to... | Read... |
| --- | --- |
| Build a solid, thermal, or dynamic analysis | [User guide](../guide/index.md) |
| Understand equations and conventions | [Theory and conventions](../reference/theory_and_conventions.md) |
| Find a complete case close to your problem | [Examples](../examples/index.md) |
| Look up one function or field name | [Theory and reference](../reference/index.md) |
| Operate AgentFEM from an AI coding agent | [For AI agents](../agents/index.md) |
