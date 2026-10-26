---
title: "Get started with AgentFEM"
description: "Set up an AgentFEM environment, create a first project and inspect simulation results through the documented finite-element workflow."
---

# Getting started

This section takes a new user from an installed numerical environment to one
inspectable finite-element result. You do not need to understand the internal
architecture before running the first model.

## Let your agent install it

Give your agent this prompt; add “use the mirror channel / 走镜像通道” if desired:

```text
Install AgentFEM using https://haoming-luo.github.io/agentfem/agentfem.json.
Detect my platform and reuse a working environment. On Windows follow the
linked short WSL/Ubuntu route; do not clone the repository first.
Do the permitted steps. If a step fails, give me one simple manual action,
then continue after it is resolved. Finish with one installed example and
briefly report the version, result location and actual verification outcome.
```

This is agent-assisted installation, not a promise to bypass Windows consent
or restart. [Windows manual/agent path](../windows_agent_installation.md).

## Recommended sequence

1. Follow [Installation and first project](../getting_started.md) to create a
   compatible FEniCSx environment and install AgentFEM.
2. Run `agentfem doctor` to record the actual numerical stack.
3. On WSL, run `agentfem workspace --protect` once so projects survive runtime
   replacement or distribution removal.
4. Create a project with the `static-solid` template.
5. Read the generated `case.py`, then run `agentfem check` and `agentfem run`.
6. Inspect `result.json` and the field output instead of relying only on a
   successful process exit.
7. Continue to the [user guide](../guide/index.md) for the nearest physical
   problem and the [examples](../examples/index.md) for executable evidence.

## First project

```bash
cd ~/AgentFEMProjects  # recommended durable location on WSL
mkdir beam
cd beam
agentfem init --template static-solid .
agentfem check
agentfem run --name baseline
agentfem runs
agentfem show latest
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
outputs/001-baseline/
├── execution.json       process and failure record
├── result.json          quantities, fields, histories, trust, and artifacts
├── summary.md           disposable human view of result.json
├── fields.xdmf          field metadata and time-series structure
├── fields.h5            numerical field arrays
└── logs/                captured execution logs
```

The readable directory name is not the scientific identity. Every run still
carries an immutable, collision-resistant `run_id` in `execution.json` and
`result.json`. Numbered names help people browse; the internal ID protects
provenance and automation. Older `outputs/<project>/<run-id>/` projects remain
discoverable.

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
