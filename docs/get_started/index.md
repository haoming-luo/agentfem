---
title: "Get started with AgentFEM"
description: "Set up an AgentFEM environment, create a first project and inspect simulation results through the documented finite-element workflow."
---

# Getting started

Install once, then **edit → check → run → view**. Start with a small working
project; learn the underlying concepts as your model needs them.

## Let your agent install it

Give your agent this prompt; add “use the mirror channel / 走镜像通道” if desired:

```text
Install AgentFEM using https://haoming-luo.github.io/agentfem/agentfem.json.
Detect my platform and reuse a working environment. Follow the linked route.
Complete the permitted steps; ask me only for approval or an action you
cannot perform. Check the installed environment, report its version, location
and readiness, then stop. Ask whether I want to try a first example before
creating a project or running a simulation.
```

Prefer manual setup? Use [installation](../getting_started.md), or the
[Windows WSL/Ubuntu route](../windows_agent_installation.md). After installation,
check the environment once with `agentfem doctor`. On WSL, prepare
`agentfem workspace --protect` once and keep projects in `~/AgentFEMProjects`.

Installation is complete when the environment is ready. You can stop here.
Your agent may offer a first example; it should wait for your choice.

## First project

When you want to try a small model, run these commands from your project directory:

```bash
agentfem init beam --template static-solid
cd beam
agentfem check
agentfem run --name baseline
agentfem show latest
agentfem verify
```

`check` checks project structure; `run` executes the model; `show` summarizes
the latest result; `verify` checks the saved record's provenance.
The generated `case.py` defines the physics, while `agentfem.toml` holds the
project name, entrypoint and output location.

## View and change the result

Open the reported XDMF file in ParaView. For displacement/stress plots, apply
Warp By Vector to the reference geometry and color by the stress field.
Keep the companion HDF5 file next to the XDMF.

```text
outputs/001-baseline/
├── execution.json       run status and failure details
├── result.json          quantities, fields, evidence and artifact locations
├── summary.md           readable summary
├── fields.xdmf          ParaView entrypoint for this template
├── fields.h5            field arrays
└── logs/                execution logs
```

To change the load or material, edit `case.py` and repeat `check` and `run`.
Each run gets its own folder. Use `agentfem runs` to browse earlier results,
or `agentfem show latest --files` to list artifact paths.
Other analyses may use different field filenames; follow the result record.

## Check what matters

For the first beam, compare displacement and reaction with the expected
physical behavior. For engineering conclusions, add the relevant reference
comparison and mesh/time convergence checks. A provenance check confirms the
saved record, not the physical accuracy of every model.

A healthy installation does not need to be checked or rebuilt for each edit.
Use `doctor` after environment changes or a runtime failure; use the
[upgrade guide](../project_upgrades.md) when changing versions or migrating an
older project.

## Where to continue

| If you want to... | Read... |
| --- | --- |
| Build a solid, thermal, or dynamic analysis | [User guide](../guide/index.md) |
| Understand equations and conventions | [Theory and conventions](../reference/theory_and_conventions.md) |
| Find a complete case close to your problem | [Examples](../examples/index.md) |
| Look up one function or field name | [Theory and reference](../reference/index.md) |
| Operate AgentFEM from an AI coding agent | [For AI agents](../agents/index.md) |
