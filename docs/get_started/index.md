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

## Restricted homes and run diagnostics

For an agent, shared server or restricted home directory, use the same project
workflow with explicitly local runtime storage:

```bash
agentfem check --project . --local-runtime --runtime --writable --json
agentfem run --project . --local-runtime --trace-after 60 --json
```

`--local-runtime` places managed runtime state, XDG caches, Matplotlib
configuration/cache and temporary files under `.agentfem-runtime/` in the project.
It overrides their environment paths for this invocation and child processes;
Python callers recover their previous environment afterward. It does not relocate
installed packages, project results or arbitrary third-party write paths, and does
not enable telemetry. Use this option on both check and run. Add
`.agentfem-runtime/` to project version-control exclusions; keep `outputs/` when
archiving scientific results. An existing library imported by an embedded Python
caller may already have selected its cache: the installed CLI is the intended
fresh-process entry point.

`--writable` explicitly creates the declared storage directories and tests writes
with a temporary file. It does not execute `case.py`. A passed syntax/runtime check
is not a built or validated physical model. Use `model.validate()` after defining
materials, constraints and loads; then run one coarse case before a mesh study.
For the installed material signature use, for example,
`agentfem capabilities --api constitutive.isotropic_elastic --json`.
Density remains a required material property; it does not affect an ordinary
static case without inertia or gravity.

`--trace-after 60` writes repeated Python stack traces to standard error while a
run remains active, including a non-daemon thread join during shutdown. It does
not terminate the calculation or change its numerical settings. For a hard wall
clock limit and an independent observation of the child-process exit, use the
built-in supervisor:

```bash
agentfem run --project . --local-runtime --supervise --timeout 300 --trace-after 60 --json
```

For a distributed case, add `--mpi N` to the same command. The supervisor writes
`outputs/<project>/<run-id>/process.json` outside the numerical child process.
Its `status`, `return_code` and `process_exit_observed` establish whether the
process exited; `execution.json` and the result manifest establish whether the
case returned and result files were saved. A timeout is accepted only with
`--supervise`, so a wall-clock policy cannot be mistaken for a numerical setting.

A published result, return from `case.py`, and successful process exit are distinct.
Do not infer successful exit from the last JSON record alone. A supervisor timeout
is a failed process even when a previously written result is intact; inspect that
result separately. Preserve the execution record and diagnostic stderr before
retrying. The installed `agentfem templates` and `agentfem init --template ...`
provide version-matched starting cases; keep their visible modeling sequence.
