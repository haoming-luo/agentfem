---
title: AI-assisted finite-element simulation with Python and MCP
description: Create, run and inspect a cantilever model with AgentFEM. Learn where an AI agent fits into a readable Python finite-element workflow.
---

# AI-assisted finite-element simulation

You want to know how a component deforms under load. An AI assistant can help
prepare the model and organize the work, but the answer still needs a stated
geometry, material law, load, constraint, and numerical solve.

AgentFEM connects those two sides: an agent operates the workflow; the
FEniCSx-based numerical runtime evaluates the declared model. You can read and
edit the same Python project without using an AI service.

## Start with one cantilever

Install a compatible runtime using the
[installation guide](../getting_started.md), activate that environment, and
check it:

```bash
agentfem doctor
```

In a new, empty project directory, run:

```bash
agentfem init --template static-solid .
agentfem check
agentfem run --name baseline
agentfem show latest
agentfem verify
```

The template supplies a small linear-elastic structural example. Read its
`case.py` before changing it: that file is the model, not a hidden prompt or a
set of settings held by a chat application. The generated project README and
[solid-mechanics guide](../guide/solid_mechanics.md) explain the modeling
conventions.

## Ask the agent for a bounded task

Connect the official adapter using the
[MCP installation guide](../agents/mcp.md). It exposes seven lifecycle tools,
including project creation, checks, run submission, and result inspection.
The adapter requires the numerical runtime above; it does not install that
runtime for you.

Use a request with an observable outcome:

> Create a new AgentFEM static-solid project inside the approved project
> directory. Explain its dimensions, material, fixed boundary, load, and
> units before running it. Check the model, run the baseline, and show the
> displacement result together with the result file and convergence status.
> Do not replace existing projects.

The useful outcome is not just an answer in chat. It is a project you can
open, rerun, change, and share.

## What to look at after the solve

| Question | Where to look |
| --- | --- |
| What did we actually model? | `case.py` and named regions |
| Did the numerical process finish? | The run's execution record |
| What displacement and stress were computed? | `result.json` and its field artifacts |
| What checks support this result? | Convergence and verification evidence |
| How do I see the field? | The recommended visualization artifact; see [results](../guide/results.md) |

## Make one controlled comparison

For an unchanged, small-displacement linear-elastic model, doubling the applied
load should double displacement and stress. Ask the agent to prepare a second
case, preserve the baseline, and compare the same output definition in both.
This is a useful consistency check, not a substitute for mesh convergence or
experimental validation.

That workflow illustrates the role of AI here: less repetitive setup and
navigation, while the equations, assumptions, and numerical evidence remain
available to the engineer.

[Explore more cases](explore.md) · [MCP source](https://github.com/haoming-luo/agentfem-mcp)
