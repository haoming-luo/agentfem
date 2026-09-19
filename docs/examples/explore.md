---
title: Explore AgentFEM — AI-assisted simulation and structural design
description: Try a structural-design demo, connect an AI agent to finite-element simulation, or generate a learning dataset with AgentFEM and FEniCSx.
---

# Explore AgentFEM

An engineering question can become a model, a collection of simulations, or an
interactive design tool. These three starting points show how those pieces fit
together. Choose the one closest to what you want to do today.

## See the design decision

**How much material can a support lose while limiting its deflection?**

Rotate a three-dimensional support, compare alternatives under the same load,
and inspect the weight–stiffness trade-off in the
[AgentFEM × GINO structural-design lab](structural_design_lab.md).
The browser demonstration requires no installation.

[Explore the interactive case](https://lab.haoming-luo.com/structural-design/)

## Give an AI agent a real simulation workflow

**Can an agent operate a finite-element model without hiding the model from you?**

The [AI-assisted finite-element quick start](ai_assisted_fem.md) follows one
cantilever from project creation to an inspectable result. The same project is
ordinary Python, whether a person or an MCP-compatible agent operates it.

[Start with a cantilever](ai_assisted_fem.md)

## Turn simulations into learning data

**How do repeated finite-element runs become a surrogate model?**

The [simulation-to-surrogate walkthrough](simulation_to_surrogate.md) varies
one material property, solves the corresponding models, and trains a compact
baseline. It is deliberately small enough to understand before scaling to
geometry variation, field prediction, or neural operators.

[Follow the data workflow](simulation_to_surrogate.md)

## Keep exploring

- [Executable example catalog](index.md): elasticity, heat transfer, dynamics,
  material behavior, and public benchmark comparisons.
- [Installation](../getting_started.md): Linux, macOS, and Windows through WSL2.
- [AgentFEM-Learning](https://github.com/haoming-luo/agentfem-learning): optional
  scientific-learning providers and their current evidence.
- [Source and community](https://github.com/haoming-luo/agentfem): code, issues,
  releases, and discussions.

AgentFEM is Apache-2.0 licensed and builds on FEniCSx/DOLFINx, PETSc, and MPI.
The examples show concrete workflows, not a claim that every engineering
problem has already been qualified.
