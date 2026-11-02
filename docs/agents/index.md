---
title: "Use AI agents with AgentFEM finite-element projects"
description: "Connect AI agents to readable Python models and structured commands for setup, execution and result inspection, with explicit scientific checks."
---

# Agent entry

AgentFEM exposes one public workflow to people, scripts, IDEs, GUIs, and AI
agents. An agent should construct or revise the readable `case.py`, use the
structured CLI for operations, and accept results only through their explicit
status and scientific evidence.

An ordinary terminal is sufficient. For a direct tool connection, the optional
[AgentFEM MCP companion](mcp.md) exposes seven typed lifecycle operations
to Codex, Claude, and other compatible hosts while leaving the model source,
solver, and evidence in normal AgentFEM projects.

## Machine-readable entrypoints

For installation, use `installation.onboarding` in `/agentfem.json`. Inspect
the host OS and retain the user's selected mirror. Windows follows
[WSL → Ubuntu → environment](../windows_agent_installation.md);
Linux/macOS start at the environment. No repository checkout is a prerequisite.
Try each permitted stage once; on failure show its exact blocker and **one**
manual action. Resume after that blocker is resolved, not by reinstalling.
An AI chat without authorised terminal tools provides the same manual path.

| Resource | Purpose |
| --- | --- |
| `/llms.txt` | Short discovery document for language-model tools |
| `/agentfem.json` | Versioned documentation and command manifest |
| `installation` in `/agentfem.json` | Tested runtime and official/mainland-China environment commands |
| `src/agentfem/knowledge/catalog.json` | Scientific cards, formulas, evidence, consumers, and maturity |
| `agentfem doctor --json` | Environment capability check |
| `agentfem workspace --json` | Runtime-independent project custody check |
| `agentfem capabilities --json` | Public API, providers, maturity, and benchmark evidence |
| `agentfem check --json` | Static project and upgrade check |
| `agentfem run --json` | Addressable execution result |
| `agentfem inspect --json` | Result and artifact discovery |
| `agentfem telemetry status --json` | Exact privacy mode, queue and reviewed delivery routes |
| `agentfem telemetry route auto|global|china --json` | Optional transport preference without location inference |
| `agentfem inspect-abaqus model.inp --json` | Side-effect-free legacy-deck inventory |
| `agentfem inspect-user-material material.for --json` | Fingerprinted UMAT/UHYPER source inventory and migration route |
| `agentfem migrate-abaqus model.inp ./project --json` | Fail-closed Abaqus migration project |
| `agentfem lower-abaqus ./project --reviewed-by NAME --unit-system SI --json` | Inactive reviewed native draft for the eligible subset |

## Use only the steps the task needs

```text
Existing project: read case/instructions → edit if requested → check → run → inspect
Existing result:  show → answer the user's question
New installation: environment → doctor → readiness summary → offer an optional example
```

Reuse the environment and capability information already checked within the
task. Run `doctor` after environment/version changes or a runtime failure;
`check` already reports project migration findings. Use `upgrade` when those
findings need review, rather than adding a migration audit to every run.
Installation and workspace setup are one-time steps, not a routine preamble.
Stop when the environment is ready. Ask before creating or running a first
example unless the user already requested it. Suggest one useful next step;
do not automatically turn a suggestion into a task.

Start with the relevant example and `agentfem capabilities materials` (or
another topic). Read the full provider contract when needed. The installed
version's API is authoritative; online development documentation may be newer.
Architecture and extension guides are for extending the software, not a
prerequisite to run a beam example. See `task_guidance` in `/agentfem.json`.

Summarize the requested result, its artifact path and the relevant check.
`verify` checks provenance; numerical acceptance uses convergence, balance or
reference evidence suited to the model. Extra benchmarks, tuning and new
installations are separate tasks.

## Specialized operations

On WSL, an agent must not replace or unregister a distribution until
`agentfem workspace --json` reports that project data is protected from
distribution removal. Use `agentfem workspace --protect`; for the Complete
Runtime, use its lifecycle scripts rather than reproducing destructive WSL
commands.

For an Abaqus migration, inspect before creating a project. Preserve the
complete source graph and review every issue in `migration.json`; never remove
an element suffix, collapse Part/Instance scopes, or execute a generated
material candidate merely because its syntax was recognized.
When the native subset is eligible, record reviewer and unit interpretation,
inspect the inactive draft and decision fingerprint, and activate separately.
If a deck references user material, inspect its Fortran source separately;
never interpret an `adapter_candidate` report as executable compatibility.

## Start here

- [Installed project workflow](../getting_started.md)
- [Connect with AgentFEM MCP](mcp.md)
- [Agent and GUI integration](../agent_gui_integration.md)
- [Scientific trust and verification](../scientific_verification.md)
- [Project upgrades](../project_upgrades.md)
- [AgentFEM skill](skill.md)
- [Agent acceptance contract](acceptance.md)
