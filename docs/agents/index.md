# Agent entry

AgentFEM exposes one public workflow to people, scripts, IDEs, GUIs, and AI
agents. An agent should construct or revise the readable `case.py`, use the
structured CLI for operations, and accept results only through their explicit
status and scientific evidence.

## Machine-readable entrypoints

| Resource | Purpose |
| --- | --- |
| `/llms.txt` | Short discovery document for language-model tools |
| `/agentfem.json` | Versioned documentation and command manifest |
| `src/agentfem/knowledge/catalog.json` | Scientific cards, formulas, evidence, consumers, and maturity |
| `agentfem doctor --json` | Environment capability check |
| `agentfem capabilities --json` | Public API, providers, maturity, and benchmark evidence |
| `agentfem check --json` | Static project and upgrade check |
| `agentfem run --json` | Addressable execution result |
| `agentfem inspect --json` | Result and artifact discovery |

## Safe operating sequence

```text
discover → doctor → init/open → inspect project → edit case.py → check
         → run → inspect structured result → verify policy → publish or revise
```

An agent must not infer scientific validity from a zero exit code. It should
inspect convergence, requested outputs, quality policy, applicability limits,
benchmark evidence, and any explicit failure record.

## Start here

- [Installed project workflow](../getting_started.md)
- [Agent and GUI integration](../agent_gui_integration.md)
- [Scientific trust and verification](../scientific_verification.md)
- [Project upgrades](../project_upgrades.md)
- [AgentFEM skill](skill.md)
- [Agent acceptance contract](acceptance.md)
