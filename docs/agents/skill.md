# AgentFEM skill

The repository ships a standards-compatible AgentFEM skill that teaches coding
agents the public workflow, project lifecycle, result contract, scientific
validation boundary, and safe extension rules.

The canonical source is
[`skills/agentfem/SKILL.md`](https://github.com/haoming-luo/agentfem/blob/main/skills/agentfem/SKILL.md).
It is versioned with the code so that agent behavior can evolve alongside API
and project-format changes.

The directory is self-contained: keep `SKILL.md`, `agents/`, and `references/`
together. A skill-aware agent can load it directly from the repository. For a
personal Codex installation, copy or link the complete `skills/agentfem/`
directory to `$CODEX_HOME/skills/agentfem/` (or
`~/.codex/skills/agentfem/` when `CODEX_HOME` is unset); copying only
`SKILL.md` omits the scientific and implementation references.

The main skill stays concise and routes detailed material progressively:
workflow guidance is read for modeling tasks, while concepts, module ownership,
validation, and extension rules are loaded only when the task requires them.

## What the skill is for

- discover and validate an AgentFEM installation;
- create and operate projects through the public workflow;
- interpret structured failures and simulation results;
- preserve the distinction between execution and scientific verification;
- route reusable finite-element functionality to the correct module;
- upgrade older projects without silently rewriting scientific intent.
- inspect declared private extensions without automatically executing or
  installing untrusted packages.

The skill does not guess API keywords. It begins with
`agentfem.public_api("core")`, `models.model_api("core")`, and the provider
contracts reported by `agentfem capabilities --json`. Compatibility names may
remain executable during 0.2.x, but new cases use the canonical vocabulary.

The skill complements the machine-readable `/agentfem.json` manifest and
scientific `src/agentfem/knowledge/catalog.json`; it does not replace the public API or the
solver's deterministic checks.
