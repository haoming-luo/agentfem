# AgentFEM skill

The repository ships an AgentFEM skill that teaches compatible coding agents
the public workflow, project lifecycle, result contract, and safe extension
rules.

The canonical source is
[`skills/agentfem/SKILL.md`](https://github.com/haoming-luo/agentfem/blob/main/skills/agentfem/SKILL.md).
It is versioned with the code so that agent behavior can evolve alongside API
and project-format changes.

## What the skill is for

- discover and validate an AgentFEM installation;
- create and operate projects through the public workflow;
- interpret structured failures and simulation results;
- preserve the distinction between execution and scientific verification;
- route reusable finite-element functionality to the correct module;
- upgrade older projects without silently rewriting scientific intent.
- inspect declared private extensions without automatically executing or
  installing untrusted packages.

The skill complements the machine-readable `/agentfem.json` manifest and
scientific `knowledge/catalog.json`; it does not replace the public API or the
solver's deterministic checks.
