---
name: agentfem
description: Install, build, run, inspect or extend AgentFEM finite-element projects using the public workflow and declared capabilities. Use for AgentFEM modeling, results, migration and solver extensions.
---

# AgentFEM

People and agents use the same readable `case.py` and installed CLI. A prepared
project runs without AI. Choose the shortest path that satisfies the request.

## Start with the user's task

- **Install:** read `installation.onboarding` at
  https://haoming-luo.github.io/agentfem/agentfem.json. Reuse a working environment
  and the user's chosen channel. Finish with installed identity, doctor readiness
  and WSL workspace protection. Report version and environment location, then
  offer an optional example. Create or run one only when requested; an existing
  request to install and try a model is sufficient.
- **Run or edit a project:** read its instructions, `case.py` and
  `agentfem.toml`; use `check → run → inspect`. Model edits follow the workflow
  reference below. `check` already includes migration findings.
- **View results:** use `agentfem show latest --json` and the recorded artifacts.
  Do not run a new simulation just to answer a result question.
- **Diagnose or upgrade:** inspect the specific failure or migration findings.
  Use `doctor --json` after environment/version changes or runtime failures.
- **Extend:** read the relevant ownership, validation and extension references
  before implementing numerical functionality.

Reuse context already checked within the task. Read one relevant capability
topic before requesting the full inventory. The installed API/options are
authoritative when online documentation describes a newer version.

## Reference Routing

References are relative to this skill directory. Keep the whole directory when
installing the skill; read only what the current task needs.

1. `references/workflow.md`: constructing or changing a model.
2. `references/concepts.md`: selecting physical objects or maturity levels.
3. `references/scientific_workflows.md`: material history, contact, thermal
   transfer, Abaqus migration, restart, campaigns and specialized output.
   Search for the current topic; this is not an installation checklist.
4. `references/module_map.md`: locating code and ownership.
5. `references/validation.md`: changing scientific code or evaluating a
   numerical claim.
6. `references/extension_rules.md`: adding a public helper, provider, material
   family or extension boundary.

## Common decisions

- Prefer Study → Model → materials/boundaries → `model.step` → Result.
  Keep physics in `case.py` and operational settings in `agentfem.toml`.
- Use public APIs and provider option contracts; keep lower-level operators and
  FEniCSx available when the task needs them. Reuse framework state, rollback,
  output and restart instead of writing a project-owned solver.
- Publish through Result and use its artifact paths; avoid case-owned
  connectivity or XDMF rewrites. `verify` checks provenance, while physical
  acceptance needs relevant reference, balance and convergence evidence.
- On WSL, protect project storage once with `agentfem workspace --protect`.
  Recheck custody before replacing/removing a runtime, not before every solve.
- If permissions or a private password entry are needed, give one concrete
  handoff. Resume from the completed stage instead of repeating installation.
- Offer one relevant next step after completing a task. Suggestions are not
  permission to run extra benchmarks, tune a model, install integrations or
  publish data. Keep support/account actions voluntary and explicitly requested.
