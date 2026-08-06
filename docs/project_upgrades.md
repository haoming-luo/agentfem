# Project Compatibility and Upgrades

AgentFEM projects are ordinary Python, but they are not anonymous scripts.
`agentfem.toml` identifies the entry point and an independently versioned
operational schema. This lets a terminal, GUI, or AI agent distinguish the
version of the installed package from the version of the project contract.

## Before running an existing project

Use three read-only checks:

```bash
agentfem doctor --json
agentfem check --project ./my-case --json
agentfem upgrade --project ./my-case --json
```

`doctor` reports the exact Python executable, imported AgentFEM directory,
installed-distribution directory, and whether a source checkout shadows the
environment. `check` validates the operational project, Python syntax, and
schema compatibility. `upgrade` adds a source-aware migration plan with stable
diagnostic codes and line numbers.

The upgrade status has four values:

- `current` — no known migration finding;
- `review_recommended` — the project remains runnable, but a newer public
  workflow is available;
- `migration_required` — the operational schema must be migrated;
- `blocked` — the project is newer than the runtime or its schema is invalid.

Advisory does not mean deprecated or wrong. Expert UFL and low-level I/O remain
valid for new physics and custom output. A finding means AgentFEM now has a
standard reusable path for that particular workflow.

## Safe automation boundary

```bash
agentfem upgrade --project ./my-case \
  --apply-safe \
  --write-plan upgrade.json
```

`--apply-safe` is intentionally narrow. It may add deterministic operational
metadata and creates an adjacent `.bak` copy before the first change. It does
not rewrite Python physics.

Changes involving regions, tags, loads, constraints, materials, weak forms,
constitutive laws, output meaning, or solver choices are marked
`semantic_review=true`. A human or coding agent can use the JSON plan, inspect
the named lines, run the case, and compare its scientific evidence. This is a
better automation contract than treating a successful text replacement as a
successful model migration.

## Compatibility policy

AgentFEM follows three rules while the public language matures:

1. keep package version and project-schema version independent;
2. warn and document before removing a public path whenever practical;
3. automate a migration only when finite-element meaning is preserved.

New installed-use templates record both `schema_version` and `created_with`.
The latter is provenance, not a requirement that a project only run with one
exact package release.

The policy follows established compatibility ideas from the
[Python backwards-compatibility policy](https://peps.python.org/pep-0387/)
and the dry-run/automatic-fix distinction used by
[Rust edition migration](https://doc.rust-lang.org/edition-guide/editions/advanced-migrations.html),
adapted to the stronger semantic risks of scientific simulation.
