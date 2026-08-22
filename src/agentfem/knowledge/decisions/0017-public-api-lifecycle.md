# ADR 0017: Machine-readable public API lifecycle

Status: accepted

## Context

AgentFEM has one public engineering language for people, agents, IDEs, and
future GUI clients. During initial development, clearer verbs replaced some
historical `add_*` registrations and material/procedure-specific Step
factories. Keeping every spelling equally prominent would fragment the
language; removing working spellings without a migration contract would make
scientific projects unnecessarily brittle.

## Decision

1. The public Model vocabulary is classified as `core`, `advanced`, or
   `compatibility` from one dependency-free contract.
2. Every discoverable method carries a lifecycle, preferred replacement, and
   semantic-review flag through Python and capability JSON.
3. Compatibility methods remain executable throughout the 0.2.x line, but
   new examples and generated projects use the core language.
4. The upgrade tool detects compatibility calls and reports stable,
   addressable advice. It does not rewrite scientific Python when the change
   may alter Study, procedure, material, boundary, or solver meaning.
5. Public roadmaps describe direction and capability maturity. Exact release
   gates, risks, sequencing, benchmark tactics, and commercial decisions
   belong to the private engineering record.

## Consequences

- Humans and agents can distinguish a supported escape hatch from the
  recommended beginner language without relying on prose scattered across
  documentation.
- A future removal requires an explicit deprecation decision and migration
  evidence; a compatibility label alone is not a runtime warning.
- IDE and GUI integrations can consume the same contract as the CLI.
- Scientific migrations remain reviewable rather than being disguised as
  mechanical renames.

## Executable evidence

- contract tests assert the lifecycle and replacement for representative core
  and compatibility methods;
- capability JSON contains the same Model contract;
- the upgrade scanner reports a specialized Step call, marks semantic review,
  and leaves the source file unchanged;
- provider tests prove the recommended `model.step(...)` route bypasses
  compatibility methods.
