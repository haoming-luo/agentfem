# Decision 0015: One product-language contract

## Decision

AgentFEM declares its progressive public module tiers, Model verbs, CLI
commands, and machine workflow stages once in `_api_contract.py`. This module
has no numerical imports. Runtime discovery, capability JSON, generated
documentation, the reusable Agent Skill, IDE support, and future GUI clients
consume that contract instead of maintaining independent inventories.

The contract describes the discoverable product language; it does not contain
finite-element implementations, solver dispatch, or scientific maturity
claims. Step providers remain the executable authority for whether a specific
Study, field, material, procedure, and option set can actually be lowered.

## Why

AgentFEM is operated through Python, command-line tools, agents, documentation,
and eventually graphical clients. If each surface copies its own vocabulary,
the platform can appear simpler or more capable in one place than it is in
another. That is a product-integrity failure even when the numerical kernel is
unchanged.

A dependency-free contract makes discovery cheap and deterministic while
preserving lazy imports of DOLFINx, PETSc, MPI, visualization, and learning
stacks. It also gives human and machine clients the same progressive-disclosure
boundary: core first, advanced when needed, expert by deliberate choice, and
compatibility names only for migration.

## Consequences

- A new public module, Model verb, CLI command, or workflow stage must enter the
  shared contract and an executable consumer in the same change.
- Generated manifests and documentation are tested against the contract; a
  hand-maintained competing list is a defect.
- Compatibility names remain executable during their declared lifecycle but
  are excluded from new examples and beginner-facing generation.
- The contract must stay free of numerical imports and problem-specific case
  vocabulary.
- Scientific capability and maturity continue to be governed by providers,
  function cards, benchmarks, and verification evidence rather than mere name
  presence.
