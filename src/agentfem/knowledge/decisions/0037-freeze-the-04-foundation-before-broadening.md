# Freeze the 0.4 foundation before broadening the feature catalog

## Decision

The 0.4 line is an architectural consolidation, not a feature-count release.
Its stable middle layer is:

```text
Model -> Operator -> Procedure -> State -> Result / Verification
                         |
                      Backend
```

FEniCSx, PETSc, Basix, and MPI continue to own element tabulation, quadrature,
assembly, distributed degrees of freedom, and linear algebra.  AgentFEM owns
engineering semantics, numerical procedure, state lifetime, structured
evidence, and the boundary shared by people and agents.

The following changes are required before adding another broad physics family:

1. the eager internal import graph remains acyclic and layer violations fail
   CI;
2. operator selection is separate from primitive operator definitions;
3. provider discovery is lazy and cannot create registry import cycles;
4. nonlinear incremental procedures share one resource-safe Newton lifecycle
   while keeping constitutive equations, line search, and commit/rollback in
   their scientific owner;
5. finite-strain material providers have an optional atomic batch route with a
   scalar compatibility fallback;
6. execution events and fracture evidence are backend-neutral records rather
   than objects owned by PETSc solver or fracture god modules;
7. prepared numerical allocations have an explicit terminal lifetime;
8. CI chooses owner tests for local changes and reserves complete serial/MPI,
   installed-wheel, and platform suites for release boundaries.

## Non-goals

- no second finite-element kernel or speculative second backend;
- no new public configuration language;
- no universal Newton class that hides formulation-specific state semantics;
- no cache keyed only by Python object identity;
- no silent result, checkpoint, or material-state migration;
- no version promotion based on refactoring tests alone.

## Scientific identity and reuse

Reusable compiled forms, matrices, material artifacts, and checkpoints must be
keyed by durable scientific identity: formulation, mesh topology and geometry,
function spaces, regions, constraints, coefficients, provider revision, and
solver policy as applicable.  Reuse must be observable in performance evidence
and invalidation must fail closed.  Performance evidence never upgrades a
scientific trust state.

The v2 executable mesh identity therefore binds the active coordinate-element
family, variant, degree, mapping, and dimensions in addition to lossless
physical coordinates and ordered connectivity. Modal and harmonic identities
also retain the structured solution-element identity. Historical manifest
field names remain readable, but a changed identity schema produces a changed
fingerprint and cannot silently authorize checkpoint or allocation reuse.

## Verification ladder

Development uses the smallest owner suite first.  A stabilized architectural
slice then runs cross-owner workflow tests.  A 0.4 release candidate must pass:

- complete serial tests;
- representative two-rank state, nonlinear, output, and checkpoint tests;
- wheel installation and clean-environment acceptance;
- Linux and macOS acceptance, with Windows runtime acceptance tracked by its
  own product gate;
- unchanged public examples and compatibility imports;
- benchmark evidence appropriate to every capability whose maturity changes.

## Consequences

This decision deliberately slows namespace growth while making future mesh,
element, material, fracture, composite, and learned-constitutive work cheaper.
Large files are split only when ownership or dependency direction becomes
clear; line count alone is not a reason to manufacture another abstraction.
