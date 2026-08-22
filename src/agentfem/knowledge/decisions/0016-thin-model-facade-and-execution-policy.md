# ADR 0016: Thin Model facade and one execution-policy snapshot

Status: accepted

## Context

`Model.step(...)` is the stable public language for selecting an analysis
procedure. As the scientific core grew, material- and procedure-specific
construction accumulated in the same `Model` class even though providers
already owned capability selection. Solver, output, history, progress, and
checkpoint controls also existed as mature individual objects but lacked one
shared inspection boundary.

Leaving both trends unchecked would turn `Model` into a second solver layer
and force agents, IDEs, GUIs, result lifecycles, and provenance tools to infer
cross-cutting execution intent from unrelated keywords.

## Decision

1. `Model` remains the public registry and facade. It owns engineering assets,
   validation, model-first operators, Step registration, and `model.step(...)`.
2. Built-in material/procedure construction moves to internal scientific
   builders selected by `step_providers.py`. Public providers call those
   builders directly. Historical Model builder methods remain thin 0.2.x
   delegates and do not retain parallel implementations.
3. `StepRequest` normalizes declared solver, output, history, progress, and
   checkpoint controls into one immutable `StepExecutionPolicy`.
4. The policy is retained by `StepExecutionContext` and summarized in every
   supported completed result. It is an inspection/provenance record, not a
   second configuration language; users continue to pass readable keyword
   arguments to `model.step(...)`.
5. An omitted value means the selected provider owns the default. Resolved
   numerical values remain authoritative in the executable Step summary.

The extraction covers linear static/steady conduction, finite-strain and
mixed hyperelasticity, global J2 plasticity, global implicit creep, explicit
dynamics, finite-strain explicit dynamics, and implicit structural dynamics.
Each public provider lowers directly to an internal builder; the historical
Model methods remain compatibility delegates only.

## Consequences

- New material families do not justify new implementation bodies on `Model`.
- Provider capability, option validation, scientific construction, and result
  completion have distinct owners.
- Transient providers that advertise `history` may consume construction-time
  requests automatically through `solve_result()`.
- Compatibility methods can be audited, versioned, and eventually migrated
  without changing provider internals.
- A third-party provider may use the same Step/result lifecycle without
  importing AgentFEM's internal built-in builders.

## Executable evidence

- regressions prove provider lowering does not call linear-static,
  hyperelastic, or finite-strain-explicit compatibility methods;
- static providers reject unsupported history instead of silently discarding
  it;
- heat accepts a construction-time history request and records its policy in
  result metadata;
- linear-static, J2, creep, full serial, two-rank MPI, three-rank sparse
  cohesive, wheel-payload, documentation, and release-gate tests remain green.
