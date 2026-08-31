# Reliability stays outside scientific state

## Decision

AgentFEM observes coarse product reliability only after a public command has
completed. The automatic channel has one exact schema and excludes models,
meshes, parameters, source, names, paths, messages, tracebacks and results.
It cannot own or mutate Model, Procedure, State, Backend, SimulationResult or
Verification.

Rich diagnosis is a separate local support artifact. It is sanitized and
integrity-sealed for a human or AI agent, and it leaves the machine only after
an explicit user action. Repeated failures are recognized by a version-scoped,
message-free fingerprint rather than a user or installation identity.

## Reason

Real usage evidence can create a maintenance flywheel, but a scientific code
must not turn proprietary geometry, material data or results into telemetry.
Nor may a failed collector delay, alter or invalidate a deterministic solve.
Separating the two channels gives the product useful reliability evidence
without making scientific state an analytics payload.

## Consequences

- basic reporting is transparent, bounded, inspectable and permanently
  opt-out;
- a release without a reviewed project-owned collector keeps only a bounded
  local queue and says delivery is unavailable;
- the collector stores daily aggregates, not raw events or source IPs;
- GitHub issue creation remains authenticated and user initiated;
- reliability evidence never changes a result's scientific trust level.

## Evidence

- strict client and collector schema validators;
- tests that inject secret messages, paths, source lines and material names;
- queue, opt-out, repetition and sanitized bundle regression tests;
- installed-wheel CLI and documentation release gates.
