# ADR 0004: One public step semantics across human and agent workflows

Status: accepted

## Context

AgentFEM exposes expert operator and backend routes, but its public model path
must not assign different meanings to the same Study, field, load history, or
completed result. Such divergence is difficult for engineers to review and is
especially unsafe for agents, campaigns, and future GUI clients.

## Decision

1. `model.validate()` and `model.step()` consume the same provider predicates.
   Capability checks include the target-field role and the material protocol
   required by default lowering, while a complete explicit operator system
   remains a documented expert path.
2. Amplitude coordinates belong to the Step: single-solve static uses the
   normalized step end, nonlinear static uses normalized step time, and
   transient procedures use physical time. An amplitude-driven nonlinear load
   is not multiplied by a second implicit ramp.
3. Transient heat, Standard dynamics, and Explicit dynamics return the same
   `SimulationResult` lifecycle. One call may solve, write XDMF/HDF5, retain
   accepted-time evidence, and attach field artifacts.
4. MPI completion is collective. A failure on any rank produces one failed
   execution record with rank-addressable errors; successful ranks cannot
   publish overall completion first or wait indefinitely at a final barrier.

## Consequences

- Beginner code and machine clients can trust the public path without
  reconstructing hidden backend assumptions.
- Expert UFL/DOLFINx/PETSc access remains available, but it is explicit rather
  than a silent exception to model validation.
- Procedure-specific energy and checkpoint state remain future extensions of
  the shared transient lifecycle; this decision does not claim that they are
  already unified.

## Executable evidence

- negative validation tests for missing thermal material and incompatible
  target fields;
- named load and prescribed-value amplitude tests;
- Neo-Hookean normalized amplitude-path regression;
- heat, Explicit, and Standard transient XDMF/HDF5 result tests;
- a two-rank injected CLI failure test that verifies collective exit and the
  structured rank error record.
