# ADR 0005: Shared transient history and restart envelope

Status: accepted

## Context

Heat transfer, Explicit dynamics, and Standard dynamics already shared field
output and structured progress, but they could not pause and resume through a
common scientific contract. A raw array dump would be unsafe: it does not say
which procedure, time coordinate, mesh partition, field layout, or diagnostic
history the state belongs to.

## Decision

1. Every transient Step owns `completed_steps`, accepted physical times,
   structured execution events, diagnostic histories, and typed checkpoint
   records.
2. `run(until_step=...)`, `save_checkpoint(...)`, `load_checkpoint(...)`, and
   `solve_result()` have the same meaning for first- and second-order
   procedures.
3. A checkpoint contains the procedure and time contract, state-field names,
   mesh/function-space identity, prior evidence, and one state shard per MPI
   rank. An incompatible procedure, time increment, total step count, MPI size,
   mesh partition, or field layout is rejected before state mutation.
4. Schema v1 is deliberately `portable=false`: it supports serial and
   same-partition MPI restart. Cross-partition portability requires stable
   global mesh/dof/cell and quadrature identities and is a separate acceptance
   gate.
5. Standard and Explicit dynamics sample mechanical energy from their visible
   M/K operators. Transient heat samples `1^T C T` and calls it thermal content,
   avoiding a stronger conservation claim when flux/source work is not yet
   integrated.

## Consequences

- An interrupted transient solve can be resumed without losing the scientific
  execution trace or silently resetting its time axis.
- GUI, CLI, campaign, and agent clients can inspect one restart schema rather
  than special-case each time integrator.
- Same-partition MPI restart is useful now, while the manifest states exactly
  why it is not yet portable to another process count.
- Full energy/work/heat balances and automatic checkpoint cadence remain
  extensions of this contract, not parallel implementations.

## Executable evidence

- uninterrupted versus restarted Explicit central difference;
- uninterrupted versus restarted Newmark dynamics;
- uninterrupted versus restarted implicit-Euler heat transfer;
- incompatible time-contract rejection;
- two-rank same-partition heat restart with rank-sharded state;
- collective failure evidence when one MPI state shard is missing.
