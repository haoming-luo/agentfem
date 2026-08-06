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
4. Schema v2 publishes generation-specific state shards before atomically
   replacing the manifest. Every shard records its byte size and SHA-256
   digest, so interruption leaves the previous manifest intact and silent
   corruption is rejected collectively. The loader remains compatible with
   schema v1.
5. The checkpoint is deliberately `portable=false`: it supports serial and
   same-partition MPI restart. Its identity includes local topology as well as
   geometry and function layout. Cross-partition portability requires stable
   global mesh/dof/cell and quadrature identities and is a separate acceptance
   gate.
6. A resumed step may write a truthful continuation XDMF/HDF5 segment through
   `solve_result(output=...)`. Result metadata records the segment start time;
   a completed step still refuses to fabricate unrecorded earlier frames.
7. Standard and Explicit dynamics sample mechanical energy from their visible
   M/K operators. Transient heat samples `1^T C T` and calls it thermal content,
   avoiding a stronger conservation claim when flux/source work is not yet
   integrated.
8. Heat, Standard dynamics, and Explicit dynamics evaluate the same
   `results.history(...)` and `results.probe_history(...)` requests after each
   accepted increment. Their values, units, descriptions, and channel names
   enter the same `SimulationResult` and checkpoint history. Continuation
   rejects a changed channel schema instead of producing ragged evidence.
9. Automatic checkpoint paths contain one unambiguous increment identity.
   `keep_last=N` publishes the newest manifest and every rank shard before
   pruning older scheduled generations. Pruning reads exact shard names from
   each manifest, deletes the manifest last, and never removes an explicit
   restart source or recursively cleans a directory.

## Consequences

- An interrupted transient solve can be resumed without losing the scientific
  execution trace or silently resetting its time axis.
- GUI, CLI, campaign, and agent clients can inspect one restart schema rather
  than special-case each time integrator.
- Same-partition MPI restart is useful now, while the manifest states exactly
  why it is not yet portable to another process count.
- Interrupted writes and corrupted state shards fail explicitly rather than
  silently mutating fields.
- Compact sensors and engineering histories can feed reports, fatigue,
  learning, or online-monitoring clients without application-owned time loops.
- Checkpoint storage can be bounded without weakening atomic publication or
  turning user directories into disposable solver state.
- Full energy/work/heat balances remain extensions of this contract, not
  parallel implementations.

## Executable evidence

- uninterrupted versus restarted Explicit central difference;
- uninterrupted versus restarted Newmark dynamics;
- uninterrupted versus restarted implicit-Euler heat transfer;
- incompatible time-contract rejection;
- two-rank same-partition heat restart with rank-sharded state;
- collective failure evidence when one MPI state shard is missing;
- checksum failure evidence when a state shard is modified;
- resumed `solve_result(output=...)` with an explicit continuation boundary;
- shared heat, Standard, and Explicit probe-history requests;
- custom-history continuation equivalence and changed-schema rejection;
- serial and two-rank bounded retention with restart-source preservation.
