# Accepted history checkpoints and continuation audit — 2026-10-29

## Decision

The generic finite-strain history Step reuses `CheckpointPolicy`, atomic archive
publication, typed checkpoint records and schema-checked serial retention. It
does not add a project callback or another numerical state owner. The policy
is construction-time operational configuration, not material identity; a fresh
compatible Step can restore with a different cadence/destination.

Only accepted increments count. Optional final saving includes a requested
pause endpoint. Publication follows constitutive/global acceptance and precedes
field-output/progress callbacks. A write/reporting failure is not a numerical
cutback. Pruning occurs only after successful payload and sidecar publication;
manual checkpoints, restart sources and other processes' files are excluded.
Scheduled files are published without replacement, including a racing writer.
Manual `save_checkpoint` retains its explicit replacement behavior.

Counts are positive integers, flags are booleans and a prefix is not a path.
The four-file example declares its required constructor keyword, so an older
same-version package fails project preflight instead of late execution.

## Two interruption orderings

1. Checkpoint newer than field output: a restored Step publishes its accepted
   boundary before advancing, without reintegration. A shared boundary is
   checked and deduplicated in the continuation view.
2. Field output newer than checkpoint: continuation recomputes an interval.
   Explicit `overlap="identical"` verifies every saved field at every overlapping
   station. Changed meshes, schemas, time partitions or values are rejected.
   The default remains one identical boundary only. This visual view neither
   establishes checkpoint ancestry nor selects a preferred solution branch.

## Evidence

- Final targeted regression: **496 passed**, 68.39 s, real FEniCSx environment,
  one numerical thread. Includes the material workflow, plane-stress reduction,
  checkpoint capabilities/retention, local material checks, live/continued
  output, project CLI and architecture/documentation contracts.
- Two educational laws exercise actual child-process exit between checkpoint
  publication and field publication. Restored state and all joined frame arrays
  match uninterrupted results exactly on this environment.
- Forced archive, sidecar, retention, output and reporting failures preserve
  recoverable accepted state. Material rejection never schedules a trial.
  Adaptive material cutback counts accepted increments only.
- Two concurrent fresh-archive writers produce one winner and one refusal;
  neither a racing writer nor an existing destination is overwritten. Manual
  archive replacement and arbitrary payload keyword compatibility remain intact.
- Installed acceptance in an isolated virtual environment sharing the existing
  FEniCSx dependencies, with `PYTHONPATH` unset: `agentfem check/run`, pause,
  restart from the **automatic** checkpoint and continuous runs pass for both
  educational laws. Each merged view contains 16 unique frames / 64 cells;
  maximum field difference is 0.0, final checkpoint arrays match, each run keeps
  two scheduled archives, and actual VTK/Warp preserves stress, history and F.
  The first installed check preceded exclusive-publication and optional
  multi-station-overlap hardening. A fresh isolated installed check now repeats
  both laws with those changes and the material-verification audit included:
  the same 16-frame/64-cell, exact-field/state and real VTK/Warp checks pass.
  Candidate wheel SHA256:
  `fe23d7d6c03f993c5b679919d4a47c2359adad4b70449b8109a8482be033ab2d`.
  No source-path injection or user-environment replacement was used. Same-host
  exact equality is evidence for this fixture, not a cross-platform bitwise claim.
- Correctness lint and generated documentation checks pass. Remote CI is a
  separate gate; this record does not assign earlier parent CI to a new HEAD.

The temporary acceptance wheel disabled unrelated native accelerators and did
not replace the user's installation. This is not a Windows fresh-install test,
an installer/distribution release or a composite speed benchmark.

## Limits

Serial same-layout restart only; the material workflow remains experimental.
No MPI/contact/MPC/regional/thermal-material admission changes. Unsaved intervals
must be recomputed. Retention is per current Step, not a quota over all old runs.
A source file that never existed cannot be recreated by a presentation index.
Process interruption is tested; arbitrary storage corruption or power-loss
recovery is not certified. The no-replacement publication requires filesystem
hard-link support and fails rather than falling back to unsafe replacement.
