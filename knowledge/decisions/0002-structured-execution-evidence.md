# ADR 0002: Structured execution evidence is part of the scientific interface

## Status

Accepted for the 0.1 release line.

## Context

A finite-element job previously exposed progress in three partly independent
forms: terminal text, optional status files, and result histories assembled
after the solve. Sparse progress printing could therefore discard increments
from the machine-readable record, and a future agent could observe different
evidence from the human operator.

AI-native CAE cannot mean only that an agent can write Python. The execution
must also expose deterministic, typed evidence that the human, an agent, a
scheduler, a benchmark, and a dataset gate can inspect without parsing backend
logs. This evidence must preserve failed attempts and cutbacks, not only the
final successful state.

## Decision

`SolveEvent` is the common execution stream for Standard nonlinear paths,
explicit dynamics, implicit dynamics, first-order transient analysis, and
stateful J2 plasticity. Events are JSON-safe and restartable. A reporter group
fans the same event out to:

- immediate rank-zero progress;
- a flushed status file;
- an in-memory complete trace;
- `SimulationResult` histories and manifest metadata;
- future scheduler, service, and agent observers.

The `display` flag changes only terminal/status visibility. It never removes an
event from the trace. Accepted increments are projected to monotone histories;
failed attempts and iterations remain in the complete event record.

Checkpoints remain procedure-specific state serializers registered through the
common `CheckpointRecord`. A common record must not imply that arbitrary state
is already portable across MPI partitions.

## Consequences

- Human and agent observers share one source of truth.
- Sparse console output no longer weakens reproducibility.
- Result manifests carry convergence and failure evidence without embedding
  terminal prose.
- Stateful restart must preserve execution history as well as field arrays.
- Event schema migrations will be required when the contract evolves.
- This is one component of the AIR layer: public scientific language,
  validation, execution evidence, results, benchmarks, and knowledge assets.
  It is not equivalent to AF-IR alone.

