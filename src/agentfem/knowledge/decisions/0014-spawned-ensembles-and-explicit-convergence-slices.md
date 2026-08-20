# Decision 0014: Spawned ensembles and explicit convergence slices

## Decision

Independent local Campaign cases use a dedicated `spawn` process provider.
They do not use Python threads and do not fork a process after MPI, PETSc, or
FEniCS has initialized. Across-case local processes and within-case MPI are
separate execution modes and cannot be nested implicitly.

Multi-axis convergence is evaluated as explicit one-at-a-time slices. Every
non-refined parameter is fixed by the user, and every observable declares a
relative, absolute, or exact comparison. Missing, failed, duplicated-size, or
insufficient sequences are inconclusive.

Campaign provenance distinguishes runtime identity, declared/built scientific
input identity, execution evidence, and result/artifact integrity. Public IR
or summary contracts and content-hashed files/arrays contribute stable input
identity. Opaque inputs remain named coverage gaps.

## Why

FEM libraries and MPI runtimes are not generally safe under arbitrary threads
or post-initialization process forks. Scientific sweeps also fail in ways that
must remain attached to individual cases. A spawned worker boundary provides
useful workstation parallelism while preserving fresh case construction,
deterministic IDs, resume, and future scheduler replacement.

Convergence cannot be inferred from a successful solve or an unqualified plot.
Explicit slices prevent changes in mesh, time step, loading, and material
parameters from being mixed into one apparent refinement sequence.

## Consequences

- Local-process callables must be importable and serializable; closures and
  notebooks should use serial execution or move reusable functions to a module.
- `fail_fast` stops after the first bounded worker batch containing a failure;
  already running cases in that batch remain valid evidence.
- Separate MPI jobs continue to use deterministic Campaign shards until a
  scheduler provider owns their launch and collection.
- Convergence certificates can compare scalar/vector quantities and exact
  event/topology records without silently dropping failed cases.
- Objects without a scientific identity contract do not block exploratory
  execution, but they prevent complete input-fingerprint coverage.
