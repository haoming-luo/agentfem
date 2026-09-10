# Own every prepared solve through one terminal lifetime

## Decision

Reusable numerical allocations conform to one private `PreparedSolve`
protocol: `solve()`, `summary()`, `closed`, and idempotent `close()`. The
execution scope that prepared the allocation must close it on every terminal
success or failure path. Model objects do not own backend destruction.

The protocol describes lifetime only. It does not erase the distinct
scientific and algebraic contracts of ordinary linear, exact-MPC, harmonic, or
future prepared problems.

## Reason

PETSc and SLEPc destruction can be collective. Deferring it to rank-local
Python cyclic garbage collection can place ranks in different collectives and
deadlock a later solve even though each individual numerical result is
correct. A named minimal lifecycle makes ownership reviewable without creating
one generic solver abstraction over incompatible formulations.

## Consequences

- `closed` is visible and terminal; a solve after close fails immediately;
- `close()` is safe to call more than once;
- owned backend handles are detached before destruction so later Python
  finalizers cannot destroy them again;
- scientific summaries and borrowed solution/constraint objects survive
  close;
- context managers remain the preferred scope for repeated solves and
  one-shot owners close in `finally`;
- the protocol stays private until more than lifetime semantics are genuinely
  common.

## Verification

- ordinary prepared linear solves retain one matrix across changing right-hand
  sides, then expose a preserved summary and reject reuse after close;
- exact-MPC and direct-harmonic prepared solves retain their existing
  idempotent close and borrowed-object tests;
- the complete two-rank modal, harmonic, MPC, and result sequence terminates
  without interpreter-shutdown collectives.
