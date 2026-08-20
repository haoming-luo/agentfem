# Decision 0013: Scientific experiments share Campaign evidence

## Decision

Reusable loading modes, finite-difference response operators, threshold
events, and future convergence orchestration build on the existing Campaign,
SimulationResult, and provenance contracts. They do not create a parallel
optimization project model or hide case construction inside `Study`.

Runtime identity, scientific input identity, and result integrity remain three
separate records. A frozen runtime can warn or stop before execution, while
paths remain diagnostic evidence rather than portability gates.

The first response provider is finite difference. Future tangent-linear or
adjoint providers must return the same response report. Irreversible damage,
contact activation, and topology events are not presumed differentiable.

## Why

Dynamic control, inverse problems, parameter identification, uncertainty
studies, and learning datasets all need the same practical mechanisms:
deterministic case identity, failure visibility, restart, units, observers,
and provenance. Building each one as a research script duplicates logic and
makes human and agent decisions difficult to audit.

## Consequences

- Named amplitude bases are serializable scientific assets; custom callables
  remain explicit non-serializable escape hatches.
- Finite-difference perturbations are ordinary Campaign cases and inherit its
  cache and failure semantics.
- A failed perturbation produces an incomplete response report, never a silent
  partial Jacobian.
- First-passage output retains its sample bracket and censoring state.
- Scheduler execution and tangent/adjoint methods can evolve behind stable
  experiment contracts without changing ordinary FEM case files.
