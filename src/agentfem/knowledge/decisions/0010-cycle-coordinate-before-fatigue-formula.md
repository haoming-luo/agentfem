# Decision 0010: Establish the cycle lifecycle before selecting one fatigue law

## Decision

Fatigue cycle count is an independent physical coordinate. AgentFEM will not
encode cycles as Explicit time increments or infer them from saved frames. A
stateful cyclic-fatigue procedure owns peak/valley equilibrium, adaptive cycle
blocks, exact landing cycles and begin/commit/rollback semantics.

The cohesive material API composes a monotonic envelope with a replaceable
cyclic evolution law. The first power-law opening-range model is a transparent
reference implementation, not the platform definition. All cyclic laws must
declare threshold, local load-ratio convention, damage variable, monotonic
limit, dissipation and restart fields.

## Why

Different materials and experiments support different fatigue cohesive laws.
Hard-coding one equation into a cylinder solver would produce a brittle case
implementation and make later calibration choices architectural changes.
Conversely, wrapping the existing S--N postprocessor would not create a
solver-coupled crack model.

Cycle lifecycle, irreversible state, rollback, restart, energy evidence and
named interface identity are common requirements independent of the selected
fatigue equation. They are therefore the reusable product layer.

## Consequences

- `ForceCycle` accepts physical minimum/maximum force and waveform data.
- `SolutionProcedure.control` includes `cycle_increments`.
- `CyclicCohesiveTransaction` has distinct monotonic and cycle-block trials.
- constant-extrema cycle blocks are integrated analytically at material-point
  level; `GlobalCyclicFatigueStep` supplies the structural peak/valley,
  post-damage re-equilibration, feedback check and automatic cutback lifecycle;
- local opening extrema supply the material-point load-ratio effect;
- multiple named interfaces retain independent state and source identity;
- disjoint named interfaces are split atomically onto one solver mesh;
- physical-facet checkpoint schema v2 stores all cyclic fields;
- the accepted cycle ledger, bulk fields and named interfaces share one global
  restart envelope, while bulk cross-partition portability remains future work;
- connected surface cracks retain physical-facet-based identity across cycles,
  and merge/split events create auditable ancestry rather than unstable labels;
- Paris-law fitting is postprocessing evidence from accepted crack histories,
  not a hidden crack-advance rule in the cyclic cohesive solver;
- the first cylinder and double-crack examples calibrate only from the single
  crack and must retain separate prediction evidence.
