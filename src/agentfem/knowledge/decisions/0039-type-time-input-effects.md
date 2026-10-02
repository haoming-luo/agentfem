# Type time-input effects before a Procedure selects reuse

## Decision

Every time-dependent Model asset or custom callback declares which numerical
owner it can invalidate:

- `right_hand_side` changes forcing, lifting, or prescribed values while the
  assembled operator remains valid;
- `operator` changes a bilinear form, Jacobian, mass, damping, or constraint
  operator;
- `state` changes accepted history that can alter residuals or tangents;
- `output` changes observation only.

Model composes these declarations into one ordered `TimeInputPlan`. Procedure
consumes the plan and owns the reuse decision. Result/Verification records the
selected policy and the complete declaration. A bare user callback is treated
conservatively as potentially having every effect; AgentFEM does not inspect
callable source code or guess from a variable name.

## Reason

The former `update_load(time)` callback erased the distinction between a load
amplitude and a time-varying material coefficient. Both ran before the solve,
so a fixed effective matrix could be reused even when arbitrary callback code
had changed an operator coefficient. That was fast but not fail-closed.

UFL deliberately has no privileged time object; time dependence is represented
through ordinary coefficients or constants. DOLFINx can reuse packed constants
and coefficients when forms are assembled repeatedly, but only the application
knows which values changed. PETSc likewise distinguishes repeated solves with
the same matrix from matrices whose values changed; preconditioner reuse after
an operator change is an explicit numerical choice, not a default scientific
assumption.

## Consequences

- amplitude-scaled natural loads, fixed-DOF prescribed values, and supported
  ambient-temperature histories declare `right_hand_side`;
- linear implicit dynamics and linear implicit-Euler heat transfer
  automatically reuse their prepared operator only for RHS/output-only plans;
- nonlinear first-order residuals remain per-step assembled and reject an
  explicitly requested reuse policy;
- operator- or state-changing plans select per-step refresh under `auto`;
- forcing `operator_policy="reuse"` against such a plan fails before solving;
- explicitly typed custom callbacks remain possible through
  `agentfem.time.input_update`;
- callback identities are normalized as finite JSON data before entering
  result evidence, so an opaque Python object cannot break an archive;
- transient checkpoint schema v5 stores the complete plan and refuses both a
  changed identity and a callback whose restart identity is absent;
- existing bare callbacks remain callable but take the conservative path;
- the same plan is included in transient Step summaries and therefore in
  `SimulationResult` evidence.

This contract classifies invalidation. It does not promise that every Procedure
already implements every possible optimized refresh strategy.

## Verification

- ordered composition retains all declared effects;
- invalid or empty declarations fail early;
- a bare callback makes implicit dynamics and linear heat transfer refresh
  their matrix every step;
- a declared RHS callback retains one effective matrix and reassembles one RHS
  per step;
- operator and state declarations either refresh automatically or reject an
  explicitly unsafe reuse policy;
- the selected policy and declarations survive into lifecycle evidence.
- matrix-free Explicit records that it reevaluates the residual after every
  input update and distinguishes a fixed preflight stability bound from an
  operator-changing path that the caller must bound conservatively;
- ordinary incremental nonlinear procedures record the same plan, state that
  residual and tangent are assembled for every attempt, and restore the
  accepted load coordinate after a rejected attempt;
- architecture audit output reports owned-module counts and the deliberately
  unowned utility roots, so a clean dependency graph cannot hide accidental
  ownership growth.
- restart rejects a changed load identity before mutating any field, while an
  unbound callback cannot publish a misleading checkpoint.
- serial and two-rank first-order runs retain the same selected policy and
  assembly evidence.

## References

- UFL form language, time differentiation through ordinary constants:
  <https://docs.fenicsproject.org/ufl/main/manual/form_language.html>
- DOLFINx assembly API, repeated constants and coefficients:
  <https://docs.fenicsproject.org/dolfinx/v0.10.0/python/generated/dolfinx.fem.html>
- PETSc KSP manual, successive systems and preconditioner reuse:
  <https://petsc.org/main/manual/ksp/>
