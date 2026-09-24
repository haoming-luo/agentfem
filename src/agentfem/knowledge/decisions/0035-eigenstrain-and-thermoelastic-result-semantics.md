# ADR 0035: Eigenstrain and thermoelastic result semantics

## Status

Accepted.

## Decision

AgentFEM owns one physical small-strain stress field, `S`. It does not expose
parallel `mechanical_stress`, `total_stress`, and `thermoelastic_stress`
fields whose additivity and meaning would depend on a solver convention.

Stress-free strain is an explicit scientific asset:

\[
E_\mathrm{MECH}=E_\mathrm{TOTAL}-E_\mathrm{EIGEN},\qquad
S=\mathbb C:E_\mathrm{MECH}.
\]

`eigenstrains.thermal(...)` is the first standard source. Expert-prescribed
sources use `eigenstrains.prescribed(...)`; cure shrinkage and phase
transformation can add constructors without changing Model, Procedure, or
Result ownership.

Model owns source registration and the material--region map. Operator lowering
creates regional equivalent virtual work. Procedure only solves the lowered
system. Result recovery consumes the same sources and regions, retains
discontinuous scientific fields, and records the source identity.

## Consequences

- `model.thermal_expansion(...)` is region aware.
- A registered source can be lowered automatically by a static-solid Step.
- `S` always means physical Cauchy stress. Positive `C:E_EIGEN` remains
  operator evidence and is never labelled physical stress.
- `E` remains a compatibility alias for total infinitesimal strain.
- Scientific projection remains discontinuous by default.
- Multi-material coverage fails before assembly on gaps or overlaps.
