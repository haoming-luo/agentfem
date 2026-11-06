# Co-located tensor history to creep assessment

## Scope

Selected material-point recording owns raw tensors and provenance. The
existing assessment previously required users to hand-copy one tensor
component into a new scalar history, losing the direct association with the
point identity. Add one optional `stress_component=(i,j)` argument to
`assessments.creep_damage_from_result`; do not introduce a parallel assessment
pipeline, a new material law or automatic stress criterion.

## Ownership and admission

- Read the existing Result histories without mutating arrays or metadata.
- Preserve signed component values; declared stress measure remains an
  engineering choice, checked against a supplied tabulated curve as before.
- When either input is a recorded material-point channel, require both to
  identify the same point and accepted clock. Check recorded shape/unit before
  invoking a user-owned rupture relation.
- Keep ordinary manually supplied scalar histories backward-compatible.
- Retain a detached copy of the source point, mesh, quadrature, channel
  conventions and scientific identity in the assessment record.
- Continue using existing trapezoidal rate integration and explicit dwells.
  No calibration, life extrapolation, constitutive damage or source data changes.

## Verification

Synthetic tests exercise exact damage, negative component preservation,
different-point rejection before callback, mixed tracked/untracked rejection,
changed clock/shape/unit, component index validation and detached provenance.
An ordinary regional creep Step supplies two distinct point histories for the
real numerical integration check. The same-point input passes; another point's
temperature fails. These tests establish data flow, not material qualification.

The initial real Step check deliberately failed when the test assumed the
history's label was `MPa`: the model retains `tonne/(mm*s^2)`. The example and
test now consume the recorded unit directly, without relabeling source arrays
or weakening the rupture-curve unit contract. The regional/history/assessment
regression passed 127 checks; subsequent alias and diagnostic checks also pass.
