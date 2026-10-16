# AgentFEM 0.4.3

This release strengthens the installed project and result workflow and adds a
bounded experimental finite-strain J2 contact composition. The ordinary
`model.step(...)` path now keeps the material update and one rigid-tool contact
contribution in the same Newton solve and accepted-state lifecycle.

## Finite-strain J2 with a rigid tool

- One frictionless analytical plane or triangulated rigid tool may contact a
  three-dimensional finite-strain J2 solid in a serial displacement Step.
  The triangulated route recomputes closest-point projection at each Newton
  trial and adds its current-branch residual and tangent before the solve.
- A prescribed rigid translation/rotation may share the normalized Step load
  path. Accepted increments retain tool pose, generalized force and moment,
  path work, contact potential, material state, result and checkpoint identity.
  Rejected trials do not advance that ledger.
- An active triangulated-tool point is admitted only on a planar facet or a
  coplanar interior seam. Free edges, noncoplanar creases and ambiguous
  projections fail closed; they do not receive an invented smooth tangent.
  Periodic MPC plus contact, friction, mixed displacement--pressure, multiple
  pairs and distributed implicit contact remain unsupported combinations.

The bounded benchmark compares analytical and triangulated planes under
nonuniform deformation and prescribed motion. It checks force--work
conjugacy, tangent, rollback, serial restart and integral-output sensitivity
to increments, mesh and penalty. The spatial scan is not yet mesh independent,
and a passed plane limit does not validate curved tooling or industrial
forming. Completed results retain their capability-specific trust status.

## Installed workflow and evidence

Accepted response snapshots can be exported as a single-grid process series
without changing history or checkpoint cadence. A declared strain-history
VUMAT profile adds named parameters, explicit state/kinematic/work semantics
and native fusion while preserving the user material as a separately reviewed
asset. Neither feature implies arbitrary Abaqus compatibility or a universal
speedup.

Project execution now provides checked mesh inputs, sibling-module imports,
preparation and failure evidence, and WSL storage/resource advisories. It does
not move a user's project or tune scientific controls without consent. Native
explicit body/contact transport has bounded local regressions, but no
cross-machine or full-run throughput claim is attached to this release.

The exact release artifacts must pass the repository's installed-wheel,
platform and scientific gates before a tag is published. The candidate scope
and outstanding evidence are recorded in the
[0.4.3 decision record](internal/release_0_4_3_candidate_gate_20261016.md).
