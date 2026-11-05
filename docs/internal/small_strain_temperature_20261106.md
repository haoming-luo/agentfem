# Small-strain prescribed temperature composition

Reuse the frozen uniform temperature input already used by finite-strain
history materials. The small-strain material driver already accepts temperature
and its increment; the structural consumer must supply them from the same
accepted physical clock as strain and history. No additional solver or material
model is introduced.

Scope: existing serial homogeneous 3D physical-time route, positive scalar
kelvin or a scalar linear temperature history covering the full duration.
The provider owns thermal strain and declares its initial state stress-free
at the supplied initial temperature. No thermal equilibrium, heat balance,
P92 parameter interpolation or engineering life qualification is inferred.

Every Newton trial receives the endpoint temperature and the difference from
the accepted start, including cutback. Increments land on thermal and mechanical
knots. Checkpoint identity binds the frozen thermal path; accepted material-point
histories include TEMP alongside stress and total strain. Old no-temperature
identities remain unchanged.

Acceptance: analytic restrained thermal stress, free expansion, time-knot
landing, fixed-old-state trial semantics, exact interrupted/continuous histories,
atomic rejection of changed thermal input, and unchanged nonthermal regressions.
Implementation is not admitted until these checks pass.

## Local evidence

The first composed regression passed 157 tests with one serial skip (the
existing dedicated two-rank admission check). The temperature-specific group
passed 12 tests, including free expansion, restrained hydrostatic stress,
cutback, frozen input, changed-path atomic rejection and exact restarted
material-point histories. These counts overlap. A real VTK read and Warp By
Vector preserve the mesh, TEMP, S_CELL and AGE_CELL; the exported displacement
channel is U, independent of the internal Function name.

TEMP is derived from the frozen prescribed input at the accepted physical time,
not an extra mutable constitutive state. Output does not require extra material
evaluations, and restart does not infer missing stress or history. Local evidence
does not promote MPI or P92 temperature-dependent parameters.
