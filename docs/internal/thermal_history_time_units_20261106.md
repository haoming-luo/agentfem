# Prescribed-history time-unit boundary

P92 assessments expose a common h/s transfer risk: temperature histories
declare kelvin but previously could not declare their time-coordinate unit.
Add an optional `coordinate_unit` to FieldHistory, preserve it through scalar
and portable archives and scientific identity, and compare it with the declared
physical time unit when freezing uniform temperature for user-material Steps.

No conversion is implicit. A declared source unit requires the same target
unit; an undeclared legacy history retains its existing model-unit convention.
Omit the new identity key when absent so existing history identities remain
unchanged. Keep numeric history interpolation and material integration intact.
The bounded admission here is the finite- and small-strain user-material
temperature consumer, not a universal dimensional-analysis engine.

Test scalar and nodal archive round trips, unchanged legacy identity,
unit mismatch before any material evaluation, matching hour-based paths,
and the existing thermal/cutback/restart suites.

The focused unit/archive/two-consumer/thermal regression group passed 65 tests
in 5.99 s in the real `fenicsx-env`. The tests include scalar and nodal archive
round trips, mutation after identity caching, a genuine hour-labelled
small-strain solve, and finite-strain temperature admission. Existing numerical
thermal tests pass unchanged; this is not a new P92 temperature calibration.

The broader physical-time, point-history, documentation and CI-routing group
passed 208 tests in 18.49 s; one two-rank-only material-admission test was
skipped in that serial invocation. Separately, the existing portable field
history driver passed both 2→1 and 1→2 rank restoration, preserving the
declared seconds unit and the interpolated spatial temperature field. MPI
jobs used `agentfem mpi-run`, 120 s timeouts and single-thread numerical
libraries, sequentially. This tests field-history portability, not MPI
admission of either physical-time user-material Step.
