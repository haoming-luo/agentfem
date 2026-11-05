# P92 rupture-table admission

Reproduce Table 2 of [Khayatzadeh et al. (2017)](https://strathprints.strath.ac.uk/62915/1/Khayatzadeh_etal_MSEA_2017_Influence_of_thermal_ageing_on_the_creep_behaviour.pdf)
through the public tabulated-rupture API, outside the runtime material catalogue.
The author PDF has SHA-256
`7abf1fe6555897b3e81e9838852d40eaf199cdeb5c0e4561aa2054772e2e8085`.
Pages 3–4 were text-checked; page 4 was rendered and visually checked.

The tests use axial parent-metal specimens remote from a pipe weld, at 650 C
and constant load. Distinguish the un-aged material from specimens additionally
aged 1000 h at 650 C. Each condition has only two reported rupture endpoints;
the other stresses were interrupted tests with no rupture time in this table.
Do not interpret blanks as zero, infinite life, or rupture observations.

Use separate 120–170 MPa curves, nominal tensile stress and hours. Log/log
interpolation between endpoints is our explicit numerical choice, not a
published regression or a validated prediction. Reject other temperatures,
compression and stress extrapolation. Reproduce endpoint time fractions and
explicit hour/second equivalence; do not claim an independent validation from
reproducing the supplied nodes.

This batch is not identified with the 2019/2020 fatigue sources. Do not combine
its rupture curve with their fatigue fits for a qualified interaction result.
Nominal test stress is also not automatically the tensor reference's Cauchy
stress after appreciable deformation. No component life, scatter bound or
failure-strain evolution is inferred.

## Executed checks

The P92 reproduction, generic rupture-table, time-resolved creep damage and
CI-routing test group passed 132 tests in 0.25 s in `fenicsx-env` on
2026-11-06. Supplied rupture times are recovered exactly; half-duration
time fractions are 0.5. Explicit hour/second conversion agrees at the nodes
and an interpolated query. Off-domain stresses and temperatures are rejected.
These checks verify data plumbing and arithmetic, not an experimental holdout.
