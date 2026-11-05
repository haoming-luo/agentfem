# P92 repeated structural-path screen

Extend the short structural checks to repeated published-waveform tensile
holds. Keep the 600/650 C parameters fixed, use a homogeneous hexahedral
specimen through ordinary `model.step`, and retain free lateral contraction.
No custom global solver, fitted Poisson ratio or damage-to-failure law is added.

Use the source's 0.001/s ramps, initial zero-to-half-range ramp, tensile dwell,
and subsequent full-range compression/tension ramps. Visit all waveform knots
and resolve ramps and holds separately. Compare each peak/end-hold stress with
the independent continuous scalar Radau reference. Halve both ramp and dwell
increments without changing physical duration. A restart at an accepted dwell
endpoint must preserve the complete selected history exactly.

First run a small screen to verify wiring and cost. Then run a bounded repeated
path with fixed parameters; do not repeatedly tune tolerances or call this
experimental life validation. Report wall time, increment count, numerical
stress discrepancies, lateral stress and strain-boundary errors. The point
history is verification evidence, not a multiaxial fatigue equivalent strain.

## Initial discriminator

The 600 C, two-cycle, 5 s short-hold smoke first failed the preselected 15 MPa
continuous-reference gate: coarse maximum error was 18.1273 MPa. Halving ramp
and hold increments reduced it to 10.3602 MPa (84 accepted increments).
The threshold is unchanged; the regression now requires both the absolute
fine-grid gate and an error ratio below 0.7. The coarse failure remains evidence
of time-discretization sensitivity, not a reason to retune material parameters.

## Executed 20-cycle paths

Both temperatures ran the same 0.8% total strain range and 60 s tensile holds,
with 856 coarse and 1712 refined accepted increments. Maximum errors below
refer to tensile peaks and hold endpoints, not all interior trajectory times.

| Temperature | Coarse error (MPa) | Refined error (MPa) | Complete restarted history |
| --- | ---: | ---: | --- |
| 600 C | 4.121080 | 2.147799 | Exactly equal |
| 650 C | 3.018525 | 1.558788 | Exactly equal |

The largest lateral stress was below 3.8e-7 MPa and the prescribed axial-strain
error below 1.7e-16. Each refined run saved after cycle 10 and restored a fresh
Step, retaining all selected stress/strain/temperature/internal-state samples.
Coarse runs took about 21 s each; refined runs including a complete second
execution through restart took 73.35 and 72.17 s. These are small homogeneous
verification timings, not industrial-model performance claims.

The raw numerical reports are preserved in
`evidence/creep/2026-11-06-p92-structural-cycles.json`. A focused regression of
the tool, tensor structural consumer and CI routing passed 85 tests in 24.70 s.
Experimental cyclic discrepancies and independent lifetime qualification remain
separate gates; the integration check does not erase either.

## Long-cycle discriminator

At frozen source commit `cf55d6f652733f36f62dbf7cf4a57802aa6eb3a2`, the
650 C / 60 s hold path was continued for 435 cycles, half the published
870-cycle experimental life. That cycle count is a prescribed verification
endpoint, not a predicted life. Both runs used one MPI rank and one numerical
thread, with explicit wall-clock limits and no concurrent numerical job.

| Resolution | Accepted increments | Maximum endpoint stress error (MPa) | Wall time (s) |
| --- | ---: | ---: | ---: |
| Coarse | 19116 | 3.018525 | 535.46 |
| Halved ramp/hold increments | 38232 | 1.558788 | 941.19 |

The maximum error remained the early-cycle maximum observed in the 20-cycle
screen; it did not grow above that value over this longer path. The refined
maximum lateral stress was 1.96e-7 MPa or less and the prescribed strain error
below 3.7e-15. Timing includes the independent scalar Radau reference and FEM
solve. These long runs did not repeat checkpoint recovery: the exact-restart
evidence remains the separately executed 20-cycle paths above.

The complete endpoint comparison records are retained in
`evidence/creep/2026-11-06-p92-435cycles-coarse.json` and
`evidence/creep/2026-11-06-p92-435cycles-refined.json`. This establishes a
longer numerical path, not independent experimental accuracy, damage-to-failure,
multiaxial qualification or engineering design life.
