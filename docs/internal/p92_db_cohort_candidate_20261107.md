# DB P92: coherent-cohort data and fixed-parameter audit

The original [Hoey thesis](https://repository.nottingham.ac.uk/handle/123456789/30506)
was retrieved from Nottingham's migrated repository. The earlier old-link
failure is resolved. SHA256:
`87db5240927faf09049ea4c9258c54ac5e392f15070ea4f005447e2d2e2cddb3`.
Repository identity, original download URL, page locations and checked numeric
transcriptions are retained in `evidence/creep/p92-db-cohort.json`.
The source PDF is not redistributed.

## Admitted observations

The uncoated cohort comes from one supplied pipe section. Twelve creep tests
cover 650–700 C; three fatigue and three tensile-dwell tests are at 650 C.
Fatigue strain runs from zero to a positive peak: the reported peak is the
range, not the amplitude. Fracture or a stress reduction exceeding 50% ends
the test. These semantics differ from the service-aged Imperial cohort.

Visual checks covered PDF pages 85, 101–103, 177, 180–181, 183–185. Source constants
use MPa and hours. They were fitted to these observations; comparison with
the same tests is reproduction, not independent validation.

## Executed numerical audit

Run `python tools/p92_db_cohort_screen.py` from a source checkout with AgentFEM
installed. It uses the existing `KachanovRabotnovCreep`, without fitting or
changing source constants. Tests independently integrate the two evolving
states with DOP853 and compare the printed closed form. Split and direct
constant-stress updates agree. No new P92-specific solver is introduced.

The 650 C / 120 MPa prediction is 868.98 h against 1219 h observed (−28.71%).
At 675 and 700 C, treating strain as dimensionless with the printed constants
produces pre-endpoint strains over ten times the reported failure strain.
This is a source/interpretation discrepancy, not justification to silently
divide coefficients by 100. The 650 C constants do not exhibit the same
scale relationship. No such correction is applied or promoted.

The 600 s dwell row also has a duration discrepancy: 1811 × 606 / 3600 gives
304.8517 h, while the table reports 304 h. Both values remain recorded; neither
the cycle count nor waveform is adjusted to force agreement.

## Next admissible step

The experimental rupture observations also exercise `TabulatedRuptureCurve`:
exact supplied-node recovery, MPa/h to Pa/s conversion, and rejection of
out-of-range stress or temperature. A separate sparse-node screen retains
105/120/140 MPa at 650 C, endpoints at the other temperatures, and omits five
interior observations. Predicted-life errors are respectively +16.39%, −0.79%,
+26.96%, −14.10%, −38.53%. This is a within-table diagnostic, not blind
validation, and is not tuned to reduce those residuals. Its strongest warning
is that sparse interpolation cannot be assumed accurate merely because all
retained nodes match exactly.

154 targeted tests passed across the source screen, CI selection, existing
constitutive models and rupture-curve contracts. Documentation and lint checks
passed. An initial duration-rounding assertion failed on the 600 s dwell row;
the test now explicitly preserves that discrepancy instead of normalizing it.

A coupled-life
prediction additionally needs credible within-hold stress relaxation and an
explicit mean-strain/interaction treatment. Peak-stress-versus-cycle curves
do not provide that missing time history. Do not extrapolate the 650 C
105–140 MPa rupture interval into the substantially higher cyclic peaks.

This advances source admission, not engineering qualification. Retain
observational scatter and outcome definitions when deciding the next model.

## Application boundary from the source discussion

The later discussion (PDF pages 186–190, sections 5.9.1–5.9.2) reinforces this
boundary: the cyclic tests deliberately use stresses above the creep-test
regime, and the high-temperature/high-stress tests are accelerated laboratory
conditions. A shared pipe source therefore does not by itself make a rupture
curve applicable to the cyclic peaks or to long-service extrapolation.
At 650 C the author reports one repeat each at 110 and 120 MPa, with 10% and
14% scatter. Those two repeat summaries are useful pedigree information, not
a statistical confidence interval for all temperatures or a universal
acceptance tolerance. No additional coupled-life formula or within-dwell
relaxation history is supplied by that discussion.
