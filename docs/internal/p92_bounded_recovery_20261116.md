# P92: bounded recovery, dwell life and observed scatter

## Decision

The bounded recovery hypothesis improves late RF softening, but **does not pass
cross-protocol material validation**. Keep it in repository research tools, not
the installed material catalogue. A completed response path is not a failure
prediction; neither this law nor the original five-state response law has a
rupture variable.

The earlier direct linear-softening substitution failed at cycle 475. Its
[failure record](p92_softening_discriminator_20261116.md) remains part of model
selection, not an experimental rupture result.

## Bounded mechanism experiment

`tools/p92_bounded_softening.py` modifies the dynamic recovery of both backstress
components by `phi = 1 + (limit-1)*(1-exp(-rate*p))`. The frozen early-RF base
parameters are unchanged; `limit=2`, `rate=0.1` are sensitivity choices, not
independently calibrated P92 properties. The saturating form is informed by
bounded cyclic-recovery models, including the [public HOWEFLEX report](https://edocs.tib.eu/files/e01fb24/1895473101.pdf),
pp23–24. That report concerns MarBN; its material parameters are not transferred.

An analytical Jacobian, identity limit, restart, and exact nominal SLR path tests
check the implementation. Each cycle uses the previous accepted state. SLR
includes its 1800 s zero-strain hold and opposite starting direction; it is not
RF with an appended pause. SLC remains stress controlled and is not admitted by
this driver.

| RF cycle | Predicted peak, MPa | Measured peak, MPa |
|---:|---:|---:|
| 10 | 268.9058 | 276.94 |
| 100 | 222.5428 | 237.26 |
| 200 | 210.7273 | 220.75 |
| 400 | 198.3649 | 203.76 |
| 500 | 194.3929 | 198.74 |
| 1000 | 184.4500 | 181.74 |

The cycle-1000 error falls from the original bounded-strength baseline's +9.46%
to +1.49%. This does not erase earlier errors: all 118 available recorded cycle
extrema through cycle 1000 are retained. The 115 post-training recorded cycles
give 8.23691 MPa RMS and 17.30568 MPa maximum absolute peak/valley error.
Nominal input is simulated continuously through missing recorded cycles; those
cycles are not described as measured histories.

Tightening Radau relative tolerance from 1e-8 to 1e-10 changes the largest sampled
peak by 1.84054e-8 MPa. This numerical discrepancy is negligible compared with
the experimental residual. An exploratory 2054-cycle run also completed, but
completion at the observed failure count is not predicted life.

SLR fails the transfer screen: at cycle 10, the predicted tensile peak is
231.1517 MPa versus 263.33 MPa (-12.22%). Across cycles 1–10, peak/valley RMS
error is 23.6777 MPa. The first nominal zero-strain hold ends at 110.3334 MPa;
the discrepancy in low-stress relaxation is not solved by the bounded recovery.
RF and SLR are already inspected development records, not new blind holdouts.

Reproduce the numerical comparison without refitting:

```sh
python -m tools.p92_bounded_softening evidence/creep/2026-11-16-bam-positive-strength.json --cycles 1000 --limit 2 --rate .1 --output coarse.json
python -m tools.p92_bounded_softening evidence/creep/2026-11-16-bam-positive-strength.json --cycles 1000 --limit 2 --rate .1 --rtol 1e-10 --output fine.json
python -m tools.p92_candidate_report coarse.json fine.json /path/Data_RF.lis
python -m tools.p92_bounded_softening evidence/creep/2026-11-16-bam-positive-strength.json --cycles 10 --protocol SLR --limit 2 --rate .1 --source /path/Data_SLR.lis --output slr.json
```

Evidence: `2026-11-16-bounded-recovery-screen.json` and
`2026-11-16-bounded-recovery-slr.json` under `evidence/creep`. Raw experimental
files remain external and checksum verified.

## Life and uncertainty, separately verified

`tools/p92_dwell_life_reference.py` reproduces [Wang et al. 2019](https://doi.org/10.3390/met9020183)
Table 2 at 600 C and equations 10/11 with Table 5 hold parameters. The 600 C,
0.004-amplitude tensile/compressive hold predictions have observed/predicted
ratios retained individually; prediction/observation ranges from 0.8573 to
1.5713. These are same-paper parameter checks, not independent validation.
The printed temperature equation repeats a quadratic term, so it is not silently
corrected or used for interpolation. Combined holds and untested dwell domains
are rejected. The empirical factor is not constitutive stiffness damage.

`tools/p92_replicate_scatter.py` uses the existing generic survival API for the
18 reported pure-fatigue tests, kept in 11 temperature/amplitude groups. At
600 C and amplitude 0.004, the four lives are 1426, 2045, 2120 and 2186 cycles.
At 2045 cycles the empirical survival is 0.5; the approximate pointwise 95%
Greenwood interval is [0.05785, 0.84486]. Four tests plainly do not establish
high-confidence design reliability. No synthetic runouts or pooled-condition
scatter are introduced. The source's unresolved failure definition remains
visible; this is not a component allowable or an individual-life interval.

## Next model/data gate

The [2025 Grade 92 work](https://opus.hs-offenburg.de/files/11628/SeifertSchlayerMaier_PlastCrpFat_FVWHT_2025.pdf)
supports investigating separate time-scale inelastic mechanisms and validating
creep, cycling and mixed-control paths together. It does not supply a complete
numeric parameter card/raw dataset for this BAM cohort. Do not bolt an arbitrary
extra creep term onto the current law and call it validated.

The NIMS 48B [official metadata](https://doi.org/10.11503/nims.1052) was checked:
the public repository page has no attached numeric file. The linked data service
requires registration/use permission; no access controls were bypassed. Existing
2019 prior-damage publications reuse the already audited tests, not new specimens.

The next defensible gate is a jointly calibrated, explicitly identified
multi-protocol response model plus independent life observations with compatible
failure definitions. The present evidence establishes reproducible numerical
experiments and quantified deficiencies, not engineering qualification.

## Standard-informed acceptance, not a universal error band

### Bounded joint-protocol calibration

`tools/p92_joint_protocol_fit.py` fits only RF and SLR cycles 1–3, with
separate virgin hidden-state assumptions and continuous history within each
specimen. The six scaled parameters are modulus, positive strength margin,
viscosity, softening target/rate and isotropic static-recovery scale. The
original constitutive equations are unchanged. Later inspected records remain
development checks, not blind validation.

The initial 40-evaluation budget was exhausted and is archived without later
predictions. One bounded extension to 120 evaluations converged at 45. Its
time-weighted objective misses the RF cycle-10 peak (244.894 vs 276.94 MPa),
although SLR cycle-10 time-weighted RMS error is 14.530 MPa. The maximum
BE/Radau stress difference across both ten-cycle paths is about 1.25 MPa;
the experimental residual is larger. The scaled Jacobian singular values span
roughly five orders of magnitude. This is a sensitivity warning, not a
statistical confidence interval or a global-optimality certificate.

A second, predeclared 50:50 physical-time/total-strain-variation objective gives
rapid loading weight without discarding holds. It converged in nine evaluations
but reached several parameter bounds. RF cycle-10 peak becomes 283.724 MPa,
while its time-weighted RMS error increases to 33.675 MPa; SLR cycle-10 RMS
error is 25.636 MPa and its maximum error is 71.605 MPa. Costs under these two
different objectives are not directly comparable. Neither result is promoted.

All three records are archived as `2026-11-16-joint-fit-*.json`. This finite
experiment rejects simply retuning this parameter subset as the next product
step; it does not prove that every possible parameterization fails. A separate
time-scale mechanism needs its own equations, identifiable calibration data and
multi-protocol validation before core integration.

```sh
python -m tools.p92_joint_protocol_fit /path/Data_RF.lis /path/Data_SLR.lis --max-nfev 120 --output joint-time.json
python -m tools.p92_joint_protocol_fit /path/Data_RF.lis /path/Data_SLR.lis --max-nfev 120 --balance-strain --output joint-balanced.json
```

### Applicable test scopes

The official [ASTM E2714-26 scope](https://store.astm.org/standards/e2714)
distinguishes cyclic deformation and crack formation, recommends complementary
cycling and creep data for the same material/temperature, and limits its scope
to specimens rather than complete components. [ISO 12106:2017](https://www.iso.org/standard/64687.html)
covers axial-strain-controlled fatigue; [ISO 204:2023](https://www.iso.org/standard/82162.html)
covers uniaxial tensile creep. Their publicly available descriptions do not
establish a universal 5%, 10% or factor-two model-acceptance band. We have not
audited the paywalled full procedures, and do not claim standard certification.
The ISO 12106 successor was still listed as FDIS on this review date.

Model admission therefore requires an explicit intended use and compatible
failure definition, test-condition coverage, independent specimens, integration
error below experimental/model error, and reported prediction/scatter limits.
A good stress peak does not by itself qualify crack-initiation or rupture life.
