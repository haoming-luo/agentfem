# Isolated primary-creep discriminator

Predeclared research hypothesis, not a published P92 card or a production law.
Keep magnitude dynamic recovery and the fast recoverable branch unchanged.
Replace only the slow Kelvin branch by irreversible, strain-hardening primary
flow. Reuse its two positive coefficients H and tau; add no fitted coefficient.
The candidate has its own model/state identity: slot 11 is accumulated primary
strain p, not recoverable signed strain. Never reinterpret an old checkpoint.

For stress sigma in MPa, set s0=120 MPa (the existing flow normalization),
v=sigma/(H*tau)*exp(-H*p/s0), pdot=abs(v). The primary element stores no energy;
its dissipation is sigma*v>=0. At constant stress from p=0,
p=(s0/H)*log(1+abs(sigma)*t/(s0*tau)). Thus initial rate is finite,
decelerates continuously, and gives logarithmic primary creep. Reversal
changes flow sign but does not erase accumulated history; zero stress gives
zero primary flow. These are explicit hypotheses, not measured P92 facts.
H and tau now control irreversible kinetics, not recoverable
compliance. This replacement can therefore fail unloading/hold tests.

Motivation: Khayatzadeh et al., Engineering Fracture Mechanics 175 (2017),
DOI 10.1016/j.engfracmech.2017.02.008, reports that primary creep matters to
P92 relaxation and compares strain/time hardening. Its 650 C data and
monotonic law do not establish this cyclic 620 C extension. The regularized
law above is our bounded discriminator, not a transcription of their equation.
Source: https://strathprints.strath.ac.uk/60381/

More directly, Alam, Yamamoto and Odette (2025), Metals 15, 354,
https://doi.org/10.3390/met15040354, Eq. (1), describes an initial excess rate
decaying exponentially with strain, above a minimum creep rate. Here the
existing backbone retains the other flows, and only the replaced branch uses
this excess-rate form. Our stress-proportional initial rate, s0/H strain scale
and absolute-history extension to reversal remain explicit modeling choices.
Their 375--550 C, 10--80000 h database is NOT calibration evidence for our
620 C, seconds-to-minutes mixed-control cohort.

An exploratory rational-decay version was stopped during its derivative audit
(before optimization) after this more direct source was located. Its seed cost
was 0.005264396768738119. It is not a completed matched comparison, and no
acceptance conclusion follows from it. The exponential version and its
analytic checks below supersede that provisional choice before identification.

Before fitting: constant-stress analytic check, stress reflection, zero-load
memory, nonnegative dissipation, integrated work balance, split continuation,
independent BDF integration, strict state identity. Then use the existing
early cycles 1--3 objective, 22 candidate/10 retained coordinates, norm .3,
40 optimizer-evaluation ceiling and unchanged two-scale derivative gate.
Keep all three protocols, actual ramp shape and inherited SLC force control.
Freeze for cycles 4--10, which are inspected development data, not blind data.
No altered response tolerance, added parameter, directional-law stacking,
merge or release. Report a failed derivative gate rather than fit through it.

## Implementation checks

The new kernel has nine focused tests: analytic constant-stress solutions of
both signs and zero stress, independent integration, split continuation,
instantaneous and integrated mechanical balance, reflection, no history reset
at zero stress or reversal, distinct decoding and prohibited combinations.
All 661 P92 tests pass in the real `fenicsx-env` (42.22 s); correctness lint
and generated-document checks also pass. An earlier unprivileged broad-test
process was terminated with exit 143 and is not counted as a test success.
These are software/numerical checks, not experimental validation.
After adding the archive identity/budget/failure regression, the final batch
passes all 662 P92 tests (42.65 s), plus lint, generated docs and diff checks.

## Matched identification

The unchanged two-scale gate passes: relative Jacobian difference
0.00046072435 and retained-subspace distance 0.0033220820. The early-only
fit terminates normally by `ftol`, with 13 optimizer evaluations and 323
actual residual evaluations. The baseline used 8 and 238 respectively;
the budget ceilings, not the actual consumption, are identical. Two retained
directions reach bounds, which were not enlarged. The final objective is
0.0034764816557672994, **47.26% above** the magnitude-control objective
0.0023607800272608744. This is not an improvement in joint early prediction.

The archived matched JSON is byte-identical to the generated result
(SHA256 eb631069b2ba20fddf5fd42a00c399168abdcee8d03adcf7d43c81162db04cfe).
Reproduce with `tools.p92_spectrum_recovery_joint`, the archived
`2026-11-21-spectrum-frozen.json` seed, checksum-verified source directory,
and `--primary`. Freeze with `tools.p92_recoverable_frozen` and that matched
JSON. The first frozen run was intentionally restarted after adding the new
kernel to its code-identity guard; the fit itself and material equations did
not change. Only the complete identity-guarded run supplies frozen evidence.

## Frozen outcome: reject this replacement

All three continuous ten-cycle paths pass independent BDF and waveform
refinement checks. The maximum independent stress-state difference is
1.13e-6 MPa, and mechanical balance residual magnitudes are below 6.7e-12 MPa.
All three experimental response screens nevertheless fail.

| Observable | Magnitude control | Primary replacement |
| --- | ---: | ---: |
| Failed checks RF / SLR / SLC | 21 / 40 / 74 | 17 / 56 / 63 |
| SLR cycle-10 peak error | 8.93% | 12.38% |
| First SLC post-creep hold anchor error, MPa | -0.302 | -7.674 |
| Same hold early anchored-shape RMS, MPa | 5.756 | 3.903 |
| Same hold late anchored-shape magnitude, MPa | 13.850 | 12.010 |
| Same hold early actual RMS, MPa | 5.903 | 8.075 |
| Same hold late actual error, MPa | -14.152 | -19.684 |

The shape improvement is real in the anchored diagnostic, but the worsened
incoming state makes the actual hold response worse. Failure counts are
diagnostic, not interchangeable severity weights or an acceptance score.
Neither an anchored improvement nor fewer SLC failed checks overrides the
unchanged absolute response criteria. The frozen cycles are inspected
development data; this is not independent experimental qualification.

This rejects the bounded direct replacement, not primary creep in general.
It does not identify irreversible primary creep as the unique missing cause.
The next discriminator must preserve the recoverable response and jointly
constrain the inherited creep-hold stress/strain and subsequent hold entry,
before attributing improvements to relaxation kinetics. Any parameter
substitution must first audit weak directions; do not add coefficients or
repeat the rejected static-history adjustment without new evidence.

Evidence under `evidence/creep/`: `2026-11-22-primary-exp-matched.json`,
`2026-11-22-primary-exp-frozen.json`, `2026-11-22-primary-exp-residual.json`.
No production promotion, response-gate change, merge or release.
