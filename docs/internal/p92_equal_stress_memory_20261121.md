# Same-stress hold diagnostic: distinguish kinetics from prior history

The [BAM experimental report](https://doi.org/10.1016/j.ijfatigue.2022.107381)
reports distinct subsequent load/hold responses after long stress- and
strain-controlled dwells. Its abstract motivates a history discriminator;
it does not provide a constitutive parameter card. The separate open data
description and checksum-verified records remain the quantitative inputs.

## Comparison, not a new fit or tolerance

Use positive strain holds in cycles 1–3, common stress bands centered at
190, 210 and 230 MPa with half-width 5 MPa. Retain only contiguous windows
of at least three actual observations outside switch brackets. Compare the
observed nonelastic increment `delta(strain)-delta(stress)/E` per elapsed
time with a frozen trajectory over exactly the same stress endpoints. Find
the model's own crossing times; do not impose observed response times or
reset internal states. Missing or nonunique crossings remain unresolved.

The modulus is the frozen model value, not a separately measured modulus.
Recorded strain drift is retained. Perturbing each endpoint by one recorded
strain/stress quantum gives a deterministic resolution sensitivity, not an
experimental uncertainty or confidence interval. This diagnostic does not
replace the original hold, peak, strain or numerical acceptance screens.

## First-cycle evidence

At the 210 MPa band, SLR's inferred nonelastic rate is 1.286e-5 /s and
the model gives 1.185e-5 /s. SLC's observed rate is 4.645e-6 /s, whereas
the model gives 1.116e-5 /s. Their one-quantum sensitivities are 4.328e-6
and 1.677e-6 /s respectively. At 230 MPa, the corresponding observed
SLR/SLC rates are 2.996e-5 and 1.103e-5 /s; model rates are 3.066e-5
and 3.022e-5 /s. Thus the frozen reference largely loses the observed
post-creep suppression of subsequent tensile relaxation at similar stress.
These are finite-band averages, not instantaneous material functions.

The separate endpoint rate budget shows that branch B is inactive at the
first SLC creep-hold endpoint, while both recoverable rates have essentially
decayed; branch A supplies about 2.484e-8 /s. This explains the model's
own endpoint behavior, not the physical share of a measured mechanism.
The measured late SLC secant is resolution-limited and must not become a
new high-confidence calibration target.

## Consequence for the next mechanism decision

First finish the already-declared matched static-recovery experiment. An
aggregate cost reduction alone is inadequate: recheck the SLC post-creep
hold and same-stress distinction with frozen parameters. The constant
energetic coupling probe improves some inherited stresses but does not
recover this distinction sufficiently; do not launch an unrestricted fit.

If the tradeoff remains, investigate interaction in history-dependent
resistance/recovery between the slow and fast mechanisms, rather than a
protocol-specific rate multiplier. A candidate must have a defined stored
energy/dissipation treatment, bounded history, a zero-interaction limit,
signed/rotation-appropriate symmetry and independent integration checks.
No new internal state is admitted solely because it fits these three paths.
The two-mechanism thermodynamic literature permits interactions, but does
not identify which microscopic process governs this BAM cohort.

Evidence: `2026-11-21-equal-stress-hold.json` and
`2026-11-21-spectrum-rate-budget.json` under `evidence/creep/`.
PR #197 remains open, unmerged and unreleased.
