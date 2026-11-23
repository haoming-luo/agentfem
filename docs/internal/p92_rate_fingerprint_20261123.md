# Separate hold kinetics from reverse-loading strength before another fit

The aging-resistance replacement reduced the objective but worsened SLR
peaks and SLC hold entry. Do not continue that fit. The existing incoming-state
reconstruction and history/kinetics parameter derivatives are different
diagnostics and must not be repeated under a new name.

## Primary evidence and proposed discriminator

[Yaguchi and Takahashi / CRIEPI T96017](https://criepi.denken.or.jp/hokokusho/pb/reportDetail?reportNoUkCode=T96017)
separated backstress, overstress and aging stress using rate changes and strain
dip tests in modified 9Cr–1Mo steel. This motivates measurements, not the
transfer of their constants to P92. The P92 study
[Khayatzadeh et al. (2017)](https://doi.org/10.1016/j.engfracmech.2017.02.008)
also distinguishes forward creep from stress-dependent primary relaxation;
its 650 C heat and elastic follow-up tests are not this 620 C cohort.

Before a mechanism replacement, compute a **frozen-state rate fingerprint**
at causal positive-stress hold entry and exit in cycles 1–3 of RF, SLR and SLC.
Keep the additional SLR zero-strain 1800 s hold separate from its positive-
strain 180 s peak hold. Record duration and imposed strain with every station.
Use the completed matched original and aging cards without fitting. At each
state, query both signs of stress at the same internal history. Invert net
irreversible flow at prescribed magnitudes 1e-7 through 1e-3 /s. Report the
positive/negative stresses, their center and half-width at every rate.
The center is not automatically a measured backstress; with two branches it
can itself depend on rate. Width is not automatically a yield stress.

Repeat using total nonelastic strain rate, including both recoverable modes.
Their difference tests whether a zero observed strain rate or a finite-rate
dip can be mistaken for zero irreversible flow. An exact common inactive
stress interval is reported when it exists; otherwise opposing irreversible
branch flows can cancel without either branch being inactive. Do not assign
a unique backstress to such cancellation or to a zero-rate plateau.

Stress scans are independent instantaneous counterfactual queries, not a
physical stress jump, state reset, energy-free transition or continuous
validation path. The archived histories remain unchanged. No measured stress
or strain response is overwritten. Numerical root verification uses Brent
and independent bisection with unchanged prescribed rate levels. Inputs,
driver hashes, root residuals and state identities are retained.

## Admission, not automatic mechanism selection

The fingerprint first asks whether a candidate changes low-rate hold response
without moving high-rate reverse-flow strength in the wrong direction, and
whether the distinction comes from irreversible or recoverable response.
Only the first three cycles inform this diagnostic. The archived ten-cycle
response failures remain failures. A synthetic fingerprint alone cannot
validate a mechanism: it must be consistent with the recorded creep,
hold-entry, hold-shape and reversal observations before another fit.

If the evidence does not discriminate a replacement, do not launch another
optimizer, vary coefficients or relabel this as frozen experimental validation.
An admitted replacement retains both recoverable modes, replaces rather than
adds independent coordinates, and uses the original same-start 22/10-direction,
radius-0.3 / 40-evaluation comparison. Freeze before later-cycle checks.
PR #197 only; no merge or release.

## Completed diagnostic and decision

Both frozen cards completed 24 stations (RF 6, SLR 12, SLC 6), each with
five rate magnitudes, two signs and two channels. The initial test assumed
only 18 stations and failed: SLR also has a positive-stress zero-strain
intermediate hold. The test and coverage declaration were corrected, not the
physical history; neither failed command published evidence.

At first peak-hold entry, the original card's paired-stress centers at
1e-6 /s are 40.577 MPa (SLR) and 42.523 MPa (SLC) for irreversible flow,
but 193.726 and 179.050 MPa for total nonelastic flow. Recoverable history
therefore strongly masks a dip interpreted from total rate alone. These
are simulated rate fingerprints, not measured internal stresses.

At 1e-3 /s the irreversible half-width changes from 275.573 to 279.166 MPa
for SLR and from 275.267 to 279.522 MPa for SLC with the rejected aging card.
At 1e-6 /s those width changes are only +0.645 and +1.214 MPa. The candidate
has not selectively corrected low-rate holds while leaving faster reverse
flow unchanged. Between-card differences include refitted coefficients and
changed incoming states; they cannot all be attributed causally to aging.

Evidence: `2026-11-23-rate-fingerprint-control.json` and
`2026-11-23-rate-fingerprint-aging.json` in `evidence/creep/`. Inputs and
driver identities are hashed; every inverse is checked by independent
bisection, not only solver success. Neither diagnostic changes parameters.

**Decision:** no new mechanism is admitted by this comparison alone, and no
repeat identification is launched. Next selection must explain the earlier
observed-stress replay conflict as well as these separate channels. Require
a physically justified kinetic/history replacement with unchanged coordinate
count before the same-budget fit; then retain the original later-cycle gates.
The existing six response failures remain failures.

Additional primary P92 evidence supports distinguishing effects rather than
assuming a single strength multiplier: [Khayatzadeh et al. (2017), thermal
ageing](https://pureportal.strath.ac.uk/en/publications/influence-of-thermal-ageing-on-the-creep-behaviour-of-a-p92-marte/)
reported changed creep behaviour and rupture life but no evidence of an
ageing effect on relaxation rate in their tests. This is a different heat,
650 C and thermal pre-ageing, not a validation of our dynamic-aging model.

Local verification: 134 focused diagnostic/classifier/aging tests passed;
the complete P92 test selection passed 1103 tests in 71.49 s in fenicsx-env.
These selections overlap and must not be added as distinct test counts.
Ruff, generated-document freshness and whitespace checks passed. These
software checks do not overturn the failed physical response screens.
