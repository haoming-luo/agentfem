# Incoming state versus hold kinetics: conditional diagnostic

This is **not** a new material fit, a permitted state reset, or a continuous
prediction. All material coefficients remain frozen. The purpose is to test
whether a small admissible change in the inherited state can substantially
alter the conflicting first post-dwell relaxation shape under the existing
rate law. A successful conditional reconstruction would still need a causal
history evolution to produce that state. A failed local reconstruction does
not prove global impossibility.

NASA/TP-20250003948 discusses state-space trajectories and the distinction
between exploratory, characterization and validation experiments:
<https://ntrs.nasa.gov/citations/20250003948>.
An indexed corrected edition, NASA/TP-20250003948/CORRECTED, p.38, explicitly
discusses loading-rate and step-relaxation tests for history dependence.
The report landing page and these indexed passages were accessible; complete
PDF downloads returned 403. No unverified equation or material coefficient
from that report is implemented here. This diagnostic is our own bounded
state-sensitivity experiment, not NASA certification or a published P92 law.

Use the frozen original spectrum and its causal first-cycle states for SLR
and SLC, immediately before their post-long-dwell positive strain hold.
Keep stress, total strain, accumulated inelastic strains, work and dissipation
fixed. Perturb four backstresses inside their existing invariant intervals.
Exchange fast and slow recoverable strain by equal and opposite increments,
so their sum remains unchanged. This fifth direction changes memory spectrum
but not the instantaneous elastic/inelastic strain partition. Backstress
energy can change, so each counterfactual trajectory uses its own initial
stored energy; it is not assigned a fictitious physical transition or work.

Scale backstress directions by their saturation values; scale the strain
exchange by the smaller of 120/K_fast and 120/K_slow. Bound each coordinate
by 0.3 and the invariant interval, without expansion. Examine the two-scale
Jacobian before a bounded, at-most-20-evaluation conditional reconstruction.
Use measured hold stresses, endpoint and resolved early/late secants with the
existing project stress and secant scales. No later cycles inform this test.
Record individual unchanged gates, rank and input/output identities. Do not
replace global response evidence with these isolated reconstructed holds.

Decision: if inherited-state changes cannot materially reduce the mismatch,
do not keep blaming only reverse loading or state preparation. If they can,
inspect the required directions and their reachability from the preceding
creep/reversal path before designing another evolution rule. No direct
injection of reconstructed states into a frozen validation run is allowed.

## Outcome

Both local derivative audits and independent integrations pass. The bounded
reconstruction improves SLC's normalized early-secant error from -1.697 to
-0.285 and endpoint error from -1.090 to -0.503 (absolute values <=1 satisfy
those individual original limits). Its late-secant error remains 2.065,
from 2.519 originally. All five coordinates reach bounds: the four
backstresses increase and strain moves from fast to slow recoverability.
No bound was expanded. These counterfactual states are not demonstrated to
be reachable from the preceding loading path.

For SLR the late-secant error falls from 1.404 to 1.195, still outside the
same limit; endpoint and RMS errors worsen slightly while remaining within
their individual limits. Neither conditional hold passes all checks.
Maximum independent stress-state differences are below 3e-7 MPa.

This bounded local experiment does not establish a global impossibility
theorem. It does show that correcting only inherited peaks or backstresses
has not removed the late-kinetics conflict in the explored neighborhood.
Keep investigating rate evolution as well as state preparation. The original
continuous prediction remains unchanged. Evidence:
`evidence/creep/2026-11-22-hold-state-discriminator.json`.

## Optimizer derivative re-audit

Review found that the optimizer used relative `diff_step`, although the
pre-fit derivative checks used absolute scaled-coordinate steps. At a zero
coordinate the former can fall back to a very small default step. Replace it
with an explicit bounded absolute 0.0005 Jacobian and test zero and boundary
coordinates against an analytic linear function.

The same two reconstructions still converge in 10 and 7 optimizer evaluations.
Their costs change by less than 1e-8; normalized late errors remain 1.19489
and 2.06507. Independent integration still passes. This correction improves
the numerical method but does not explain the physical response discrepancy.
The original evidence is retained alongside
`2026-11-22-hold-state-absolute-jacobian.json`, with separate driver/raw hashes.
