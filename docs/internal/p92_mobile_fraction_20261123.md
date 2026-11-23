# Mobile-fraction replacement: prospective bounded test

The preceding frozen-rate fingerprint separates recoverable strain rate from
irreversible flow. A strength-memory replacement also widened faster reverse
flow and failed both protocols. Test mobility rather than another strength
multiplier, without interpreting simulated rate centers as measured backstress.

## Literature and our reduction

[Basirat et al. (2015), equations 7, 9–14](https://doi.org/10.3390/met5031487)
describe mobile-dislocation density and generation/annihilation in modified
9Cr–1Mo creep. This motivates a kinetic population, not transfer of a P91 weld
card to P92. [Zheng et al., anelastic recovery](https://www.sciencedirect.com/science/article/abs/pii/S0020740316305690)
supports retaining recoverable deformation in P92 cyclic holds. Neither source
publishes the following reduced equation or validates it for our 620 C cohort.

Let z be an immobilized fraction (initially zero), m=1-z, and multiply both
original irreversible reference rates by m. Keep the four signed backstresses,
both recoverable modes, strengths and stored-energy function unchanged:

    dz/dt = a (1-z) - b (p_dot_A+p_dot_B) z.

Here a is a time-controlled immobilization rate and b a strain-controlled
remobilization coefficient. This is an explicitly phenomenological reduction,
not a measured dislocation density. It changes dissipation mobility, assigns
no new stored energy, and does not use protocol labels, hold detection, load
sign switches or measured response as input. At z=0 the derivative is a;
at z=1 all irreversible rates vanish and the derivative is zero. The domain
[0,1] is invariant. For steady positive *actual* speed v the equilibrium
fraction is a/(a+b*v); high v reduces blocking. This does not guarantee that
a transient reverse peak stays unchanged: that is a required test.

For a fixed backbone state with unblocked speed V, substitution v=(1-z)V
gives dz/dt=(1-z)(a-bVz). Its mobile steady branch exists only for bV>a,
where v=V-a/b; otherwise z tends to one. Thus this hypothesis introduces a
kinetic low-rate cutoff and an absorbing fully blocked boundary. It is not
universal smooth creep and cannot be presented as proven microscopic physics.
Finite rests retain m=exp(-a*t) in the zero-flow limit, but long-rest numerical
resolution and reverse transient strength need special scrutiny. This
analytical limitation is recorded before identification results are examined.

Replace A.static1 and A.static2 by a and b, setting the removed rates to zero.
The independent coefficient count remains 22; one dimensionless state is
added and both recoverable strains are retained. Initial screening uses
a=1/1800 per second and b=250 per strain, fixed before replay, with original
and deletion-only controls. These are diagnostic starting scales, not fitted
measurements. No parameter sweep is allowed to rescue a failed screen.

## Gates

1. Verify zero-immobilization baseline, sign symmetry, positive dissipation,
   mechanical work identity, invariant bounds, continuation, and independent
   Radau/BDF integration. No mixed mechanism options.
2. Replay the actual first three cycles for RF, SLR and SLC continuously with
   one card. Compare SLC entry, early/late holds and creep accumulation along
   with SLR peaks. Report deletion separately. A lower objective alone is not
   admission. If the fixed-screen mechanism worsens both target conflicts,
   reject it without another optimizer run.
3. Only a promising screen enters the existing 22-coordinate/10-direction,
   radius-0.3, maximum-40-evaluation identification. Reuse the same seed,
   observations and gates, freeze before ten-cycle development checks. No
   later-cycle observation enters the objective; these cycles are not blind.

PR #197 only. No core material promotion, merge, release or life claim.

## Fixed early screen

All three arms complete RF/SLR/SLC cycles 1–3. Objective values are original
0.002239828765, deletion-only 0.002238959861 and population 0.004465280612.
The first SLC post-creep endpoint residual improves from -11.8097 to -1.9517
MPa; early secant improves from -5.37028 to -4.65510 MPa/s against -3.79373.
However entry residual grows from +1.1160 to +8.4632 MPa, and SLR cycle-3 peak
increases from 282.3525 to 301.4327 MPa. All nine original response screens
fail. This is a mixed mechanism discriminator, not an improvement claim.

One predeclared bounded joint-identification run tests whether the favorable
SLC kinetic direction can coexist with the peak constraints. No changed start,
weight, expanded range or subsequent rescue fit is permitted. The original
same-budget control already exists in
`2026-11-23-aging-resistance-control-matched.json`: its seed hash equals this
run's `2026-11-23-rate-recovery-control-frozen.json`, and the unchanged original
kernel/control path remains regression-tested. However the strict paired
report also requires equal driver hashes, including the identification
dispatcher changed by this batch. Reuse alone does not meet that gate. A fresh
original control is therefore run with this exact driver, rather than weaken
the report's identity check or relabel the old evidence. The archived control
remains a reproducibility cross-check, not the new pair's substituted input.

Local engineering checks before completion of identification: 1225 P92 and
CI-classification tests passed in 74.88 s; Ruff, generated-document freshness
and whitespace checks passed. This does not admit the physical model.

## Frozen candidate outcome

The candidate converged in 6 optimizer evaluations / 216 residual calls,
cost 0.002247963385. Its local observability audit passed: two-step Jacobian
difference 1.1100e-4, retained-subspace distance 3.2883e-4, residualized
history-direction ratio 0.35012, minimum retained-coordinate Gram eigenvalue
0.13836. One retained coordinate reaches its bound; no bound is expanded.
Frozen rates are a=0.000388590501248 /s and b=287.501101091 /strain.

All three ten-cycle independent-integration/refined-waveform checks pass;
all three original response screens fail. At the first SLC post-creep hold,
entry residual is -0.21836 MPa, endpoint 166.71688 MPa against 174.88 MPa,
early secant -4.71912 against -3.79373 MPa/s, and late secant -0.085841
against -0.181928 MPa/s. Preceding creep endpoint strain is 0.00101196
against 0.00106. Despite these improvements over the original matched card,
SLR cycle-10 peak error is 10.06745%, versus the archived original 7.6481%.

**Reject this bounded shared-population replacement.** It has not separated
SLC kinetic correction from SLR reverse strength sufficiently. Its initial
steady high-rate limit did not guarantee a benign transient history. This
rejects this specified reduction, not every dislocation-population model.
Do not relaunch with a different time scale, expand bounds, ignore late
secants, or promote it to a production P92 material. The ten-cycle data remain
inspected development data, not independent validation or life evidence.

The useful next question is branch-resolved *causal* selectivity, not whether
another shared positive multiplier can lower aggregate loss: test whether
the recorded histories distinguish the carrier of the SLC excess flow from
the carrier of SLR reload strength before choosing another equation. The
previous observed-stress replay already implicates both branches, so merely
renaming one branch "creep" is not evidence. Do not launch another fit without
discriminating evidence; preserve the unchanged admission gates.

## Same-driver control and paired closure

The fresh original control converges in 3 optimizer evaluations / 153
residual calls to 0.0022088403742237975, exactly reproducing the archived
control cost. Its three frozen numerical checks pass and three response
screens fail. The paired report checks identical seed, source, driver hashes,
22/10 parameter-direction budget and radius, then re-evaluates every physical
gate from retained observations. Candidate objective is 1.7712% higher, not
a speed or accuracy improvement. All six card/protocol numerical checks pass;
all six physical response screens fail. No subsequent fitting was performed.

The six `2026-11-23-mobile-fraction-*.json` records retain fixed screening,
both matched fits, both frozen histories and the strict paired comparison.

Final local regression: all `tests/test_p92_*.py` and the CI-scope tests
passed, **1,227 tests in 71.61 seconds**. This verifies implementation and
retained evidence contracts; it does not change the failed physical screens.
