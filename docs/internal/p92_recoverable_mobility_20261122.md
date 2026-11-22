# Prospective recoverable-memory mobility discriminator

## Observable conflict and distinct hypothesis

At the approximately 210 MPa first positive-hold interval, the original card's
slow recoverable strain is about 1.68e-4 in SLC and 0.86e-4 in SLR. That is a
larger history distinction than their almost equal accumulated strains. The
B flow dominates the excessive predicted SLC relaxation. An interaction
through the recoverable potential was tested separately and did not improve
the declared joint screen. A positive state-dependent mobility is a different
hypothesis: it changes kinetics, not stored energy or conjugate stress.

Retain both Kelvin strains, both irreversible branches and the original
directional rule. Let x=K_slow*a_slow/120 MPa and f=1/(1+x^2). The 120 MPa
normalization is the existing backbone rate normalization, not a fitted new
scale. Compare three fixed laws: unchanged; multiply B reference flow by f;
multiply B static-recovery rates by f. No new coefficient or state is added.
Do not combine the two changes or try further powers/scales after seeing
results. Both signed strain directions give the same positive mobility.
Stored energy remains unchanged and each dissipative term remains nonnegative.

This is an explicitly proposed scalar kinetic hypothesis. The
[two-mechanism thermodynamic framework](https://journals.ub.ovgu.de/index.php/techmech/article/download/766/743)
supports distinguishing energetic and dissipative interaction, but does not
supply this function or validate it for P92. It is not a derived microscopic
law or a reproduction of the Benaarbia P91 model. The potential's positivity
is necessary, not sufficient for physical acceptance.

## Before evaluation

Use the unchanged spectrum seed, measured ramp controls, cycles 1–3 and all
three protocols. Validate zero-memory identity, signed symmetry, both control
modes, instantaneous/integrated power accounting, independent integration and
failure retention. Preserve creep accumulation, hold entry, unshifted hold
stress and resolved secant gates. Fit only if a declared arm completes and
improves the joint objective with compatible SLC hold/creep behavior; a later
fit must retain the existing parameter count, radius and evaluation ceiling.
No later-cycle data or artificial state reset enters this screen.

## Fixed-card result and bounded next experiment

The unchanged, flow-feedback and static-recovery-feedback costs are respectively
0.00257608010, 0.00269979936 and 0.00248316475. Reject the flow-feedback arm.
For static recovery, the first SLC positive-hold end rises from 159.025 to
163.743 MPa (observed 174.88 MPa); its RMS falls from 11.634 to 10.804 MPa.
The preceding creep-end strain changes from 0.00103459 to 0.00102895
(observed 0.00106), while its RMS improves. The early relaxation secant
improves slightly, but the late secant becomes worse: -0.08312 rather than
-0.09179 MPa/s, against observed -0.18193 MPa/s. This is not acceptance.

Admit only static-recovery feedback to the unchanged 22-coordinate,
10-retained-direction, 0.3 log-radius and 40-optimizer-evaluation experiment.
Do not fit the normalization, exponent or additional history coefficients.
All three early protocols remain in the objective. After convergence, freeze
the card and check cycles 1–10 with the original stress, strain and resolved
secant gates and independent integration. A lower cost cannot override a
failed hold-slope or later-cycle gate. This decision precedes that fit.

## Matched identification and frozen result

The fit converges in six optimizer evaluations and 216 residual evaluations,
with unchanged 22 candidate / ten retained coordinates and radius 0.3.
Cost is 0.002304008534, 2.405% lower than the matched original's
0.002360780027. Three retained coordinates reach their existing bounds; they
are not enlarged. Both recoverable modes and the original directional rule
remain intact. A final constructor guard prevents accidentally combining
this mechanism with the alternate direction rule; it does not alter this
original-direction card's equations.

All three ten-cycle independent-integration and waveform-refinement checks
pass. All experimental response screens fail: RF has 20 failed checks,
SLR 45, and SLC 76 (plus ten unresolved late-creep secants). These counts
are not independent statistical samples or a quality score.

At the first SLC post-creep hold, entry error worsens from -0.302 to
-2.331 MPa, while the final observed-sample error improves from -14.152 to
-10.619 MPa. The predicted creep-end strain is 0.00101089 versus observed
0.00106. The early relaxation secant is -5.047 versus -3.794 MPa/s; the
late secant is -0.08296 versus -0.18193 MPa/s. Both relaxation-rate checks
remain outside the unchanged 20% limit. Frozen SLR cycle-10 peak error
worsens from 8.93% to 9.90%. The improvement is therefore a redistribution
of error, not acceptance or an engineering material upgrade.

Archive: `2026-11-22-recoverable-mobility-{matched,frozen,residual,hold-partition}.json`
under `evidence/creep/`. Retain this negative result; no second fit, wider
bounds, promotion, merge or release follows from it.
