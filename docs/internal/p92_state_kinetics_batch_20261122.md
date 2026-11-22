# P92: distinguish inherited state, reversible strain and hold kinetics

This batch remains on PR #197. It does not change production materials,
release status, experimental gates or the distinction between response and
life qualification. The matched original cost is 0.0023607800272608744.

## Completed discriminators

| Experiment | Evidence | Decision |
| --- | --- | --- |
| Critical-surface dynamic recovery, matched identification and ten-cycle replay | [Critical recovery](p92_critical_recovery_20261122.md) | Numerical checks pass; joint cost increases and all three response screens fail |
| Conditional incoming-memory reconstruction | [State discriminator](p92_hold_state_discriminator_20261122.md) | Even reconstructed incoming states do not resolve late-hold error; not a reachable-state prediction |
| Positive recovery-mode rotations and timescale exchanges | [Mode experiments](p92_static_modes_20261122.md) | Fixed-card costs worsen; no fitting launched |
| Recorded stress ripple and combined measured-channel replay | [Input audit](p92_stress_ripple_20261122.md) | Independent integration and refinement pass; first SLC hold endpoint remains about 14 MPa low |
| Conditional observed reversible-strain budget | [Budget](p92_observed_recovery_budget_20261122.md), [211-card sensitivity](p92_recovery_budget_sensitivity_20261122.md) | Same-stress history distinction persists over the declared finite sample; not statistical uncertainty or direct plastic-strain measurement |
| Shared fast/slow kinematic memory | [Common-memory experiments](p92_common_memory_20261122.md) | Both mapped cards worsen joint error; no fitting |
| Both published elastic-modulus values | [Modulus audit](p92_modulus_discriminator_20261122.md) | Neither fixed substitution removes the conflict; sonic and mechanically inferred values are not interchangeable specimen measurements |
| Backstress-to-isotropic cross-resistance substitution | [Cross-resistance](p92_cross_resistance_replacement_20261122.md) | Joint error increases about 31%; no fitting |
| Consistent rate-independent fast branch plus creep | [Plastic consistency](p92_inviscid_branch_20261122.md) | Better isolated SLC early slope but worse entry and total error; no fitting |
| Causal history/direct-kinetics derivative split | [Derivative evidence](p92_history_kinetics_split_20261122.md) | All 18 numerical checks pass; incoming-history effects can reverse locally beneficial kinetics changes |
| Shared first compression/hold before protocol divergence | [Common prefix](p92_common_prefix_20261122.md) | Measured mean difference -11.42 MPa; recorded-ramp fixed-card prediction +0.30 MPa; no scatter distribution or residual correction inferred |

Failure of a fixed mapping does not exclude its entire constitutive family.
These negative experiments prevent promoting an attractive isolated endpoint
or a new equation name without continuous-path evidence.

## Diagnostic correction

The original first SLC post-creep hold predicts about 160.728 MPa versus
174.880 MPa at the **last sampled point**. This is not its entry error: the
first sampled stress residual is approximately -0.302 MPa. Hold summaries
now name both residuals and the anchored shape separately. The anchored view
never replaces the original absolute errors or response gates.

This distinction matters: a model that improves only the endpoint while
moving the entire hold downward at entry is not an improvement of the same
prediction. Output validation also rejects invalid sample shapes/times.

## Completed bounded experiment

The [existing recovery-shape substitution](p92_recovery_shape_20261122.md)
uses the two B ratcheting exponents in place of two weak A static-recovery
coordinates. It retains all reversible strains, does not create parameters,
and uses a same-start additional-budget control if fitting is admitted.
Derivative checks preceded fitting and frozen response checks followed it.
The same-budget control cost is 0.002275663761, versus 0.002275795828 for the
replacement. The apparent gain over the older seed is continued optimization,
not a new-mechanism benefit. All six numerical checks pass; all six response
screens fail. [Nominal-rate controls](p92_autonomous_controls_20261122.md)
leave the first SLC hold endpoint about 12.4 MPa too low. The
[signed flow budget](p92_hold_flow_budget_20261122.md) shows that modifying
recoverability alone cannot explain the frozen-path mismatch: early signed
irreversible flow is too fast, late flow too slow even before recovery offsets.
This is an accounting discriminator, not a global model-family exclusion.

Do not replace the unresolved response claim with a life-uncertainty claim.
Reliable life inputs still require credible stress–strain history. Keep
already inspected cycles labelled development evidence and reserve genuinely
independent conditions for a later validated use claim.

## Local engineering checks

The serial fenicsx-env P92 suite plus CI-scope tests passed **1056 tests in
61.79 seconds** on this batch. Changed Python correctness lint, generated-doc
consistency and local links in the 14 affected/new roadmap documents passed.
These checks include the bounded absolute-Jacobian correction and preserve
all negative scientific conclusions. They are not experimental acceptance.
Those counts describe the earlier checkpoint, before the final paired report,
common-prefix and autonomous-control diagnostics; final batch checks follow.

The completed paired/common-prefix/autonomous/signed-budget batch passes
**1091 tests in 61.61 seconds** in `fenicsx-env`. The first restricted run
was terminated (143), had no completed result and is not counted; the
serial rerun completed. The new source-equation audit then passes six focused
tests in 0.40 seconds. It uses existing core creep integration and checks a
separate public source, not a new BAM acceptance result.

After archive regeneration, the source audit, matched report, signed budget,
batch evidence and CI-scope tests pass **134 tests in 0.90 seconds**.
Correctness lint and generated documentation consistency also pass.

Both the matched comparison and signed budget now reproduce from compact
repository archives, without requiring the original temporary trajectory files.

## Standards cross-check

The public [ASTM E2714-26 scope](https://store.astm.org/standards/e2714)
describes specimen creep-fatigue deformation/crack-formation testing and its
use in evaluating constitutive models. It is not a prescribed percentage
accuracy limit for this simulation, nor a certificate for component life.
The original BAM records predate that edition; no retrospective test
conformity is inferred from the current catalogue entry.

[ASME VVUQ 10.2](https://www.asme.org/codes-standards/find-codes-standards/the-role-of-uncertainty-quantification-in-verification-and-validation-of-computational-solid-mechanics-models)
explicitly relates validation requirements to experimental and simulation
uncertainties. The deterministic recorded-channel perturbations in this batch
do not supply those probability distributions. Keep the existing project
accuracy gates and distinguish numerical verification, experimental agreement
and a qualified engineering use; do not turn uncertainty into an error waiver.

## Recoverability and data sufficiency

The indexed Table VII, printed page 35, of the
[NASA corrected material-characterization report](https://ntrs.nasa.gov/api/citations/20260002420/downloads/TP-20250003948-CORRECTED.pdf?attachment=true)
separates sub-threshold creep/relaxation from above-threshold unloading and
recovery, and recommends repeated specimens to characterize scatter.
This supports keeping reversible and irreversible mechanisms separately
observable. It supplies neither P92 coefficients nor permission to infer a
scatter distribution from the two service-like specimens. The table was
verified in the public search index; direct full-PDF retrieval was blocked,
so the rest of this report is not treated as reviewed evidence.

The present three protocols do not include a separate unloaded-recovery
experiment. Existing Kelvin states are therefore preserved, and fitting
cannot establish their unique physical interpretation. New frozen response
checks must still pass the original absolute and kinetic gates; this data
limitation is not a replacement acceptance criterion.

An additional [Chen, Kitamura and Feng study](https://doi.org/10.1016/j.ijfatigue.2018.03.010)
reports precisely a warning relevant to our discriminator: a static-recovery
term can improve relaxation curves while misrepresenting inelastic-rate
evolution under alternating stress/strain holds. Its public abstract and
introduction were reviewed; its material is 316L at room temperature, not
P92, and its parameters cannot be transferred. Retain the separate creep,
hold-entry, early/late-rate and reversal observations rather than selecting
by an endpoint alone.

The [2026 comparative SRT study](https://doi.org/10.3390/machines14090967)
also compares intermediate histories, but its section 2.3.1 explicitly turns
static recovery off outside a holding period. That protocol-aware switch is
not adopted: the AgentFEM material must evolve from physical state and time,
not from knowing a test's named segment. It concerns room-temperature 304
steels and fitted conditions, not an independent P92 validation. Existing
power-law and accumulated-history mechanisms already cover its relevant
generic equation families; this search does not justify duplicating them.

The [primary-source equation audit](p92_primary_source_audit_20261122.md)
checks a retrieved author manuscript rather than relying on its abstract.
The literal printed strain-hardening expression does not recover its own
constant-stress primary curve. Existing core integration reproduces the
forward relation with explicit hour/second conversion. Retain this source
limitation; do not adopt an ambiguous equation to force a new material fit.
