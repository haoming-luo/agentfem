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

## Next bounded experiment

The [existing recovery-shape substitution](p92_recovery_shape_20261122.md)
uses the two B ratcheting exponents in place of two weak A static-recovery
coordinates. It retains all reversible strains, does not create parameters,
and uses a same-start additional-budget control if fitting is admitted.
Derivative checks precede fitting; frozen response checks follow it.

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
The recovery-shape identification is still in progress at this checkpoint.
