# Observed-path recovery budget, declared before evaluation

The fixed-coefficient input interaction and discrete recovery-time assignments
did not resolve the post-creep hold. Before changing another material law,
isolate a question that does not require identifying plastic backstresses:
under the current **linear Kelvin hypothesis**, how much of the observed
nonelastic increment could be recoverable, and what signed remainder must
the other mechanisms supply?

For each existing Kelvin branch, integrate its linear equation against the
recorded piecewise-linear stress history analytically. Then evaluate
`delta strain - delta stress/E - sum(delta recoverable strain)` on the
existing early/late hold windows. Both measured stress and strain enter this
diagnostic, so it is not a forward prediction, calibration, or validation.
The remainder is not independently measured plastic strain or dissipation.
The selected material modulus and Kelvin coefficients are conditional inputs.

Do not silently assign a virgin starting state. Report the zero-memory
reference together with the exact influence of an explicit initial-memory
box, `abs(a_i(0)) <= max(abs(recorded stress))/K_i`. This box is a sensitivity
assumption, not a physical bound for unknown prehistory. Report its decay,
not a fitted initial state. Use one recorded stress/strain quantum for a
deterministic resolution envelope, not a statistical confidence interval.
Do not differentiate quantized measurements pointwise, bridge missing cycles,
or promote the prescribed measured stress to response agreement.

Check the analytic constant-stress and ramp solutions, small-step numerical
stability, subdivision invariance, signed symmetry, initial-memory influence,
and comparison with an independent ODE integrator. Evaluate early cycles
1--3 only, retaining the original response acceptance policy elsewhere.
No new material coefficient, changed tolerance, or production API is added.

The need to distinguish recoverable and unrecoverable contributions is
consistent with the primary study of prior-loading effects in 10%Cr steel,
[Hosseini et al. (2020)](https://doi.org/10.1016/j.ijmecsci.2020.105546).
That study does not supply this P92 parameter card; the convolution and
uncertainty bounds here are derived directly from the stated linear ODE.

## Outcome

Evidence: `evidence/creep/2026-11-22-observed-recovery-budget.json`.
At the first positive hold's 210 MPa band, the conditional remainder rate is
1.29579e-5 /s for SLR and 4.48994e-6 /s for SLC. The respective deterministic
intervals are [8.62026e-6, 1.72956e-5] and [2.80873e-6, 6.17114e-6] /s.
They do not overlap under the declared starting-memory box and resolution
assumptions. This is neither a statistical significance test nor independent
identification of plastic strain. It supports checking the history dependence
of the remaining flow, rather than attributing the entire discrepancy to
the fixed linear recoverable branches. Modulus and Kelvin parameter uncertainty
are not covered by these intervals and remain a separate diagnostic question.
