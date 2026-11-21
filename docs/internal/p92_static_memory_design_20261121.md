# Static recovery driven by accumulated inelastic history

[Zhang et al. 2020, equations 9–11](https://doi.org/10.3390/met10030307)
uses a positive saturated-plus-exponential coefficient of static backstress
recovery. Its single-flow 600/650 C experiments do not specify how to couple
our two-flow BAM 620 C hypothesis. Transfer the functional idea, not constants
or a qualification claim.

For each branch multiply its existing static coefficient by
`rho+(1-rho)*exp(-b*p_history)`; retain the existing static power, energy,
strength and dynamic recovery. Own history uses p_A or p_B. Shared history
uses p_A+p_B, the sum of accumulated branch magnitudes, NOT the accumulated
magnitude of the total signed strain rate (cancelling branches differ).
The history definition is a declared hypothesis. Neither contains reversible
Kelvin strain, cycle number, elapsed time or a control-protocol selector.

The multiplier is positive in [rho,1]. Static dissipation is scaled by the
same factor; stored energy is unchanged. Defaults rho=1 preserve the prior
model exactly and introduce no state slots. Parameter identity retains both
the history definition and coefficients.

Prespecify five fixed cards: unchanged control; own/shared history at rho=.5
and .1 on both branches, b=250 per strain. These are mechanism screens, not
measured constants. Keep the original best spectrum card and actual controls.
Verify analytical inactive decay at fixed p, sign symmetry, power balance,
zero-amplitude reduction, shared/own distinction and independent continuation.
Then compare cycles 1–3 of RF/SLR/SLC: holds, inherited SLC stress, forward and
reverse extrema together, retaining the existing gates. No bounds expansion.

Freeze the rho=.5 own/shared pair before checking cycles 4–10, independent
integration and waveform refinement. This fixed comparison is not a fit and
does not turn inspected development cycles into blind validation. No model is
promoted merely because the aggregate cost decreases. Identification requires
compatible improvement in the conflicting early observables first.

## Results

| History / ratio | Joint early cost | SLR cycle-3 peak error, MPa | SLC first post-creep hold end, MPa |
| --- | ---: | ---: | ---: |
| Control | 0.00257608 | 15.915 | 159.025 |
| Own / 0.5 | 0.00257523 | 15.530 | 159.424 |
| Own / 0.1 | 0.00260705 | 15.103 | 159.769 |
| Shared / 0.5 | 0.00260474 | 15.077 | 160.123 |
| Shared / 0.1 | 0.00287691 | 13.669 | 161.228 |

The SLC observed hold end is 174.88 MPa. Shared history moves the conflicting
observables in compatible directions, including inherited hold stress
(123.37 MPa at ratio 0.5 versus baseline 123.65; observed 121.70), but the
remaining hold error is large. The stronger ratio-0.1 intervention increases
aggregate error. Neither qualifies as an accepted upgrade.

The preselected ratio-0.5 own/shared cards both completed ten continuous cycles
in all three protocols. Independent BDF and waveform-refinement comparisons
pass for every path and all twelve state entries; maximum independent
stress-state discrepancy is below 5.6e-7 MPa. All six experimental response
screens fail. Cycle-10 SLR peak errors are +9.846% (own) and +9.787% (shared),
against +10.177% for the original spectrum card. This is modest directional
evidence, not resolution of the model discrepancy or life validation.

Local validation: 380 affected P92 and CI-scope tests passed in 11.35 s;
Ruff and diff checks passed. The prior defaults reproduce the old law, and
the parameter decoder retains the history definition. Fixed-card validation
is explicitly marked unfitted; it does not fabricate optimizer success.

Next decision: shared history is the more discriminating hypothesis for the
conflicting holds, but an identification experiment must first audit whether
its history-rate/amplitude directions can be separated from existing static
rates. Replace weak directions rather than expanding the parameter space;
keep the same early observations, unchanged gates and a matched baseline.
Do not use these inspected later cycles to tune parameters or label a blind
validation. No optimization or parameter-bound expansion occurred in this batch.

Evidence: `evidence/creep/2026-11-21-static-memory-{probe,own-frozen,shared-frozen}.json`.
