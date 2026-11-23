# Nonlinear static recovery: fixed-card discriminator

The shared-drag candidate failed frozen SLC continuation and is not adopted.
Return to the best mechanically verified recoverable-spectrum card. Do not
change its fitted constants, loading waveforms, initial states or response gates.

Zhang et al., [Metals 2020, 10, 307](https://doi.org/10.3390/met10030307),
uses a nonlinear backstress static-recovery term for P92. Its experiments are
at different temperatures and amplitudes; neither constants nor qualification
transfer to BAM. Our previous spectrum uses linear static recovery. Test the
structural difference before another identification exercise.

Use `Xdot_static=-r*(abs(X)/a)^m*X`, with `a` the declared saturation scale.
All powers agree with the baseline at abs(X)=a. At smaller backstress,
positive m slows recovery and preserves more reversal memory. This is an
explicit normalized hypothesis, not reproduction of the paper's complete law.
Stored energy is unchanged; the corresponding nonnegative dissipation is
`r*(abs(X)/a)^m*X^2/H`. No load-protocol labels enter the material law.

Predeclare branch powers (0,0), (1,0), (0,1), (1,1), (0,2), (2,2).
These are six mechanism probes, not fitted material cards. Run all three
protocols through cycles 1–3 continuously. Preserve every probe, early-hold
diagnostics and unchanged gate results. Compare hold entry, decay and subsequent
peak separately; aggregate residual reduction alone does not select a model.
Do not use cycles 4–10 to choose powers. If the mechanism improves compatible
early observables, only then consider shared early-only identification and
freeze its parameters before development-cycle evaluation.

Verification first: exact inactive-flow polynomial decay, zero-power reduction,
sign symmetry, nonnegative dissipation and mechanical balance. Independent
integration and continuation checks remain required for any selected card.

## Fixed-card result and matched identification decision

The branch-A power has negligible effect, consistent with its tiny static rate.
Branch-B power 1 reduces the third SLR peak error from 15.92 to 13.75 MPa and
raises the first post-creep SLC hold end from 159.02 to 160.75 MPa (observed
174.88). SLC creep-entry stress moves from 123.65 to 120.82 MPa (121.70 observed).
These directions are compatible, but total cost worsens from 0.002576 to
0.002895 and the post-creep hold remains outside tolerance. This is not a pass.
Power 2 further lowers SLR peaks but worsens aggregate cost and entry stress.

Proceed with one matched comparison: baseline powers (0,0) versus (0,1), both
starting from the same spectrum card. Fix powers, E, and waveform; audit all
22 positive log-coordinate directions (18 backbone plus four recoverability
coefficients). Fit only ten leading directions if two central-difference scales
agree and their leading subspaces are stable. Both hypotheses get a log-norm
bound 0.3, maximum 40 optimizer evaluations and identical early-only residuals.
Do not compare a refitted hypothesis solely to an unfitted baseline. No new
cycle-dependent coefficient or condition-specific parameter is introduced.
Freeze converged cards and retain all original response gates; any selection
based on already inspected cycles remains development, not blind validation.

## Observation and numerical cross-check

A separate fixed-card audit of cycles 1, 3 and 10 found no omitted measured
SLR extrema and no difference between SLR solver-point and observation-point
predicted extrema. RF/SLC predicted sampling differences reach about 1.93 MPa;
they cannot explain the dominant SLR/post-creep mismatch. The response gates
are unchanged. Cycle peaks/valleys themselves now enter the independent
numerical comparison. All three baseline protocols pass this strengthened
check, with maximum independent cycle-extrema difference below 4.8e-7 MPa.
Evidence: `evidence/creep/2026-11-21-spectrum-extrema-check.json`.

## Matched-budget outcome

Both fits terminated after eight optimizer iterations. Linear recovery used
238 residual calls and reached cost 0.0023607800; nonlinear recovery used 258
and reached 0.0023958776. The nonlinear hypothesis did not beat its matched
control. Both derivative-scale/subspace audits passed their predeclared gates.

Frozen ten-cycle continuations completed in RF, SLR and SLC. All independent
integration and refinement checks passed, including full internal state and
cycle extrema. All three experimental response screens still failed for each
card. No tolerance, observation selection or late-cycle calibration was changed.
Keep this as a negative model-discrimination result, not an accepted material.
The four `2026-11-21-recovery-matched-*` evidence files retain identification,
source/implementation identities, response screens and numerical comparisons.
