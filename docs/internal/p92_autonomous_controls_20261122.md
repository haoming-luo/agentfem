# Autonomous nominal-rate control check

The existing measured replay explicitly reaches each ramp target over its
recorded interval. After SLC creep, that gives a rate dependent on predicted
creep strain. This is a declared source-aligned replay, not a controller
prediction. The source Methods specify 0.001/s ramps. Earlier nominal-path
checks concerned the older two-mechanism card, not the present recoverable
spectrum. Reuse the published RF/SLR path definitions and replace only SLC's
zero-strain long dwell with a captured-entry-stress dwell.

Freeze the current original and recovery-shape cards. Run all ten cycles at
nominal ramp rate, retaining predicted creep strain and every internal state.
Ramp duration is abs(target - current strain)/rate; reject an overshot SLC
positive target instead of manufacturing a reverse ramp. Hold durations retain
the source-aligned replay convention: 180/1800 s except the already recorded
RF fourth-cycle negative-hold departure at 180.096 s. This isolates the ramp
experiment instead of additionally changing hold observations or dropping
out-of-support samples. Stress is captured from the calculated incoming state. No
observed creep strain, measured stress, or cycle-specific coefficient drives
the material.

Compare peak stresses and hold responses at the same **elapsed hold times**
as the source, recording the resulting different absolute switch times.
These are protocol-relative diagnostics, not exact recorded-time predictions
or a substitute for the existing frozen screen. Preserve original gate
values and a separate numerical check with BDF; test exact elastic limits,
state carry-over, overshoot rejection and subdivision before using real cards.
This check changes the control experiment, not the material or fitting budget.

## Recovery-shape card: frozen result

All three ten-cycle nominal-rate paths complete, and their independent BDF
checks pass. All three experimental response screens still fail. For the
first SLC post-creep positive hold, the measured-time replay predicts
162.410 MPa at the last observation; autonomous nominal-rate loading predicts
162.464 MPa, against 174.880 MPa observed. The change is only 0.054 MPa.
Its first-sample residual is -0.909 MPa, early secant -5.142 MPa/s versus
-3.794 observed, and late secant -0.08245 MPa/s versus -0.18193 observed.
Thus this control approximation does not explain the observed hold-shape
conflict for this frozen card. This is not a universal conclusion about
arbitrary material models or loading records. These early holds entered
calibration; “frozen” describes the parameter policy, not statistical holdout.

Evidence: `evidence/creep/2026-11-22-recovery-shape-nominal-rate.json` retains
all checks, hold diagnostics, switch times and source/card/driver identities.
Only redundant sampled segment arrays are omitted from its compact archive;
the original full artifact digest is retained. No experimental value drives
the predicted creep strain or sets the post-creep ramp duration.
