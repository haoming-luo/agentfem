# Recorded hold-strain sensitivity, not a new loading specification

The dual-channel resolution audit makes tiny late SLR rates questionable as
isolated mechanism discriminators. Test the input effect at fixed original
and mobility cards, not by recalibrating or loosening response limits.

Replay cycles 1--3 with the same measured switch times and observations.
Compare the existing ramp-shape/nominal-hold path with piecewise-linear
recorded strain on strain-controlled segments, including dwells. A force
dwell still predicts creep freely. Its following ramp starts at that
predicted strain and reaches the recorded strain endpoint without a state
reset. Its interior waveform retains endpoint-relative recorded shape;
it is not an exact absolute measured history when its initial strain differs.

Recorded strain is quantized instrument output, not an authenticated actuator
command. This is a deterministic sensitivity experiment, not a replacement
protocol or an uncertainty distribution. No measured stress or creep strain
is imposed as a new control. No filtering constant or material parameter is
added. Keep all original nominal-control failures.

For each card and protocol check the perturbed path against BDF at tighter
tolerance and the same piecewise-linear history with bisected intervals.
Use the existing full-state numerical limits. Record unchanged response
screens separately; a sensitivity improvement is not engineering acceptance.
Stop on any failure without repairing the control direction or resetting
material history. Do not fit to the perturbed path in this batch.

## Result

The first replay used the common pre-fit seed, retained explicitly as
`seed-recorded-hold`, not mislabeled as the matched baseline. The subsequent
paired comparison uses the matched original and mobility cards. All three
cards complete the three protocols; independent integration and knot
refinement pass. All response screens still fail. Source/card hashes keep
the seed and matched comparisons distinct.

For the matched original, the second SLR long-hold late slope changes from
-0.001298 to +0.002557 MPa/s (observed +0.001729). Recorded strain variation
can change this tiny rate's sign, but the changed rate is still outside the
unchanged relative gate. It does not establish the true controller history.

The first SLC post-creep hold end changes only from 160.728 to 160.570 MPa
against 174.880 MPa observed. Its early slope becomes more negative, -5.604
instead of -5.081 MPa/s, versus -3.794 observed; its late slope stays about
half the observed magnitude. Thus the input sensitivity does not remove
the large SLC discrepancy. It is neither a new fit nor a qualification.
