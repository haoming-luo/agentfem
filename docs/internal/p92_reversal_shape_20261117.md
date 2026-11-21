# P92 reversal shape and hold-entry stress

## Prespecified conditional test

Keep PR #197 open. The prior time-shape candidate improves peaks and late
creep, but predicts an excessive hold-entry stress and subsequent early creep.
Do not correct these by imposing measured hold strain or resetting history.

Use the existing first-backstress law in each branch. Vary `a1` by a factor
`10**z`, and divide `C1` by the same factor. This preserves `C1*a1`, the signed
hardening coefficient, while changing dynamic recovery/saturation behavior.
It does not preserve the complete tangent, static-recovery equilibrium or
elastic modulus by implication; E itself is held fixed. All other parameters
and the slow/fast time-recovery terms remain exactly those of the frozen
time-shape card. Initial states remain zero only at initial loading, never at
reversal, hold entry or cycle boundaries.

Two coordinates, each in [-0.3, 0.3] (approximately factors 0.5–2), start at
zero. Retain the same three protocols, cycles 1–3, weights and 40-evaluation
budget. Freeze before cycles 4–10, independent integration and switch-bracket
checks. Do not widen bounds after results. These remain conditional development
experiments, not a jointly identified or independently validated material.

[Zhang et al. (2020), equations 6 and 9](https://www.mdpi.com/2075-4701/10/3/307)
distinguish kinematic saturation and the rate of approach, and test hysteresis
and relaxation rather than peak stress alone. This motivates the paired
coordinate test. Our bounded two-branch/Jiang hypothesis is not their complete
law, and their temperatures/specimens do not supply a BAM parameter card.

## Reproduction

```sh
python -m tools.p92_transient_history evidence/creep/2026-11-17-time-shape.json /path/to/BAM kinematic-shape.json --direction kinematic_shape
```

Inspect all hold-entry stresses, early and late strain secants, and peak/valley
errors. Good peak agreement alone never admits response or life predictions.

## Result: trade-off, not an admitted replacement

The two-coordinate fit converged in six optimizer / 31 residual evaluations;
cost fell from 0.01329958 to 0.01071853. Fast a1 scale hits the prespecified
upper bound (about 1.995), while slow a1 scale is about 0.609. Other material
parameters are unchanged. This is not evidence for expanding the bound.

Frozen cycle-10 RF/SLR/SLC peak errors are -0.64%/+3.93%/+2.36%. Extrema RMS
over cycles 4–10 are 8.52/6.61/14.86 MPa. Thus RF improves, while SLR/SLC
cycle-10 peaks worsen relative to the prior time-shape card. All three paths
complete ten cycles and pass the existing Radau/BDF numerical gates.

First SLC hold strain RMS improves from 8.549e-5 to 6.580e-5, but its inherited
stress worsens from 138.74 to 143.06 MPa versus the last observed 121.70 MPa.
The third-cycle early strain secant is 5.656e-5/s versus 3.441e-5/s observed;
cycle 10 is 5.712e-5/s versus 3.186e-5/s. A smaller scalar objective conceals
these incompatible improvements. The candidate is not promoted as a fix.

## Control-window diagnostic correction

While examining switch uncertainty, the upper-bracket hold observation window
was found to include samples already on the following reload ramp. Its last
observed first-hold stress was then 146.59 MPa instead of the actual pre-ramp
121.70 MPa. That is an observation-classification artifact, not relaxation.

Bracket comparisons now select the **same source indices** inside the common
unambiguous interval: after the previous switch's upper bound and before the
next switch's lower bound. The perturbed simulated intervals remain unchanged.
Nominal fitting samples, residual weights and parameter identification are
unchanged. Early/late metrics keep source indices and actual relative times.
Tests require matching lower/upper observation identities and in-segment times.

The corrected first-hold predicted stresses still span only about
142.34–143.27 MPa across timing bounds. The roughly 21 MPa bias therefore
cannot be removed by these declared switch brackets. This does not exclude
other loading-model errors or unknown experimental history.

Next distinguish the actual strain-controlled reversal waveform from the
piecewise-linear ramp approximation before adding an additional material
memory term. Any measured-waveform replay must retain continuous state and
separate prescribed control channels from response observations. Do not impose
measured stress during a strain-controlled ramp or reset stress at hold entry.

Evidence: `evidence/creep/2026-11-17-kinematic-shape.json`. Engineering and life
admission remain false. Repeat verification preserves the original
identification-driver identity instead of relabelling it as the latest replay.
