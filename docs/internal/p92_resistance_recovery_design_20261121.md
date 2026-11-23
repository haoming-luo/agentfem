# Distinguish mobility history from strength recovery

## Evidence and limits

[Zhang et al., Metals 2020, 10, 307](https://doi.org/10.3390/met10030307),
equations 9–13, separately evolve backstress and isotropic strength. Their
discussion explains that rapid subsequent plastic loading can erase the
effect of backstress recovery on peaks; an isotropic recovery term addresses
a different observable. The experiments are at 600/650 C, not BAM 620 C.
Their constants cannot be transferred or the model called validated here.
[Hosseini et al., IJMS 2020](https://doi.org/10.1016/j.ijmecsci.2020.105546)
uses drag and backstress to describe prior-deformation effects in 10%Cr steel.
Only its abstract has been checked; it supplies motivation, not our equations.

The previous fixed-card kinetic interaction increased joint error. Do not
fit it by widening bounds. First compare where a shared history acts.

## Matched mechanistic experiment, declared before runs

For each branch introduce s in [s_min,1], initially 1:
`sdot = b*(s_min-s)*abs(flow) + r*(1-s)`.
Plastic flow lowers resistance and static recovery restores it. This is a
bounded scalar hypothesis, not a transcription of Zhang's full model.
No cycle number, control label or measured stress enters this equation.

Strength variant: `k_eff = k*s`, q unchanged.
Mobility variant: k unchanged,
`q_eff = q*(1+k*(1-s)/120 MPa)^n`.
At fixed backstress and original overstress 120 MPa the two variants have
exactly the same flow. Away from this anchor they differ: strength changes
activation, mobility does not. The anchor is the existing reference law's
stress normalization, not a fitted experimental value. Report it explicitly.

s is a dissipative resistance state with no additional stored-energy term;
positive thresholds and positive mobility preserve effective-stress power.
The existing elastic, backstress and recoverable-strain energies remain.
This convention is not a microscopic energy interpretation of isotropic
hardening. It must not be substituted for an energetic hardening potential.
Require constant baseline strength/hardening evolution to avoid hidden double
counting. Use a distinct 14-state identity and retain variant in parameters.

First test zero-amplitude identity, matched-anchor flow, activation threshold,
sign symmetry, power balance, exact inactive recovery, invariant bounds,
independent integration and split continuation. Then freeze the best original
spectrum card and run cycles 1–3 of RF/SLR/SLC with measured ramps.
Predeclare s_min=0.9, b=250/strain, r=1/180 or 1/1800 per second,
for both variants, plus the zero-amplitude control. These scales correspond
to a strain excursion and existing hold durations; they are screening scales,
not calibrated material parameters. No expansion after seeing outcomes.

Implementation stores loss `1-s`, initially zero, as the two added states.
The sign of the restoring term is our hypothesis; Zhang's incomplete recovery
target does not establish restoration to virgin strength for BAM. Failure of
this bounded experiment therefore cannot exclude all isotropic recovery laws.

All original response gates remain. Examine positive/negative extrema,
inherited SLC hold stress, hold endpoints and whole-hold errors together.
A candidate must improve compatible observables before identification is
considered. Any accepted fit then freezes all parameters for continuous
cycles 4–10 and independent integration/refinement. Those inspected cycles
remain development validation, not a blind holdout or industrial certification.

## Early outcome and fixed continuation decision

| Variant | Joint cost | SLR cycle-3 peak error, MPa | SLC first post-creep hold end, MPa |
| --- | ---: | ---: | ---: |
| Control | 0.00257608 | 15.915 | 159.025 |
| Strength, 180 s | 0.00255503 | 16.491 | 158.544 |
| Strength, 1800 s | 0.00261849 | 17.114 | 158.076 |
| Mobility, 180 s | 0.00256404 | 16.020 | 158.896 |
| Mobility, 1800 s | 0.00254558 | 15.920 | 158.778 |

The observed SLC end is 174.88 MPa. All four modifications worsen this hold
and do not reduce the third SLR peak error. A small aggregate-cost reduction
therefore does not justify identification. No optimizer is launched and no
bound is enlarged. Freeze the lower-cost representative of each variant
(strength/180 s and mobility/1800 s) solely for a ten-cycle negative-hypothesis
continuation with independent BDF and waveform refinement. This selection is
based only on the early experiment and is not scientific admission.

The mechanism implementation has 15 local checks covering matched-anchor
flow, distinct activation, power, sign symmetry, exact inactive recovery,
invariants, independent integration, continuation and input rejection.
The frozen driver tests separately ensure failure retention and coverage of
all 14 state entries, distinguishing resistance fractions from strain.

## Frozen result and next admissible hypothesis

Both fixed cards complete all ten cycles for all three protocols. Independent
Radau/BDF and waveform-refinement checks pass for all six paths; maximum
independent stress-state discrepancy is below 6e-7 MPa. All six experimental
response screens fail. Cycle-10 SLR peak errors are +10.40% (strength/180 s)
and +10.19% (mobility/1800 s). Neither is an accepted replacement.

The failure is specifically that bounded resistance restoration, as constructed
here, does not retain the required post-creep distinction. It is not evidence
that either whole model family is impossible. A next source-supported test is
the accumulated-inelastic-strain dependence of static backstress recovery
in Zhang equation 10, rather than another resistance-amplitude fit. Before
adding it to the current non-unified law, declare whether history is branch-local
or shared; the single-flow published equation does not settle that choice.
Compare the consequent backstress at reversal and same-stress hold flow,
alongside inherited hold stress, full hold errors and both extrema. Preserve
the present no-fit results as controls. Do not combine several untested
evolution terms or copy different-temperature constants.

The [NEML Chaboche interface](https://neml.readthedocs.io/en/dev/hardening/non/chaboche.html)
also separates isotropic and backstress channels and strain/time evolution;
its hardening sign convention is opposite to the current research reference.
This is an implementation reference only; no NEML cross-code run is claimed.

Local batch acceptance: **369 tests passed in 10.26 s** in fenicsx-env,
including the full affected P92 control group and CI scope tests. Ruff and
diff checks passed. No change to production constitutive behavior, numerical
tolerances, experimental gates, data split or fitted parameter bounds.

Artifacts: `2026-11-21-resistance-recovery-probe.json`,
`2026-11-21-resistance-strength-frozen.json`, and
`2026-11-21-resistance-mobility-frozen.json` under `evidence/creep/`.
