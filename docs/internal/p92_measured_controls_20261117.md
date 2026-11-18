# P92 measured early-cycle identification

## Scope and provenance

This continues the [first-cycle joint fit](p92_joint_identification_20261117.md),
now using cycles **1–3 of RF, SLR and SLC**. All three BAM files retain their
published checksums and sample indices. Cycles 4–10 have already been inspected:
they are frozen-parameter development checks, not blind validation. The model
remains a scalar research hypothesis, not a qualified P92 material or life law.

Sources: [BAM records](https://doi.org/10.5281/zenodo.7198219),
[experimental description](https://doi.org/10.1016/j.dib.2023.109333).
The experiment uses combined strain/stress control, not a single imposed
strain history. Hidden states are assumed virgin separately for each specimen;
initial recorded stress and strain are retained. There are no cycle resets.

## Control reconstruction

Peak and zero-strain crossing times are interpolated between recorded samples.
The 180 s and 1800 s dwell durations come from the source protocol. Each dwell
departure must have a nearby measured ramp, identified using a 0.0004/s slope
separator against the nominal 0.001/s ramp. This separator is a reconstruction
assumption, not a material parameter or a universal controller detector.
We retain the supporting sample indices and time brackets.

Where the duration-derived time lies outside the observed departure bracket,
use the bracket midpoint and explicitly record the inconsistency. This occurs
once in the first ten RF cycles, at the fourth cycle's negative-hold departure.
These records are not exact controller logs. Sensitivity below uses the full
observed brackets, including relatively wide intervals from sparse hold data.

For the first SLC cycle, the inferred stress-hold start/end are 184.2105 and
1984.2105 s; the next positive strain crossing is 1985.215 s. The measured
cycle ends at 2167.233 s rather than nominal 2168 s. The post-hold strain ramp
starts from the **predicted** creep strain and reaches the strain target over
the measured interval. No observed creep strain overwrites model state.
The SLC dwell holds the predicted entry stress constant; small measured stress
fluctuations remain response residuals, not prescribed stress perturbations.
Consequently this is source-aligned endpoint replay, not exact servo-controller
reproduction or an autonomous prediction of observed ramp duration.

## Identification and freeze

The same six log parameters and bounds as the previous experiment are used.
The first-cycle seed's parameter identity and reconstructed branch cards must
match the base calibration. Equal protocol/segment weights and the existing
100 MPa / 0.002 strain residual scales are retained; they are not uncertainty
estimates or admission tolerances. Observed responses are not interpolated.

Identification converged in 24 optimizer evaluations / 264 residual evaluations,
cost 0.0321656843. Tightening local integration changes the largest scaled
training residual by 1.09e-9. The parameter Jacobian condition ratio is about
6.51e6, and the fast static-recovery scale is near its lower search bound:
convergence does not establish parameter identifiability.

During optimization only, incompatible trial ramps may continue to define a
fixed-size residual vector. **Frozen prediction never continues a reverse
ramp**: it records the failed hold endpoint and excludes the partial cycle.
The frozen fit is saved before prediction; all control windows are preflighted
before identification. This fixes a development run that completed fitting
but failed on late RF timing reconstruction before saving its parameters.
That lost run was rerun once, not presented as additional validation.

## Frozen results: not admitted

| Protocol | Complete cycles | Extrema RMS after cycle 3 | Outcome |
|---|---:|---:|---|
| RF | 10 | 16.49 MPa (cycles 4–10) | Cycle-10 peak error −5.15% |
| SLR | 10 | 26.71 MPa (cycles 4–10) | Cycle-10 peak error −9.95% |
| SLC | 4 | 72.99 MPa (cycle 4 only) | Cycle-5 creep strain 0.002097 exceeds target 0.002 |

The autonomous nominal-path check also stops SLC at cycle 5. These RMS values
use different windows from the previous first-cycle fit and are not directly
comparable as a single improvement percentage.

Independent BDF integration at 1e-10 matches the same failure cycle. The largest
stress/state discrepancy is 2.10e-6 MPa; existing numerical screening limits
of 1e-4 MPa and 1e-8 accumulated/total strain remain unchanged and pass.
This is numerical realization evidence, not material acceptance.

Moving all switches to their recorded bracket bounds changes predicted extrema
by up to 0.45 MPa (RF), 20.01 MPa (SLR), and 27.07 MPa (SLC). These deterministic
perturbations are not confidence intervals and may vary programmed dwell
duration. SLC still fails at cycle 5 for both extremes. Timing uncertainty
therefore matters to response accuracy, but does not remove this failure.

No engineering or life qualification is granted. Next isolate transient
recovery/softening observability using the early hold and reversal residuals;
do not expand the parameter search blindly, refit on later checks, or call a
second branch an implementation of an entire published material model.

## Reproduction

```sh
python -m tools.p92_measured_fit evidence/creep/2026-11-16-bam-positive-strength.json evidence/creep/2026-11-17-dual-fit-scaled.json /path/to/BAM --output measured-fit.json
python -m tools.p92_measured_fit evidence/creep/2026-11-16-bam-positive-strength.json evidence/creep/2026-11-17-dual-fit-scaled.json /path/to/BAM --verify-frozen measured-fit.json --output verified-fit.json
```

The second command verifies frozen parameters without fitting. The checked-in
`evidence/creep/2026-11-17-measured-controls-fit.json` retains full parameter
cards, training indices, switch brackets, accepted and failed states,
independent-integrator checks, timing sensitivity and implementation hashes.
It is about 310 KiB; raw source files and personal header metadata are absent.

## Development resource decision

#194 was blocked by a one-line merge conflict, not failed numerical CI. After
resolution, 43 local tests and all applicable CI passed; it was merged as
`82901c3c`, without a release. The subsequent reconstruction, fit and CI-policy
changes were developed as one local batch. Known scalar research tools now
select their transitive tests; unknown numerical owners retain broader gates.
Scientific failure records are retained without creating a new CI run per fit.
