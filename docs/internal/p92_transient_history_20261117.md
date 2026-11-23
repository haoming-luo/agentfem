# P92 early-hold transient and reversal-history discriminator

## Decision before computation

Continue on PR #197; no merge or release. The preceding dual-time fit closes
the ten-cycle control-envelope failure but not SLC response accuracy. Test
whether its fixed kinematic transient is a limiting assumption before adding
another mechanism, an initial-state fit, or a damage variable.

Only fast and slow first-backstress evolution coefficients `C1` vary, by
log10 scales in [-1, 1]. All other parameters of the archived dual-time card
remain fixed, including `a1`, `C2`, flow, time-recovery and strength parameters.
The two-coordinate experiment is conditional on that card, not a globally
identifiable two-parameter material. Start at unit scales, allow 40 optimizer
evaluations, retain the same residual weights and cycles 1–3. Do not widen
bounds after seeing results. C1 changes hardening and dynamic-recovery rates
per accumulated inelastic strain; it is not an independent physical clock.

Retain continuous states from initial loading through every reversal and hold.
At each early hold entry, record signed branch flow and recovery contributions.
Compute additional instantaneous probes with either backstress or isotropic
state zeroed while holding stress and all other coordinates fixed. These
artificial states are diagnostic only: they are never integrated or installed
as a corrected history. Their rate differences are not experimental causal
attributions, strain partitions, stored energy or damage.

The first SLC long hold already follows compression and a return to zero total
strain; it is **not a virgin uniaxial creep test**. Distinguishing an initial
creep transient from reversal memory cannot be done by renaming its first
ten-second secant. Compare the first peak relaxation and the post-reversal
hold, preserve stress-controller approximation error, and retain the continuous
entry state. Matched virgin/reversed experiments would be stronger evidence.

After convergence freeze the full card, then replay cycles 4–10, independent
Radau/BDF integration, timing brackets and nominal controls. These later cycles
have previously been inspected and remain development checks, not blind
validation. Numerical completion does not admit engineering response or life.

## Literature and applicability

- [Khayatzadeh et al., 2017, Engineering Fracture Mechanics](https://doi.org/10.1016/j.engfracmech.2017.02.008)
  compares primary/secondary creep and relaxation models for P92 at 650 C.
  It motivates checking the early transient rather than fitting only late
  holds. Its temperature and specimen differ from the BAM 620 C histories;
  none of its constants are transferred here.
- [Zhang et al., 2020, Metals](https://doi.org/10.3390/met10030307)
  tests static recovery in kinematic hardening for P92 fatigue–creep response.
  It motivates separately inspecting backstress and isotropic evolution.
  The present bounded dual-branch hypothesis is not a reproduction of their
  complete law or parameter identification.
- [Shutov et al., 2017, ZAMM](https://doi.org/10.1002/zamm.201600286),
  [public manuscript](https://arxiv.org/abs/1612.04620), models a reversal-related
  transient creep-rate increase. Its validation is aluminium cyclic torsion,
  **not P92**. It supports a history-sensitive diagnostic, not material transfer.
- [BAM experimental description](https://doi.org/10.1016/j.dib.2023.109333)
  remains the source of the three 620 C protocols. Existing checksum-verified
  raw histories are reused; no personal source headers are redistributed.

## Reproduction

```sh
python -m tools.p92_transient_history evidence/creep/2026-11-17-both-time-identification.json /path/to/BAM transient-history.json
```

Output retains baseline and fitted early-hold diagnostics, source and card
identities, parameter bounds, convergence and frozen development checks.

## Observed result

The conditional fit converged in 12 optimizer / 49 residual evaluations.
Its cost fell from 0.02224865 to 0.01555038. Fast C1 reached its lower scale
bound (0.1); slow C1 scale is approximately 1.325. The two singular values are
0.12679 and 0.02096. This conditional conditioning does not identify the
remaining frozen parameters or supply a statistical uncertainty interval.
Tightening training integration changes scaled residuals by at most 1.71e-10.

| Observable | Previous dual-time card | Conditional C1 fit | Observed |
| --- | ---: | ---: | ---: |
| First SLC hold early strain secant, 1/s | 5.350e-6 | 1.002e-5 | 3.568e-5 |
| First SLC hold last sampled strain | 0.0005960 | 0.0007487 | 0.001060 |
| First SLC hold strain RMS error | 0.0004433 | 0.0002878 | — |
| Tenth SLC hold last sampled strain | 0.001571 | 0.001208 | 0.000870 |
| Tenth SLC peak stress relative error | -22.84% | -10.87% | — |

First-hold entry flow changes little (sum 1.468e-5/s to 1.505e-5/s), whereas
the early finite-interval secant increases substantially. Within this model,
that supports inspecting **decay during the hold**, not just increasing entry
flow. The fitted entry backstresses remain negative after the compressive
prehistory: their sums are -45.37 and -30.32 MPa. Conditional removal of these
backstresses reduces the instantaneous summed flow to 2.23e-7/s. This is
model-internal sensitivity evidence, not a measured decomposition.

The improvement does not close the cycle trend: the second and third early
SLC secants become 5.36e-5/s and 5.69e-5/s versus observed 3.57e-5/s and
3.44e-5/s. One constant rate adjustment cannot yet reproduce both the first
and subsequent histories. No physical history is reset to repair that error.

Frozen cycle-10 RF/SLR/SLC peak errors are -4.12%/-3.35%/-10.87%.
All three ten-cycle paths, nominal controls and both timing brackets complete;
Radau/BDF comparisons pass the existing numerical limits. Engineering response
and life admission remain closed. The next bounded mechanism test should
separate accumulated-strain-dependent kinematic saturation/recovery from the
first-hold transient, with early-loop shape and early-hold secants retained
separately; do not infer a missing damage law from this two-coordinate fit.

Evidence: `evidence/creep/2026-11-17-transient-history.json`. Raw sources and
personal headers are not included. No public material catalog was changed.
