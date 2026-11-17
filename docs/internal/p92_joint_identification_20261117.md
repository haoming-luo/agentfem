# P92 three-protocol joint identification

## What was fitted

The bounded two-mechanism research model now has a reproducible joint
identification route. Six dimensionless parameters control fast/slow flow at
120 MPa overstress, each positive isotropic-strength margin, fast static
recovery, and the slow branch's second backstress saturation. Fast exponent 6
and other constants are fixed hypotheses, not separately identified constants.
The 120 MPa pivot is numerical parameterization, not a calibration measurement.

Training uses **cycle 1 only**, separately for RF, SLR and SLC. Nominal input
segments are explicit approximations to the measured controller. Actual
recorded observations are selected at log-spaced target times without
interpolating the measured response. Missing first-cycle coverage is rejected.
Each specimen starts with its own recorded stress and assumed virgin hidden
state. A first-cycle strain offset is not inferred as plastic history.

The objective assigns equal protocol and segment weights. Stress is scaled by
100 MPa; SLC's constant-stress hold also compares total strain scaled by 0.002.
Both channels share that segment's weight. These are objective-balancing
scales, **not measurement standard deviations or relaxed acceptance limits**.
SLC training stops at the first 1800 s stress-hold endpoint, before the
strain-dependent next ramp. RF and SLR include their full first nominal cycle.
No later-cycle observation enters the optimizer.

## Numerical experiments

1. Relative finite differences and unscaled coordinates exhausted 30 optimizer
   evaluations (192 residual evaluations), cost 0.0344677. No later-cycle
   predictions were emitted for this failed fit.
2. Absolute central differences of 0.001 in dimensionless log coordinates,
   Jacobian scaling, and tighter local integration converged in 14 optimizer
   evaluations (146 residual evaluations), cost 0.0344556. The allowed budget
   was 80, not 80 evaluations actually used. Tightening material integration
   from 1e-8 to 1e-9 changes the largest scaled training residual by 9.87e-11.

The second experiment is a numerical-method check with the same data, bounds
and model family; it does not introduce later data or claim a global optimum.
Its Jacobian singular values range from 0.17284 to 5.756e-7, a ratio about
300,000. The first-cycle data weakly constrain some parameter combinations.
Optimizer convergence is not parameter identifiability, an uncertainty
interval or material acceptance.

## Frozen subsequent-cycle prediction

Parameters and full branch cards are hashed before running cycles 1–10.
No refit occurs after inspecting predictions. Cycle 1 retains the calibration
label for all three protocols; later observations are **previously inspected
development checks**, not independent experimental validation.

| Protocol | Completed cycles | Later-cycle extrema RMS | Failure |
|---|---:|---:|---|
| RF | 10 | 34.686 MPa, cycles 2–10 | Response errors remain |
| SLR | 10 | 44.329 MPa, cycles 2–10 | Response errors remain |
| SLC | 3 | 72.199 MPa, cycles 2–3 | Cycle 4 creep strain 0.00211217 exceeds target 0.002 |

RF cycle-10 peak is 11.84% low; SLR cycle-10 peak is 17.46% low.
The SLC partial fourth cycle is retained as failure and is not counted as a
completed cycle. Checks of the actual first ten SLC cycles show recorded
strain maxima 0.00201–0.00203, so the nominal envelope failure is not explained
by an unrecognized large shift of the measured positive strain target.

Independent BDF integration at 1e-10 preserves the same SLC exit cycle.
Including the failed hold endpoint, maximum stress differences are
4.84e-7 MPa (RF), 3.54e-7 MPa (SLR), 3.25e-7 MPa (SLC); strain differences
are below 1.90e-10. These checks address numerical realization, not experiment
agreement. `verify_frozen` rejects modified parameter identities.

**Decision: not admitted.** This finite six-parameter fit does not show that
every two-mechanism model fails. It does show that declaring a second branch,
obtaining optimizer success and delaying the envelope exit do not satisfy
cross-protocol validation. No life law or engineering allowable is produced.

## Reproduction and next discriminating experiment

```sh
python -m tools.p92_two_mechanism_fit evidence/creep/2026-11-16-bam-positive-strength.json /path/to/BAM --max-nfev 30 --output joint-budget.json
python -m tools.p92_two_mechanism_fit evidence/creep/2026-11-16-bam-positive-strength.json /path/to/BAM --max-nfev 80 --scaled-jacobian --output joint-scaled.json
```

Repository evidence keeps both fits, exact training observation indices,
parameter cards, source identities, the failed partial cycle and completed
cycle endpoint states. Intermediate segment endpoints of completed cycles are
omitted from compact evidence; raw-report SHA256 identifies the original.
The numerical baseline is `2026-11-16-bam-positive-strength.json`, canonical
SHA256 `e70095c61a15f76466ff7365694c1004d67ff3ef3424ae9fe0dc0847925631a9`.

Next prioritize identifiability rather than another unconstrained large fit:
use additional early **measured** mixed-control cycles with explicit control
transition alignment, inspect which recovery/softening parameter combinations
are observable, and then declare a new frozen training/prediction split.
Do not let an optimizer invent reverse ramps when nominal creep overshoots.
Changing the training split is a new development experiment, not a way to
relabel failed holdouts. Independent specimens and compatible lifetime
definitions remain separate engineering admission gates.
