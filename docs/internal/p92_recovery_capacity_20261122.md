# Conditional recoverable capacity, before another spectrum fit

For a linear Kelvin mode during constant stress, with a fixed hold-entry
state, write A=sigma/H-a(0). Its late secant is
A[exp(-t0/tau)-exp(-t1/tau)]/(t1-t0). For positive A and t1>t0>0,
the maximum over all positive tau occurs at the logarithmic mean
tau=(t1-t0)/log(t1/t0). Negative amplitudes have supremum zero, not a
positive contribution. Summing positive amplitudes gives an explicit bound.
This follows directly by differentiating the exponential solution; tests
compare it with independent scalar optimization, including close times.

The bound fixes the entry state and moduli. It is NOT an upper bound for
a complete refit, a changed preceding path, nonlinear recoverability or
irreversible creep. It does not prove that moving relaxation times will
improve a continuous mixed-control prediction.

For the matched original card, the first SLC long hold's two amplitudes are
7.1195e-5 and 3.2736e-4. Over the recorded late interval, the optimum time
is approximately 1611 s and the conditional maximum recoverable secant is
9.082e-8/s. The current times give only 3.08e-14/s. This confirms that
long-time recoverable capacity is not exhausted by the current short-time
spectrum, but does not identify a missing physical branch.

Crucially, the measured late strain change is below the existing five-quantum
resolution gate in all three early SLC cycles. For cycle one, the inferred
nonelastic secant is 1.106e-7/s with a one-quantum endpoint perturbation scale
5.705e-8/s. That scale is not a confidence interval. These samples cannot
justify a new precise long-time parameter or an extra passing slope gate.
Retain the unchanged creep accumulation and resolved hold-stress constraints;
do not launch another spectrum fit from this diagnostic alone.

Evidence: `evidence/creep/2026-11-22-recovery-capacity.json`, linked by hash to
the full frozen trajectory. No fitting, new material parameter, acceptance
change or runtime-package promotion is made by this audit.
