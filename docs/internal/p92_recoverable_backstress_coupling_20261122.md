# Prospective recoverable/backstress interaction discriminator

## Why this is different

The existing energetic-coupling experiment couples the A and B backstresses.
It does not couple delayed recoverable strain to the force opposing irreversible
flow. The present diagnostic asks whether that missing interaction can retain
the creep-created memory through the subsequent ramp without discarding
recoverability. This is our scalar constitutive hypothesis, not a reproduced
P92 law from a paper. The thermodynamic interaction framework of
[Wolff, Bohm and Taleb](https://www.math.uni-bremen.de/zetem/cms/media.php/262/report1005.pdf)
motivates using one potential and conjugate forces rather than independently
altering a flow rate. It does not validate the proposed coupling or its sign.

For each of two B hardening coordinates b and recoverable strains a, take
psi = H*b^2/2 + K*a^2/2 + c*a*b, c=kappa*sqrt(H*K), |kappa|<1.
The state stores X=H*b+c*a and a. Then
psi = X^2/(2H) + K*(1-kappa^2)*a^2/2,
Y = c*X/H + K*(1-kappa^2)*a.
Keep the existing B flow driven by sigma-sum(X), set
a_dot=(sigma-Y)/(K*tau), and
X_dot=H*v_B-decay*X+c*a_dot.
The A branch and directional recovery rule are unchanged except that its two
weak static-recovery coefficients are removed (set exactly zero) to replace
them with the two coupling coefficients. No extra coefficient is fitted.
The original card with its tiny A static rates remains a separate control;
the zero-coupling, zero-A-static card measures the deletion alone.

The dissipation is the original nonnegative flow and backstress-recovery
contribution plus sum((sigma-Y)^2/(K*tau)). This must be checked against the
potential derivative, both control modes, sign reversal and independent
integrators before any protocol experiment. Old uncoupled rectangular X
bounds are not valid after adding c*a_dot; do not reuse that assertion.

## Declared fixed-card screen

Use the same existing spectrum seed and measured early cycles 1–3. Arms:
original; deletion-only; equal couplings +0.1, +0.25 and -0.1. Keep the full
three-protocol objective and unchanged response gates; no new loading data,
branch reset, range expansion or later-cycle selection. A same-budget fit
is considered only if at least one declared arm completes and improves the
joint objective without hiding a failed loading continuation. Inspect creep
accumulation, hold entry and relaxation before choosing a fit. Success at this
screen would not constitute material admission. PR197 remains unmerged.

## Fixed-card outcome

All five arms complete the measured early paths. Joint costs are 0.0025760801
(original), 0.0025747723 (deletion only), 0.0025964174 (+0.1), 0.0027230543
(+0.25), and 0.0026238453 (-0.1). Every nonzero interaction arm worsens both
the original and deletion-only objective. The tiny improvement from deleting
the weak static rates is not evidence for interaction. No interaction fit or
coupling-range extension follows. Retain this as a rejected declared screen,
not proof that all possible interaction potentials are invalid.

Eleven synthetic checks cover potential equivalence, dissipation, both control
modes, zero-coupling reproduction, time splitting, Radau/BDF agreement, sign
symmetry, failed-trial input retention and analytical off-load recovery without
irreversible flow. These verify the research equations, not a P92 material.
