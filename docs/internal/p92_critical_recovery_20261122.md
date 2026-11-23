# Critical-state recovery: prospective discriminator

Research only; PR #197 remains unmerged and unreleased. Preserve both
recoverable strain modes, both inelastic flows, all control waveforms, and
experimental admission tolerances. Do not add fitted parameters.

## Basis and distinction

Ohno and Wang's original 1992 formulation, printed p. 489, equations 4--9,
defines a critical backstress surface and recovery activated at that surface.
The primary page was visually inspected again for this experiment:
<https://doi.org/10.1299/kikaia.58.488>.
Equation 10 is the smooth model II, whose directional scalar reduction has
already been tested here. It must not be confused with model I.

For each scalar backstress use the constrained evolution

    Xdot + N_[-a,a](X) contains H*v - r*X.

Inside the interval, Xdot=H*v-r*X. At either endpoint, remove only the outward
part of this unconstrained rate. Without static recovery this is the scalar
critical-state rule. Including r inside this projection is our separately
tested static-recovery extension, not an equation attributed to the paper.
It avoids the ill-defined rapid switching produced by adding static recovery
after a Heaviside dynamic law. Here H, a and r are the existing coefficients;
the smooth recall exponents are no longer operative, not refitted.

With psi_X=X^2/(2H), the backstress contribution to dissipation is
X*(H*v-Xdot)/H. It equals r*X^2/H in the interior; at an outward constrained
endpoint it equals X*v, nonnegative because outward H*v-r*X has the sign of X.
The existing effective-stress flow dissipation and Kelvin dissipation remain.
This establishes a continuous scalar power identity, not structural or
engineering validation.

## Numerical and scientific gates

Use event-located boundary entry and release, rather than silently clipping
accepted trajectories. Check analytic constant-flow saturation, release on
reversal, pure static decay, sign reflection, work/energy/dissipation, split
continuation, independent Radau/BDF integration, and strict invalid-input
rejection. Preserve the old model as an unchanged control.

Only after these gates, replay early measured RF/SLR/SLC with fixed original
parameters. Record failures, including event or domain failures. If this
isolated discriminator is viable, use the existing 22-coordinate candidate,
10-direction retained identification budget, radius 0.3, 40 optimizer
evaluation ceiling and two-scale derivative checks. Piecewise smoothness is
not permission to relax the derivative gate. Freeze before later-cycle
checks; cycles 4--10 are inspected development data, not a blind holdout.
No coefficient-range expansion or combination with other unvalidated laws.

## Matched and frozen outcome

The fixed original card completes, with cost 0.00248088191 versus
0.00236078003 for its unchanged smooth-recovery control. The two-scale
derivative check of the matched identification passes (Jacobian discrepancy
0.00100403; subspace distance 0.00261899). The same-budget fit converges in
six optimizer evaluations, 216 residual evaluations including audits, at
cost 0.00237077495. This is 0.42% worse than the matched smooth control.
Bounds were not enlarged; zero-sensitivity critical saturation directions
were not presented as identified material constants.

All three frozen ten-cycle paths complete and pass independent BDF and
waveform-refinement checks. All experimental response screens fail, with
20 RF, 40 SLR and 69 SLC failed checks. Counts are diagnostics, not a ranking
that can replace the individual limits. SLR cycle-ten peak error is 9.00%.
The first post-creep SLC hold ends at 160.136 MPa versus 174.88 MPa;
early and late stress secants are -5.04572 and -0.0971441 MPa/s versus
-3.79373 and -0.181928 MPa/s observed. Numerical agreement does not resolve
these material-response discrepancies. Do not promote this candidate.

Compact matched/frozen evidence is stored in
`evidence/creep/2026-11-22-critical-recovery-{matched,frozen}.json`.
Full generated trajectories are identified by their original SHA-256 hashes.
The initial fixed probe exposed an empty-query bug in the piecewise dense
output evaluator; it was corrected and the probe rerun. Simultaneous critical
surface entry/release is tested explicitly rather than hidden by clipping.
