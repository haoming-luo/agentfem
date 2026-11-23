# Static strength recovery versus restoration to virgin strength

This independent discriminator keeps both linear recoverable branches and
the original magnitude dynamic-backstress rule. It does not stack nonlinear
recoverable kinetics or the rejected irreversible primary-creep replacement.

## Source and hypothesis

The NEML ChabocheVoceRecovery documentation explicitly separates plastic
hardening toward Rmax and static recovery toward Rmin:
https://neml.readthedocs.io/en/stable/hardening/non/special_chaboche.html.
For a linear static exponent its scalar strength kinetics is affine in
strength at a fixed plastic rate. NEML's negative hardening convention and
tensor normalization must not be silently imported into our positive scalar
strength convention. This is a reduced hypothesis informed by that equation,
not a numerical reproduction of NEML or a supplied P92 parameter card.

The prior bounded resistance discriminator used plastic softening and static
restoration to virgin strength. Its failure does not test static softening
with strain-driven rehardening. Here retain loss L=1-s and write

    Ldot = r*(1-smin-L) - b*pdot*L,
    k_effective = k*(1-L).

Initially L=0. Both branches share the same accumulated-rate driver
pdot=abs(flow_A)+abs(flow_B), but no recoverable rate enters it. The existing
two loss slots are retained for compatibility and evolve identically from
equal initial states; they are not two independently fitted mechanisms.
No protocol name, cycle index or measured stress enters the law. The interval
0<=L<=1-smin is invariant. Strength remains positive. This is a kinetic
resistance variable without added stored energy; it is not a quantitative
dislocation-density or heat-release model. Existing mechanical dissipation
and recoverable energies must remain correctly accounted for.

The BAM original discussion (DOI 10.1016/j.ijfatigue.2022.107381, sections
3.1--3.3) motivates examining competition between deformation and recovery
under different holds, but does not identify this specific evolution law.

## Prospective fixed arms and decision

Use the same archived original spectrum seed and early cycles 1--3, all three
protocols and measured ramps. Predeclare a zero-amplitude control, followed by
smin=.9 with (b,tau)=(250,1800),(1000,1800),(250,180), tau=1/r in seconds.
These are screening scales, not fitted or literature-derived constants.
Do not expand them after the result. Validate inactive exponential softening,
boundedness, signed symmetry, mechanical power, unchanged zero-amplitude
limit, split continuation and independent integration before the data run.

Any subsequent identification would replace, not add to, the two weak
branch-A static-backstress coordinates. The amplitude remains fixed at .1;
only b and r replace those coordinates. Keep the declared 22/10 direction
budget, norm .3, 40 optimizer evaluations and all original response gates.
Do not fit if the fixed arms only trade a lower SLR peak for worse creep or
post-creep SLC relaxation. PR #197 remains unmerged and unreleased.

## Fixed-screen result: do not identify this candidate

All four arms complete the early continuous paths. Their unchanged objective
values are 0.002576080097253627 (control), 0.004197033618348164,
0.0031054742635773947 and 0.008127678895494601, respectively. The zero-amplitude
control agrees with the original spectrum objective to 3e-13. At the third
SLC peak, errors change from -3.187 MPa to -25.821, -11.006 and -51.325 MPa.
Thus this particular initial-high-strength / static-softening construction is
not selected for identification. The conclusion does not reject all Voce
hardening or the NEML model: our reduced mapping and initial condition are
explicit additional assumptions. Retain the failed arms, not only the best one.

The 24 focused tests covering this discriminator and the preceding resistance
implementation pass in fenicsx-env. This establishes numerical and regression
checks, not experimental accuracy. No parameter budget or response threshold
was changed to obtain these results.
