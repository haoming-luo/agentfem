# Transient strain hardening with static recovery: prospective discriminator

The static-softening screen starts at its maximum strength and reduces it
during a hold. It worsens SLC. That initial condition is not equivalent to
building additional strength during creep and recovering it during a long
low-stress dwell. Test the latter separately, retaining both recoverable
strains, their energies, and the original dynamic-backstress rule.

Source: NEML's ChabocheVoceRecovery equation,
https://neml.readthedocs.io/en/stable/hardening/non/special_chaboche.html,
separates deformation-induced hardening and static recovery. Our bounded
scalar reduction is not a reproduction of its full tensor formulation:

    k_i_effective = k_i * (1+h)
    hdot = b*(abs(v_A)+abs(v_B))*(hmax-h) - r*h
    h(0)=0, 0<=h<=hmax.

This differs from the previous cross-drag experiment, which reduced only
branch B's mobility using branch A's flow and branch B erasure. Here both
strengths respond to total irreversible flow. Recoverable strain does not
drive h. It is a kinetic resistance, not a measured dislocation density or
an assigned extra stored-energy term. The existing positive strength ensures
nonnegative flow dissipation. No protocol label, held stress measurement,
cycle number or special hold switch enters the equation.

The fixed amplitude hmax=.1 is a screening hypothesis, not a fitted constant
or literature value. Predeclare control and (b,tau)=(250,1800), (1000,1800),
(250,180), with tau=1/r. Do not extend this list after inspecting outcomes.
Use only early cycles 1--3 and the same archived spectrum seed. Any later
identification would replace the two weak A static-recovery coordinates with
b and r; hmax stays fixed. No fitting-dimension increase, budget increase or
acceptance change. A lower aggregate objective alone is insufficient: inspect
SLC creep accumulation, the subsequent hold entry and its actual relaxation,
and SLR peaks jointly. Do not identify a candidate with a clear adverse tradeoff.

Before data evaluation verify invariant boundaries, zero-amplitude reduction,
zero-flow exponential recovery, sign reflection, power balance and independent
integration with split continuation. This is a separate research model/state
identity, not an added production material option. PR #197 stays open.

## Fixed screen and bounded identification decision

Control cost is 0.0025760800969908517; the three declared arms give
0.0031796601794222, 0.003957455268976496 and 0.0028048204173581072.
None is accepted. The first arm changes the SLR third peak error from
15.915 to 13.360 MPa, and the first SLC post-creep hold endpoint from
159.025 to 160.854 MPa (observed 174.88). Its early secant moves from
-4.815 to -4.744 MPa/s (observed -3.794). However, the SLR first hold
endpoint moves from 156.445 to 158.726 MPa (observed 148.34), and SLC
creep endpoint strain from .00103459 to .00102864 (observed .00106).

This is a measured tradeoff, not evidence of improvement. Unlike the
subthreshold replacement, it completes all early paths and moves the two
targeted discrepancies in the desired direction without losing the creep
endpoint gate. Test whether the tradeoff persists under the already-declared
bounded joint identification, using the original first arm as center, not
selecting a new center from later-cycle results. Substitute b and r for the
two A static coefficients; keep 22 candidate and 10 retained directions,
norm .3, at most 40 optimizer evaluations, the same residual definition,
derivative and observability gates. Count actual residual evaluations too.
The .1 amplitude is fixed, not an additional identified coefficient. Both
recoverable strains and all original backstress evolution remain intact.
Freeze the outcome for ten-cycle evaluation even if the early cost worsens.

Parameter accounting: the two A static coefficients retain their seed values
rather than being identified. The two replacement coordinates identify b and
r. Those two frozen A rates remain in the law; freezing does not delete them.
The fixed maximum .1 is also an additional constitutive assumption, so this is
equal **fitting dimension**, not an assertion that the full material record
contains the same number of constants. A favorable result would still require
review of that assumption; it cannot be promoted under an unqualified
"no additional material parameters" claim.

This decision was made after inspecting the fixed early screen. It is an
amendment to the original qualitative "no clear adverse tradeoff" selection
rule, not a wholly preregistered successful screen. Keep that distinction in
the evidence: the bounded fit is a diagnostic of the tradeoff, and the same
unchanged response gates still decide rejection. Neither its convergence nor
a lower cost may be reported as material admission.

## Matched fit and frozen result: rejected

The fit converged by the unchanged gradient criterion: 13 optimizer evaluations,
363 actual residual evaluations, cost 0.0025483909798906827. This is 7.947%
worse than the stronger matched original control (0.0023607800272608744).
All three frozen ten-cycle trajectories pass independent integration and
waveform-refinement checks but fail the unchanged response screen. The SLR
tenth peak remains 8.269% high. The first SLC creep endpoint is .00100170
against .00106; its following hold ends at 159.600 MPa against 174.88 and has
an early secant -5.270 against -3.794 MPa/s. Do not adopt this candidate.

Its replacement-space audit has minimum projection-Gram eigenvalue .0133085
and weakest-combination reachable half-width .0226158. This qualifies the
extent of the bounded exploration; it does not turn failed responses into
passes. The fit, frozen response, residual partition and geometry are archived
as the four `2026-11-22-transient-strength-*` evidence records. The subsequent
constructor-only restriction against combined research mechanisms changes no
equation or coefficient of this completed card.
