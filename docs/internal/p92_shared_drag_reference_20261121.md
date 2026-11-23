# Shared backstress and evolving drag: research design

## Reason for a structural comparison

The independent-flow spectrum improves initial BAM SLR/SLC responses but still
misses cyclic peaks and post-creep relaxation. Fixed resistance, modulus, and
signed-recovery modifiers did not resolve this tradeoff. Rather than append
more modifiers, implement a separate, published shared-backstress architecture.

Source: Li et al., International Journal of Mechanical Sciences 190 (2021),
106044, [open manuscript](https://doi.org/10.1016/j.ijmecsci.2020.106044),
scalar equations 12–17 and Table 2. The original material is 10%Cr at 600 C,
not BAM P92 at 620 C. The table card is a transcription reference only; no
transfer or engineering qualification is implied. Full PDF text was available;
direct PDF download and screenshot failed, so table visual verification remains
open. Do not call a graph reproduction independently verified yet.

Use two power-law rates sharing effective stress `sigma-sum(X_i)` and positive
drag D, three backstresses, signed flow strain v and accumulated flow p.
`f=sum(A_j*(abs(sigma-sum(X))/D)^n_j)*sign(sigma-sum(X))`.
`Xdot_i=C_i*f-(gamma_i(abs(v))*abs(f)+K_i)*X_i`.
`Ddot=Cdrag*abs(f)-(gamma_drag(p)*abs(f)+Kdrag)*D`.
Each gamma interpolates positively between its initial and asymptotic values
using an exponential history law. Integrate `log(D/D0)` to preserve positivity
without clipping the physical drag evolution. Optional two-time recoverability
is an explicitly separate addition, disabled for the original reference card.

For our verification ledger, choose `psi=sigma^2/(2E)+sum(X_i^2/(2C_i))`
plus the enabled Kelvin energies. Treat drag as dissipative resistance with no
assigned stored energy. Then mechanical dissipation is
`(sigma-sum(X))*f+sum((gamma_i*abs(f)+K_i)*X_i^2/C_i)` plus Kelvin dissipation,
which is nonnegative. This ledger is our declared scalar accounting, not a
claim of reproducing a published thermodynamic proof or a life law.

## Gates before a BAM comparison

1. Explicit new state identity; no reuse of independent-branch restart states.
2. Exact zero-load drag relaxation and constant-state rate checks.
3. Mechanical power identity, sign symmetry, p>=abs(v), positive drag.
4. Split continuation and independently tightened BDF/Radau agreement.
5. Same measured controls and response gates, no forced SLC entry stress.
6. A published-card calculation is a formulation check, not P92 validation.
7. If re-identifying for BAM, use only cycles 1–3 jointly, record all bounds,
   then freeze for cycles 4–10, already inspected development observations.

The shared replay needs only named observable slots (stress, strain, work,
dissipation); it must not require other models to mislabel their internal state
as old branch variables. Numerical comparison must cover the dimensionless
log-drag coordinate explicitly, rather than silently treat it as MPa.

## Initial feasibility and prospective joint comparison

The text-transcribed table card produces implausibly large instantaneous flow
under our ideal stress jumps, and its independent numerical difference exceeds
the strict strain/energy limits. The paper's finite loading transitions and
table units still require checking; do not label this successful reproduction.
Its BAM use also leaves the subsequent SLC ramp direction. Retain the failure,
do not fit or reinterpret those units to obtain a desired curve.

Separately define a dimensional, explicitly non-literature starting card for
the shared architecture. The first flow scales (1e-4,1e-6)/s leave the SLC
continuation; reducing both by ten still exceeds the next target in cycle 2.
Balancing initial drag hardening against dynamic softening at D0=200 MPa,
Cdrag=10000 MPa, gamma0=50 yields a feasible three-protocol start. Its cost
0.0543800 is poor, not an improvement. These three feasibility attempts are
retained; none is calibrated or an experimental validation.

Identify only early cycles using shared parameters. Fix E, the two exponents,
dynamic-history rates, and the separately identified recoverable spectrum.
Audit fourteen positive log-coordinate directions: two flow coefficients,
three kinematic moduli, three static recovery rates, drag hardening/static
recovery, three initial/final kinematic-recovery paired scales and one paired
drag-recovery scale. This avoids independently changing every coefficient.
Require central-difference refinement consistency and a stable leading
eight-dimensional subspace before bounded fitting. The log-coordinate norm
is at most 1.2 (a wider *new-model starting-point* bound, not relaxation of
response gates). Allow forty optimizer evaluations and retain invalid trials
as rejected numerical-domain steps. A failed control path is never accepted
as a response. A converged candidate must undergo frozen full-state checks and
all original response gates. Later cycles never enter the objective.

## Rejected frozen candidate

The bounded identification terminated at cost approximately 0.02552, versus
0.002576 for the independent-branch recoverable spectrum. Optimizer convergence
is not response acceptance. RF and SLR frozen ten-cycle paths passed independent
integration and waveform-refinement checks but failed response gates. SLC left
the declared subsequent ramp direction at segment 39, cycle 6: predicted strain
0.0020132564 exceeded the next target 0.002, at predicted held stress
115.142239 MPa. No forced stress, state reset, or reversed ramp is substituted.
This candidate is rejected; adding shared drag did not solve the discrepancy.

The public table's indexed PDF text and indexed HTML disagree on the exponent
of the second power-law exponent (278 versus 0.0278). Neither representation
has been visually verified against the original table. The provisional card
must not be called a verified published parameter set; changing that exponent
to obtain a plausible response would not resolve provenance. The separate BAM
candidate uses explicitly declared dimensional starting values, not this card.

On 2026-11-22, the indexed [author thesis, section 3.4, Table 3.4.2](https://www.research-collection.ethz.ch/server/api/core/bitstreams/fbedddb7-9ebe-4add-9bb2-e212c838c133/content)
also reports the second exponent as 278 and distinguishes signed flow magnitude
from accumulated flow in the two recovery laws. Direct download returned 429;
the original journal repository timed out and the publisher returned 403.
This corroborates the text extraction but does not close visual verification
or the loading/time-unit reproduction. Keep the card unverified and unchanged;
do not substitute a more convenient exponent or silently rescale time.
