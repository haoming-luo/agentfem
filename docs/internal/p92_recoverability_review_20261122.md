# Recoverability and mixed-control mechanism review

## What the current evidence identifies

The BAM protocols are three different loading histories, not three materials.
RF holds the positive and negative strain extrema. SLR adds a zero-strain
relaxation dwell. SLC instead adds a constant-force creep dwell. The latter
changes the state inherited by the next reload and relaxation. One shared
material card must predict that inheritance; measured creep strain cannot be
inserted as an input. The sources explicitly describe the long service-like
holds as non-standard experiments, not a universal certification test:
https://doi.org/10.1016/j.dib.2023.109333 and
https://doi.org/10.1016/j.ijfatigue.2022.107381.

At the same stress in the first post-creep hold, the current scalar model
predicts similar SLR and SLC rates while the observations differ substantially.
Signed branch-flow accounting closes the nonelastic strain increment. It
locates the model's excessive irreversible flow, but does not experimentally
identify individual microscopic mechanisms. A recovered strain may contribute
negatively; normalizing absolute component rates into positive fractions would
misrepresent that physics.

## Reliable sources and what they do not establish

| Source | Verified scope | Consequence for this experiment |
| --- | --- | --- |
| Zhang et al., 2017, [anelastic recovery](https://doi.org/10.1016/j.ijmecsci.2017.01.018) | Publisher abstract and introduction: stress-controlled P92 at 898 K, recovery during unloading and off-load periods | Retain delayed recoverable strain. This is not the BAM heat at 620 C and supplies no verified transferable coefficients here. |
| Benaarbia et al., 2018, [thermodynamic decomposition](https://doi.org/10.1016/j.ijplas.2018.03.015) | P91 model combines viscoelastic and viscoplastic mechanisms | Separate recoverable and irreversible response; do not equate a positive energy balance with a uniquely identified decomposition. |
| Zhang et al., 2022, [hybrid stress/strain control](https://doi.org/10.1016/j.ijmecsci.2022.107369) | Accessible publisher abstract/introduction studies P92 and Inconel 718 under mixed control | Pure strain-controlled calibration may transfer poorly. Full equations/card were not obtained, so our subthreshold screen is not a reproduction of its Walker model. |
| [NEML Walker documentation](https://neml.readthedocs.io/en/main/walker.html) | Full documented equations for Alloy 617, including deformation-driven hardening and static recovery; notes that its implementation omits the original overshoot contribution | Distinguish verified equation structure from material transfer. Do not label an arbitrary history multiplier as the published complete model. |
| [NEML ChabocheVoceRecovery](https://neml.readthedocs.io/en/stable/hardening/non/special_chaboche.html) | Explicit deformation hardening/static recovery equation and sign convention | Supports a bounded strength-history discriminator; not its amplitude, initial condition or a P92 parameter set. |
| Gong et al., 2026, [primary creep regeneration](https://doi.org/10.1016/j.ijfatigue.2025.109395) | Publisher abstract: 316H at 650 C; reverse plastic deformation and cyclic hardening affect renewed primary creep | Supports distinguishing reversal memory from a clock restarted at each hold. Only the abstract was accessible; neither its Gorash modification nor coefficients are implemented here, and 316H is not P92. |

The [Wolff, Bohm and Taleb technical report](https://www.math.uni-bremen.de/zetem/cms/media.php/262/report1005.pdf)
provides a thermodynamic treatment of interacting inelastic mechanisms. This
is not a newly untested direction in this repository: the existing
`p92_coupled_recovery_reference.py` couples the A/B backstress energies and
evolution, while retaining separate Kelvin strains. It does **not** couple
Kelvin strain directly to backstress. Its fixed-card negative evidence is
already in `p92_mechanism_coupling_20261121.md`. A different coupling matrix
cannot be presented as a new mechanism without first showing the mathematical
difference and a discriminating physical observable.

## Decisions, not an expanding collection of fits

Final parameter accounting distinguishes fitting dimension from constitutive
complexity. Freezing two old coefficients while identifying two new ones is
not a strict deletion/replacement of constants. The nonlinear-kinetics and
transient-strength diagnostics do this and are not coefficient-neutral
promotion candidates. Their negative evidence stays visible. The final
recoverable-memory mobility law has no added independent coefficient/state;
the explicit backstress-coupling experiment deletes the two replaced rates.
None of these distinctions changes the failed scientific screens.

1. Preserve both recoverable branches. The previous irreversible replacement
   worsened hold entry and subsequent actual stress even when an anchored
   shape looked better.
2. Reject the present nonlinear-dashpot replacement as an upgrade: its matched
   objective is worse and all frozen response screens fail. Keep its analytic
   verification and failed evidence.
3. Reject initially high strength followed by static softening, and reject
   the declared rate/slope-matched zero-threshold arms. The latter can exceed
   the next prescribed strain target; do not repair the loading to hide this.
4. The bounded transient-strength comparison is now complete: its joint cost
   is 7.95% worse than the matched original, and all frozen response screens
   fail despite numerical agreement. Its extra fixed maximum-strength
   hypothesis is not advertised as unchanged total parameter count.
5. Three-time recoverable bandwidth and the declared one-flow/three-recovery
   replacement also fail their fixed-card screens. The latter exceeds the
   next measured strain target; no continuation repair or subsequent fit.
6. The [recoverable/backstress interaction](p92_recoverable_backstress_coupling_20261122.md)
   deletes two weak static rates rather than adding coefficients. All three
   nonzero interaction arms worsen the original and deletion-only controls;
   no fit follows. This differs from the earlier A/B backstress coupling.
7. The [replacement-directed fit-space experiment](p92_fit_subspace_20261122.md)
   converges but does not outperform the ordinary shared-history space.
   All frozen experimental screens fail despite numerical agreement. Do not
   enlarge the radius or blame the retained geometry alone.
8. The [recoverable-memory mobility discriminator](p92_recoverable_mobility_20261122.md)
   retains the two recoverable states and all existing coefficients. Static
   recovery feedback, but not flow feedback, improves the fixed early cost.
   The unchanged-budget fit now converges with 2.405% lower joint cost, but
   all frozen response screens fail: SLC hold entry worsens, its early/late
   secants remain outside the gate, and SLR cycle-10 peak error grows to
   9.90%. Reject promotion; do not enlarge the reached coordinate bounds.
9. The [conditional recovery-capacity bound](p92_recovery_capacity_20261122.md)
   separates a possible long-time spectral contribution from what the recorded
   late creep samples can resolve. It does not justify another fit or a new
   experimental pass.
10. The [matched activated-flow replacement](p92_activated_flow_20261122.md)
    preserves the threshold and local reference rate/tangent but fails the
    first SLC continuation. No subsequent fit or loading repair follows.
11. The [recoverable-memory resistance discriminator](p92_recoverable_resistance_20261122.md)
    improves early relaxation but increases joint cost by a factor of 3.24
    and worsens late relaxation. Its prospective admission conjunction fails;
    no fit follows, despite the attractive isolated early-rate improvement.

The final-code original-control replay reproduces all ten-cycle sampled
responses and end states exactly for RF, SLR and SLC. Its independent
integration and waveform-refinement checks also pass. This is code-regression
evidence, not a new experimental pass; the original response failures remain.
See `evidence/creep/2026-11-22-control-replay-audit.json`.

The [timescale and regeneration audit](p92_timescale_evidence_20261122.md)
revisits an already admitted Grade 92 source and separates inherited memory
from flow-driven regeneration. It is not a new independent experimental
dataset. Compact archived diagnostics retain source identities and numerical
status; none of these rejected hypotheses is promoted to the runtime package.

The fitting residual already samples early hold stresses and stress-hold
strains, but it is a least-squares compromise, not a simultaneous feasibility
certificate. Resolved secant gates are evaluated separately. The prior L4
gate-weighted experiment is already recorded in
`p92_recoverable_reference_20261121.md`; do not rediscover it as an untried fix.
Unresolved quantized late-creep slopes are not passes. Nor should they be
silently fitted as precise derivatives. Later inspected cycles remain
development checks, not an independent validation set.

The next scientific decision must follow the completed paired comparison,
not another unconstrained parameter expansion. PR #197 remains open, with
no production material promotion, merge or release.

## Input uncertainty recheck

The recorded temperature is not exactly constant. Across cycles 1--3 the
checksum-admitted records span 891.55--894.05 K (RF), 887.35--896.95 K
(SLR), and 887.15--896.95 K (SLC). However, the first positive-hold fitting
samples span only 892.85--894.35 K for SLR and 892.85--893.65 K for SLC;
their last sampled temperatures are both 892.85 K. Do not use the whole-cycle
temperature range as though it were a simultaneous difference at that hold.

The main paper's experimental section calls the extensometer observations
mechanical strains and describes induction heating with a spot-welded type-S
thermocouple. It does not justify silently subtracting a guessed thermal
strain from these data. The present material experiments are nominally
isothermal; temperature dependence remains an input/model limitation, not a
new fitted explanation of the residual. No thermal coefficient, activation
energy or altered strain history is introduced by this check.

At the first post-creep positive hold, the matched original card changes
stress by -98.630 MPa versus recorded -84.780 MPa. The recorded endpoint
strain drift corresponds to E*Delta-epsilon=-4.655 MPa. A purely algebraic
subtraction of that elastic scale would enlarge the change discrepancy from
-13.850 to -18.505 MPa, not remove it. This subtraction is not a valid
nonlinear controller replay or an acceptance correction; it only rules out
the claim that this simple endpoint correction explains the excess relaxation.

The fixed-card switch audit then replays all-lower and all-upper recorded
switch brackets with identical observation indices in both arms. For the
original card, the first SLC positive-hold end remains 160.74--160.80 MPa
against 174.88 MPa observed, with late slopes -0.09032 to -0.09029 MPa/s
against -0.18193. For the fitted mobility card, the end is 164.28--164.39 MPa
and late slopes -0.08299 to -0.08282. These two prescribed perturbations do
not remove the late discrepancy. They are not an exhaustive switch envelope,
a stochastic uncertainty interval, or evidence about every possible mixed
switch perturbation. Early secants use the common bracket-safe support and
must not be subtracted directly from the nominal-support diagnostic. No
parameter or acceptance limit changes. The source identities, support and
failures are retained in the two `2026-11-22-*-switch-audit.json` records.

## Conditional flow correction, not an identified material coefficient

Keeping the archived average A and Kelvin contributions fixed, a diagnostic
B multiplier is `(observed_total - A - Kelvin_fast - Kelvin_slow)/B`.
For the original first 210 MPa bands this gives 1.301 for SLR and 0.262 for
SLC. Applying the recorded one-quantum endpoint sensitivity gives respectively
0.708--1.895 and 0.040--0.484. These are deterministic sensitivity intervals,
not confidence limits. The cross-protocol stress endpoints differ slightly;
each model interval is matched to its own observed endpoints.

This conditional accounting argues against another common B-rate multiplier.
It does not prove global infeasibility: changing kinetics also changes the
trajectory, duration and all other state contributions. In the fitted mobility
card's first SLC 230 MPa band the corresponding interval is entirely negative
(-0.571 to -0.125): even removing positive B flow would not reconcile that
fixed decomposition. An admissible new mechanism must alter the coupled
history, rather than interpret a negative multiplier as a physical flow law.
The following [local feasibility audit](p92_local_feasibility_20261122.md)
tests the retained parameter directions without inventing a new coefficient.

The [joint input-channel resolution audit](p92_control_resolution_20261122.md)
also shows why stress-only resolution is insufficient for interpreting tiny
late SLR rates. It leaves all original gates and failures intact; it is not
a revised acceptance policy or a claim that input uncertainty explains the
large post-creep SLC discrepancy.

The subsequent [fixed-card recorded-strain sensitivity](p92_recorded_hold_sensitivity_20261122.md)
confirms this distinction: it can reverse the tiny second SLR late-rate sign,
but does not remove the substantial SLC hold-end or relaxation-shape error.
All numerical checks pass while all response screens still fail. The
observations are not relabeled as authenticated controller commands.

The local signed-constraint audit also separates a diagnostic artifact from
a constitutive failure. Differentiating almost-zero absolute errors caused
the mobility derivative audit to fail; equivalent positive/negative
constraints restore derivative agreement. The same-budget nonlinear checks
still reject all three proposed fractions in both arms. No experimental
failure is removed and no additional frozen candidate is justified.
