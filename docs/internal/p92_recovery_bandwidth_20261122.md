# Recovery bandwidth: conditional discriminator, not another fitted state

Benaarbia, Rouse and Sun's [2018 model](https://doi.org/10.1016/j.ijplas.2018.03.015)
uses three viscoelastic components alongside viscoplasticity for P91 at high
temperature. The institutional full-text search index exposes this decomposition;
direct PDF retrieval returned 403 in this audit. This motivates testing time
bandwidth, not copying its parameter table or claiming a BAM P92 validation.

The present nonlinear Kelvin dashpots changed stress dependence but failed the
matched comparison. A distribution of linear retardation times is a distinct
hypothesis. Preserve both existing recoverable channels: split the slow one
into two positive components, without changing its equilibrium compliance or
initial constant-stress creep mobility:

    H_minus = H_plus = 2 H_slow
    tau_minus = tau_slow / (1+s)
    tau_plus  = tau_slow / (1-s),  0 <= s < 1.

Thus sum(1/H_j)=1/H_slow and sum(1/(H_j*tau_j))=1/(H_slow*tau_slow).
The zero-spread limit exactly reproduces the two-mode response when each split
strain starts at half the original slow strain. For positive spread the
intermediate constant-stress creep response is smaller, not larger, than the
original mode. This is a testable constraint, not a freely adjustable curve.

Use a separate research identity and a third recoverable state. Keep positive
quadratic stored energy, nonnegative dashpot dissipation, the original A/B
flow and dynamic/static backstress rules, and continuous mixed controls. No
protocol, cycle counter or measured creep response enters the law. Do not
combine this screen with transient strength or nonlinear recovery kinetics.

First verify zero-spread equivalence, signed constant-stress creep and unloaded
recovery analytically, both compliance/mobility invariants, power balance,
reflection, independent integration and split continuation. Preparing those
tests does not authorize selecting a new material card while the current
transient-strength comparison is running.

The early-only screen has exactly four arms: s=0, .25, .5, .75. No fitting or
new range. Scheduling amendment before running this screen: its four fixed
arms may be computed independently while transient-strength identification
finishes. They do not use that fit's parameters or intermediate results, and
cannot select a replacement before both comparisons are evaluated. Inspect actual
SLC creep accumulation, subsequent hold entry and relaxation together with
RF/SLR peaks. A failed control continuation is a failure, not a reset.
Only a coherent improvement would justify identification. Such a later fit
must replace the two weak A static-recovery coordinates with the extra mode's
modulus/time, preserving the existing 22-candidate/10-retained budget and all
gates. Adding a recoverable state does not authorize extra fitted parameters.

For this fixed screen the split moduli and times are tied to the original
slow mode and a declared discrete spread; they are not independently fitted.
Any subsequent free extra-modulus/time fit needs a separate full parameter
accounting: freezing two old coefficients is not the same as removing them
from the material law. Do not describe such a fit as an unchanged total
material-parameter count merely because its optimization dimension stays ten.

This construction is our scalar discriminator. It is neither the complete
published tensor model nor an industrial material card. PR #197 stays open.

## Fixed-screen outcome: do not identify this split

All four declared arms complete. Their costs are 0.002576080096994543,
0.0025899390498696703, 0.0026607106008599063 and 0.002879286304705975.
The strongest spread improves the first SLC creep-segment RMS strain error
from 7.7444e-5 to 5.7901e-5, but decreases its endpoint from .00103459 to
.000998339 (observed .00106). The following hold endpoint only changes from
159.0245 to 159.5772 MPa (observed 174.88), while its early secant worsens
from -4.8147 to -5.0540 MPa/s (observed -3.7937). The SLR third peak error
also increases from 15.9153 to 16.3528 MPa.

This is not a coherent improvement of accumulation, entry and relaxation.
Keep the original spectrum; do not follow this screen with another fit or
enlarge the spread range. It rejects these fixed, moment-preserving splits,
not every possible generalized recovery spectrum. The raw report is archived
with its source hash as `evidence/creep/2026-11-22-recovery-bandwidth-probe.json`.
