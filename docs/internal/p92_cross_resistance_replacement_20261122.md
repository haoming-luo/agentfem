# Replace a directional memory by bounded shared resistance

## Why this differs from the rejected hypotheses

The original two-flow spectrum retains four signed backstresses. Common
kinematic-memory probes changed which signed stresses enter each flow; they
failed. Previous transient-strength hypotheses added a resistance state, and
recoverable-memory feedback inferred resistance from Kelvin strain. Neither
isolates replacing one existing directional memory by an unsigned resistance
with its own existing time scale.

Sonntag et al., https://doi.org/10.1016/j.ijfatigue.2022.107381, pages 6--9,
discuss competing hardening and recovery and their different effects during
long creep versus relaxation. Hosseini et al.,
https://doi.org/10.1016/j.ijmecsci.2020.105546, distinguishes isotropic resistance
from signed pile-up memory in its accessible introduction. These are physical
motivations, not published equations or coefficients for the following law.
Christopher and Choudhary, https://doi.org/10.1080/14786435.2014.944608,
study P92 work hardening at 300--873 K; this is not validation at BAM's 893 K.

## Exact replacement and thermodynamic check

Replace B's slow backstress X_B2 by R >= 0, retaining A's two backstresses,
B's fast backstress, both accumulated absolute flows and both Kelvin strains.
Use the deleted component's existing H, a and r as R's hardening coefficient,
upper bound and static decay. Its dynamic-recovery exponent is removed from
the active law; no new coefficient is introduced. Start R=0 with the same
virgin-state assumption; old evolved checkpoints are not compatible.

Let s_A=sigma-X_A1-X_A2, s_B=sigma-X_B1. Keep the two power laws but replace
their thresholds k_j by k_j+R. Write v_j for signed flow, w_j=abs(v_j):

    Rdot = H*w_A - (H/a)*(w_A+w_B)*R - r*R.

The low-threshold branch builds resistance; both flow magnitudes and time
remove it. No loading-control label enters this equation. It is our declared
cross-mechanism hypothesis, not a measured dislocation density. At R=0 the
rate is nonnegative; at R=a it is -H*w_B-r*a <= 0. Thus [0,a] is invariant.

Use stored energy R^2/(2H) in place of X_B2^2/(2H). With the retained
backstress and Kelvin energies, subtract R*Rdot/H from the corresponding
flow power. Since active A flow has |s_A|>=k_A+R, its residual dissipative
power is at least k_A*w_A+R^2*w_A/a. B's flow and the other recovery terms
are nonnegative as well. Stress reflection reverses stress, signed memory
and strain but leaves R unchanged. Unloaded Kelvin strains still recover.

## Prospective experiment

Test energy-rate identity for both controls, reflection, invariant boundaries,
inactive analytic recovery, independent integration, split continuation and
invalid states before real data. Compare the original and one mapped card on
RF/SLR/SLC cycles 1--3 with unchanged controls and gates. Retain all outcomes.
This is a topology discriminator, not a parameter fit: a worse fixed-card
cost cannot disprove the family. It also cannot justify automatic repeated
fits. If the fixed response gives a clear simultaneous improvement without
breaking continuation, declare one unchanged-budget identification separately;
otherwise retain the failure and do not broaden bounds. No runtime promotion,
merge, release, engineering acceptance or life qualification is implied.

## Fixed-card outcome

All three continuous early paths completed. The replacement cost increased
from 0.0023607800 to 0.0030951389 (about 31%). In SLC the post-creep hold
endpoint fell from 160.728 to 158.936 MPa versus 174.88 MPa observed. The early
relaxation rate also worsened, although the late rate moved toward the
observation. The creep endpoint fell from 0.000991685 to 0.000977349 versus
0.00106 observed. Thus no simultaneous improvement was obtained.

The prospective stopping rule applies: no identification of this candidate,
no parameter-bound expansion and no frozen-cycle success claim. Six focused
tests passed, including power balance, reflection, invariant boundaries,
independent integration and continuation. Those verify implementation, not
the material hypothesis. The archived paired probe preserves all three
failed response screens and its original implementation hashes. A later
driver-only change binds loop variables explicitly for lint clarity; the
archive hashes are not rewritten to pretend it preceded the experiment.
