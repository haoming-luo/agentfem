# Isolate common kinematic memory without a new drag law

The independent-flow card loses the experimentally observed suppression of
post-creep relaxation at similar stress. Its high-threshold flow has no direct
hardening from the lower-threshold creep flow. The prior shared-drag model
changed the flow law, drag evolution, coefficients and backstress structure
together; its failure did not isolate sharing kinematic memory alone.

The common-effective-stress organization is present in
[Li et al. (2021), equations 12--17](https://doi.org/10.1016/j.ijmecsci.2020.106044).
[NEML's Chaboche interface](https://neml.readthedocs.io/en/dev/vp_flow/chaboche.html)
also separates the flow surface from nonassociative strain/time hardening.
Neither source establishes the following parameter-preserving P92 hypothesis.
Do not equate it with the earlier published drag model or claim a NEML run.

## One discrete replacement

Keep the original two threshold/power flow functions, four backstress
coefficients, two recoverable strains and twelve state entries. Replace
branch-specific effective stress by `s = sigma - sum(all four X)` for both
flow functions. For each backstress use the **total signed flow** as its
hardening drive and total absolute flow as its dynamic-recovery clock.
Accumulate each branch's own absolute flow in its existing history slot.
Keep static rates, saturation law and Kelvin kinetics unchanged. First admit
only constant strength/hardening/saturation and the original linear Kelvin
card, so no undefined history coupling is smuggled into this comparison.

The stored energy remains `sigma^2/(2E)+sum(X_i^2/(2H_i))+sum(K_j*a_j^2/2)`.
The irreversible power is `s*(f_A+f_B)` plus the nonnegative backstress
recovery terms. Each flow has the sign of s; dissipation is nonnegative.
The original rectangular backstress bounds remain invariant because at a
saturation boundary the total-flow hardening and dynamic recovery cancel or
point inward, and static recovery points inward. Kelvin recoverability stays.
No new coefficient, fit coordinate, state, scale or protocol label is added.
The state meaning changes, so assign a separate model identity; never reuse
old checkpoint states as though they belonged to this law.

## Prospective gate

Compare exactly two fixed-card arms: independent memory control and common
memory. Verify zero-state agreement, analytic stress-controlled inactive
recovery, mechanical power, sign symmetry, invariant boundaries, continuation
and independent integration first. Replay early RF/SLR/SLC cycles 1--3 with
unchanged recorded-ramp controls. If aggregate error or the simultaneous
creep/hold/reversal evidence worsens, do not identify this card or broaden
ranges. A fixed-card failure is not proof against all common-memory models.
If a useful result emerges, use the existing bounded same-budget derivative
audit/identification, then freeze for later development cycles. No merge or
release, and no change to scientific acceptance thresholds.

## Complete common-memory arm and a distinct slow-memory discriminator

The original control cost is reproduced (0.00236078003); sharing all four
components raises cost to 0.03300267184. The first SLC held stress drops
from 123.087 to 93.292 MPa before creep. The subsequent hold endpoint rises
from 160.728 to 184.061 MPa, but its late rate becomes still too small
(-0.053506 rather than observed -0.181928 MPa/s). Do not select this based
on its attractive early rate or identify it. Preserve the failed complete arm.

One narrower structural question remains: common **slow** memory while each
fast backstress follows its own flow. Declare this separately before solving,
not as a fitted mixing fraction. Effective stresses become
`s_A=sigma-X_A1-X_A2-X_B2` and
`s_B=sigma-X_B1-X_A2-X_B2`. Both slow backstresses are driven by total
signed flow; fast components retain their own signed flow. Dynamic recovery
uses the corresponding sum of absolute flows, so invariant bounds and positive
dissipation remain valid even when the two effective stresses have opposite
signs. The same energy gives flow dissipation `s_A*f_A+s_B*f_B`.
This shares long-lived kinematic memory without also making the very stiff
fast B component harden under A creep. It is an independently derived
structural hypothesis, not a published P92 law. No extra parameter or fitted
mixing weight is introduced. Exactly one original/slow-shared fixed comparison;
retain the same simultaneous response gates and no fitting of worse evidence.

The slow-only comparison also completes all three early protocols but raises
joint cost to 0.02707874269 from 0.00236078003. Both discrete hypotheses are
rejected at this fixed-card gate; neither enters identification. Preserve both
arms in `2026-11-22-common-memory-probe.json` and
`2026-11-22-slow-common-memory-probe.json`. This isolates a failed topology
replacement, not a proof that every jointly calibrated shared-memory model fails.
