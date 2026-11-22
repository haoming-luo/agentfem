# Separate parameter effects through incoming history and local hold kinetics

The earlier state reconstruction varied admissible incoming states, but could
not establish which changes a preceding loading path would actually produce.
The previous all-path Jacobian mixed two effects: changed history at hold entry
and changed rate equations inside that hold. Resolve this distinction before
another mechanism or fit. The gate-weighted L4 experiment and local minimax
audit already failed; repeating them is not an untried solution.

For a parameter perturbation, replay the real preceding path and compare:

1. the changed material with its own causal incoming state;
2. the original material with the changed incoming state;
3. the changed material with the original incoming state.

Only the first is a continuous material prediction. The other two are
counterfactual derivative diagnostics, with their own initial stored energy;
they do not imply a physical jump with zero work. They must never replace
accepted response evidence or be continued as a validated prediction.

At first order, the derivative of (1) equals the sum of (2) and (3).
Check this chain-rule closure and finite-difference stability at log10 steps
0.001 and 0.0005. Keep the card, objective and gates unchanged. Examine the
six existing coordinates q_A, q_B, r_Bfast, r_Bslow, K_slow and tau_slow,
using first-cycle RF/SLR/SLC paths. Report first sampled stress, final sampled
stress and both resolved hold secants separately, normalized by their existing
diagnostic scales. Do not confuse the first sampled point with the exact switch.

This is a finite-difference causal sensitivity audit, not parameter inference,
a new material law, statistical identifiability or an accuracy improvement.
Retain both contributions and cancellation; absolute component magnitudes
cannot be presented as independent physical fractions. A failed derivative
or chain-rule check stops interpretation of that row.

## Outcome

All 18 coordinate/protocol checks pass: maximum two-scale relative difference
2.50e-6 and chain-rule residual 5.58e-7. The archive is
`2026-11-22-history-kinetics-split.json`, with the frozen card, source identities
and driver hashes retained. This is numerical verification of derivatives,
not verification of the material response.

For SLC, the normalized early-secant derivative with respect to log10 slow
Kelvin modulus is -7.263: its incoming-history contribution is -8.092 and its
direct-kinetics contribution +0.828. For log10 slow time these are -0.373,
-0.713 and +0.340. Increasing either coordinate has a locally helpful effect
on the overly negative early slope, but the causally changed incoming state
reverses that benefit. The modulus also strongly changes the first sampled
stress (normalized derivative +6.125). Thus a hold-only adjustment is not a
valid route to continuous prediction.

No parameters were fitted or changed. No failed response gate was removed.
The next mechanism must be assessed through its preceding creep and reversal,
not by accepting the counterfactual state substitutions as a physical model.
