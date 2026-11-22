# Conditional recovery-budget sensitivity, before another constitutive law

The observed-path decomposition at 210 MPa distinguished first post-dwell
SLR and SLC hold rates, conditional on the fixed two-Kelvin card. It did not
include modulus or relaxation-time sensitivity. Before using that distinction
to justify another mechanism, check its dependence on those existing values.

Use the same recorded windows and exact piecewise-linear convolution. Sample
each of the two log10 modulus and two log10 time coordinates at -0.15, 0,
0.15. Include E=151 GPa, the original 155.176 GPa and 162 GPa. These two
literature points have different provenance, documented in the
[modulus audit](p92_modulus_discriminator_20261122.md); they are not endpoints
of a measured confidence interval. Retain only combined five-coordinate
Euclidean norms <=0.3. This produces 211 fixed cards including the original,
without optimizing an objective or enlarging the prior coordinate radius.

For each card subtract its elastic increment and exact linear recoverable
increment from the observed strain increment. Retain initial-memory and
quantization sensitivity exactly as in the original budget. Compare SLR
minus SLC rate intervals at the same nominal stress band. No observed
response is supplied as a boundary condition for a forward material test.

This is a finite sensitivity lattice, not an exhaustive bound over the
continuous neighborhood, a plastic-strain measurement, a statistical interval,
or a new material fit. Any loss of separation must be reported rather than
discarded. All original response gates remain unchanged. No runtime API,
material coefficient or state variable is added by this diagnostic.

## Outcome

All 211 sampled contrasts remain positive. The smallest lower contrast is
4.5571e-7 /s, versus approximately 2.4491e-6 /s for the original-card interval.
Thus the diagnostic distinction survives these samples, but its margin is
substantially sensitive to the recoverable partition. This does not prove
positivity everywhere in the continuous ball, or identify a microscopic
mechanism. The archived record retains every card and both original windows;
no favorable subset or optimized coefficients are selected.
