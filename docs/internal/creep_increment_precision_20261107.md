# Constant-stress creep update: small-increment precision

Review of the public Kachanov–Rabotnov material-point path found cancellation
in the exact update, not an error in its constitutive equations. With the
reviewed 650 C coefficients, 110 MPa and a `1e-12 h` interval, the previous
implementation returned zero strain and damage increments. The independent
90-digit reference gives approximately `1.3483637994e-17` and
`2.7625999849e-17`, respectively.

## Numerical design

For old integrity `I=1-D`, power `p=phi+1` and normalized stress `s`, retain
`f=p B s^m dt/I^p` and evaluate `log(I_new/I)=log1p(-f)/p`. Compute integrity
loss and the integrated creep increment using `expm1`, rather than differences
between powers near one. These are standard cancellation-resistant primitives;
see the [Python math reference](https://docs.python.org/3/library/math.html#math.expm1).
No coefficients, exponent, stress measure or time unit change.

The same expression stabilizes the prescribed failure threshold and rupture
time. A supplied terminal state remains terminal instead of losing damage
when the configured threshold is lower than its existing damage. This remains
a bounded constant-stress material-point model, not a new global damage solver.

## Acceptance

The dedicated tests compare a 90-digit Decimal integration at small and
ordinary increments, positive/negative/zero integral exponents, nonzero prior
damage, tensor direction, repeated subdivision, explicit stress/time reference
scales, tiny failure thresholds, and terminal-state monotonicity. Existing
constitutive, analytical structural references and release golden contracts
are also checked: 76 targeted tests passed, including the actual thermal-wall
example, with lint and generated-document checks passing. This numerical
repair does not resolve discrepancies in
published material coefficients or qualify P92 engineering life.
