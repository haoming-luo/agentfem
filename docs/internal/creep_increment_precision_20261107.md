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

The same audit found cancellation in `PowerLawCreep.constant_stress_increment`:
at a stored start time of `1e10`, end time `start + 1e-5`, coefficient
`1e-12`, stress exponent 3 and stress 100, subtracting accumulated strains
underestimates the interval increment by 4.63%. The actual representable
duration is `9.5367431640625e-6`; this is the reference interval, not an
assumption that floating-point time has retained all requested decimal digits.
For time-independent rates, integrate rate times duration directly. Otherwise,
factor the power difference around the upper endpoint and use a bounded
`-expm1(-p*log(end/start))` fraction with `log1p` for nearby endpoints.
Zero-origin and zero-duration paths remain explicit; invalid clocks are rejected.
This avoids cancellation without changing the power-law integral or resetting
the physical clock.

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
are also checked. After adding the power-law repair, 184 targeted tests passed,
including the actual thermal-wall example, creep history, manufactured field,
elastic follow-up and prescribed-trial paths. The 47 dedicated precision tests
also cover wide time intervals, nearly singular integrable initial rates,
zero duration and the distinction between requested and representable clocks.
An additional zero-damage/high-power underflow regression passes in a 92-test
material-reference run. It keeps the zero-damage branch independent of the
unused damage exponent and avoids dividing by an underflowed integrity power.
Critical lint passes. This numerical
repair does not resolve discrepancies in
published material coefficients or qualify P92 engineering life.
