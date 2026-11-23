# Recorded stress ripple: diagnostic input, not response validation

The SLC force-controlled dwell contains small recorded stress variations.
The nominal material comparison inherits its predicted stress at the switch
and holds that stress. A diagnostic now applies only the recorded *relative*
stress variation about that predicted entry; it never resets entry to the
measured stress or prescribes measured creep strain.

For piecewise-linear stress variation add sigma_dot/E to the total strain
rate and sigma*sigma_dot/E to external power. Material evolution still uses
the instantaneous stress. Update cumulative work and retain the original
energy, dissipation and state-invariant checks. The control cursor advances
only after successful integration. Unknown material kernels are rejected:
direct smooth-ODE integration must not silently bypass the critical-state
kernel's event algorithm.

The audited implementation reproduces the earlier local diagnostic. In the
first SLC creep hold, endpoint strain increases from 0.000991685 to
0.001031434 (observed 0.00106). The subsequent hold endpoint changes only
from 160.728 to 160.825 MPa (observed 174.88). Its early secant changes
from -5.08144 to -4.83052 MPa/s (observed -3.79373), while its late secant
is essentially unchanged: -0.0902691 to -0.0903808 MPa/s, versus -0.181928.
The independent integrator agrees within 9.71e-7 MPa and 1.77e-11 strain.

Thus force variation explains part of creep accumulation, not the remaining
post-creep relaxation error. Recorded response is not an authenticated
controller-command history. In particular, its prescribed stress agreement
is not a new validation pass. Keep the nominal comparison as control,
all original gates, and the distinction between input sensitivity and model
admission. No fitting or material change is performed here.

Evidence: `evidence/creep/2026-11-22-stress-ripple-audit.json` includes the
full source/input/code identities, numerical comparison, unchanged response
screens and hold diagnostics. Dense trajectories remain local and are linked
by the original report hash. The shared implementation is
`tools.p92_stress_ripple_audit`, with analytic elastic-work, continuation,
independent-integration and transactional-cursor tests.

## Paired input interaction, declared before replay

The recorded-strain and recorded-stress diagnostics were initially separate.
Their effects need not add in a history-dependent material. Complete one
four-arm, fixed-card SLC comparison: nominal, recorded strain only, relative
stress only, and both. Reuse the existing continuous recorded-strain helper
and the audited relative-stress wrapper. Never prescribe SLC creep strain or
reset its inherited stress. Verify the combined arm with BDF and identical
piecewise-linear histories subdivided in both control channels. Compare the
same early cycles 1--3, with unchanged response limits and no refitting.
Report the nonlinear interaction rather than assuming separate small effects
prove a small combined effect. This closes an input-sensitivity question;
it does not replace authenticated controller records or qualify a material.

The paired replay completes. Nominal, strain-only, stress-only and combined
first post-creep hold endpoints are 160.728126, 160.570399, 160.825047 and
160.670304 MPa, versus 174.88 MPa observed. The combined-minus-additive
endpoint interaction is only 0.002984 MPa. Combined early/late secants are
-5.355012/-0.089035 MPa/s; the original targets remain -3.793735/-0.181928.
All four experimental screens fail. Combined creep strain is 0.001028995,
versus 0.00106 observed. Independent integration agrees within 3.52e-6 MPa
and 3.61e-11 strain; subdividing the same two input histories agrees within
2.29e-9 MPa. This closes this particular input-interaction check, not every
possible controller or experimental uncertainty. Archive:
`evidence/creep/2026-11-22-mixed-control-audit.json`.
