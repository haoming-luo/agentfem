# Signed flow budget before another kinetic replacement

The matched recovery-shape experiment did not outperform its same-start
original control. Do not launch another fit on the assumption that all
inelastic deformation contributes to relaxation in the same direction.
During a strain hold, recovery of previously accumulated reversible strain
can oppose irreversible flow. The model's accumulated branch variables are
nonnegative path lengths, not signed strain increments.

For each first positive hold in the frozen original control, retain its exact
incoming state and integrate with unchanged parameters. Independently
quadrature-integrate the **signed** A/B flows from the dense solution; take
the two reversible-state changes directly. Check the identity

`delta(epsilon) = delta(sigma)/E + integral(flow_A + flow_B) + delta(a_fast + a_slow)`.

Use exactly the existing early and late observed secant windows, and verify
the endpoint against the original frozen continuation. Require strain closure
below 1e-9, quadrature estimate below 1e-10, and independent Radau/BDF secant
agreement below 1e-5 MPa/s. These are diagnostic numerical tolerances, not
changes to experimental gates. Verify positive/negative flow, exact two-time
creep/recovery and out-of-support rejection first.

The contribution obtained by algebraically omitting reversible flow is only
frozen-path accounting. A modified material would change the entire path;
the omitted-component value is not its response or a bound on all material
models. This diagnostic introduces no parameters, fitting, state reset,
protocol-dependent material terms or new experimental acceptance criterion.

## Frozen original-control outcome

All six window decompositions pass quadrature/strain closure and independent
integration; maximum absolute closure is below 5.5e-14 strain. In the first
SLC positive hold:

| Window | Observed stress rate | Predicted | Signed A+B flow contribution | Reversible contribution |
| --- | ---: | ---: | ---: | ---: |
| Early | -3.7937 | -5.1834 | -4.6409 | -0.5425 |
| Late | -0.18193 | -0.08231 | -0.10565 | +0.02334 |

Units are MPa/s. Reversible back-flow opposes late relaxation, but even simply
omitting that contribution on this **fixed trajectory** does not reach the
observed late rate. Early irreversible flow alone is already faster than
observed. This rules out attributing the whole frozen-path discrepancy to
reversible recovery; it is not a global impossibility proof or a proposed
removal of recoverability. A new mechanism must alter the time distribution
and inherited irreversible state as well, while retaining creep accumulation.

The recorded late-window strain endpoints are equal; the early endpoints
decrease by 3e-5 strain. At the frozen modulus their elastic secant contribution
would be -0.501 MPa/s, not the positive correction needed to fix overly fast
early relaxation. This arithmetic does not turn instrument variation into a
controller command or establish experimental uncertainty. The original
hold assumptions and gates remain unchanged.

Evidence: `evidence/creep/2026-11-22-signed-hold-flow-budget.json`.

The diagnostic replays the first cycle causally from the public observations
and frozen material card. It checks the cycle end against the compact frozen
archive before using the inherited hold state; no hidden full-trajectory file
is required. The recorded `frozen_sha256` identifies that compact input file,
not the earlier, larger run from which it was archived.

```sh
python -m tools.p92_hold_flow_budget \
  evidence/creep/2026-11-22-recovery-shape-control-frozen.json \
  /path/to/p92-public-source-cache-7198219 \
  /new/path/signed-hold-flow-budget.json
```

The output path must not already exist. Source hashes, frozen parameters and
driver hashes are retained alongside the numerical checks and failed physical
interpretation; the diagnostic does not promote the material to validated.
