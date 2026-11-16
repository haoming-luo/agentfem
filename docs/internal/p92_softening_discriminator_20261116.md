# P92: test the softening mechanism before changing the material

The continuous RF response plateau is not a life prediction. The five-state
2020 model has no failure variable. Its isotropic equation, with static recovery
disabled, integrates exactly to

`R = (Q + H/b) (1 - exp(-b*p))`.

Thus the `H*pdot` term in that rate equation shifts saturation; it does not
produce the persistent linear contribution in `R = Q*(1-exp(-b*p)) + H*p`.
These are different equations, not interchangeable implementations.

References: [2020 response model](https://doi.org/10.3390/met10030307),
[2013 cyclic-softening work](https://doi.org/10.1016/j.ijpvp.2013.08.001).
The original reproduction is unchanged.

## Frozen-parameter discriminator

`tools/p92_softening_discriminator.py` tests a hypothesis with separate
saturating `S` and linear `H*p` contributions. Recovery acts on `S`; flow sees
their sum. This extension is not claimed to reproduce either publication.
It uses the frozen early-RF calibration, without refitting, on the continuous
nominal RF path. Each cycle commits only after all five segments succeed.

```sh
python -m tools.p92_softening_discriminator \
  evidence/creep/2026-11-16-bam-positive-strength.json \
  --cycles 1000 --output split-softening.json
```

Radau, relative tolerance `1e-8`, absolute tolerances `1e-8 MPa` for the four
stress-like states and `1e-12` for accumulated inelastic strain:

| Completed cycle | Sampled tensile peak (MPa) |
|---:|---:|
| 1 | 263.4233 |
| 10 | 269.0020 |
| 100 | 223.1768 |
| 200 | 211.9361 |
| 400 | 200.8705 |
| 474 | 198.0300 |

The next cycle fails in zero-based segment 2 when the yield strength reaches
zero. The last committed strength is `0.0292710 MPa`. The command returns
nonzero and retains cycle 474, not a partially advanced cycle 475.

**Decision: reject direct substitution with old parameters.** Neither clipping
strength nor treating this mathematical failure as experimental rupture is
acceptable. Any subsequent candidate needs bounded softening, new calibration
and independent response tests. The cycle-1000 baseline error remains open.

Eight targeted tests cover both exact scalar solutions, analytical Jacobians
at active/inactive states, time partition/restart agreement, and failed-cycle
transaction behavior. These verify the experiment implementation, not the
material's engineering applicability.
