# P92 creep/relaxation mechanism discriminator

## Decision

Do not admit the current candidate as an engineering P92 material or life law.
A Jiang-type recovery correction improves early cyclic extrema but does not
resolve the mixed-control failure. The original calibration and acceptance
requirements are unchanged. Research code remains outside the installed API.

## Source and equation mapping

[Seifert, Schlayer and Maier (2025), pp37–41](https://opus.hs-offenburg.de/files/11628/SeifertSchlayerMaier_PlastCrpFat_FVWHT_2025.pdf)
uses two inelastic mechanisms, each with its own relative stress and recovery,
coupled by the same elastic stress. Its Jiang weight reduces dynamic recovery
below backstress saturation. The reported calibration distinguishes short- and
long-time primary creep, not simply high- and low-stress branches. It covers
primary/secondary creep, not tertiary damage. Its BED/519n, 600 C data are not
the BAM 620 C cohort and no complete transferable parameter card is supplied.

The present discriminator implements **only** a Jiang-type weight on the
previous five-state bounded-recovery candidate. It is not a reproduction of
that two-mechanism paper. With positive-backstress convention:

`xdot_i = C_i*a_i*flow - C_i*phi*W_i*x_i*pdot - static_recovery_i`

`W_i = (abs(x_i)*phi/a_i)**chi`, `phi = 1+(2-1)*(1-exp(-0.1*p))`.

The instantaneous saturation used in this hypothesis is `a_i/phi`.
No clipping conceals a saturation excursion. `chi=0` recovers the original
candidate exactly. Exponents 0, 1 and 3 are fixed sensitivity choices, not
estimated P92 constants. Stress/strain control share the same rate equations;
stress hold evolves total strain, without resetting the hidden state.

## Frozen comparison

Run `python -m tools.p92_ratcheting_screen` with the existing
`2026-11-16-bam-positive-strength.json`, checksum-pinned BAM directory and a
new `--output` path. All nine experiments retain the frozen base calibration.
RF metrics below cover cycles 4–10; SLR covers cycles 1–10. Both are previously
inspected development records, not independent validation specimens.

| Exponent | RF extrema RMS, MPa | SLR extrema RMS, MPa | SLC cycle-2 hold strain |
|---:|---:|---:|---:|
| 0 | 9.26834 | 23.67770 | 0.01900105 |
| 1 | 4.27244 | 17.34114 | 0.01410402 |
| 3 | 2.45353 | 12.96934 | 0.01161506 |

All SLC experiments exceed the nominal positive target 0.002 in cycle 2.
The failed cycle is completed only for diagnosis and no further cycles run.
Its extrema are not an admissible protocol prediction. Improving two columns
does not erase the third failure or qualify life prediction.

Radau refinement for exponent 3, relative tolerance 1e-8 to 1e-10, changes
maximum cycle-end stress by 5.664e-9 MPa (RF), 2.379e-9 MPa (SLR), and
2.175e-9 MPa (SLC). Maximum accumulated-strain differences are below 2.206e-12.
Both tolerances retain the same SLC failure. These are endpoint integration
checks, not a full-path error bound or an experimental acceptance tolerance.
Tests additionally cover identity, recovery direction, analytical constant-flow
tension/compression, restart, invalid parameters and failure nonpublication.

## Next bounded model experiment

1. Represent two flow mechanisms with independent backstress/time scales and
   one shared stress: `sigma_dot=E*(strain_rate-flow_1-flow_2)`; under stress
   control `strain_rate=flow_1+flow_2`. Do not add independently computed stress
   responses or introduce damage to conceal relaxation errors.
2. Verify zero-mechanism and identical-split limits, derivatives, control
   switches and restart before fitting. Preserve the existing material/State
   boundary; research integration is not a second public FEM solver.
3. Identify early rapid ramps and creep/relaxation holds jointly. Declare
   separate stress/strain residual scales and inspect sensitivity rank; a
   large parameter set with an apparently low cost is not identification.
4. Freeze before predicting later RF/SLR/SLC cycles. Since these records have
   already guided development, reserve independent specimens for validation.
   Keep creep strain, relaxation histories and peak/valley evolution together.
5. Only after response passes, assess a failure model against compatible
   specimen endpoints and censoring. Do not equate a completed cycle history
   with predicted rupture life or claim a universal standard error band.
