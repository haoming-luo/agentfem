# P92 shared-stress two-mechanism research

## Scope and sources

The new local integrator couples two independent inelastic histories through
one stress, rather than summing two separately solved stresses:

`sigma_dot = E*(epsilon_dot - flow_A - flow_B)`.

For prescribed constant stress, `epsilon_dot = flow_A + flow_B`. Each branch
retains its own two backstresses, isotropic state and accumulated strain.
No state is reset when changing control. Both branches share one elastic
modulus and their rate equations are integrated simultaneously.

This follows the composition principle in [MOOSE's multiple-inelastic
update](https://mooseframework.inl.gov/source/materials/ComputeMultipleInelasticStress.html),
not its finite-strain implementation. [NEML's Chaboche documentation](https://neml.readthedocs.io/en/dev/hardening/non/chaboche.html)
separates strain/time recovery and warns about its opposite hardening-stress
sign convention. The present equations retain positive backstress conventions.

[Seifert, Schlayer and Maier (2025)](https://opus.hs-offenburg.de/files/11628/SeifertSchlayerMaier_PlastCrpFat_FVWHT_2025.pdf)
motivates separate short/long primary-creep mechanisms and Jiang recovery.
Our branches reuse the previous bounded-recovery hypothesis, including its
isotropic softening and nonlinear static recovery. Consequently this is **not
an exact reproduction** of their two-strain Chaboche-Jiang law or parameters.
Neither the paper's 600 C BED/519n cohort nor its claimed agreement establishes
accuracy for these BAM 620 C specimens. No tertiary damage or failure law is
introduced. The implementation stays in research tools, outside the material
catalogue and global Step provider.

## Design and verification

`p92_two_mechanism.py` owns only a local material-point rate integration.
`Mechanism` freezes branch parameters and enables an exact disabled-branch
limit. State has ten entries: stress, four variables per branch, total strain.
Radau and BDF are distinct time integrators using numerical rate Jacobians;
this does not provide a consistent global FEM algorithmic tangent.

Tests cover the single-branch limit, pure elasticity, analytical positive and
negative constant-stress creep, shared stress rate, branch permutation,
subdivided restart, control switches, invalid properties, failed-solve
nonpublication, and the actual SLC envelope failure. Old arrays remain intact.
This verifies the numerical implementation, not its experimental parameters.

## Protocol experiments and decision

Five frozen hypotheses are evaluated on RF, SLR and SLC, first with Radau
at relative tolerance 1e-8 and then BDF at 1e-10 (30 runs total). Each specimen
starts with its own measured initial stress and an explicitly assumed virgin
hidden state. Nominal paths are not reconstructed measured histories.

The hypotheses are: single-branch reference; additional branch with threefold
viscosity scale; zero static recovery in the fast branch; and two follow-ups
with fast flow exponents 6/12. The latter retain the flow at a numerical pivot
overstress of 120 MPa. None is jointly calibrated or an identified P92 card.
Increasing viscosity alone does not prove physically separated time scales.

| Candidate | RF extrema RMS MPa, cycles 4–10 | SLR RMS MPa, cycles 1–10 | SLC first exit cycle |
|---|---:|---:|---:|
| Single reference | 2.45 | 12.97 | 2 |
| Added slower flow | 6.12 | 19.06 | 2 |
| Retained fast backstress | 27.50 | 34.40 | 2 |
| Retained fast, n=6 | 20.75 | 29.96 | 3 |
| Retained fast, n=12 | 16.15 | 28.04 | 3 |

SLC exits when creep strain exceeds the next positive target 0.002. Integration
stops at that hold: partial cycles do not become complete-cycle extrema, and
no reverse ramp is invented to hide the failure. The failure remains in every
report even if preceding-cycle extrema appear accurate. Exits are identical
under both integrators. Maximum completed-cycle endpoint stress discrepancy
is 6.08e-7 MPa; failed-hold strain discrepancy is below 2.55e-10. This numerical
agreement does not rescue the much larger experimental discrepancies.

All hypotheses fail cross-protocol admission. The negative result rejects
simple superposition and these parameter ablations, **not every possible
two-mechanism model**. Previously inspected records remain development data;
no independent validation or life qualification is claimed.

Reproduction:

```sh
python -m tools.p92_two_mechanism_screen evidence/creep/2026-11-16-bam-positive-strength.json /path/to/BAM --output dual-radau.json
python -m tools.p92_two_mechanism_screen evidence/creep/2026-11-16-bam-positive-strength.json /path/to/BAM --method BDF --rtol 1e-10 --output dual-bdf.json
```

Compact evidence: `evidence/creep/2026-11-17-two-mechanism.json`. Full reports
include all branch parameters, initial assumptions, source/calibration hashes,
completed-cycle states and the failed partial-cycle state. Raw specimen header
personal information is not copied.

## Next admission gate

The remaining task is joint identification, not more arbitrary manual sweeps:
use measured rapid-ramp and long-hold histories, residual scales declared before
fitting, and sensitivity-rank inspection to determine which parameters the
available experiments can identify. SLC must supply strain residuals under
stress control, not be disguised as a strain-controlled stress fit. Freeze
parameters before later-cycle prediction. If the current branch equations
cannot fit these observables together, replace the mechanism law rather than
loosening criteria. Independent specimen validation and a compatible life
endpoint remain required before engineering admission.
