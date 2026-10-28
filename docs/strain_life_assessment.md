# Calibrated scalar strain-life assessment

`constitutive.CoffinMansonBasquinCurve` describes the combined elastic and
plastic strain-amplitude relation

```text
eps_a = (sigma_f / E) (2 N)^b + eps_f (2 N)^c
```

`N` is full cycles to failure; `2 N` is reversals. The strain contributions
are added, not the lives. Strain is dimensionless, not percent. The stress
coefficient and Young's modulus must use the same declared unit. The
[COMSOL theory description](https://doc.comsol.com/6.4/doc/com.comsol.help.fatigue/fatigue_ug_sme.4.21.html)
documents this conventional relation and the importance of the selected
strain measure and mean stress. AgentFEM's initial scope is narrower: one
signed scalar total mechanical strain component, not a multiaxial criterion.

```python
from agentfem import constitutive

# Synthetic values for demonstrating the interface, NOT design data.
curve = constitutive.CoffinMansonBasquinCurve(
    young=200_000.0,
    fatigue_strength_coefficient=1_000.0,
    fatigue_strength_exponent=-0.1,
    fatigue_ductility_coefficient=0.5,
    fatigue_ductility_exponent=-0.6,
    calibrated_cycles=(1.0, 1.0e8),
    stress_unit="MPa",
    source="synthetic example; not material calibration",
)
assessment = constitutive.assess_strain_history(
    [-0.01, 0.01, -0.01],
    curve,
    strain_measure="mechanical_total",
    source="signed axial mechanical strain at the selected material point",
    mean_stress_correction="none",
)
print(assessment.as_dict())
```

## Input and failure contract

- Supply a signed mechanical strain history with enough accepted samples to
  resolve its reversals. Sparse visualization frames are not automatically a
  sufficient fatigue history. Thermal strain must already be excluded.
- Do not use accumulated PEEQ/CEEQ or a positive strain norm as a substitute
  for a reversing signed component. The caller declares the measure; the
  software cannot recover a missing tensor or infer physical semantics from
  unlabeled numbers.
- Calibration-cycle bounds and provenance are mandatory. A counted amplitude
  outside those bounds raises an error; there is no hidden clipping,
  extrapolation, or endurance-limit inference.
- The caller must explicitly select `mean_stress_correction="none"`. Means
  remain visible in the counted cycles. This is not evidence that mean-stress
  effects are negligible. Morrow/SWT require their own paired stress/strain
  contract and are not implemented here.
- The existing scalar rainflow algorithm is reused. Remaining excursions count
  as half cycles. Miner damage is a postprocessing assumption, not constitutive
  damage. A constant sampled history returns no counted damage; strict JSON
  reports `repeated_history_life: null`, not a claim of infinite service life.

## Composition and evidence

Pass this assessment explicitly as `fatigue=` to `assessments.creep_fatigue`,
together with source-identified creep time fractions and the intended
interaction diagram. The existing `creep_fatigue_from_result` convenience
function remains stress-life based; it does not silently reinterpret its
history as strain. The reference linear interaction is transparent, not a
design-code acceptance curve. A combined assessment does not feed damage back
into equilibrium, and does not prove absence of double-counting when the
supplied strain-life calibration already includes dwell effects.

The current tests cover equal-exponent closed-form inversion, round trips,
stress-unit invariance, full/half-cycle conventions, calibration rejection,
source retention and composition. They do not validate a material's fitted
parameters, nonisothermal life, multiaxial damage or component service life.
