# P92 prior-damage remaining-life audit

## Result and provenance

Two public author manuscripts provide a useful sequential-loading discriminator:

- [Microstructural damage mechanics-based model for creep fracture of 9%Cr steel under prior fatigue loading](https://strathprints.strath.ac.uk/68319/1/Zhang_etal_TAFM_2019_Microstructural_damage_mechanics_based_model_for_creep_fracture.pdf), Table 1, PDF p5.
- [Evaluation of the effect of various prior creep-fatigue interaction damages on subsequent tensile and creep properties of 9%Cr steel](https://strathprints.strath.ac.uk/67692/1/Zhang_etal_IJF_2019_Evaluation_of_the_effect_of_various_prior_creep_fatigue_interaction_damages.pdf), Table 1, PDF p22.

Both tables were rendered and visually checked. `evidence/creep/p92-prior-damage.json`
retains PDF hashes, 15 model-table observations and published predictions, plus
eight links to the experimental table. PDFs and figures are not redistributed.
The two manuscripts reuse experiments; they are not independent datasets.
Retain the differing 1143/1144 h and 186/185 h entries explicitly. Do not average
them or treat a difference as a new replicate.

The experiment identifies a normalized/tempered 105 mm OD, 24 mm wall P92 pipe,
650 C in air, prior strain amplitude 0.4%, strain rate 0.001/s and subsequent
creep at 130 MPa. Its prior-life fractions use the shortest of at least three
reference fatigue tests per hold condition, not a mean fatigue curve. The model
table additionally includes prior LCF at 0.25% amplitude. No common heat with the
existing thermally aged rupture data or 600/650 C dwell-life table is established.

## Reproducible arithmetic, not an executed constitutive law

Run from the source root:

```bash
python tools/p92_prior_damage_audit.py
```

The author used as-received creep at 125/130/140 MPa and prior 0.4% LCF to
identify parameters (sections 5.1-5.4); the table's corresponding rows are
1, 6-9. Rows 2-5 and 10-15 are the author's transfer group. This audit does not
re-fit anything and does not call these an AgentFEM blind validation split.

| Published prediction group | Mean absolute relative error | Largest relative error |
| --- | --- | --- |
| Calibration-associated table rows | 3.123% | 7.952% |
| Transfer rows, all retained | 51.956% | 326.357% |

At 50% and 70% prior C-F life fraction with 180 s holds, published predictions
550/479 h are compared with observed 129/186 h: ratios 4.264/2.575. Section 5.4
attributes the additional degradation to surface cracks beyond microstructural
recovery. These are useful failure cases, not outliers to discard. Neither these
ratios nor a factor-of-two plot is an engineering acceptance criterion.

## Next bounded implementation

1. Reproduce the published material-point equations with fixed parameters before
   any new calibration. Check equation units, initial conditions and the rupture
   event explicitly; an integrator failure or cutback is not physical rupture.
2. Preserve calibration-associated and transfer groups. Include all high-damage
   failures. Measured lath width and a fraction of known fatigue life are inputs
   to this conditional prediction, not outputs from an arbitrary service history.
3. Investigate whether a crack-sensitive damage description is needed for the
   failed regime. Do not invent a correction from those two points and then
   reuse them as independent validation. First locate additional compatible
   crack/history observations or restrict the model's admitted domain.
4. Only after numerical reproduction, connect the local law through the existing
   material/state/Step interface and verify homogeneous structural response and
   restart. Do not create a P92-specific global solver or claim a component life
   allowable from this single-stress dataset.

The earlier 2017 ResearchGate landing page currently exposes only an abstract;
search snippets suggesting full text were not treated as numeric evidence.
The public institutional manuscripts above are the actual reviewed sources.
