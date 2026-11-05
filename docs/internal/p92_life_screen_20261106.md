# P92 life-data screen: fixed calibration and bounded holdout

## Sources and scope

[Zhang et al., Metals 10 (2020) 307](https://doi.org/10.3390/met10030307),
Table 2, reports eight life observations at two temperatures, two total axial
strain ranges and three tensile dwell durations. Section 2 defines failure by
the authors' 10% peak-stress-drop criterion. These are not silently relabeled
rupture times, crack-size lives, or allowable component lives. Source values,
heat treatment, loading rate and split are retained in
`evidence/creep/p92-life-table.json`.

The separate [BAM/Zenodo data](https://zenodo.org/records/7198219) contain
service-like cyclic deformation histories at 620 C. They are a different
experiment and are not pooled with this table as though heat/batch, loading
and failure definitions were identical.

[ASTM E2714's public scope](https://store.astm.org/e2714-26.html) distinguishes
specimen creep-fatigue deformation/crack formation from full-component testing,
and recommends complementary continuous-cycling and creep data for the same
material and temperatures. This screen does not implement the paid standard
or claim code compliance.

## One deliberately simple baseline

Before numerical evaluation, use IDs 1,3,4,5,7,8 for calibration and 2,6 for
within-source validation. Fit only the training rows in log-life coordinates:

```text
ln(N) = a + b*ln(total_strain_range/.008)
          + c*ln(1+tensile_dwell_seconds/60) + d*(T_C-600)/50
b, c, d <= 0
```

This is an explicitly chosen empirical screening ansatz, not a published
constitutive law or a calibrated damage mechanism. The four parameters have
six fitting observations. Held-out cases share the low strain range and an
intermediate dwell: they test interpolation within this source, not a new
material batch or an unseen strain regime. No alternative model shopping,
resplitting or refitting to the holdout is used.

Reproduce with `python tools/p92_life_screen.py`. The script prints complete
JSON with source, split, coefficients, every prediction, ratios and limitations.
14 tests verify source transcription, fixed baseline, no holdout-label leakage,
monotonicity, rank admission and rejection outside the declared input domain.
The regression's numerical expectations do not constitute scientific acceptance
thresholds selected before the experiment.

## Result

The two held-out observed lives are 1190 and 853 cycles; predicted lives are
961.892 and 788.484 cycles, respectively. Signed relative errors are -19.17%
and -7.56%; multiplicative error factors are 1.237 and 1.082. Calibration
predictions and their residuals remain in the report rather than being combined
with validation accuracy. This establishes a reproducible, modest within-source
prediction baseline, not an uncertainty bound or engineering qualification.

This is already a **combined creep-fatigue** life relation. Feeding it as a
pure-fatigue curve into a time-fraction-plus-fatigue damage sum would count
the dwell effect twice. It is intentionally not wired into the production
assessment API as a P92 material card.

## Stop conditions and next admission gates

- Keep the baseline fixed; further fit iterations on these eight rows cannot
  create independent validation or repeat-specimen scatter.
- Admit only 600 or 650 C, total strain range .008-.012, tensile dwell 60-600 s,
  and the recorded loading rate/environment. No temperature interpolation,
  zero-dwell prediction, compressive dwell, weld or multiaxial claim.
- Obtain compatible pure-fatigue and creep-rupture/ductility data if separating
  damage mechanisms. Record heat treatment, specimen, failure criterion and
  units before combining sources; do not borrow missing parameters by name.
- Test additional independently selected lives, then state prediction scatter
  and domain. Engineering allowable life needs a justified conservative margin,
  not merely a factor-of-two scatter plot.
- Co-located structural histories are a separate data prerequisite. Total
  strain cannot automatically replace the mechanical strain required by a
  strain-life assessment. A material response model alone has no fracture clock.

Status: research baseline calibrated and held-out evaluated; engineering life
qualification remains open with explicit data requirements.

## Cross-publication audit

[Wang et al., Metals 9 (2019) 183](https://doi.org/10.3390/met9020183),
Table 1 (page 4, visually checked), lists CF600S04T01/T03/T10 with lives
1303/1190/675. At 600 C, amplitude .004 and tensile dwells 1/3/10 minutes,
these exactly match rows 1/2/3 after amplitude-to-range and minute-to-second
conversion. Publication independence is not specimen independence: exclude
these observations from external validation pending specimen provenance.
This includes the current held-out row 2; it remains a within-source holdout,
not external evidence. The JSON report exposes this overlap warning.

The same table also contains zero-dwell repetitions, compressive holds and a
60-minute tensile hold. Those are useful future mechanism/domain tests, but
are outside this baseline's admitted loading domain. Do not extend the domain
just to obtain additional validation rows. No coefficients were refitted.
