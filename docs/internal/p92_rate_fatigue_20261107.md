# P92 service-aged fatigue: fixed-coefficient rate screen

Source: [Imperial thesis, DOI 10.25560/60640](https://doi.org/10.25560/60640).
Tables 4-3/4-4 and the failure definition were visually checked at PDF pages
94/101/80. The source hash, seven valid observations and two invalid tests are
retained in `evidence/creep/p92-service-aged-fatigue.json`.
This is Mat02 service-aged parent steel at 600 C, not the 650 C prior-damage
cohort. The source failure criterion is a trend-relative stress drop, not final
rupture. Invalid thermocouple tests are excluded, not assigned zero life.

Run `python tools/p92_rate_fatigue_screen.py`. The existing public strain-life
API evaluates the two printed coefficient sets; no new law or fit is added.
The tabulated sigma_f/E ratio is dimensionless despite its printed stress-unit
label. E=1 normalization preserves this ratio and does not supply elastic
properties for a structural material. The alternative pooled-fit coefficient
0.42 is not substituted for 0.57. Each curve retains its own observed cycle
bounds; out-of-domain inversions are reported, not silently admitted.

## Computed diagnostics

At amplitude 0.006, the high-rate curve predicts 928.368 cycles: its own-rate
observation is 965, the intermediate-rate observation is 851 and the low-rate
observation is 771. Corresponding errors are -3.80%, +9.09% and +20.41%.
At amplitude 0.004 the same curve predicts 2173.962 cycles versus 1640 at the
low rate (+32.56%). The low-rate curve predicts 775.285 cycles at amplitude
0.006 versus its own-rate 771 (+0.56%).

These are fixed-coefficient, within-source comparisons, not a blind holdout or
an engineering acceptance threshold. Rate affects transfer error even without
an explicit dwell. Do not use the curve labelled for one rate as a universal
fatigue relation, or add a creep fraction to compensate without reviewing
double-counting and failure definitions. The bounded screen adds new source
evidence but does not close component-life qualification.
