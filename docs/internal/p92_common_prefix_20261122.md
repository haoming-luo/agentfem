# Shared loading prefix before SLR/SLC divergence

Question: does the observed cross-protocol difference start only after the
1800 s strain-controlled versus stress-controlled dwell? No. The first
compression ramp and 180 s negative strain hold precede that divergence.
The [BAM source article](https://doi.org/10.1016/j.dib.2023.109333) describes
separate specimens and supplies stress, strain and temperature histories.

The checksum-checked first holds are aligned by reconstructed elapsed hold
time. Piecewise-linear interpolation uses only common support, 0.035 to
179.403 s; there is no extrapolation or sampling-density weighting. The
time mean and RMS integrate the linear difference exactly. Source switch
brackets remain in the record; they are not authenticated controller logs.

| Recorded quantity, SLC minus SLR | Result |
| --- | ---: |
| First stress difference | -9.749 MPa |
| Last stress difference | -10.180 MPa |
| Time-mean stress difference | -11.422 MPa |
| Maximum absolute stress difference | 15.160 MPa |
| Total relaxation, SLR / SLC | 113.711 / 113.280 MPa |
| RMS strain difference | 3.212e-6 |
| Maximum strain difference | 1.999e-5 |
| RMS / maximum temperature difference | 0.155 / 0.800 K |

The frozen original card is then replayed from its declared initial state
through the common compression, hold and reload, both with linear ramps and
with recorded ramp deviations. Neither path fits or resets internal state.
The predicted time-mean stress difference is +0.296 / +0.297 MPa respectively,
not the measured negative difference. Independent BDF integration agrees
within 1.60e-7 MPa at the hold observations. This test does not replay measured
hold strain ripple or identify temperature coefficients; it therefore does
not rule out every control or temperature contribution.

The companion [experimental paper](https://doi.org/10.1016/j.ijfatigue.2022.107381),
section 2.2, calls the reported extensometer quantity mechanical strain.
Do not subtract an assumed thermal expansion from that channel without
establishing its acquisition/compensation convention. Recorded temperature
differences alone do not identify temperature-dependent material coefficients.

Interpretation: attributing *all* cross-specimen discrepancy to the later
long-hold memory is unsupported. Nevertheless this two-specimen observation
is not a distribution of material scatter, does not establish a cause, and
does not justify a specimen-specific parameter, initial-state reset, or
subtracting 11.4 MPa from later validation residuals. In particular, the
existing stress tolerance bands can overlap: this is not proof that the
unchanged acceptance gates are mathematically impossible. Keep absolute
stress, anchored hold shape, creep accumulation and later-cycle gates.

Evidence: `evidence/creep/2026-11-22-common-prefix.json`; reproducible driver
`tools/p92_common_prefix.py`. Numerical agreement is not material admission.
