# P92 published-figure screen: design and evidence

## Decision before implementation

Keep this scientific screen outside the constitutive catalogue. The existing
published-parameter ODE reference can now be checked against independently
read plot coordinates, not just another integrator using the same equations.
Use the version-2 PDF of [Zhang et al., Metals 10 (2020) 307](https://doi.org/10.3390/met10030307),
already pinned by SHA-256 in the equation reference.

Two different targets must remain separate:

- First-cycle **model dashes** in Figure 11(b): a transcription/implementation
  comparison against the authors' plotted calculation.
- First-cycle **experimental hollow squares** in Figures 11(b), 11(d) and
  14(b): a sparse held-out-condition screen at 600 C / 60 s, 600 C / 600 s and
  650 C / 600 s. Section 5 states that the authors fitted the 0.8% range,
  180 s dwell tests; this screen does not refit those parameters.

Retain original embedded-image size/hash, tick calibration and selected symbol
centres. Use clearly identifiable, uncluttered symbols; exclude legends,
overlapping ambiguous marks and points outside the dwell. Declare a combined
four-pixel coordinate-reading bound. It is a digitization sensitivity bound,
not experimental repeatability or material uncertainty. Keep the original
strain-rate/startup assumption visible. Images/PDFs need not enter the repo.

The implementation must report signed errors, absolute relative errors and
coordinate-reading sensitivity separately. This sparse nonuniform sample is
not the article's one-second-grid MRE and must not be presented as such.
Do not select/reject samples by model agreement or fit parameters after viewing
the errors. Numerical checks test axis calibration, integration agreement and
immutable source/parameter identities; experimental disagreement is retained,
not converted into a green engineering capability claim.

This screen does not certify multiaxial behavior, full cyclic history, fatigue
life, structural discretization or another P92 batch/temperature. It narrows
the next gate: distinguish implementation error from the published model's
remaining discrepancy before creating a general structural material provider.

## Reproduction and visual audit

`evidence/creep/p92-published-figure-points.json` contains the observations and
calibrations. The PDF's first embedded image on printed page 12 is Figure 11
(3036 x 2178); the first on page 13 is Figure 14 (3091 x 2178). These are not
the page-render dimensions. With `pypdf`, inspect `reader.pages[11].images[0]`
and `reader.pages[12].images[0]`, respectively. The record includes the PNG
hashes produced by saving those decoded images through Pillow. The PDF hash,
pixel dimensions and source coordinates are the primary provenance, since
different PNG encoders can produce different byte hashes for identical pixels.

Selected centres were overlaid on the extracted source images and visually
checked: 24 experimental hollow squares, six separate black model dashes,
no legend entries or adjacent-cycle symbols. The four-pixel reading bound
corresponds to about 1.3 MPa, with time sensitivity about 0.2 s for the 60 s
panel and 2 s for the 600 s panels. Model changes over those time intervals
are included; experimental scatter is not invented from that reading bound.

Run `python tools/p92_dwell_figure_screen.py --output <new-json-path>`;
repeat with `--method BDF` and a different new path. Existing evidence is never
overwritten. The complete source-figure record can be reviewed without running
FEM or downloading a dataset during normal software installation.

## Executed first-cycle screen

Published Table 4 parameters remained unchanged. The source-model dash
comparison has maximum absolute difference 1.32 MPa; all six differences are
inside the declared coordinate-reading sensitivity intervals. This provides a
separate plot-transcription check, unlike two integrators sharing one rate law.

| Held-out condition | Selected experimental time coverage | Mean absolute relative error | Maximum absolute error |
| --- | --- | --- | --- |
| 600 C, 60 s dwell | 5.86-56.70 s | 5.80% | 32.61 MPa |
| 600 C, 600 s dwell | 35.46-571.98 s | 4.99% | 20.19 MPa |
| 650 C, 600 s dwell | 25.08-569.82 s | 6.10% | 18.38 MPa |

These are eight identifiable markers per panel, not uniformly sampled complete
curves. They do not reproduce the paper's reported MRE or define an acceptance
tolerance for engineering. Some experimental differences exceed digitization
uncertainty; retaining that discrepancy is part of the result. No fitting,
initial-state tuning or transfer to the different BAM 620 C material occurs.

Radau/BDF predictions agree within 0.001 MPa at all selected samples. The
equation, analytical-discriminator, figure-screen and CI-routing group passes
98 tests; Ruff passes. The numerical/source checks do not assert that the
experimental discrepancy is acceptable. Full cyclic curve comparison and a
verified multiaxial structural provider remain separate gates.
