# P92 long-cycle response: retain discrepancies, do not refit

## Scope and reproducibility

Reuse the already executed Radau/BDF campaigns of the fixed published equations.
Read five experimental peak-stress square markers in each Figure 10 panel of
[Metals 10 (2020) 307](https://doi.org/10.3390/met10030307), printed page 11.
The pinned PDF hash, exact Poppler crop command, axis ticks and marker pixels
are retained in `evidence/creep/p92-cyclic-figure-points.json`. A five-pixel
coordinate sensitivity covers manual reading of the dense marker band; it is
not a confidence interval. Only peak stress is sampled, not blue relaxation
markers or red predicted curves. No experimental raw-data precision is claimed.

```bash
python tools/p92_published_recovery_reference.py --published-campaign --output campaign.json
python tools/p92_cyclic_figure_screen.py campaign.json
```

Reuse an existing campaign rather than launching the first command again.
The plot's horizontal coordinate is cycle divided by observed half-life.
The observed lives 1303 and 675 are normalization inputs, not model outputs.
Fractional cycle coordinates use linear interpolation of the computed peaks;
reading intervals must lie entirely within the run, with no endpoint clamping.
Seven inexpensive tests verify axes, scope and inconsistent input rejection.

## Results with unchanged coefficients

| Tensile dwell at 600 C, range .008 | Peak sample mean absolute relative error | Maximum absolute error |
|---|---:|---:|
| 60 s | 4.03% | 15.32 MPa |
| 600 s | 1.70% | 9.19 MPa |

Several discrepancy intervals exclude zero. Radau and BDF predictions differ
by at most 5.26e-6 MPa at these sample positions, much smaller than experimental
figure reading sensitivity or response discrepancy. Tightening numerical
tolerances is not a demonstrated remedy. Do not tune to these points and then
reuse them as validation. Sparse long-cycle agreement complements the existing
first-cycle dwell screen; it does not establish a full cyclic experimental
history, multiaxial response, fracture or engineering allowable life.

## Life criterion and next data gate

Figure 2's 10% peak-stress loss is relative to the extrapolated intermediate
softening trend shown in the schematic, **not 10% below the first-cycle peak**.
The rapid initial cyclic softening must not be mistaken for a failure clock.
The published reference does not contain a validated crack-initiation law.

NIMS data sheet 48B has authoritative T92/P92 creep-rupture data
([catalogue](https://doi.org/10.11503/nims.1052)), but the old public PDF link now
redirects to a service migration notice. The current
[CDS access page](https://cds.nims.go.jp/) requires registration/application and
sets reuse conditions. No authenticated data were acquired or copied. Obtain
authorized data with heat identity, temperature, stress and rupture definition;
do not substitute another P92 heat or optimized Grade 92 silently.

Remaining engineering gate: mutually compatible fatigue, creep and interaction
evidence, independent specimens/loads and a justified uncertainty allowance.
The software composition and research screens can advance without pretending
that missing experimental qualification is a coding defect.
