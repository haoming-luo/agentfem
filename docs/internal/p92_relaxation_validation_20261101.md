# P92: first executed public-data relaxation comparison

## Why this source

The [BAM dataset, Zenodo 7198219 v1](https://doi.org/10.5281/zenodo.7198219)
by Sonntag, Jürgens, Uhlemann, Skrotzki and Olbricht (2022), CC BY 4.0,
supplies digital time/stress/strain/temperature records, not just life tables.
The [associated Data in Brief article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10319173/)
explains nominal stress, percent strain, Celsius and sparse cycle recording.
This is P92 at nominal 620 °C; it is **not Alloy 617** and does not replace
the separate Alloy 617 source audit. No private applicant or customer data is used.

## Fixed, deliberately small experiment

`tools/validate_p92_relaxation.py` reads checksum-pinned original files. It
converts percent to strain and Celsius to kelvin without resampling or filling
missing cycles. Select cycle 1, raw time 8–180 s, avoiding ramp/servo settling
and reversal; retain the actual first/last samples. RF tensile holding fits
the effective law

```text
dσ/dt = -r200 sign(σ) (|σ|/200 MPa)^n
```

Trapezoidal **time** weights prevent dense early sampling from dominating the
fit. Three starting points converge to the same least-squares cost. Freeze
the parameters before predicting the separate SLR compressive hold. The
measured initial stress is an explicit initial condition in each hold; it is
not a predicted peak. No held-out curve values enter optimization.

This identifies `E*A` and `n`, not E and A separately. Full-cycle plasticity,
cyclic softening, damage, thermal sensitivity, initial loading and lifetime
remain outside the fit. Measured strain/temperature drift is reported but
neglected by this ideal constant-strain/isothermal baseline. Experimental
nominal stress is approximated by small-strain axial stress, explicitly.

## Executed evidence

The frozen fit gives `r200 = 1.590673 MPa/s`, `n = 9.687639`, with no active
optimization bound. Time-weighted stress RMSE is **0.502629 MPa** for RF and
**6.496178 MPa** for the held-out SLR hold. Maximum SLR error is **8.328164 MPa**;
its endpoint prediction is **−149.954699 MPa**, against **−142.85 MPa** measured.
These are descriptive errors, **not a passed experimental acceptance criterion**.
No uncertainty model or independently justified material tolerance has yet
been established. Do not retune against this holdout and still call it held out.

The ordinary `creep_solid → model.step` workflow realizes the same conditional
law on a free-lateral unit cube. E=100000 MPa is a numerical realization,
**not a measured P92 modulus**; set A=r200/E and initial strain=σ0/E. An
offline test doubles E while preserving E*A and verifies the same stress.
No new solver, P92-specific core material, or project Newton loop is added.

| Increments | Max FEM–continuous time error (MPa) | Endpoint error (MPa) |
| --- | ---: | ---: |
| 32 | 2.160319 | −0.805955 |
| 64 | 1.162444 | −0.411316 |
| 128 | 0.606418 | −0.208049 |

This is temporal verification of homogeneous equilibrium, not spatial
convergence or reproduction of a physical specimen. Raw point stress agrees
with the strong-boundary reaction; restart retains the accepted point history.
Eight offline tests cover parsing, units, sparse cycles, source identity,
time weighting, signed analytical response, synthetic parameter recovery and
actual FEM pause/restart. They do not require network or the experimental files.
The combined point-history, experiment-protocol and CI/package checks passed
**110 tests in 27.48 s**. The executable numerical gate passed with error
ratios 1.8584 and 1.9169. Full machine-readable evidence is retained at
`evidence/creep/2026-11-01-p92-conditional-relaxation.json` in the repository.

## Reproduce

Obtain `Data_RF.lis` and `Data_SLR.lis` from the versioned Zenodo record. Keep
them outside the repository; no download or package install occurs in the tool.
In the normal scientific environment, from a source checkout using the package
under test:

```bash
python tools/validate_p92_relaxation.py --rf /path/to/Data_RF.lis \
  --slr /path/to/Data_SLR.lis --fem --output /path/to/new-report.json
```

Existing reports are never overwritten. The report records source SHA-256,
tool SHA-256, versions, actual sample endpoints, fit/holdout errors and separate
numerical evidence. A failed numerical gate retains its report and exits
nonzero; a descriptive experimental comparison never promotes material trust.

## Next scientific decision

Use this baseline to locate what is missing, not to set a product promise.
Before full-cycle validation, acquire/estimate E from an independently reviewed
elastic segment, retain the full controlled loading history, choose and verify
a law that admits cyclic softening, and freeze a **new** condition/specimen
holdout. Then integrate through the existing material/Step/state contract.
The longer zero-strain SLR hold and stress-controlled SLC route are not silently
treated as the same strain-held experiment. A generic user-material temperature
contract remains a separate implementation task.
