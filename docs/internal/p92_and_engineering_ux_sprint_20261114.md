# P92 validation and engineering workflow sprint

## Decision and scope

Prioritize a traceable experimental response validation before more life fits.
Industrial fitness is application-specific, not a label earned by passing unit
tests. No new release or general P92 service-life qualification is implied.
Keep calibration, held-out prediction, numerical verification and design
allowables separate in all exported reports.

## Execution order and acceptance

1. Admit the BAM service-like P92 data, preserving checksum, specimen identity,
   units, recorded-cycle gaps, control modes and the stated failure definition.
   A truncated file must not become accepted experimental evidence.
2. Reconcile extrema against recorded histories. Distinguish observed failure
   cycles from the total test cycles; do not count cycles as independent specimens.
3. Freeze a bounded calibration/validation question before fitting: RF response
   calibration versus held-out histories/protocols only when material pedigree
   and loading support the transfer. Report peak stress, relaxation, hysteresis
   work and errors by stage. Never interpolate across unrecorded cycles to
   fabricate a continuous integration history.
4. Reuse existing material-point and structural workflows for time refinement,
   accepted-state/restart agreement and dimensional consistency. A successful
   response comparison is not independently validated life prediction.
5. Evaluate life only inside the rupture/fatigue data domain with an explicit
   interaction rule and failure criterion. Report uncertainty provenance and
   unavailable evidence, not a universal safety factor or precision percentage.

Source: [BAM dataset](https://zenodo.org/records/7198219),
[experimental protocol article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10319173/).
These are three different RF/SLR/SLC protocols at 620 C, not interchangeable
replicates. DB source constants from the existing 650--700 C audit must not be
silently transferred to this cohort. ASTM E2714 is a relevant test-method
identifier; a catalogue lookup does not constitute a full standards compliance
review. NIMS CDS access/use restrictions remain respected.

## Existing-CAD preprocessing research

Retain Gmsh/OpenCASCADE rather than create a CAD kernel. Audit the existing
import path before adding APIs. Priorities: length units, physical groups and
stable region identities; local size/curvature/distance fields; element-order
and formulation compatibility; Jacobian/quality diagnostics; thickness and
contact-region resolution; named material/boundary coverage. Successful mesh
generation alone does not establish FE accuracy. Acceptance must include a
tag-preserving import, positive valid geometry and a quantity-of-interest
refinement study, not just an attractive mesh.

Reference: [Gmsh manual](https://gmsh.info/doc/texinfo/).

## Lightweight postprocessing research

Keep XDMF/H5 and ParaView as authoritative existing outputs. A viewer is an
optional, read-only consumer of Result, not a new solver or result schema.
Initial candidates:

- vtk.js: small client-rendered view with explicit surface extraction/field
  mapping. Browser memory and topology/field association need measurements.
- Glance: useful reference/prototype, but its documented readers do not promise
  direct general XDMF/H5 FE loading; do not equate HDF5 image support with ours.
- trame with VTK: preserves mature data processing and offers local/remote
  rendering, but adds a Python server/dependency footprint. Compare against
  the client-only route rather than declaring it lightweight by default.

First scope: rotate/zoom/reset, frame selection/playback, displacement warp,
stress/strain/state selection, component/invariant and unit display, scalar
bar and two restrained themes. Fixed color ranges should be available across
frames. Show warp scale and any surface-only/recovered representation. Never
merge coincident interface nodes or silently average discontinuous cell fields.
Load only published frames; retain source IDs and original data. Offline/local
operation is preferred; no automatic cloud upload or CDN dependency.

Acceptance: compare time, counts, selected values and warped coordinates with
VTK/ParaView, including two-dimensional vectors and discontinuous interfaces;
measure cold-open, frame-switch latency, peak memory and derived file size on
a representative 25-frame result. Embedding in an agent is optional and must
not be required to run the project or view its outputs.

References: [vtk.js](https://kitware.github.io/vtk-website/ecosystem/vtk-js.html),
[Glance readers](https://kitware.github.io/glance/doc/readers.html),
[trame VTK](https://trame.readthedocs.io/en/latest/trame.widgets.vtk.html).

## Status

Initial RF source admission is complete: published MD5 matches; SHA256 is
`128e6de1854f4633e26eced6c4ea221687cde2f232db275487ef0d1201767e0a`.
The file contains 178211 history rows across 376 recorded cycles, spanning
cycle 1 to 2547. Specimen 8d has a reported failure endpoint of 2054 cycles,
distinct from the 2547 total cycles. A direct independent comparison against
all 376 MAX-MIN rows gives zero maximum stress-extrema discrepancy at printed
precision; strain-extrema discrepancies are below 2.8e-17 percentage points.
This verifies ingestion, not a constitutive prediction. The initial download
timed out and was resumed; the checksum was verified only after completion.

`tools/p92_bam_source.py` preserves recorded gaps and converts MPa/percent/C to
Pa/dimensionless/K. It retains only selected scientific header metadata, not
operator names. Seven focused importer tests pass; the combined importer and
existing survival/reliability suite passed 64 tests before the metadata addition,
followed by seven focused passes.

SLC admission subsequently passed: SHA256
`ab2953c0cfa6335619c8b699728aa73233f8d2209af39b709916ab3fd392683e`,
197070 rows, 216 recorded cycles, specimen 11d, failure at 1370 cycles versus
1437 total. Source extrema agree to printed precision. A truncated download
was resumed before its checksum could be accepted. SLR also passed: SHA256
`34e26cae1816ec4b712a4cec5811e0a0c66bad34848c10d962528085171fcabb`,
255301 rows, 276 recorded cycles, specimen 9d, failure at 1484 versus 1754
total cycles. All three audits are recorded in
`evidence/creep/2026-11-14-bam-source-and-relaxation.json`, without personal
header metadata. Their LifeObservation records use separate protocol cohorts;
they must not be pooled as three identical-condition survival replicates.

## Frozen first-hold response screen

`tools/p92_bam_relaxation_screen.py` tests an effective positive axial law
`ds/dt = -r0 (s/s0)^n`, not a general P92 constitutive model. The declared RF
split is 5--60 s of the first tensile hold for calibration, 60--182 s for
within-path holdout. Time-quadrature weights prevent densely sampled intervals
from dominating the fit; they are not statistical inverse-variance weights.
The fitted rate is 4.54833678643 MPa/s at s0 = 222.1 MPa, exponent 9.92074406637.
Calibration RMSE is 0.579569 MPa; held-out RMSE is 0.647943 MPa (maximum error
1.517676 MPa). These are sample-wise errors, not confidence bounds.

With this same coefficient and exponent, later holds are predicted conditional
on their measured initial stresses, without refitting:

| RF cycle | Initial stress (MPa) | RMSE (MPa) | Maximum absolute error (MPa) |
| --- | ---: | ---: | ---: |
| 10 | 228.94 | 1.259585 | 2.525254 |
| 100 | 197.59 | 11.690932 | 14.515483 |
| 1000 | 149.56 | 31.697217 | 38.696548 |

This is a negative transfer result: the simple first-hold fit is insufficient
for later cyclic response. It does not predict cyclic peak stresses, failure
life, or the other two protocols. The analytic solution is independently
checked against DOP853; changing held-out observations cannot change its fitted
parameters. E and a creep coefficient are not separately identifiable from
this effective relaxation rate. Correlated samples within one specimen do not
provide independent replicate uncertainty.
Repeated printed timestamps in later evaluation holds are retained, not
averaged or interpreted as independent replicates; the calibration interval
requires strictly increasing times for its quadrature weights.
The actual strain ranges within these nominal holds are 4e-5 to 6e-5;
temperature ranges are 0.4--0.7 K. Reports retain both. A constant-strain,
constant-temperature effective model neglects those excursions; its mismatch
cannot be assigned entirely to constitutive error without a controlled-input
replay. This is another reason not to promote the first-hold fit.

The original [service-like experiments paper](https://doi.org/10.1016/j.ijfatigue.2022.107381)
also reports limited transfer between strain-controlled relaxation and
stress-controlled creep dwell protocols. Next model work therefore requires
history-dependent cyclic response, explicit control modes and validation
across the full path, not more first-hold curve fitting. Existing 600/650 C
coefficients remain separate from this 620 C cohort.

## Reusable software improvement

`assessments.life_prediction_errors` compares frozen specimen predictions with
observations, retaining identities, units and failure definitions. Run-outs
give descriptive bounds, not invented exact errors. Failure-only log errors
are explicitly conditional and must not be used as censor-aware reliability
scores. The existing six-training/two-held-out P92 baseline now uses this
public interface without changing its fit or predictions. API documentation,
workflow guidance and generated knowledge were synchronized.

No new runtime speedup, interactive viewer or industrial qualification has
been verified. Existing generated site files are preserved.

## Software acceptance checkpoint

2026-11-16 continuation: the earlier CI had one failure, the missing SPDX header
on the new module (6789 other tests passed); fixed and rechecked locally before
resubmission. Distributed and macOS stages had not run, so that CI was not a pass.

The new `verification.history_errors` uses ordinary HistoryResult objects and
requires exact coordinate/unit/component alignment plus an explicit admitted
sampling interval. No interpolation or gap filling is automatic. It integrates
piecewise-linear error and squared error analytically, retaining source hashes.
The RF first-hold calibration coordinate-weighted RMS is 0.300047 MPa versus
0.579569 MPa sample RMS; held-out weighted RMS is 0.646239 MPa versus 0.647943
MPa sample RMS. The fit and predictions are unchanged. Later repeated-timestamp
samples retain their existing point-wise report; they are not silently merged
to enter the strictly ordered HistoryResult contract. 121 targeted tests pass,
including independent analytic history integrals and executable documentation.

149 focused tests passed (life comparison, specimen survival/reliability,
P92 source/response screens, assessments and documentation). Ruff, generated
documentation and whitespace checks passed. A separately built pure-Python
wheel was installed into an isolated target and imported from outside the
checkout: public error comparison and censored survival both passed. This
checks packaging of the new Python API, not native accelerators or a fresh
operating-system installation. The existing environment was not replaced.
An additional 128 structural, coupled creep/plasticity, rotating-reference and
increment-precision regression tests passed in 100.57 s, including selected
history and checkpoint restoration. These verify numerical composition, not
experimental qualification at 620 C.

## Recorded-input replay admission

### Frozen continuous-response experiment, 2026-11-16

Before calibration: use RF cycles 1--3 only; predict cycles 4--10 with the
same parameters and continuous inherited state. Use the existing five-state
uniaxial published recovery equations. Fit only E, k and Z (positive bounded
ranges 100000--200000 MPa, 50--250 MPa and 300--1500 MPa respectively); retain
the other 600 C published coefficients as an explicit transfer hypothesis,
not as measured 620 C properties. Initialize the measured first stress and
zero backstresses/isotropic history/accumulated inelastic strain; this initial
hidden-state assumption must remain visible. Replay measured mechanical strain
piecewise linearly, keeping the model isothermal at nominal 620 C and reporting
the measured temperature excursion. Minimize time-weighted axial stress error,
not sample-count weighted error. Limit optimization evaluations; retain failure.
Compare frozen predictions with refined backward-Euler subdivisions and an
independent stiff ODE method. Report per-cycle peak, valley, sampled work and
stress error. No reset at holdout entry, fitted late-cycle initial stress,
confidence interval from within-specimen samples, or life claim is allowed.

This is a bounded test of a transfer hypothesis. Failure requires improving
the material/loading evidence, not broadening parameters until the holdout fits.

Result: the frozen fit converged in nine evaluations, but Z reached the
predeclared 1500 MPa upper bound. E=146410.238 MPa and k=108.231492 MPa.
Calibration cycles 1--3 have time-weighted RMS errors 5.859, 3.368 and
4.988 MPa. Held-out cycles 4--10 increase from 8.683 to 23.302 MPa.
Doubling backward-Euler subdivisions changes stresses by at most 0.891 MPa
within these cycles. Independent Radau differs from baseline BE by at most
1.818582 MPa; refined BE reduces that difference to 0.927110 MPa. Consequently,
integration error does not explain the observed progressive mismatch. Retain
this as a failed transfer screen, not a validated 620 C material. The measured
temperature range is 2.7 K and the model remains isothermal; the screen does
not uniquely attribute mismatch to one constitutive parameter.

Reproduce with `python -m tools.p92_bam_cyclic_screen Data_RF.lis --radau`.
The compact evidence is `evidence/creep/2026-11-16-bam-continuous-cycles.json`.
The same replay split at a recorded time reproduces the uninterrupted state
exactly in tests; no experimental observations reset the holdout state.

Uncertainty verification now exhaustively enumerates binomial outcome counts
at sample sizes 1, 3, 10 and 30 and confidence levels 90%, 95% and 99%, checking
coverage at declared probability grids including near-zero/one probabilities.
This verifies the existing exact fixed-horizon interval computation against
direct probability sums, not a new P92 scatter estimate. The combined cyclic,
survival, fixed-horizon and P92 life suites passed 99 tests.

PR #188 at cee54865 passed FEniCSx, macOS installed-wheel, native distribution,
lint, documentation and optional PyTorch CI. Pages deployment and targeted
FEniCSx were skipped by workflow selection. New changes require fresh CI.

The extended local run passed 162 tests (published recovery reference, measured
replay, source admission, uncertainty, P92 life and attribution), including a
held-out-data mutation test: changing cycle 4--10 observations changes reported
errors but cannot change calibrated input, parameters or final predicted state.

Additional source search: the 2026 nuclear-material creep compilation
([paper](https://doi.org/10.1038/s41597-026-07563-y),
[public repository](https://github.com/avakanski/Dataset-of-Creep-Properties-for-Nuclear-Structural-Materials))
provides traceable creep records, but searches of its current data for P92/T92
did not locate those names. It is not admitted as P92 scatter evidence.
The P92 small-punch round-robin
[paper record](https://iss-cronfa.swansea.ac.uk/Record/cronfa40645/Details)
states laboratory data are available on request; small-punch results would
also need a separate mechanics mapping, not direct uniaxial-life pooling.
NIMS Grade 92 data remain a candidate subject to actual access/use conditions.

### Frozen cross-protocol prediction

SLR cycles 1--10 (9251 samples, through 21682.805 s) were then predicted with
the exact frozen RF parameter record. There is no fitting on SLR stress and
no hidden-state reset between cycles. A declared 6 s source-interval limit
admits the recorded maximum interval 5.981 s. Two versus four backward-Euler
subdivisions change stress by at most 0.470 MPa; time-weighted prediction RMS
remains 56.617 MPa in cycle 1 and 28.136 MPa in cycle 10. This separate-specimen,
changed-protocol transfer fails to support an engineering material claim.
It does not prove the complete constitutive family incapable: most coefficients
remain transferred from another temperature/source, Z hit its fit bound and
initial hidden state and isothermal assumptions remain unverified.

This test exposed an actual local numerical defect in the research reference:
MINPACK reported stagnation at an elastic/static-recovery root with normalized
residual 5.334e-17 and a stress correction 5.77e-15 MPa. The reference now accepts
only when residual, Newton correction and material-domain checks all pass,
rather than requiring a solver progress flag in addition to a solved equation.
Residual tolerance was not enlarged. A falsely successful status with a bad
root is tested and still rejected. 109 reference, replay, structural-material
and fixed-horizon tests passed after this correction.

Reproduce with `python -m tools.p92_bam_cyclic_screen Data_SLR.lis --frozen-rf
evidence/creep/2026-11-16-bam-continuous-cycles.json` (one command).
Evidence: `evidence/creep/2026-11-16-bam-slr-transfer.json`.

The next bounded response problem is the first ten consecutive BAM RF cycles,
not a synthetic reconstruction of all 2547 cycles. `replay_window` admits
4739 observations over 0--3680.637 s, with maximum recorded interval
1.800000000000182 s under an explicit 2 s admission limit. It returns existing
HistoryResult objects for measured strain, stress and temperature. Missing
cycles, repeated timestamps, unresolved intervals and SLC's mixed stress-control
path are rejected rather than silently converted into prescribed strain.

This gate does not establish the material initial state: the first observation
is strain 4e-5 and stress -50000 Pa, not exactly virgin zero. The full RF record
has 1623 equal adjacent printed timestamps, including 1256 with different strain
values. These cannot directly define a unique physical strain rate. Early
continuous-cycle response therefore has usable input evidence, while late
history replay needs the original controller path or an explicitly assessed
reconstruction assumption. No late-cycle state reset is permitted.

The isolated installed-wheel check also passed for `history_errors`, imported
outside the checkout: the linear-ramp weighted RMS equals 2/sqrt(3). This is
Python API packaging evidence, not a native performance claim. The 48 focused
source, replay, relaxation and history-comparison tests passed; an initial
restricted-environment run terminated during runtime initialization and was
rerun successfully with local communication permitted.
