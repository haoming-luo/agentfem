# Thermal inelasticity and continuation: bounded development plan

This record contains generic software requirements only. No private proposal,
personal information, proprietary material data or customer geometry is retained.

## Order and gates

1. Maintenance and ordinary use: PR #132 merged; add a checked zero-copy XDMF
   continuation view. A view is not a restart or a scientific continuity proof.
   Fix complete-Step restoration if corrupt later payloads can leave earlier
   fields changed. Keep existing checkpoints and core state ownership.
2. Thermal inelasticity: reuse accepted temperature histories and the physical
   creep clock; add the monotone Sinh branch to the existing creep integrator.
   Then compose creep and cyclic plasticity within one local increment, with
   one accepted state and one complete consistent tangent. Stage structural
   admission only after local tests, combined state rollback/restart and energy
   bookkeeping pass. Do not use the rate-independent provider with artificial
   normalized time to bypass physical-time semantics.
3. Damage/life: start strain-life postprocessing with explicit strain convention,
   reviewed curve identity and units; keep lifetime assessment distinct from
   constitutive stiffness degradation. Global K-R damage and mesh regularization
   remain separate future gates. No crack-length or industrial-life claim.

## Simultaneous creep/plasticity design

Retain the existing complete J2/Chaboche return map `P` and its tangent `Cp`.
At fixed previously accepted histories solve the six-component stress equation

`R(sigma) = sigma - P(eps_mech - CE_old - dt flow(sigma), plastic_old) = 0`.

The Mandel basis is orthonormal: no hidden engineering-shear scaling. With
`L = d flow / d sigma`, the local Jacobian is `I + dt Cp L`; the consistent
strain tangent solves `(I + dt Cp L) C = Cp`. Backstress makes the two flow
directions generally different, so no common radial direction is assumed.
Each line-search probe re-evaluates from *old* plastic history. Failure returns
no new state. This condensed simultaneous solve is not a one-pass split.

Initial boundary: small strain; constant shared elastic moduli; isotropic J2
or Armstrong--Frederick/Chaboche; power-law or Sinh creep; optional shared secant
thermal expansion and normalized Arrhenius power-law rate. Temperature-dependent
plastic parameters, static backstress recovery, finite strain and contact are
not inferred. A material-point prototype does not imply global admission.

Global integration must preserve separate CE/CEEQ and PE/PEEQ/backstress fields,
bind every field to checkpoint identity, and include plastic hardening storage,
material dissipation and backward-Euler contributions without calling plastic
work stored energy. Do not graft the coupled state onto a CE-only checkpoint.

## References and independent checks

- [MOOSE simultaneous creep/plasticity](https://mooseframework.inl.gov/source/materials/ComputeCreepPlasticityStress.html)
  motivates solving both mechanisms within one increment. Its default finite
  incremental formulation is not the total-small-strain formulation used here.
- [Abaqus creep integration theory](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-ratedepplast.htm)
  distinguishes rate integration from instantaneous plastic response.
- Tests use an independent scalar root for coaxial isotropic hardening, exact
  n=1 continuous Sinh relaxation under refinement, nonproportional Chaboche
  histories, fixed-old-state tangent differences, zero-mechanism limits, and
  forced failures. None is an independent commercial structural comparison.

No new full composite runtime, MPI composition, release or installer claim is
part of this record. Local targeted counts are recorded at each verified commit;
skipped rank-specific tests are not counted as passed.

## Integrated slice

The serial homogeneous 3D composition now reuses the existing implicit creep
Step rather than creating a second nonlinear solver. The joined quadrature
transaction saves CE/CEEQ and PE/PEEQ/all backstresses. Accepted plastic work,
hardening storage, material loss and discrete contributions are separate.
Both serial and portable file formats have serial interruption/continuation
checks; MPI execution remains rejected. The ordinary
`examples/thermal_cyclic_creep` project completes through `agentfem run` with
computed (not independently validated) result status.

Promotion remains gated on broader structural time/mesh convergence and an
independent multiaxial reference. Static backstress recovery and temperature-
dependent plastic parameters are not implemented. The mechanical energy
ledger excludes imposed thermal transfer; the isolated plastic energy
decomposition is not promoted into a full thermal first-law claim.

Local verification: 156 targeted tests passed in 100.98 s across coupled
creep/plasticity, Sinh creep, platform/constitutive models, checkpoint
capabilities and the generic small-strain Step. The ordinary source CLI example
completed 16 accepted increments in approximately 4.4 s; this is a workflow
smoke check, not a speed benchmark or an installed-wheel test.

Follow-up checks add an independent continuous uniaxial solution for linear
creep plus isotropic plasticity: 16/32/64-step error ratios verify first-order
time convergence. A 1/2-subdivision homogeneous patch gives matching reactions
and state; it is deliberately not called nonuniform spatial convergence.
The broad affected static/dynamic/operator/stateful selection passed 193 tests
in 109.69 s. It exposed and now guards against an old duck-typing hazard:
having E/nu must not advertise a history material as stateless elasticity.
Explicit supplied-operator workflows remain distinct and supported.

The ordinary project now solves homogeneous insulated heat first, transfers
accepted temperature frames on the same mesh, and completes the 16-increment
mechanical path in approximately 4.2 s. This records a source-CLI workflow
check only, not an installed-wheel or thermal feedback/first-law validation.
The project declares its required public API, and `agentfem check --runtime`
passes. A separate final selection of flow/platform, continuation output and
assessment regressions passed 178 tests in 11.20 s. Counts overlap earlier
selections and must not be summed as unique test coverage. Ruff, generated
documentation checks, strict MkDocs and the 57-card knowledge check pass.
Local REUSE lint was unavailable because that tool is absent in the numerical
environment; repository CI retains that gate.

## First damage/life assessment slice

The bounded scalar Coffin-Manson-Basquin postprocessor adds explicit
calibration range, source, cycle/reversal convention and an uncorrected-mean
decision. It reuses rainflow and the existing creep/fatigue interaction;
no stiffness damage or life-driven state mutation is introduced. All 68 tests
in the strain-life, existing fatigue/fracture and assessment selection passed.
The [public contract](../strain_life_assessment.md) explains scope, theory
source and the risk of dwell double-counting. This starts stage three; it does
not close global damage, regularization, multiaxial life or material calibration.

## Integration review

PR #133 first full Linux run: 5223 passed, 428 skipped, 8 failed. All eight
failures were pytest 9.2 rejecting nested `approx(list_of_dicts)` in older
restart tests; they were not reported as scientific passes. The tests now
compare complete record sequences exactly, preserving the previous nonnumeric
fallback without relaxing tolerances. The six affected files pass 84 tests
under local pytest 9.1.1; an isolated pytest 9.2.0 run including strain-life
passes 104 tests. The installed scientific environment is unchanged.

Additional assessment guards reject complex histories and distinguish numeric
life overflow from no counted damage. CI must rerun on the repair commit;
the first run's skipped macOS gate is not acceptance. No automatic merge or
release follows from these local results.

## Reference-driven follow-up (2026-10-29)

The [verification ladder](../creep_fatigue_verification.md) records official
test-method catalogue entries, open institutional literature, analytical
derivations, numerical errors and the remaining independent-material gate.
Only public catalogue scopes were reviewed for ASTM; no full-standard
compliance or qualified material fit is claimed.

Three new structural/reference cases check n=1 and n=3 Norton relaxation,
the corresponding synthetic time-fraction integral, and a perfect-plastic
closed-cycle work/strain-life reference. These use the ordinary Step. The
first test attempt found a missing required `stress_exponent` in the new
zero-creep fixture; adding its explicit n=1 setting fixed the fixture without
changing the solver. Focused checks passed 41 tests; the expanded creep,
coupled plasticity, fatigue and assessment set passed 124 tests in 106.53 s.
These counts overlap and are not additive. Ruff, generated documentation,
57 scientific-card checks and strict MkDocs pass.

The review also fixed a decision bug: zero fatigue previously allowed creep
damage beyond the supplied diagram because the out-of-range ordinate was
zero. Acceptance now requires actual abscissa coverage, with explicit
machine-readable scope/domain metadata and invalid imported damage rejection.
This changes unsafe assessment decisions, not equilibrium or material history.
PR #133's preceding full CI is still running; this local evidence does not
replace the latest-commit remote gate.

## Resolved histories and nonuniform joint fields (2026-10-29)

PR #133 at `42ed45b6aee2ea79c166ff34794d19ae0b4aa900` completed all
applicable remote checks, including FEniCSx and macOS installed-wheel
acceptance. The following changes require a new latest-commit CI run.

The public assessment now integrates the reciprocal of a supplied rupture
life along explicit, covered dwell intervals. Physical time, stress units,
kelvin temperature, original sampling gaps, relation provenance and segment
contributions are preserved. Missing coverage, overlapping holds and invalid
relation output cannot publish a partial assessment. Temperature histories
from the implicit creep Step now declare kelvin. Zero counted fatigue damage
serializes to null lifetime with an explicit status, not nonstandard JSON
infinity. Rainflow direction detection no longer loses tiny-scale reversals
through multiplication underflow; the published MathWorks 14-reversal example
provides an external counting reference, not ASTM certification.

Independent nonuniform antiplane fields exercise the ordinary Step. Pure
creep gives relative displacement L2 errors 0.2490, 0.06125 and 0.01433 on
2/4/8 cells. With simultaneous active plasticity and nonlinear creep, errors
are 0.12862, 0.03200 and 0.007810 on 4/8/16 cells at 128 increments. Exact
equilibrium stress and plastic strain are checked independently; 32/64-step
peak-creep errors 0.06348/0.03149 match the known backward-Euler sum. These
are synthetic proportional isothermal references, not alloy calibration or
nonproportional thermal-field validation. The combined reference, material,
assessment and existing NAFEMS-cylinder selection passed **135 tests in
205.80 s**. After the history-index change, **86 assessment tests passed in
0.24 s**. Counts overlap; do not add them. The ordinary thermal/cyclic project
also completed 16 accepted increments via source CLI in about 3.9 s, with
physical units and `computed`, not `validated`, result status.

The first open rupture-data audit retained source/hash/license and rejected
premature calibration because temperature/condition metadata are missing and
the landing-page/table material labels disagree. Public catalogue scopes and
institutional literature guide the next experimental gate; inaccessible full
papers are not evidence of having checked their tables. No private proposal
or personal details were copied into this repository.

### Bounded assessment scaling check

The first history implementation scanned each entire source channel for each
dwell. Ordered-time binary searches now select enclosing source intervals and
interior sample views, retaining identical complete assessment records. On
macOS arm64, Python 3.11.15/NumPy 2.4.6, one process with BLAS/OpenMP threads
limited to one, three alternating baseline/indexed measurements after warmup:

| Samples/channel; holds | Baseline seconds | Indexed seconds |
| --- | --- | --- |
| 1,001; 100 | 0.001855, 0.002078, 0.001902 | 0.001451, 0.001325, 0.001357 |
| 1,000,001; 1,000 | 3.312518, 3.476882, 3.128787 | 0.027396, 0.019692, 0.018547 |

Each hold covers two source intervals. Timing includes the full assessment
call but excludes input creation and serialization; every output record is
compared exactly outside the timed region. This deliberately exposes sparse
holds on long histories, not the cost of densely evaluating every sample.
It is **not a finite-element or full-simulation speedup**. The algorithmic
reason to retain the change is removal of the histories-times-holds scan,
not a portable fixed speedup promise. No concurrent simulation was running.

## Spatial temperature and response validation follow-up

A prescribed Arrhenius temperature profile has an independently integrable
joint plastic/creep field. Relative displacement L2 errors are 0.1379916,
0.0346244 and 0.00813975 at 4/8/16 cells; the focused check passed in 76.86 s.
It isolates local temperature sampling/rate dependence with zero expansion,
not nonproportional thermal cycling or a heat-equation verification.

Review found that the new joint quadrature consumer could cast complex inputs
to real and had no full-array validation barrier after successful local calls.
Inputs are now checked before dispatch. Returned state, stress, tangent and
energy arrays are staged and checked before any live publication. Thirteen
focused fault-injection/input checks pass, including a bad third point's
tangent, stress, energy or backstress. This is numerical-input atomicity, not
a guarantee against arbitrary process or storage failure during assignment.

The ANL-ART-168 Appendix A pages were inspected in the original PDF, including
column units and hold-direction definitions. They are a candidate life-test
source, not complete pointwise stress/relaxation histories or a qualified
AgentFEM material. The NIMS Grade 91 article was then obtained; Tables 7–9
were checked visually against extraction. Their hours/Celsius/MPa and
percent-per-hour conventions must not be mistaken for seconds/kelvin and
strain-per-second. Published fitted correlations are not independent raw
validation data. Its data have not been fitted here. No universal empirical follow-up factor
or proprietary design curve was added to core.

### Residual-only local backtracking

The condensed joint material's local backtracking previously requested a
full plastic algorithmic tangent for every residual-only probe. It now calls
the existing `linearization="none"` branch for those probes; full Newton
states and the returned consistent tangent are still computed as before from
the same accepted old state. Twenty-four focused local/material and input
checks pass. Direct material-point complex strain is explicitly rejected.

An alternating eight-observation prepared 16-increment thermal
Chaboche/creep `solve_result` comparison, after one warmup per method, measured:

| Method | Solve/observe seconds |
| --- | --- |
| Original local update | 9.729661, 9.836265, 9.953227, 9.915063 |
| Residual-only backtracking | 9.120214, 9.138382, 9.215313, 9.236852 |

Median reduction is 7.08%. All physical state arrays and non-timing histories
match exactly, and both accept 16 increments. Both methods use the same new
quadrature validation barrier; this isolates the local-update change, not a
net comparison against an older release. Preparation, compilation and output
files are outside the timer. OMP/OpenBLAS/VECLIB requested one thread in this
single-process Mac run. This is not a full-composite or portable speed claim;
the earlier ordinary-project smoke uses a different fixture and tolerance.

Final affected selection: **149 passed in 267.88 s**, covering joint local and
global updates, nonuniform mechanical/temperature references, failure atomicity,
resolved creep damage, rainflow, strain-life and Sinh creep. Counts overlap the
earlier selections. The final source CLI project accepted all 16 increments in
about 3.8 s and wrote fields, the final checkpoint and the `computed` result
record. Neither this smoke nor the synthetic reference suite calibrates an
alloy or certifies component life.

## Independent rotating-flow and knowledge review

Four additional continuous material references pass in 2.82 s: linear isotropic
and Chaboche plasticity, each with constant temperature or heating/cooling.
The reference prescribes plastic rate and rotating relative stress, integrates
the continuous Chaboche backstress convolution analytically, and independently
integrates creep strain before reconstructing the imposed total strain. It
does not reuse the discrete update. Chaboche creep and plastic axes differ.
32/64/128 increments recover first-order stress, strain and backstress accuracy;
the detailed equations and errors are in the verification ladder. This is
material-point evidence, not a newly passed structural or life-calibration gate.

Machine-facing review found stale knowledge claims that Sinh remained local
only. Those are corrected without extending its tested combinations. The new
joint composition has a separate experimental card so power-law MPI/regional
claims cannot be inherited accidentally. The resulting 58-card catalog passes
import/contract validation, and 18 documentation tests pass. No solver behavior
is changed by this knowledge correction.

The existing regional pure-creep Step also passes four series-shear elastic
follow-up checks in 9.68 s. Independent continuous relaxation at follow-up
ratios 2 and 5 and Norton exponents 1 and 3 matches first-order FE time
refinement. The elastic region has exactly zero creep, interface tractions
balance, and total-displacement compatibility predicts the creeping strain.
This reuses ordinary material regions and coefficient-zero elasticity; no
empirical follow-up parameter or new solver is added to production code.
It is a shear analogue motivated by the public ANL two-bar mechanism, not
an experimental axial-specimen comparison or joint-regional admission.

Final usage review reproduced a new-history guard defect: ten nominal 0.1 s
intervals include a floating-point gap 0.10000000000000009 and were rejected
by an exact 0.1 s limit. The comparison now has a recorded 1e-12 relative
roundoff allowance, retains the measured source gaps, and has no absolute or
large-time-origin allowance. Regressions reject real excess, large-origin
gaps and tiny-scale undersampling. This does not alter the quadrature or
invent additional source samples.

## CI budget follow-up

Run `40812073297` on `acd46f3b` passed the full serial suite: **5315 passed,
428 skipped in 2497.73 s**. Installed native kernels (310 tests), the independent
provider and several distributed suites also passed. The 45-minute job-wide
limit then cancelled the checkpoint-portability step; later MPI/demo gates
and dependent macOS acceptance did not complete. This is not a successful
whole-CI result or evidence for the later local commits. Logs identify a job
budget interruption, not a checkpoint assertion failure. Increase the bounded
budget to 60 minutes and print the slowest 20 serial tests without removing
any tests or relaxing numerical tolerances. Final HEAD requires a fresh run.
