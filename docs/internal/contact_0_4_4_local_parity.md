# 0.4.4 bounded implicit-contact sprint

Status: target and working evidence ledger, **not** a released capability or a
claim of Abaqus/Standard parity. Baseline: merged PR #117, commit `683eb429`.
Scientific input, implementation revision, independent reference and
acceptance tolerances must be recorded before a comparison is judged.

## Claim boundary

Target the ordinary `model.step(...)` route for serial three-dimensional
solids, prescribed rigid analytical or reviewed triangulated tools, finite
sliding, quasistatic implicit increments, finite-strain J2, named multiple
tool pairs and optional supported periodic MPC. Admit frictionless normal
contact first; admit a Coulomb/geometry combination only after its own
tangential-state and feature-switch tests. Comparison is on physical
observables and accepted histories at equivalent stations, not bitwise
equality with another solver or one cloud image.

Excluded from this claim: general self-contact, deformable--deformable
implicit contact, arbitrary CAD/STEP surfaces, shell thickness/offset,
distributed implicit contact, arbitrary history materials, impact and
thermomechanical contact. They remain separate capability gates rather than
being silently approximated by the bounded route.

## Live gate status

### Integration boundary and focused diagnostic sprint (2026-10-25)

**Release-scope audit finding.** The material guide had placed finite-strain
contact claims inside the small-strain J2/Chaboche description and retained
outdated one-tool/affine-friction exclusions. Checking the actual dispatch
revealed a separate defect: small-strain native J2/Chaboche and provider-neutral
material capability predicates accepted registered contact or foundation
Operators even though those builders did not assemble them. The native builder
could omit them; the neutral builder already rejected them later. Six real
model counterexamples failed the capability gate before the fix. Both built-in
predicates now reject these unsupported compositions with one stable
`AFM-SMALL-STRAIN-BOUNDARY` reason, consumed by capability and execution, before
Step registration or field mutation. An external provider may still implement
its own supported composition. No material conversion or new contact law is
introduced. The combined constitutive/global-material subset passes 50 tests
in 2.54 s, including all six counterexamples. The finite-strain numerical
implementation used by the ongoing spatial diagnostic is unchanged.

An adjacent audit finds the same missing-assembly risk in power-law creep and
quasistatic Maxwell. Four additional boundary counterexamples fail before their
admission fix. All five families now share the same reason across capability,
ordinary execution and compatibility builders. The initial expanded subset
passes 95 tests in 13.76 s; 17 existing platform creep tests pass in 6.04 s.
This rejects unsupported compositions; it does not implement new contact laws.

A fresh native wheel from `535e1eac` repeats the ordinary installed partial,
restart and continuous project after this safety fix. Its 99 selected tests
pass in 4.75 s. Displacement, material vectors, contact force, accepted factors
and contact work match both the same-wheel continuous run and the preceding
`51c4a911` frozen candidate exactly on this fixture. An independent meshio reader
finds one 384-tetrahedron block and finite fields at every exported station.
Wheel identity, commands and the reused exact project source are recorded in
`evidence/contact/2026-10-25-installed-boundary-admission.json`. The environment
still shares numerical dependencies; this is not Windows or clean-machine proof.

At entry, PR #118 head `d5347eeb` passes all applicable remote checks, including
Linux native distribution, FEniCSx and macOS installed-wheel acceptance. It
remains draft. The candidate release contract now names bounded serial periodic
J2/contact separately as **experimental** and explicitly excludes nonuniform
mesh independence, Gaussian-diagnostic energy closure, portable combined-contact
restart and transferring implicit performance gains to the explicit composite.
This is a claim audit, not a tagged release or a substitute for final acceptance.

**Equivalent material execution.** With trial `Fe = U diag(s) V^T`, the same
radial return gives `Fe_new = U diag(exp(e_new)) V^T`. Therefore
`Fp_new = V diag(exp(-e_new)) U^T F`. The batched core now applies this identity
using the existing SVD instead of reconstructing the left stretch and rotation
and factoring their product again. The scalar implementation is deliberately
unchanged as an independent arithmetic reference. No law, acceptance threshold,
state schema, public option or tangent definition changes. Tests advance both
histories independently through elastic/plastic loading, unloading, reversal,
rotation and isochoric prehistory with condition numbers up to 10^4, including
1025 points. Existing numerical-tangent and chunk-boundary tests still apply.

An isolated 32-increment screen measures 2.87% less solve-and-observe time.
A fresh 64-increment confirmation records ABBA 28.72407 / 27.53229 / 27.40486 /
28.00160 s: means 28.36284 -> 27.46857 s, **3.15% less**. Every route retains
64 accepted increments, 208 Newton corrections and identical line-search/linear
reason histories. Maximum U/state/stress difference is 5.042e-11; maximum
per-point relative tangent error is 5.355e-12. This is a small bounded gain,
not an order-of-magnitude improvement or an explicit-composite result. Timing
excludes setup/publication and includes station observation. Both reports keep
individual timings, gates, runtime hashes and fixed reference `d5347eeb`:

- `evidence/performance/2026-10-25-j2-svd-return-screen.json`
- `evidence/performance/2026-10-25-j2-svd-return-confirmation.json`

**Cross-platform precision follow-up.** Linux head `5d73ecf3` completed the
unit/interface suite with 4961 passed, 428 skipped and one failure in the new
zero-hardening, ill-conditioned cyclic-history check. One first-Piola component
out of 9225 differed by 3.335e-8 (relative 2.400e-9), just beyond its componentwise
`rtol=atol=2e-9`. Preserve the failure in
`evidence/performance/2026-10-25-j2-linux-component-gate-failure.json`; later MPI
and installed-demo gates did not run, so this CI attempt is not passed.

An optional 80-decimal-digit reconstruction probe advances its own history for
selected condition-1/100/10^4 points. On the local numerical stack, both binary64
implementations have Cauchy errors below 6.03e-10. In the condition-10^4 cyclic
case the new/old first-Piola errors reach 3.57e-8/1.63e-8: multiplication by
`F^-T` magnifies small stress/history roundoff. Neither double-precision route
is an exact reference. This is a local conditioning diagnosis, not reproduction
of Linux's exact arithmetic, a performance test or new material physics evidence.
Reconstruction tool and measured norms:
`tools/check_j2_return_precision.py` and
`evidence/performance/2026-10-25-j2-high-precision-diagnosis.json`.

The regression now compares first-Piola **tensors per material point** in the
Frobenius norm, retaining the original 2e-9 relative and absolute budgets.
Rotating a tensor no longer changes the comparison scale. Added tests reject a
real error at a soft/zero point even beside a stiff point and reject nonfinite
data. Both routes also satisfy `P F^T = J sigma` with a normwise matrix-product
roundoff allowance. Cauchy, state, energy, det(Fp), monotonicity, independent
finite differences and global timing-comparison gates are unchanged; no runtime
code or tolerance changed. All 49 installed material/spectral/protocol checks
pass locally. A fresh Linux head must still pass before integration.

**Spatial discrimination.** Accepted implicit-contact samples expose reference
positions and the integration measure alongside current positions, weights and
point IDs. Exports preserve this optional array when all providers supply it;
older providers remain usable without fabricated reference coordinates. The
square-patch diagnostic partitions pressure on the fixed reference geometry.
A sharp edge-belt indicator has mesh-dependent sampled area, so a second smooth
partition is mandatory for interpreting redistribution: center weight
`256(y-.25)(.75-y)(z-.25)(.75-z)`, edge weight its complement. Their exact areas
are 1/9 and 5/36; this is an observable, not a contact enforcement law or an
error estimator.

At the first loading station, P2 elastic controls on 4/8/12 meshes give total
reaction magnitudes 19.63306 / 18.88425 / 18.69374 N. Smooth edge contributions
are 13.71058 / 12.75983 / 12.49675 N; center contributions are 5.92248 / 6.12442 /
6.19699 N. All have reference active area 0.25, zero PEEQ and periodic mismatch
below 4e-18. Thus reaction differences persist without plastic history or a
changing active area; edge-weighted redistribution exceeds the total change
and is partly offset in the center. This narrows the diagnosis but proves
neither a constitutive bug nor continuum accuracy. Evidence:
`evidence/contact/2026-10-25-periodic-elastic-pressure-moments.json`.
The diagnostic explicitly stops at 0.25 (`complete_cycle=false`); it does not
claim reverse loading or recontact. Do not replace the nodal AL contract with
Gaussian P2 silently.

**Controlled plastic discrimination.** The matching 4/8-cell P2 paths now
both reach 0.25 within the 1200-second runner cap, using the same 0.000375
accepted-plastic limit, bulk degree 6 and contact degree 11. They accept
186/251 increments with 7/8 rejected attempts. Reactions are 4.48528146 /
4.19262706 N (6.52% relative to the coarse value). Smooth edge moments change
3.51572801 -> 3.13241176 N while center moments change 0.96955344 -> 1.06021530 N.
Unlike the elastic control, the sampled active reference areas differ:
0.22584707 / 0.25. These are quadrature-sampled indicators, not exact areas of
an independently reconstructed active-set boundary.
The common plastic-increment cap is an accuracy control, not an identical
accepted time grid: the two meshes take different adaptive paths. Consequently
the measured force difference is a controlled spatial comparison with residual
path error, not a pure spatial-error estimate. Tightening the cap is a separate
check; a third mesh alone cannot establish path independence or continuum accuracy.

On the unchanged four-cell controlled path, contact degree 7 -> 11 changes
reaction by only 0.00557%, with 186 accepted increments in both paths; the
sampled active area still changes 0.22041742 -> 0.22584707 and the peak pressure
98.3622 -> 111.9195 Pa. This separates stable integrated-force evidence from
quadrature-sensitive area/peak sampling. It does not support blaming the 6.52%
spatial force gap on contact quadrature, nor treating sampled peak pressure as
a mesh-independent engineering quantity. Degree 21 was first rejected by the
existing 5/7/9/11 admission contract; no guard was bypassed or core option
expanded. That rejection is retained, not presented as a nonlinear failure.

One independent path-control check on the four-cell mesh halves the accepted
plastic limit to 0.0001875. At the same first station the force changes
4.48528146 -> 4.48618952 N (0.02025%), mean PEEQ changes by 0.01345%, and the
sampled active area is unchanged. It accepts 369 increments instead of 186.
This makes the coarse path error an unlikely explanation for the 6.52% spatial
force difference, but does not bound path error on the finer meshes or later
reversal. See `evidence/contact/2026-10-25-periodic-plastic-half-limit.json`.

Evidence files:

- `evidence/contact/2026-10-25-periodic-plastic-pressure-moments.json`
- `evidence/contact/2026-10-25-periodic-plastic-quadrature7.json`
- `evidence/contact/2026-10-25-unsupported-contact-quadrature21.json`

The first report began on `d5347eeb` with the working source subsequently
committed as `51c4a911`; its recorded tool/runtime hashes identify that snapshot.
Later diagnostics explicitly hash the constitutive and contact implementations
as well. Elapsed diagnostic times include concurrent correctness work and must
not be used as acceleration measurements.

An independent polynomial trace oracle checks the alternative explanation of
incorrect P2/Q2 contact interpolation: on the unit reference square,
`p=2(1+y^2+z^2)` gives force 10/3, moment components +/-11/6, virtual work -1
for `delta_u_x=yz`, and quadratic tangent work 200/9. Tetrahedral P2 and
hexahedral Q2, one/two cells per edge and contact degrees 5/11 pass all eight
installed-core cases. This checks nonuniform trace force, moment, work and
tangent independently, not full elastoplastic spatial convergence.

The next bounded spatial comparison is therefore one 12-cell P2 first-station
run, holding the physical model, plastic limit and integration rules fixed.
Its purpose is to test whether the eight-cell active pattern and integrated
moments persist; it is not another general mesh sweep. It uses the existing
accepted checkpoint policy (every 50 increments, retain two) and a progress
status file. No new solver or restart implementation is introduced by the
diagnostic. A timeout must retain accepted checkpoints and remain a partial
result. Full-cycle accuracy and independent same-input comparison remain open.
The diagnostic's `--restart` option selects exactly one mesh/degree and calls
the public Step checkpoint loader without changing identity checks. It skips
already-passed reporting stations, preserves the source checkpoint and records
its SHA-256; elapsed time denotes only the new segment. Installed tests check a
resumed station against a continuous path and reject a changed penalty before
advancement. This is diagnostic convenience around the existing State lifecycle,
not a new core restart mechanism. The tool can evolve while a running segment
continues from its already-recorded source revision; numerical runtime sources
remain frozen.

Methodological sources: [Simo's multiplicative return mapping](https://doi.org/10.1016/0045-7825(92)90123-2)
and [COMSOL's multiplicative-plasticity integration description](https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_theory.06.033.html)
support keeping the discrete flow and volume-preserving update explicit; neither
provides our timing result. [Contact stability and patch-test analysis](https://www.sciencedirect.com/science/article/pii/S0045794901000487)
motivates separating integral accuracy from approximation/stability. The
publisher abstract was available, not the paywalled full text; the algebra and
new pressure moments above are independently derived and tested here.
The [penalty convergence analysis by Chouly and Hild](https://arxiv.org/abs/1204.4136)
separates penalty regularization from FE discretization; its regularity assumptions
and linear-contact analysis do not prove our finite-plastic convergence.
[COMSOL's contact guidelines](https://www.comsol.com/support/learning-center/article/structural-contact-modeling-guidelines-83831)
also distinguish accurate pressure fields from global structural response. These
support the diagnostic separation, not an automatic parameter recipe.
Its [contact-node discretization notes](https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_modeling.05.147.html)
separately describe pressure interpolation, quadrature and the update solver;
raising the displacement order does not justify reusing a lumped nodal pressure
update without a compatibility argument. This supports keeping the current P1
nodal augmentation and P2 Gaussian diagnostic distinct. It is not evidence that
AgentFEM's particular spatial discrepancy is pressure locking.

**Installed candidate acceptance.** Native wheel source `51c4a911` imports from
its fresh venv, not the working checkout; all three optional native modules
also resolve there. The environment shares the existing numerical dependency
stack, so this does not claim a clean-machine installation. A project assembled
from the existing public periodic-plane example uses ordinary `agentfem run`
for a partial 0.25 path, a restart and a continuous eight-increment path.
Displacement, committed material state, contact force, accepted coordinates and
relative contact-work arrays have zero measured difference on this machine.
Independent meshio reads show one 384-tetrahedron volume block per view and
finite displacement arrays. The partial/resumed views intentionally share the
restart station; the implicit example writes these views after its segment,
not live during Newton iterations.

Separate installed suites pass 52 output/connectivity/live-publication tests,
66 execution/history/retention tests, and 18 physical-fixture/independent scalar
plastic-contact tests. Two mixed-pair export checks additionally prove that an
older provider lacking reference coordinates does not create a partial-looking
global array. Live-view evidence comes from the live writer's independent
reader and killed-next-frame tests, not from the post-solve implicit example.
No tests were skipped. Timings here are not performance evidence because the
bounded spatial diagnostic was also active. The wheel hash, exact project
reconstruction driver, commands, tolerances and observed arrays are recorded in
`evidence/contact/2026-10-25-installed-periodic-contact.json`.

### Batched J2 tangent contractions (2026-10-24)

The existing logarithmic J2 derivative still evaluates the same nine dP/dF
directions. Its chunk-local push-forward now forms the common U^T F^-T once
and packs directions into 3x27/27x3 products instead of 27 separate 3x3 products
per point. No material law, iteration tolerance, state owner, cache lifetime or
public option changes. The 1024-point scratch bound remains unchanged.

The first global comparison failed a componentwise tangent gate: four of
746496 entries differed by at most 2.295e-9, while displacement, state, stress
and the accepted Newton path had already passed. Keep that failure in
`evidence/performance/2026-10-24-j2-tangent-block-contractions.json`, with its
source reconstructable at `a0c06894`. The revised comparison uses each point's
Frobenius stiffness scale, relative tolerance 2e-10, rather than assigning
small off-diagonal entries their own scale. This matches the existing
scalar/batch analytic relative gate and is five orders tighter than the
independent numerical-derivative gate, 2e-5. The original displacement/state/
stress gates (rtol 2e-8, atol 2e-10), force checks and exact increment/Newton/
line-search/linear-reason comparisons are unchanged. Added tests reject real
changes at individually soft points, zero-reference changes and nonfinite data.

Fresh isolated confirmation uses the public four-cell-edge cyclic P2/J2/plane
penalty/MPC diagnostic, not the explicit composite model. After two full
warmups, ABBA solve-and-observe times are 15.40445, 14.12869, 14.09898 and
15.38637 s: means 15.39541 -> 14.11383 s, **8.32% less elapsed time**.
All paths retain 32 accepted increments and 113 Newton corrections; maximum
U/state/stress difference is 3.708e-11 and maximum per-point relative tangent
error 4.543e-12. Four-station force, pressure, contact area, PEEQ and displacement
comparisons pass. The same fresh report measures about 35.78% less analytic
tangent time at 4096 material points; that kernel number is not a solver gain.
Setup and file publication are excluded; station observations are included.
Source hashes remain unchanged across the comparison. Evidence:
`evidence/performance/2026-10-24-j2-tangent-block-confirmation.json`.

Seventy-five focused material/periodic-contact/reuse/protocol tests passed
before the comparison-rule follow-up; forty material/protocol tests then passed,
including independent finite-difference derivatives and a 1025-point chunk
boundary with plastic history, repeated stretches and rigid rotation. Counts
overlap. Final expanded targeted acceptance passes 177 tests, adding standard
boundary and periodic global-J2 paths. This does not close spatial accuracy, independent external comparison,
MPI admission or explicit-composite performance. A separate SVD-based plastic
gradient reconstruction remains exploratory; it is not included in this core
change or its timing claim.

### Controlled P2 spatial-path diagnostic (2026-10-24)

The existing Gaussian-penalty diagnostic now accepts the existing automatic
plastic-increment policy; it does not own a new solver. The eight-cell-edge P2
run at source `197ba2c8`, bulk/contact quadrature 6/11 and plastic limit
0.000375 reached common stations 0.25 and 0.5 before its external 1800-second
budget expired. Its process exited and the partial scalar evidence is retained
as failed/TimeoutExpired, not completed or a numerical-convergence diagnosis:
`evidence/contact/2026-10-24-periodic-p2-controlled-path8.json`.

At 0.25 its reaction is -4.192627056 N, compared with -4.181129794 N on the
same mesh with 128 fixed increments (about 0.27% difference). The four-cell
refined-prediction reaction is -4.485281459 N (about 6.52% spatial difference
relative to that coarse value). The active reference areas also differ:
0.25 versus 0.225847069 square metres. At 0.5 the fine diagnostic opens contact,
with zero force/pressure/penetration and MPC mismatch below 1e-18. These are
two accepted stations only: reverse loading, final recontact, full-cycle path
accuracy and spatial convergence remain unverified. The Gaussian route still
has no normal-relative-work ledger; no general energy-closure claim follows.
Do not launch a larger blind sweep from this partial result. Fixed-increment
CLI validation again rejects non-positive/non-quarter-compatible partitions
before starting a model; four regressions cover that diagnostic guard.

### Accepted-plastic increment prediction (2026-10-23)

The two-hour solver sprint promotes the private work-reduction experiment into
the existing automatic-increment policy, not a separate case solver. The
optional `inelastic_prediction_safety` caps the Newton-based next proposal
using the last accepted maximum plastic increment. Acceptance still enforces
the same declared hard bound; failure cutback and mandatory load knots remain
unchanged. This is a lagged workload predictor, not a local/global error
estimator or an explicit stability bound. Defaults and old default checkpoint
identities are unchanged. Enabled identity binds algorithm, safety and the
expert affine acceptance-measure name.

Source-frozen commit `b83a3936` was compared on a public four-cell-edge P2/J2
cyclic plane-penalty/MPC fixture with limit `0.00075`. ABBA setup/solve/observe
times are 186.897, 115.284, 114.090 and 185.903 s: **38.47% less mean elapsed
time**. Accepted increments are 250 -> 232; rejected attempts 140 -> 13;
Newton corrections 1300 -> 806. This changes the discrete path and must not be
presented as bitwise-equivalent execution or combined arithmetically with the
separate 24.32% execution result. The maximum accepted plastic increment is
0.000748321, below the unchanged 0.00075 bound. A finite 512-increment reference
primes compilation and supplies four common stations, not a continuum truth.
Maximum candidate differences to it are 0.204% reaction, 0.0235% displacement
vector and 0.00448% reference-volume mean PEEQ. Zero reaction uses absolute
error, not a misleading relative error. Repeated U/state/stress/tangent pass
`rtol=2e-8`, `atol=2e-10`. Runtime and driver hashes remain unchanged throughout
the comparison; compact evidence is
`evidence/performance/2026-10-23-inelastic-increment-prediction.json`.

Serial standard and affine global-J2 paths are admitted. Single-region dispatch
was initially advertised by capability inspection although it lowered to a
material map rejected by the backend. A failing regression established that
mismatch; admission now follows the actual lowering. Regional, mixed and MPI
prediction stay unsupported, without restricting their old increment policies.
The real two-rank admission test passes on both ranks with a 120-second launcher
limit. It checks early refusal, not distributed prediction success.

The expanded 48-test focused set passes (one MPI-only test skipped in serial).
It includes ordinary/affine serial and portable-format restart, cyclic moving
AL tool state/work rollback, invalid measure rejection, policy-identity mismatch,
minimum-increment exhaustion and retained benchmark failure evidence. Portable
combined-contact restart is still explicitly refused, not silently promoted.
The independent scalar confined-J2 reference exercises loading, unloading,
reverse plasticity and recontact with one/two-cell meshes and interrupted paths.
Its original 3e-9 displacement gate first failed: the old 1e-8 complementarity
product allowed a 4.678e-9 positive gap at pressure 1.895, product 8.866e-9.
The reference fixture now uses stricter 1e-10 complementarity and 1e-11
penetration tolerances for both increment policies; original field/state/force
error gates are unchanged and pass. No runtime default or timed fixture was
changed to repair this test. Spatial convergence, general energy closure of
the Gaussian diagnostic, and explicit-composite acceleration remain separate.
The broader affected source regression passes 283 tests (one MPI-only test
skipped in serial), followed by four benchmark-protocol tests. These counts
overlap and are not summed. Generated API checks and a strict documentation
build in a separate temporary directory pass; existing `site/` edits are
untouched. The benchmark also has a `--refine-limit` mode for a prospective
fixed-reference/limit/half-limit comparison, which deliberately emits no ABBA
speed claim.

That prospective refinement is now complete at frozen `33412d59`, with no
concurrent numerical run or build and unchanged source hashes. The 1024-step
reference has final reaction -3.71250393, versus -3.69748392 for 512 steps
(0.4046% change). Limits 0.00075 / 0.000375 give final reactions -3.70501173 /
-3.71635888; the sign of their difference from a finite reference matters.
Maximum four-station reaction differences are 0.20181% / 0.10384%, displacement
vector differences 0.03834% / 0.01184%, volume-mean PEEQ differences 0.02909% /
0.001654%, and maximum PEEQ differences 0.08964% / 0.005264%. Open-contact
station forces are zero in each path. The refined prediction takes 449
accepted increments, 18 rejected attempts and 1447 Newton corrections;
its maximum accepted plastic increment is 0.000370597 < 0.000375.
The coarse path reproduces the previous 232/13/806 counts. Observed individual
times are 382.705 s reference, 114.241 s coarse prediction and 208.196 s refined
prediction, not a new ABBA performance claim. Evidence:
`evidence/performance/2026-10-23-inelastic-increment-refinement.json`.
These results support controlled limit refinement, not an exact/global error
bound. No third blind parameter/mesh sweep is started in this sprint. The next
gate is compatible volume/contact/periodic approximation with loading-path
error explicitly separated. No current experimental capability is promoted
to commercial-code parity or released based on this timing result.

### Combined implicit execution confirmation (2026-10-23)

After the individual mechanisms were selected, the combined serial
J2/plane/periodic-MPC route was timed independently with twelve cells per edge
and 48 increments. Reference A disables retained material responses and outer
workspace reuse and uses direct MUMPS; candidate B enables the owners and the
explicit managed GMRES policy. The ABBA times are 261.223, 196.909, 195.732 and
257.591 s, a 24.32% mean reduction, with no concurrent local numerical test or
build. Setup is excluded and four-station observations are included.

All paths complete 48 accepted increments and 533 Newton corrections. Each
candidate records 172 retained material responses, 435 preconditioner reuses,
98 builds and no fresh-PC retry. U, committed state, first Piola stress,
tangent, pressure and normal relative work meet the predeclared
`rtol=1e-7`, `atol=1e-9` comparison. The report retains individual times,
differences, policy and runtime/source identity in
`evidence/performance/2026-10-23-combined-affine-execution.json`.
The driver now preserves unsuccessful setup/solve/comparison evidence too.
It captures code identity before the run, retains it on failure, and refuses
to promote timing if the tool or selected runtime sources change during the
comparison. Three targeted failure/provenance tests pass. The source used by
the completed comparison remains reconstructable at commit `d1e1a368`.
Follow-up dispatch review reproduced a missed invalidation when the actual
preferred `update_array_batch` callback is replaced after a refresh. Its bound
callback identity now participates in the one-shot guard alongside scalar and
object-batch entries. The new regression fails before the correction; afterward
53 focused material-retention, adaptive-correction, workspace, automatic-restart
and benchmark-failure checks pass. No constitutive formula or default solver
policy changes. The recorded timing remains the earlier tested source result,
not a new timing claim for this guard-only follow-up.
This is a bounded implicit execution result, not continuum convergence,
external-code parity, MPI admission or a full explicit-composite speedup.

### Physical cell-recovery audit (2026-10-23)

Reference-rule averaging was correct for affine geometry but not a physical
cell average on a variable-Jacobian map. For `X(r,s)=(r*(1+s),s)`, the old
recovery gave mean `s=1/2` instead of `5/9`, a 10% relative error. The corrected
DG0 recovery normalizes `w_q*abs(det(J_mesh,q))`, keeps each owned cell separate,
uses the backend DOF map for bulk assignment, and forward-scatters only after
collective validation. It does not modify the constitutive state or infer a
current-volume factor from displacement. Curved quadratic triangle recovery
also matches the independent centroid `3/8`.

Fifty-one serial recovery/constitutive checks pass, with three MPI-specific
checks skipped; the latter then pass on each of two ranks. The first empty-rank
test used a one-cell ParMETIS graph and failed during mesh construction,
before recovery. Its MPI group timed out safely; a fail-fast diagnostic
identified the external partitioner error. The corrected test explicitly
assigns that cell to rank zero and verifies the deliberately empty rank.
No partitioner or solver behavior was changed to make the fixture run.
The wider affected source subset then passed 165 tests in 5.14 s, with the
same three MPI-only cases skipped in serial. It includes result contracts,
mixed-J2 energy diagnostics, contact traces and benchmark-failure persistence.
Embedded-surface area weighting and changed mesh coordinates are covered too.
The latter fixture evaluates physical coordinates through the backend rather
than assuming a reference axis is aligned with physical y.
The complete fourteen-test recovery file also passes on each of two MPI
ranks (0.39 s per rank). It now participates in the existing distributed
result CI group, whose launcher has a 180-second timeout.

### Fixed-reduction linear ownership investigation (2026-10-23)

The serial affine outer Procedure can own the KSP and reduced product matrix
for one augmentation attempt rather than recreate them for every frozen
equilibrium. Current full tangents and PtAP values are still assembled at each
Newton correction; there is no frozen Jacobian. Changed transformation identity,
PETSc state or solver options reject, and attempt exit releases owned objects.
The first source regressions pass 40 solver/material tests and 13 cyclic/contact
lifecycle tests. Isolated twelve-cell/32-increment ABBA times are 151.933,
149.915, 150.727 and 152.032 s: a small 1.09% mean reduction. Every observed
station array is identical, with 32 accepted increments and 355 Newton
corrections in all four runs. This is allocation/lifetime consolidation, not
a major acceleration. The evidence is
`evidence/performance/2026-10-23-affine-linear-workspace.json`.

Do not assume consecutive tangents are identical just because the starting
material response was retained. A four-cell cyclic probe finds **zero exact
consecutive matrices in 109 linear solves**: the last factorization generally
belongs to the preceding Newton iterate, not its accepted line-search endpoint.
Consequently exact-matrix factor reuse is not implemented on this hypothesis.

A separate private algorithm probe keeps the current matrix and uses an earlier
LU only as a right GMRES preconditioner. It checks the true unpreconditioned
linear residual and rebuilds on failure; it must never be used with `preonly`.
The first binding attempt failed because this petsc4py exposes reuse on PC,
not KSP; that failure record is retained. Corrected four-cell/16-increment
loading, unloading and recontact match the direct-solve fields within declared
1e-7 relative / 1e-9 absolute checks. All 109 corrections satisfy the configured
true linear residual target; 87 reuse a preconditioner, without a failed-solve
rebuild. This small model has no demonstrated elapsed-time advantage. Larger
accuracy screening precedes timing and any public option design. The twelve-cell,
32-increment screen also matches all four accepted stations (32 increments,
355 Newton corrections). It reuses 287 preconditioners, needs no failed-solve
rebuild, and uses at most 31 Krylov iterations. Its concurrent 156.23/137.19 s
observations are **not** an isolated speedup claim.

If isolated screening selects this mechanism, the design is an explicit linear
execution policy, not a contact/material tuning parameter. The owning linear
sequence must keep the current operator, verify the true residual against the
declared linear tolerance, rebuild once after a stale-preconditioner failure,
and reject if the fresh correction still fails. Direct `preonly` must never
reuse a changed factorization. Scope and cleanup follow the existing attempted
increment; acceptance, rollback and restart do not retain the PETSc factor.
Unsupported adapters (including an SNES translation that cannot enforce these
rules) must reject the option rather than discard it. The default direct route
and legacy checkpoint summaries stay unchanged. Required tests include current
matrix changes, changed layout, a failed reused solve, a failed fresh solve,
nonfinite true residual, exception cleanup, and continuous/restarted cyclic
plastic-contact agreement. Do not ship a public option solely on the present
two private accuracy probes.

The implemented opt-in public boundary is `LinearSolverOptions(preconditioner_reuse=
"adaptive")`, initially lowered only by the serial affine Newton owner. The
existing common Newton factory transports that linear policy; it does not add
a contact-specific tuning knob. A raw KSP/SNES or unreviewed nonlinear adapter
must reject it rather than silently enable PETSc lagging. The owner selects
right-preconditioned GMRES, requires explicit linear tolerances/iteration limit,
and rebuilds after an expensive correction (initial bounded threshold: eight
Krylov iterations). Result diagnostics count reuses, rebuild retries and the
maximum true residual-to-tolerance ratio. Default summaries omit the new
policy key so existing direct-solver restart identities remain unchanged.

The isolated twelve-cell/32-increment ABBA screen subsequently measured
149.995, 129.921, 130.073 and 150.193 s (13.39% less mean solve-and-observe
time), with all declared four-station comparisons passing. This selects the
mechanism for integration, not for a full-run explicit performance claim.
The integrated owner and option transport pass 131 focused serial tests;
the separate two-rank admission test rejects unsupported distributed reuse
before field mutation. Failed correction cleanup, true-residual false
convergence, one-rebuild retry, restart and default-policy compatibility
are exercised. Fresh 48-increment integrated ABBA confirmation finishes at
226.635, 195.011, 195.842 and 226.402 s: **13.73% less mean time**. Each run
accepts 48 increments and uses 533 Newton corrections. Candidate runs reuse
435 preconditioners and build 98, with no failed-solve retry. Their maximum
true residual/threshold ratio is 0.998462. All four station comparisons pass;
maximum displacement/state differences are about 1.33e-15/4.67e-14, and the
largest tangent difference is 4.39e-9 under the declared relative/absolute
criterion. See `evidence/performance/2026-10-23-adaptive-preconditioner-confirmation.json`.
Do not add this percentage to retained-material or explicit-body measurements.

The distinction follows PETSc's [preconditioner lagging](
https://petsc.org/main/manualpages/SNES/SNESSetLagPreconditioner/) and
[current-system KSP operators](https://petsc.org/release/manualpages/KSP/KSPSetOperators/).
These sources are algorithm guidance, not AgentFEM speed evidence.

### Discretization investigation after graded-mesh failure

The next bounded diagnostic compares P1/P2 displacement with the already
supported Gaussian penalty trace and explicit bulk/contact quadrature.
`tools/diagnose_periodic_contact_order.py` builds periodic equations from
**all field DOF coordinates**, not just mesh vertices, and keeps the same
finite-area physical guides and interior patch. Tests apply arbitrary reduced
fluctuations to check opposing-face jumps and the guides, including midsides.
This avoids an apparently successful P2 solve with missing periodic equations.

This is not the existing nodal augmented-Lagrangian discretization, so no
reaction difference can be attributed to polynomial order alone. First compare
P1/P2 at common penalty/integration/path settings, then separately examine
quadrature and penalty sensitivity. Failed solves remain in the scalar report;
do not produce large field archives or tune a value to match the AL reaction.
The physical-model and independent-reference gates remain open.

The first P2 diagnostic correctly failed existing initial-overlap admission:
the old rule required both field and geometry degree one. The narrowed
extension now separates these concepts for analytical planes: CG2 on linear
tetrahedral or Q1 hexahedral reference geometry uses authoritative face vertices
for the reference gap bound. High-order traces also carry off-face cell DOFs,
so using all trace node IDs would falsely reject separated faces. Two real
mesh tests exercise that distinction and exact touching/sub-picometre overlap;
curved reference geometry stays rejected. This is an initial-reference proof,
not a bound for deformed quadratic contact or permission for P2 nodal AL.

The corrected 4/8-cell, 32-increment P1/P2 Gaussian study completes all four
loading/unloading/recontact paths. Final structure forces in x are respectively
-4.05337/-3.68946 N for P1 and -3.29373/-2.73078 N for P2. These differences
are not a convergence pass. Gaussian order seven and bulk order four are held
fixed; integration sensitivity must be separated from spatial interpolation
before interpreting the result. Maximum sampled penetration is about 0.00260 m
in the final eight-cell P2 station; a converged nonlinear residual alone says
nothing about adequate penalty accuracy. Evidence:
`evidence/contact/2026-10-23-periodic-contact-order.json`.

The first reporting attempt failed after each first solve because it requested
the AL-only normal-work recorder from a penalty Step. The observer now explicitly
omits that quantity and marks work unavailable on this diagnostic route; it
does not substitute zero or claim energy closure. The regression exercises
the actual P2 solve and observer together. Source trace/initial-bound tests pass
163 checks, with eight two-rank-specific checks skipped in this serial run.

Ordinary quadratic tetrahedra can have zero equivalent pressure force at
corner nodes; [Abaqus's contact-control guidance](
https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-contactcontrolsstd.htm)
distinguishes that node-to-surface issue from surface-integrated contact.
It motivates this separate diagnostic, not automatic admission of P2 nodal AL.

The fixed eight-cell P2 sensitivity probes now separate integration from mesh
order. Increasing only the contact rule 7 -> 11 changes final x-reaction from
-2.7307783 to -2.7298206 N (0.0351%). Keeping rule 11 and increasing only bulk
quadrature 4 -> 6 gives -2.7300627 N (0.00887%). Both complete all 32 increments
and unloading/recontact. The two records are
`evidence/contact/2026-10-23-periodic-contact-p2-quadrature11.json` and
`evidence/contact/2026-10-23-periodic-contact-p2-bulk6.json`.
These differences are far smaller than the current 4 -> 8-cell force gap;
neither is a spatial-convergence certificate or a reason to increase every
user's quadrature default. Sampled maximum penetration changes with the set of
sample points and is not a continuous-face penetration bound.

### Separate path and elastic spatial diagnostics (2026-10-23)

With the eight-cell P2 mesh, contact rule 11, bulk rule 6 and unchanged penalty,
32/64/128 accepted increments give final reactions -2.7300627/-2.9171279/
-3.0121821 N. Successive differences, relative to the finer result, are 6.4126%
and 3.1557%. The maximum PEEQ changes from 0.1100375 to 0.1106489 to 0.1113156.
The trend is consistent with a remaining loading-path discretization error;
128 accepted increments are not a path-independence certificate. These are
diagnostics, not grounds to change every user's increment count. See
`evidence/contact/2026-10-23-periodic-contact-p2-time64.json` and
`evidence/contact/2026-10-23-periodic-contact-p2-time128.json`.

An elastic-limit control changes only yield stress to 1e12 Pa, holding the
geometry, boundary path, penalty and integration rules fixed. All reported
PEEQ values are zero; four increments capture the four loading stations. First
and final loaded forces agree to roundoff, and unloaded forces are below
1.1e-10 N. Under the **same Gaussian penalty formulation**, the 8 -> 12-cell
reaction change is 4.4570% for P1 and 1.0191% for P2. P2 forces on 4/8/12-cell
meshes are -19.6330568/-18.8842501/-18.6937424 N. Records:

- `evidence/contact/2026-10-23-periodic-contact-p1-elastic.json`
- `evidence/contact/2026-10-23-periodic-contact-p2-elastic.json`
- `evidence/contact/2026-10-23-periodic-contact-p2-elastic-coarse.json`

This narrows the next decision: volume/trace approximation matters even without
plasticity, and plasticity adds an appreciable path error. It does not prove
locking, identify a unique error source, validate the finite penalty as an exact
constraint, or compare P2 penalty directly to the P1 nodal-AL reaction. There is
still no independent continuum reference. Do not launch another unstructured
mesh/penalty sweep or admit P2 nodal AL without a compatible normal space.
Next use a bounded common-path refinement at the selected P2 rules, then assess
the penalty error separately. The supported nodal-AL capability remains P1.

The lifecycle audit confirms that the current path driver retains prescribed
deformation-path knots, starts each constitutive trial from committed history,
and already supports a maximum accepted plastic-strain increment. That bound
is not an error estimator. Do not conflate local return-map convergence,
global Newton convergence and path accuracy. A separate control audit found
that fractional fixed counts could be truncated and several automatic controls
admitted NaN, infinity or noninteger counts. Early typed/finite validation now
rejects those inputs while preserving valid Python policy/checkpoint spelling.
107 focused controller, automatic-restart, material-substep and adaptive-contact
tests pass (one MPI-only test skipped in serial). This changes admission, not
the integration algorithm or default path.

For a later accuracy/cost experiment, [MOOSE's radial-return documentation](
https://mooseframework.inl.gov/source/materials/RadialReturnStressUpdate.html)
distinguishes global material-limited steps from local constitutive substeps.
Local substeps may avoid some discarded global solves, but they cannot resolve
an equilibrium/contact event skipped by the global path. They are not enabled
here. A finite-strain implementation must specify its deformation interpolation,
carry state sensitivities through the full sequence and differentiate that
sequence, rather than returning the last substep's tangent; the general
consistency issue is treated by [Pérez-Foguet et al. (2001)](
https://doi.org/10.1016/S0045-7825(00)00336-4).
First compare the existing global controller against common-path refinement;
only pursue local substepping if material integration, rather than contact
events or spatial approximation, is shown to limit useful step size. No new
runtime dependency, automatic differentiation framework or hidden material
path is introduced by this research note.

### Worker-local nodal transport: coherent prototype withdrawn (2026-10-23)

A native prototype removed the complete Hex8 cell-displacement intermediate:
kinematics and material/body workers gathered their 24 local values directly
from the immutable nodal snapshot. It retained uncertain-geometry fallback,
material repair, viscosity, current forces and accepted-state ownership.
Thirty-nine arithmetic/lifecycle tests passed, and four 600-step late-state
trajectories retained identical fields, material state and per-step histories.
The ABBA times were 73.111, 73.065, 72.649 and 74.695 s. Mean reduction is
1.41%, while the two baseline runs differ by 2.17%. This does not select a
useful whole-step speed gain; the option and implementation are withdrawn,
with the prototype retained privately. Logical temporary removal is not a
measured peak-memory reduction. See
`evidence/performance/2026-10-23-nodal-transport-screen.json`.
Reconsider with a measured memory limit or a combined dependency/critical-path
change, not repeated short sampling until a preferred percentage appears.

### Joined body/viscosity candidate (2026-10-22)

An opt-in native cell pass now gathers nodal velocity locally and evaluates the
existing bulk-viscosity formula alongside body force. The public defaults are
unchanged. It retains separate material work and viscous power, current damping
stability bounds, exact material repair and the viscosity-conditioning fallback.
All physical contributions validate before the ordinary trial is published.

The isolated native build and source path pass 133 focused native/body/material/
restart tests, followed by 16 restart tests including both cross-policy recovery
directions and different worker/tile budgets. A later pure-native recheck passes
all 13 damped-dispatch tests without requiring MPI initialization. These are
source/candidate-build results, not a frozen installed release acceptance.

One real late-state pair advances increments 95,000 to 95,600 in 128.845 s and
118.889 s. Numerical U/V/A, material state/stress/gradient and the physical
history snapshot match exactly. However the comparison harness incorrectly
hashed dictionaries as NumPy object-array bytes (process-local pointers),
which aborted the intended ABBA sequence after the first pair. The harness is
fixed to fingerprint nested content and reject object-array pointer hashing;
the original failed artifacts are retained, not relabelled as a completed test.
The candidate's first-response comparison also agrees on ten body/material/
viscosity arrays, including work and damping bound.

The earlier permission blockage was resolved in the next authorized window.
A fresh complete ABBA confirmation on 95,000–95,600 records separate-pass
82.892/79.564 s versus joined-pass 76.712/76.502 s: **5.69% less mean elapsed
time for this late interval**. All accepted U/V/A, material state/stress/gradient
and physical residual history match exactly. Preparation, checkpoint loading
and serialization are outside the timing. Evidence:
`evidence/performance/2026-10-22-joined-viscosity-confirmation.json`.
Do not extrapolate this interval into an achieved full-run improvement.
Neither the reference case nor its completed 25-frame output was modified.

The next separate geometry experiment evaluates a current closest-point normal
only at a strictly interior smooth Q1 point on a reviewed embedded, connected,
outward closed surface. Edges, vertices, ambiguous numerical projections and
unresolved orientation retain the existing two-ray classifier. This follows the
smooth-surface sign argument described by
[Bærentzen and Aanæs](https://doi.org/10.1109/TVCG.2005.49), not an unverified
transfer of triangular pseudonormals to warped Q1 patches. The private prototype
passes eight geometry tests, including warped/concave surfaces, current refits,
disconnected/nested shells, reversed orientation, near-zero fallback and
unchanged rejection of distinct equally near faces. On the same 40 prescribed
late geometries, ABBA contact times are 3.261/3.257 s versus 2.911/2.932 s.
Ray queries fall from 307,140 to 5,094; all contact forces, gaps, weights and
potential match exactly. This is not a dynamic trajectory or a complete-step
performance claim.

The optional core `sign_policy="interior_normal"` now reconstructs its normal
from current Q1 derivatives rather than trusting a callback's unit-normal
transport. Connected-shell orientation is rechecked after geometry refits.
107 targeted geometry/explicit-contact tests pass, including a toroidal shell,
both cross-policy checkpoint directions and rejection of changed provider
identity. Checkpoint compatibility omits only the core sign arithmetic choice;
actual execution provenance retains it. The default two-ray route is unchanged.

A whole-step combined ABBA on the same late interval records baseline
73.950/77.405 s versus combined 72.357/71.689 s, with identical accepted fields
and physical history: **4.83% less mean time in this screen**. Rays decrease
4,676,043 to 63,682 in the first pair (including setup), with identical
4,973,416 prepared points. This does not establish an additional gain over
body fusion alone. Concurrent critical paths and changed CPU contention matter;
do not add the geometry and body percentages. Next screen process-local worker
budgets and confirm any selected combination on fresh runs.

### Explicit Q1 classification reuse: tested, not promoted (2026-10-22)

A private prototype transported a certified inside/outside classification only
while a conservative clearance exceeded query and surface motion. Every force
and projection still used current geometry; expired certificates fell back to
the existing two-ray classification. A second variant removed best-fit rigid
rotation before bounding the residual motion. Neither cached old contact forces.

On 40 evolving prescribed geometries sampled between the last two accepted
composite frames, the translated certificate reduced native classifications
from 307,140 to 294,703, but ABBA elapsed sums were A=3.865/3.910 s versus
B=4.135/4.129 s. Rotation compensation reduced them to 269,798, yet remained
slower: A=4.190/4.320 s versus B=4.481/4.539 s. Forces, gap bounds, interpolation
weights and potential matched exactly at every sample. This was geometry replay,
not a dynamic trajectory or full-solver timing claim.

The active inside clearance had median 5.57e-6 mm, whereas the next global
maximum surface motion was about 3.34e-5 mm. Nearly every inside point already
had a positive certificate, so merely increasing clearance-query coverage is
not the leading fix. Global motion bounds expire too quickly in this tested
configuration. The prototype was removed from production code and retained as
a recoverable private patch. Reconsider tighter spatially local certificates
only with complete coverage proof and a whole-step benefit; do not retain
extra cache/State complexity for a query-count improvement alone.

### Edge-graded spatial diagnostic (2026-10-22)

An optional research mesh clusters coordinates near patch/support edges while
preserving the physical patch, periodic face matching and default uniform
fixture. The independent linear exact-constraint diagnostic gives forces
27.7036, 25.5705, 24.6039, 23.7530 and 23.3888 N for 8, 12, 16, 24 and 32
cells per edge. All patch reactions remain compressive; residual/gap bounds
remain below 2.7e-13 / 1.1e-16. This is a separate discretization study, not
an equivalent-execution speedup or proof of a converged continuum value.

The complete graded 12-cell, 256-increment finite-J2 cycle also meets the
original penetration/complementarity bounds: maxima 4.711e-9 / 9.678e-8.
Final reaction magnitude is 4.13932494 N and reference-volume-weighted PEEQ
is 0.06307447. It takes 2634 Newton iterations and 962 augmentation updates.
Unlike the uniform grid, graded elements require physical-volume weighting,
not an arithmetic average of quadrature-point PEEQ. The tool now makes that
distinction explicit. One graded mesh does not close spatial convergence;
the uniform 12-cell force is 4.17988201 N and the uniform 16-cell force is
3.97145647 N, so agreement with one coarse result is not an accuracy criterion.

Evidence: `2026-10-22-independent-linear-graded.json` and
`2026-10-22-affine-j2-graded12.json` under `evidence/contact/`. The fixture's
graded option and periodic affine coordinate mapping pass 13 targeted tests
together with the existing affine equilibrium reuse regressions.

The graded 16-cell/256-increment cycle finished on 2026-10-23. All 256
increments meet the original numerical contact gates (maximum penetration
4.689e-9 m, complementarity 9.866e-8). It takes 3148 Newton corrections and
1282 outer updates. Final reaction magnitude is 3.85809815 N, versus
4.13932494 N on graded 12: **7.2893% difference relative to the finer grid**.
The first loading peak differs by 3.7488%; final volume-weighted PEEQ differs
by 0.4579%. Neither the smaller PEEQ difference nor successful nonlinear
convergence substitutes for reaction convergence. Evidence:
`evidence/contact/2026-10-23-affine-j2-graded16.json`.

This grading experiment does not close the spatial gate. Do not initiate
another expensive graded cycle solely hoping to meet a percentage. The next
discretization study must separate volume interpolation and boundary multiplier
approximation. The current AL adapter explicitly owns a coalesced P1 nodal
trace with disjoint reduced rows. Merely changing the volume to P2 leaves
unconstrained midside contact/periodic degrees of freedom or violates that
multiplier contract. A higher-order comparison therefore needs a compatible
trace and periodic constraint construction, not a degree flag in this example.
Adaptive plastic-contact examples such as
[deal.II step-42](https://dealii.org/9.7.0/doxygen/deal.II/step_42.html)
inform the next study; their small-strain formulation and estimator are not
an independent reference for this finite-J2 cyclic problem. External-code
comparison and a converged spatial reference remain open release claims.

### Retained material response, not a lagged Jacobian (2026-10-22)

The serial affine AL Procedure can request reuse of an unchanged built-in J2
response between outer equilibria. The transaction owns a one-shot, SHA-256
buffer guard scoped to that attempted equilibrium. It includes evaluated
old/new gradients (so live expression inputs are checked), material identity
and parameters, time, old/trial internal state, current stress, tangent,
energy and diagnostic fields. Raw field storage avoids copying quadrature
arrays merely to inspect them. No result array is cached or re-owned.

Every changed Newton displacement still receives its current constitutive
update and tangent. A changed writable response buffer invalidates reuse too.
Commit, rollback, restart, failure and leaving the scope discard retention;
mixed pressure, regional/user material and MPI keep ordinary refresh. A
failed optional guard falls back to that same refresh/error path. The guard
is operational evidence, not a new scientific checkpoint identity or tolerance.
Contact residuals and global tangent assembly are never reused by this change.
This differs from [PETSc Jacobian lagging](https://petsc.org/release/manualpages/SNES/SNESSetLagJacobian/),
which deliberately changes how frequently the nonlinear Jacobian is rebuilt.

The first source checks pass 24 mutation/lifecycle/trajectory tests and the
18 existing AL/MPC exact-reference, restart and failure tests. These are
scoped source results, not a new installed release or a continuum-accuracy
certificate. `tools/benchmark_retained_material_trial.py` compares the actual
implementation against normal recomputation, including the guard cost; it
does not extrapolate the implicit-path speedup to the explicit composite run.

After adding callable and field-space identity guards, the fresh 12-cell,
32-increment ABBA confirmation records A=208.220/211.357 s and B=173.141/172.245 s:
**17.68% less elapsed solve-and-observe time** on this machine and implicit route.
Both candidates reuse 116 unchanged material trials; all four runs take 355
Newton iterations and 32 accepted increments. U, committed internal variables,
first Piola stress, algorithmic tangent, normal pressure and work histories are
exactly equal at four common stations. Source hashes and thread settings are in
`evidence/performance/2026-10-22-retained-material-trial-confirmation.json`.
The smaller first screen measured 14.47%; these are different workloads, not
additive speedups and not claims about explicit composite throughput.
The final focused source group passes 37 tests covering retained response,
affine assembly reuse and independent linear-reference amplitude scaling.

### Independent spatial diagnostic (2026-10-22)

`tools/diagnose_periodic_contact_spatial.py` keeps the supported periodic mesh,
patch and affine constraint elimination, but removes both plastic integration
and normal augmentation. It independently assembles small-strain Hooke
elasticity and exactly imposes the known active nodal patch. All contact
reactions must be compressive; a tensile reaction rejects that assumed active
set. It is a research diagnostic, not a replacement public contact solver.

| Cells/edge | Contact force (N) | Edge-node share of contact force |
| ---: | ---: | ---: |
| 8 | 27.73441371 | 81.38% |
| 12 | 26.05503776 | 71.16% |
| 16 | 25.21188874 | 63.84% |
| 24 | 24.37327238 | 54.06% |
| 32 | 23.95763141 | 47.70% |

The free residual norm stays below 9e-14 and imposed-gap error below 2e-16 m.
The reaction trend therefore persists without plasticity, penalty or an AL
stopping criterion. This narrows the investigation to the spatial formulation
and physical fixture; it does not by itself distinguish bulk approximation,
boundary-patch discretization or edge regularity. The finite-Hencky material
differs from this linear law, so numerical equality with the J2 fixture is
neither claimed nor required. Keep the finite-J2 continuum-accuracy gate open.
Evidence: `evidence/contact/2026-10-22-independent-linear-spatial.json` and
`evidence/contact/2026-10-22-independent-linear-spatial-fine.json`.

### 2026-10-22 independent confined plastic-contact reference

`tests/test_contact_plastic_coaxial_reference.py` supplies a scalar reference
independent of AgentFEM's material, contact and equilibrium implementations.
For a homogeneous, transversely confined column, write
`F=diag(lambda,1,1)` and `Fp=diag(exp(p),exp(-p/2),exp(-p/2))`. Quadratic
Hencky elasticity and associative linear-hardening J2 then give
`q_trial=2*mu*log(lambda)-3*mu*p_old`,
`delta_gamma=max(0,(abs(q_trial)-yield-H*alpha_old)/(3*mu+H))`,
`p=p_old+sign(q_trial)*delta_gamma`, and `alpha=alpha_old+delta_gamma`.
The independent nominal stress is
`P11=(K*log(lambda)+2*mu*(2*log(lambda)/3-p))/lambda`.
A scalar bracketed root enforces
`P11 + penalty*max(0,u-gap) = applied_reference_traction`.

The 32-increment cycle includes compression/tension reversal, contact opening
and recontact, with genuine forward and reverse plastic flow. At every accepted
station, two P1 mesh sizes and both analytical-plane and certified closed-face
integration compare displacement, all nine Fp components, PEEQ, stored
`yield_stress*PEEQ` dissipation and integrated contact force against that
reference (2e-9 absolute / 2e-7 relative for fields/state/resultants). All four
FE paths and the scalar path-coverage test pass. No material parameter or
acceptance tolerance was fitted to the computed FE response.

This extends the earlier elastic-only spring reference to cyclic plastic
history. It does not validate noncoaxial flow, curved feature transitions,
MPC, a general energy closure or the inhomogeneous spatial convergence gate.
The underlying multiplicative-plasticity framework is described by
[Simo (1992)](https://doi.org/10.1016/0045-7825(92)90123-2); the confined scalar
fixture above is an explicitly derived test, not a claimed published benchmark.

### 2026-10-22 separated time/space refinement

The finite-area-supported periodic J2/plane fixture now has both 128- and
256-increment paths on 8- and 12-cell meshes, plus a complete 16-cell,
256-increment path. The refined study explicitly
selects MUMPS after the same-model backend comparison below; its numerical
and physical inputs otherwise match. All accepted stations meet penetration
1e-8 and complementarity 1e-7, including opening, unloading and recontact.

| Cells per edge | Increments | Final force magnitude (N) | Displacement L2 | Mean PEEQ | Signed relative-gap work (J) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 128 | 4.38449911 | 0.0110007384 | 0.0635272994 | -7.27418e-5 |
| 8 | 256 | 4.46981161 | 0.0110002343 | 0.0635578553 | -1.86169e-5 |
| 12 | 128 | 4.09973201 | 0.0111094465 | 0.0631515272 | -8.04070e-5 |
| 12 | 256 | 4.17988201 | 0.0111101705 | 0.0631811981 | -1.89452e-5 |
| 16 | 256 | 3.97145647 | 0.0111379022 | 0.0629464910 | -2.05871e-5 |

On the 12-cell mesh, 128→256 changes final reaction by 1.9175% and mean
PEEQ by 0.0470%, relative to the finer path. At 256 increments, 8→12 changes
reaction by 6.94% and mean PEEQ by about 0.60%, relative to the finer mesh.
The next 12→16 comparison changes reaction by 5.2481%, mean PEEQ by 0.3729%
and displacement L2 by 0.2490%. Reaction differences decrease, but this does
not establish an asymptotic convergence rate or close the accuracy gate.
Thus the spatial reaction gap is not explained away by the earlier coarse
increment path. Time refinement reduces signed relative-gap work magnitudes
on the 8/12-cell meshes; spatial refinement is not monotone in that quantity.
It is not plastic dissipation or a general first-law certificate.
Peak pressures/PEEQ are not substituted for these integral comparisons.

Evidence: `2026-10-22-supported-periodic-time-refinement.json` and
`2026-10-22-supported-periodic-refinement-12-mumps.json`, plus
`2026-10-22-supported-periodic-refinement-16-mumps.json`. The 12-cell paths accepted
128/256 increments with 1271/2288 Newton iterations. The 16-cell path accepted
all 256 increments with 2626 Newton iterations and 977 augmentation updates,
in 3634.224 s. Maximum accepted penetration/complementarity are
8.813e-9 / 9.971e-8, within the original 1e-8 / 1e-7 bounds. Elapsed durations
are scientific-study provenance, not an isolated speed comparison.
Gate 8 remains open: next compare spatial discretization/contact enforcement
at a fixed resolved path and retain the external same-input comparator gate.
Do not lower accuracy requirements merely to promote this release target.

### Elastic-limit diagnostic: the spatial gap is not plastic history alone

`2026-10-22-supported-periodic-elastic-limit.json` keeps the supported geometry,
elastic constants, 40,000 penalty and original acceptance tolerances, but
explicitly changes yield stress from 2 Pa to 1,000,000 Pa. This is a separate
physical diagnostic, not a replacement material or an acceleration candidate.
Both 8/12-cell, 32-increment cycles finish with exactly zero PEEQ and zero MPC
mismatch. Final reactions are 27.18411331 / 25.52207474 N: a 6.5122% difference
relative to the finer mesh despite no plastic flow. Displacement L2 changes
only about 0.0295%. The elastic first/final loaded stations agree to numerical
tolerance, unlike the history-dependent plastic cycle.

Therefore plastic integration alone cannot explain the spatial reaction gap.
The same-mesh exact active-normal elimination now supplies that discriminating
check. In `test_affine_normal_augmentation_research.py`, augmented/exact
resultants are 27.1841132431 / 27.1841132697 N on 8 cells and
25.5220747284 / 25.5220747419 N on 12 cells. Displacement, state and nodal pressure
comparisons also meet the pre-existing reference tolerances. Thus the elastic
spatial reaction gap persists under exact enforcement of the same nodal
constraints; incomplete augmentation is not its explanation. This does not
prove continuum convergence, remove edge singularities, establish a general
inf-sup condition or identify locking. The external comparison stays open.

The direct full-load diagnostic initially exhausted 60 outer updates on the
8-cell case (complementarity 3.462e-7, required 1e-7), while the 12-cell dense
diagnostic exceeded its default 4096-DOF budget. The corrected diagnostic
explicitly permits 120 outer updates and 8192 dense DOFs, without changing
physical parameters or tolerances. A separate constructor misuse in the test
harness was rejected and corrected. These are test-only budgets; the public
32-increment cycle above retains its 60-update policy. Five exact-elimination
cases pass in 101.46 s, including the three earlier plastic discrete checks.
Evidence and retained failure summaries are in
`2026-10-22-exact-normal-elimination.json`.

For the next discretization decision, compare a fixed physical patch and
integrated reaction, not only nodal peak pressure. MOOSE distinguishes
[node/face enforcement from mortar surface discretization](https://mooseframework.inl.gov/modules/contact/)
and documents the [conditioning tradeoff of penalty parameters](https://mooseframework.inl.gov/syntax/Contact/).
These are useful design alternatives, not evidence that mortar is already
needed here or that a larger penalty repairs a spatial error. First isolate
the current discrete constraint from the continuum approximation; any later
surface method must retain the same State/Result contracts and earn its own
patch, force/moment, tangent and convergence evidence.

### 2026-10-22 linear backend and option-transport audit

The public Newton adapter silently dropped explicitly supplied inner linear
`rtol`, `atol` and `max_it` on SNES; the affine adapter dropped `atol`. Both now
preserve these controls. A real SNES regression inspects its actual KSP
tolerances, not just a Python options dictionary. Explicit affected policies
receive a versioned identity; a legacy-policy checkpoint is rejected before
mutating accepted state, while a corrected checkpoint resumes equivalently.
Unspecified default policies retain their old identity. The focused solver,
affine lifecycle and standard/periodic finite-J2 suite passed 146 tests.

The sequential, same-input screen in
`tools/benchmark_affine_linear_policy.py` uses the finite-area-supported
periodic fixture, 16 increments, penalty 40,000 and unchanged nonlinear/contact
tolerances. Four common stations compare displacement, committed material
state, nodal contact force and relative-gap work. The default actual backend
was PETSc's built-in serial LU, not MUMPS.

| Cells per edge | PETSc LU | MUMPS LU | GMRES/ILU | Newton iterations for each |
| --- | ---: | ---: | ---: | ---: |
| 8 | 24.703 s | 20.011 s | 18.759 s | 149 |
| 12 | 187.329 s | 83.448 s | 70.722 s | 173 |

These are single sequential screens, not statistically established universal
speedups. The 12-cell MUMPS screen used 55.45% less solve time, while GMRES/ILU
used 62.25% less; all four stations satisfy the predeclared 1e-7 absolute/relative
comparison and the observed maximum state difference is below 5e-13.
The iterative route explicitly uses linear rtol 1e-12, atol 1e-14 and at most
1,000 iterations per correction; it performs 13,170 linear iterations on the
12-cell case. Results are in `2026-10-22-affine-linear-policy-screen*.json`.
This is an **implicit** solve finding, not an explicit-composite acceleration.
No core default or scientific input is changed. MUMPS is selected explicitly
for the subsequent refined scientific study, and its policy is stored in that
study's report. Backend choice does not close the continuum/comparator gate.

The implementation reuses the existing PETSc-backed public policy, following
[PETSc's optional MUMPS backend](https://petsc.org/release/manualpages/Mat/MATSOLVERMUMPS/)
and [GMRES](https://petsc.org/release/manualpages/KSP/KSPGMRES/).
Neither documentation promises these timings; they are local observations.

### Next implicit cost hypothesis, not a selected optimization

A one-second read-only stack sample of the 16-cell refinement finds both MUMPS
numeric factorization and small tensor algebra in the live execution. It does
not establish whole-path percentages. Source inspection confirms that
`FiniteStrainJ2StateTransaction.refresh_trial` requests the full material
response, including its tangent, on trial refresh. Before extending another
backend, count which residual/line-search evaluations actually need that
tangent. A future stress/state-only request must be explicit in the material
contract; never fill a missing tangent with zeros. A later Jacobian request
must produce the derivative of that same trial from fixed committed history.
Ordinary nonzero-BC lifting and mixed-pressure transformations currently use
the tangent even outside a matrix assembly, so they cannot inherit a shortcut
without their own dependency checks. Start with a bounded affine residual
path only if a profile establishes redundant tangent work.

This is a candidate research direction, not implemented acceleration. The
explicit composite response-only path already avoids algorithmic tangents;
neither this idea nor a different KSP can be credited as an improvement to
that explicit run. The separate elastic-limit fixture diagnostic varies only
the declared yield stress to help distinguish spatial effects associated with
plastic history; it must not replace the original plastic acceptance problem.

If factorization remains dominant on the larger mesh, collect the actual
ordering, fill, numeric-factor time and memory before selecting another
backend. [MUMPS documents ordering and scaling](https://mumps-solver.org/doc/userguide_5.8.2.pdf),
and [PETSc exposes its controls](https://petsc.org/main/manualpages/Mat/MatMumpsSetIcntl/).
These are scoped solver-policy candidates, not permission to change global
PETSc options, freeze an obsolete tangent, or assume symmetric positive
definiteness in plastic/contact corrections. Compare total nonlinear work and
the accepted trajectory, not only one successful factorization.
The serial loop already calls `PtAP(..., result=reduced_tangent)` within each
equilibrium solve; proposing that same reuse again is not a new optimization.
Any longer-lived matrix-product cache must establish invariant sparsity and
constraint reduction, and invalidate when either changes. PETSc's
[matrix-product reuse](https://petsc.org/release/manualpages/Mat/MatPtAP/)
is distinct from retaining old numerical factors.

| Gate | Current evidence | Still needed for promotion |
| --- | --- | --- |
| 1. Comparator | Internal/public-protocol tests; elastic and small-displacement published cases screened but not misidentified as finite-J2 comparators | Independent same-input finite-J2 contact run and frozen observables |
| 2. Surface | Reviewed closed/open triangle geometry, bounded initial-overlap checks, and per-face same-plane proof with one bounded fallback subdivision | Rims, unsupported curved tools, ambiguous sharp features and target-path sliding coverage |
| 3. Integration | Planar/convex partitions; certified zero-contact and smooth-branch faces; private wholly-interior and single-plane partially active reentrant edge/face local integral and tangent; bounded adaptive interior-ball cover; whole-face smooth numerical-touch admission | General active boundaries and feature sets; global signed-gap Step admission; other edge/point laws |
| 4. Normal law | Penalty Step; opt-in ordinary serial P1 fixed/translating-plane AL Step, coalesced nodal multiplier, outer Newton lifecycle, joint restart and accepted tool/relative-gap work; bounded disjoint-normal-row MPC admission with exact active-constraint reference and cyclic restart; real-mesh/reduced-rank counterexamples retained | General multiplier space and transfer, overlapping-row MPC/curved/rotating admission, spatial stability and wider-path work accuracy |
| 5. Friction | Locally planar Coulomb State; rigid-master spin transport tested; a separate smooth point tool can slide beside a frictionless certified-face tool | Admitted geometry-by-friction matrix; friction on sharp features remains rejected |
| 6. Transaction | Total-system and restart tests for admitted penalty modes; opt-in serial standard-J2 certified-face research Step with work, plastic-history rollback/restart and two named tools; smooth-subset affine/MPC total-system path, now including a prescribed moving face; certified-face plus point pair in serial standard J2 and smooth MPC, including separate point-friction State | Sharp-feature MPC, MPI, changing feature topology and wider path tests |
| 7. Result | Pair work/pressure samples; constant-normal plane AL signed relative work, prescribed tool work, discrete absolute-work bound and raw/contact-aware residuals; standard and affine/MPC research face-location extensive evidence and accepted-station coverage without point pressure | Cross-law first-law, between-station coverage and a defensible pressure interpretation |
| 8. Science | Local/oracle/installed tests; independent elastic and cyclic confined plastic histories; separated inhomogeneous space/time refinement through 16 cells and 256 increments; moving-tool/restart screens; directional refinement through 128 slave faces | Full mesh x increment x law convergence and external observables |
| 9. Release | Scoped solver screens; remote Linux and local macOS installed-wheel checks; unchanged explicit full trajectory with 25 verified frames; closing `0913ab9a` installed regression | Final frozen release-candidate acceptance and bounded contact scientific gates; explicit runtime does not certify implicit contact |

“Partial” here is not a release status. The test target, geometry and
enforcement must all match before a gate can be marked closed.
The new `test_contact_face_science_research.py` compares an opt-in certified
closed-triangle face Step against the independently solved scalar equilibrium
`1000 log(1+u)/(1+u) + k(u+0.05) = 30` for a homogeneous slender column with
Poisson ratio zero and a strictly interior initial face. At `k=500`, the
two tetrahedral meshes and one/three increments all give
`u=0.00334447328517` versus scalar root `0.00334447328519` (relative error
about `5.7e-12`); `k=250/1000` also agree to better than `9e-12` relative.
Contact potential matches the independently integrated
`k A (u+0.05)^2/2`. Homogeneous deformation is represented exactly by
both meshes, so this is a useful same-input *spring-limit* test, **not** a
general mesh convergence study or an external finite-element code comparison.
The companion Poisson-ratio-0.3 clamped column is deliberately
inhomogeneous. At one/three increments and one/two/three cells across its
cross-section, the whole-face closed tool and independently assembled
analytical-plane backend agree in displacement and penalty energy to their
declared tolerances. Its closed-tool center displacement changes from
`0.00303999393425` (one cell) through `0.00319348675388` (two) to
`0.00324430311741` (three); the penalty energy changes from
`0.00703287986482` through `0.00707388762374` to `0.00708738984529`.
This is a directional refinement screen, not proof of the asymptotic limit.
The same unchanged nonuniform setup was also run manually at four and eight
cells per edge, the latter using exactly the declared 128 slave-face cap:
center displacements `0.00326938245543` and `0.00330647510339`, and penalty
energies `0.00709406772191` and `0.00710395068898`. The closed tool and
analytical plane still agree at the eight-cell station. Differences along
the 1/2/4/8 sequence decrease, but this one path has no independent
continuum-limit solution and does not cover a changing active set or a
plastic/damaged body; it does not close the release science gate.
Nine focused inhomogeneous column comparisons now vary one, two or three
cells per edge, one or three increments and penalties 250, 500 or 1000,
according to the explicit test matrix. The independently lowered analytical
plane and closed-triangle face agree in displacement and final penalty energy
at each tested station. At two cells and three increments, the closed-tool
center displacements for these penalties are respectively `0.01346611117`,
`0.00319348675` and `-0.00958666389`; penalty energies are respectively
`0.00503498730`, `0.00707388762` and `0.00816611425`. This *sensitivity*
is why a matched penalty must be frozen for any external comparison. Agreement
between two representations of the same penalty law is not a zero-penetration
limit or an independent finite-J2 validation.
An added *plastic* clamped-column screen keeps six load increments, penalty
500, Poisson ratio 0.3 and J2 yield stress 1 across 1/2/3/4 cells per edge.
The certified closed-face and separately lowered analytical-plane backends
agree in displacement, committed material vectors and penalty energy at all
four mesh sizes. Their closed-face maximum PEEQ values are about
0.04114/0.05617/0.06391/0.06384 and final contact potentials about
0.006190/0.005772/0.005874/0.005878. This is a useful *plastic path and
spatial sensitivity* screen, not an external code or continuum-limit proof.
The three-cell first increment required 34 Newton iterations; the previous
25-iteration cap rejects that input, so this screen explicitly declares a
100-iteration cap rather than concealing the extra solver work. A four-cell
three-increment attempt also failed its line search. Contact integration
agreement must not be mistaken for robust default solver convergence;
yield-onset and increment-policy sensitivity remain open Gate 8 evidence.
One additional two-cell/three-increment plastic path starts with a zero-gap
closed triangular tool, translates it into the body and compares every
accepted station against the separately lowered moving analytical plane.
Displacement, committed J2 state, contact potential and cumulative tool work
agree at the declared tolerances; tool work is nonzero. A checkpoint after the
first accepted increment resumes to the same final displacement, plastic
state, work and face-path evidence as the uninterrupted closed-tool run. This
tests joint moving-tool/material lifecycle for a smooth face, not a changing
edge/vertex feature or an external-code result.
In a separate three-cell trial, automatic cutback from 1/6 with the
25-iteration cap exhausted eight cutbacks at a line-search failure; simply
reducing the increment is therefore not a demonstrated remedy for this
near-yield path. With the same declared automatic controls but a
50-iteration cap, it completed in seven accepted increments; the highest
Newton count was 34. A fixed-old-State directional difference at the
converged first-increment displacement, probed on both sides of the
elastic/plastic return and restricted to free DOFs, matches the *total*
J2-plus-certified-face tangent. The initial diagnostic appeared inconsistent
only because it mistakenly perturbed strong Dirichlet DOFs while comparing a
BC-eliminated matrix; the corrected check is now a regression. This case
does not establish that the default 25-iteration globalization is robust.
[PETSc SNES][petsc-snes] offers several globalization rules rather than one
backtracking policy; [deal.II step-42][dealii-step42] couples plastic-contact
active sets with a safeguarded Newton path, while [MOOSE's radial-return
documentation][moose-return] distinguishes *material-point* substepping from
discarding whole global increments. None is a drop-in fix for this observed
case. The same three-cell/six-increment input now has a paired globalization
regression: the existing `basic` Newton choice completes with the declared
25-iteration cap, whereas `backtracking` needs a declared 50-iteration cap.
Their accepted displacement, committed plastic state, contact potential and
external work agree to the test tolerance. A second two-cell/three-increment
regression starts from exact zero gap and enters active contact; the same
accepted-state comparisons pass under both globalization choices. On the first
increment of the initially closed three-cell case, the
observed Newton counts were 10 and 34 respectively; a separate four-cell,
three-increment screen also completed its first increment in 11 iterations
with `basic` where backtracking had failed its line search. This identifies
strictly monotone backtracking as a *candidate* source of difficulty on this
near-yield path, not as a generally inferior algorithm. `basic` can accept
temporary residual growth, so the default and release claims remain unchanged.
The next gate is a frozen matrix of load paths, active-set transitions,
rollback/restart and negative controls before recommending any default
globalization change. Raising a cap or changing one line-search option on a
single case would conceal rather than resolve this gate.
The same benchmark also starts from zero gap and advances into active contact.
It permits a zero-force/open-side-tangent trial only when a whole-face
coplanar coverage/competitor proof succeeds and the remaining vertex gap is
within `8 epsilon * world_scale` with the closed projector reporting zero.
That numerical convention is recorded on the local certificate and refuses
gaps of `1e-8` in either direction; it is not an exact-arithmetic theorem,
and it rejects coordinates whose roundoff scale exceeds `1e-8` of the local
face size. The initial contact tangent selects the zero/open-side generalized
derivative at the penalty kink; every subsequent trial recertifies its branch.
These statements are limited to the opt-in research route,
and unproved rims or sharp transitions still fail closed.
For a face that has a *separate whole-face smooth-plane certificate*, the
two-feature research Operator now reuses the existing affine active-triangle
clipper. Its P1 normal gap is affine and its penalty potential is quadratic,
so the positive degree-two rule integrates the active reference area,
residual and fixed-branch tangent without a conic two-feature partition.
This is per-face dispatch from a fresh current-configuration proof, not a
cached contact projection or an assumption that every tool is planar. A
partly active face passes a re-partitioned potential-gradient and tangent
finite difference; the reentrant two-feature regressions still take their
original branch. A bounded immutable Gauss-rule cache also removes repeated
Legendre-node generation without changing the rule. The reproducible
`tools/benchmark_certified_face_step.py` alternates both certified routes on
the *same* two-cell, three-increment J2 model, checks equal displacement and
contact energy, and after two warmup pairs measured eight pairs on one
Apple Silicon FEniCSx host: medians `0.7324 s` (certified two-feature
reference) versus `0.3043 s` (smooth-face dispatch), about `2.41x` for this
**small research Step**. The ordinary analytical-plane Step on that model
remains faster; this is neither a production speed claim nor an estimate for
sharp-feature models, which correctly retain the more expensive rule.
The certified-face provider now keeps only its last immutable trial and
reuses it when the displacement, reference positions, rigid-motion factor,
pair/tool identities and numerical settings match exactly. A changed Newton
state or pose forces a new whole-face proof; the cache is neither checkpoint
State nor a contact-neighbor approximation. The paired
`tools/benchmark_certified_face_trial_cache.py` disables just this reuse in
its reference run. On one Apple Silicon FEniCSx host, three-increment serial
research Steps with 2/4/8 cells per edge had recompute/reuse median times
`0.306/0.167`, `1.215/0.674` and `5.740/3.582` seconds respectively,
about `1.83x`, `1.80x` and `1.60x`. The 8-cell case reaches the declared
128-slave-face cap. Geometry-proof calls dropped from 32/34/35 to 13/15/16;
the paired script checks the same solution, committed material vectors,
contact energy, face path and work history. These are small bounded research
solves, **not** an Abaqus comparison or a production/full-geometry speed
guarantee. Other models may spend little time in redundant face proofs.
The reentrant-tool sample carries an engineering quality failure whenever a
sampled slave face crosses nearest features without a certified transition
integral. Release additionally requires a certified integral even if no
switch or active point happened to be sampled. Exploratory inspection remains
possible; an absence of sampled switches is not a whole-face certificate.
For a reviewed closed solid, a current first-order slave face can now instead
be certified **strictly separated** by an adaptive 1-Lipschitz signed-distance
bound. A rigid-pose-aware complete AABB separation is checked first, so
obviously distant faces require no nearest-feature/winding point query. That
proves its normal contact integral is exactly zero at that accepted
configuration. A manufactured counterexample with three apparently separated
ordinary Gauss points and a penetrating face corner is correctly *not*
certified. An unresolved budget, ambiguous projection,
or any near/active contact leaves the integral uncertified. This is not a
between-increment path-intersection proof and does not admit active nonconvex
contact, friction, or lower-dimensional edge/point laws. Its reference
geometric cost also is not a large-CAD performance guarantee.
There is now a separate, deliberately stricter **active** subset: for a
reviewed closed tool, each current affine slave triangle must project
inside a completely covered coplanar master patch; its nearest-plane distance must
beat every noncoplanar master facet by more than twice the slave-face radius
(plus numerical margin). The difference of two distance-to-set functions is
2-Lipschitz, so this inequality certifies the closest feature over the
*whole* face, not just at Gauss points. Distinct slave faces may select
distinct certified master planes, but one face cannot cross a sharp feature.
The patch check subtracts the union of coplanar master triangles from the
projected slave polygon with a bounded halfspace clipper; an overlapping
master-patch area sum is not accepted as coverage. It admits a flat
triangulation seam but rejects a positive-area gap or proof-budget exhaustion.
This is a bounded floating-point coverage certificate, not an exact-arithmetic
CAD topology proof. It follows the classic reentrant polygon-clipping idea
of [Sutherland--Hodgman][polygon-clipping], adapted to subtract a triangle
union rather than to integrate a second contact surface.
The current Newton trial may then reuse the already proved analytical-plane
active partition, with the closed solid's own signed-gap projection checked
against each plane. Geometry proof is
repeated on each trial and rigid pose. A real serial finite-J2 solve has
matched the corresponding analytical plane in displacement, PEEQ and tool
work, with a contact directional tangent, interrupted restart and explicit
reentrant-edge rejection; the small runnable example exports one XDMF grid
and unsmoothed contact samples. This does **not** solve a feature switch or
physical edge/vertex contact: those remain rejected. The current per-face
smooth-branch and 1024-slave-face/4096-master-facet bounds are intentionally narrow,
and this is not an external commercial-solver comparator.
The closed-solid projector now takes the reviewed facet normal directly on a
smooth face, including a coplanar seam. Dividing a nearly zero
point-to-foot vector by the gap previously amplified tangential roundoff and
made strict projection/plane comparisons fail near closure. The geometric
edge/vertex branch and the signed-gap classifier are unchanged. A near-gap
two-sided normal regression and a serial finite-J2 seam-crossing Step exercise
this numerical correction.
The whole-face smooth-plane proof has now also been exercised on a reviewed
**open** oriented triangle sheet through a distinct
`reviewed_planar_branch_partition` mode. It uses the same complete coplanar
coverage and noncoplanar competitor bound, but its gap is the sheet's
one-sided oriented gap, not the closed solid's signed inside/outside distance.
Serial finite-J2 tests compare displacement and tool work with an analytical
plane through a coplanar seam, moving rotation, two simultaneously active
named tools and interrupted restart. A face extending beyond the sheet's rim
is rejected. A periodic-MPC J2 solve also checks the total reduced residual
against the analytical plane. This advances the reviewed-surface slice of
Gate 2; a true edge/rim law and general active nonconvex feature switch remain
open Gate 3 work.
The ordinary serial Step now also accepts an exact-name mapping from rigid
contact pairs to their already admitted integration modes. A missing or
unknown pair name is an early capability rejection, never a fallback to
sampled Gauss integration. Each lowered pair retains its own mode in
checkpoint identity; a two-tool mixed-mode finite-J2 path and interrupted
restart exercise this composition. A separate two-active-tool regression now
checks independently integrated pair forces, positive per-tool work, total
work sum and joint restart while one tool uses the certified closed smooth
branch and the other uses an analytical planar partition. A small rigid
rotation of the closed tool also matches its equivalent analytical plane
in a real finite-J2 Step. The serial periodic-MPC J2 route now admits this
*one named scalar mode* on a wide reviewed smooth patch, with total-residual
reduction, analytical-plane limit, interrupted restart and explicit
reentrant-edge refusal tested. The affine route now also uses the same
exact-name per-pair mode contract: one closed smooth branch and one analytical
plane can enter a *single* total MPC system, survive interrupted restart and
refuse a changed mode identity. This remains serial and does not admit a
sharp-feature transition merely because the second tool is supported.
The same-plane proof now uses the immutable triangle BVH to *discover* only
facet boxes that can intersect the projected slave polygon or lie within
`abs(gap_at_center) + 2 * slave_radius + numerical_margin` of its center.
The 2-Lipschitz exclusion inequality and exact closest-triangle narrow phase
are unchanged; a rejected box is a lower bound, never an assumed old contact
state. A paired, alternating eight-repeat serial component screen on one
Apple Silicon FEniCSx host, with 1024 small slave faces, measured median
certificate times of 0.203/0.203 s (old/new, 1728 master facets) and
0.299/0.241 s (old/new, 3888 master facets). The latter is about 1.24x for
this **geometry-proof component** near its declared 4096-facet cap. It does
not measure full finite-J2 solve speed, improve the cap, or establish
portability to another mesh or computer. The broad-phase candidate lists
are also compared against exhaustive facet boxes in a deterministic test.
For a nearby facet that fails the center/`2r` exclusion, the proof now has
a second conservative route: the other facet's *infinite plane* is no
farther than its finite triangle. The difference between its squared plane
distance and the selected plane's squared distance is a quadratic over a
P1 slave face. Six degree-two Bernstein coefficients bound that difference
over the whole face. A strictly positive lower bound excludes the competitor
without subdivision; an inconclusive bound retains the old subdivision or
fail-closed result. One open-sheet and one closed L-wall fixture that
previously needed refinement now pass with refinement depth zero, while a
manufactured truly competing facet still rejects. A three-vertex-only
distance test is separately shown to miss an interior competitor. This is
still a single-smooth-branch *certificate*, not a conic feature-switch
integral. For a finite competing triangle whose nearest-point KKT conditions
place the *whole* slave face in one edge or vertex Voronoi region, the exact
squared distance to that fixed feature is also quadratic. A second six-value
bound may therefore exclude that finite competitor even when its infinite
plane intersects the slave face. If the Voronoi region changes or the bound
is within the numerical safety margin, the old subdivision/fail-closed route
remains. This admits a formerly rejected near-reentrant face that is actually
on one smooth plane; a face truly crossing the reentrant feature is still
rejected. The near-edge numerical-budget fixture also still refuses an
unproved branch. Edge/vertex **competition** here is not an active edge or
vertex contact law. A six-increment serial finite-J2 `model.step` now also
exercises a reviewed open sheet with this finite-vertex competitor: the
whole-face certificate passes at refinement depth zero, disabling only the
finite-feature bound restores the old coarse refusal, and the accepted
displacement and tool work match an analytical plane. This checks the
Model-to-Result route for this smooth subset, not a transition law or an
external solver comparison.

The reproducible paired component screen
`tools/benchmark_planar_branch_certificate.py` alternates the original
subdivision route (both quadratic shortcuts disabled) and the new bounds at
the same maximum refinement depth. On one Apple Silicon host, 200 measured
pairs after 20 warmup pairs gave `0.911/0.290 ms` medians for the infinite
plane competitor and `1.649/0.309 ms` for the finite-vertex competitor
(reference/new). Both modes certify the same plane. The ratios, about
`3.1x` and `5.3x`, apply **only to these individual geometry proofs**,
not the full nonlinear solve. Other geometry can incur the six-value checks
and show no gain; no full-solve speed improvement is claimed.
An **independent smooth-geometry branch** is now under bounded evaluation.
Even with an exact analytical sphere/cylinder projector and curvature tangent,
three fixed Gauss points on a P1 slave face can all be open while a small
positive-area circular contact patch is active. The opt-in
`radial_active_partition` derives the active region from the radial
distance-squared quadratic, splits the reference triangle at conic tangency
and edge events, and uses positive 12/24-order quadrature. It screens the
per-face potential, nine nodal residual components and 81 tangent entries
against each other; inconclusive integration, an axis/center singularity, a
bounded-cylinder rim or a point budget refuses the trial. The paired-order
screen is **not a rigorous error bound** and cannot license general curved
contact. A declared initial-penetration limit now uses exact affine-face
radial extrema (closest point to the center/axis and maximum vertex radius),
not sampled points; the 0.001/0.0009 sphere-overlap regression catches a
patch that ordinary Gauss sampling missed. Independent circular-patch area,
potential, force and tangent
formulas at three heights and full-nodal finite differences check the local
operator. A vertex-centered quarter-circle, a patch clipped by a slave edge,
rigid-frame and length-scale changes, and a real-Step directional tangent
exercise the less convenient geometries. Real serial finite-J2 Steps
demonstrate both a sphere patch and a
cylinder strip missed entirely by ordinary three-point Gauss; prescribed
sphere motion and interrupted checkpoint/restart also pass. This is an
analytical smooth P1/frictionless/scalar-penalty subset, not a sharp-feature
solution, mesh-converged hard-contact result or external solver comparator.
In the one-cell sphere regression the opt-in mode initially has 1152
positive-weight active points and finishes with maximum displacement
`1.75747e-5` and contact potential `4.94207e-7`; the otherwise identical
three-point Gauss mode reports zero active points, displacement and contact
potential. Those values expose a sampling failure in this manufactured
geometry, not an accuracy/performance benchmark for a production model.
The conic split now lives in a geometry-neutral
`contact_quadratic_region` primitive. On the independent research oracle
where a finite edge and a vertex exchange nearest-feature ownership along
a **parabola**, its two positive branch integrals reproduce the oracle's
potential, nodal residual and within-branch tangent at zero and nonzero
in-plane nodal displacement. This validates the region integration building
block only. A separate positive coarea rule on the quadratic equality
boundary reproduces the oracle's **moving-boundary tangent** at those two
states; its circular special case also matches the exact derivative of disk
area. A vertical reference-coordinate boundary is handled by transposing the
triangle coordinates, with an independent moving-area derivative test; a
component singular in both orientations is still refused. The physical
feature-specific residual jump and branch admissibility
are still supplied by the oracle. The *total* tangent needs this term in
addition to the two bulk tangents; no general sharp-feature Step is admitted
by these geometric primitives.
This division follows a useful but limited literature lesson: [Farah--Popp--
Wall's segment/element mortar comparison][mortar-integration] treats contact
zone boundaries and integration cost together, rather than assuming that more
fixed Gauss points always resolve a moving interface. It does not validate
our non-mortar P1 penalty rule. More importantly, [Konyukhov--Schweizerhof's
closest-point analysis][closest-point-domain] shows why a general nonsmooth
surface needs explicit edge/corner projection domains and uniqueness checks.
Our conic integral handles a *given* feature division, not the proof that the
two supplied physical branches exhaust a real tool's nearest features.
A private, solver-neutral `contact_feature_pair_research` operator now
combines that division with the complete bulk-plus-moving-boundary tangent
for the squared distance to the union of two **infinite affine** features.
At zero and perturbed nodal displacement it matches the independent
parabolic edge/vertex oracle; all six nodal potential derivatives and force
derivatives agree with re-partitioned finite differences, and a three-
dimensional rigid-frame test checks covariance. The distinct-feature
separator has no artificial boundary force, and an exact tie is rejected.
All six triangle-connectivity permutations preserve the reordered residual
and tangent. The integral's translational resultant and moment about the
origin are checked against finite differences of a rigidly translated and
rotated pair of features, giving a local virtual-work identity; this is not
yet an accepted *path-work* ledger for a moving tool.
This is deliberately *not* a unilateral contact law: finite edge/face
projection domains, closest-feature exhaustiveness, inside/outside sign,
initial overlap, physical pressure/work and accepted-state lifecycle are not
conferred by an accurate squared-distance integral. Those are the next
admission tests before any sharp-feature Step can use this operator.
One of those checks now has a private, deliberately bounded implementation:
`contact_feature_pair_geometry_research` takes a finite triangle union and
certifies that each of two selected triangles keeps the *same* face, edge or
vertex closest-point branch over the whole current P1 slave face. Its
strict KKT inequalities are affine on that face, so checking all three
vertices proves the branch region; distance to each fixed affine feature is
convex, so its largest vertex distance bounds the selected pair from above.
Every other finite master triangle is excluded over the whole slave face by
its 1-Lipschitz center/radius lower bound, a common finite feature, a positive
quadratic Bernstein bound, or an affine-feature dominance inequality over
both convex hulls; otherwise the proof refuses. A separate
brute-force nearest-triangle sample grid corroborates an edge/vertex fixture;
an intervening third facet, a changing selected branch, degenerate face and
unresolvable world-coordinate precision are rejected. This certifies the
*unsigned distance to a finite triangle union* in a tiny selected-facet
subset, not the sign of a closed tool or the contact traction. A reviewed
surface can have many competing features; forcing its geometry into two
selected facets would be unsound. This is a bounded floating-point margin
check, not an interval-arithmetic or exact-CAD topology certificate.
For a separately attested closed, outward-oriented tool, a winding-classified
interior point and a boundary-distance ball now also certify that the *whole*
slave face is inside at the current configuration. Composed with the two-
feature certificate, the private local Operator has an actual frictionless
reference-area penalty interpretation: on a wholly penetrated face,
`g=-distance` and `k*g*dg/dx` is the derivative of `k*distance²/2`.
A four-facet closed tetrahedron test exercises this face/face switching law
against an independent exact linear-clip integral of `min(x,y)²`, while a
face reaching the boundary refuses the interior-ball proof. This is a
**narrow local physical subset**, not a normal project Step or a finite-slip,
restart or accepted-path-work validation.
The conservative competitor check now also handles an additional facet
whose fixed finite feature is exactly the same shared edge, or whose entire
fixed-feature squared-distance difference has a positive quadratic
Bernstein bound. A reviewed **concave L-prism** thereby admits one small,
wholly interior slave face where its bottom face and true reentrant vertical
edge exchange closest-feature ownership. Independent closest-point queries
match the two-feature distance over a deterministic reference grid; all nine
nodal potential derivatives and 81 tangent entries match re-certified,
re-partitioned finite differences, including the nonzero moving-boundary
term. The tool is attested embedded, with reference winding sign. This is
local evidence for one nonconvex reentrant transition in a deliberately
deep-penetration subset. A nearby face crosses the L-prism bottom boundary:
its exterior portion is inactive, and the finite vertical reentrant edge
ends at a bottom **vertex** there. The wholly-inside proof still refuses it.
A second *private* path now clips the P1 reference face by one selected,
transversely crossed tool-facet plane. It certifies constant opposite winding
signs on the two resulting convex polygons by separating every other tool
facet plane from their interiors. The finite feature pair is proved nearest
only on the active polygon; a selected finite edge may end on the cut only
when its endpoint lies on that same plane. The other facets are excluded by
the bounds above, including affine-feature dominance for the adjacent wall.
The private discovery helper proposes candidate facet identities from current
nearest-point queries but returns an integral only after the complete polygon
and competitor proof. It does not require a user-labelled special pair of
triangle IDs; a missed proposal fails closed rather than falling back to
ordinary sampled integration. A small facet budget and a second crossing
tool boundary are explicit rejection tests.
The first (activation-face) distance and force vanish on the cut, so its
moving activity boundary has no distributional force/tangent contribution;
the *different* moving nearest-feature boundary still has its coarea term.
The partially active L-prism test checks the winding sign over a grid,
re-clips and re-certifies at every finite-difference perturbation, and matches
all nine nodal energy gradients and 81 tangent entries. Local tool force
matches virtual work for all three small translations; the vertical-axis
tool moment and both tilted-axis moments match small rigid rotations after
fresh geometry proofs. Exact shared vertices that define a rotated master
edge remain shared despite floating projector roundoff; unrelated near edges
are not merged. This does **not** yet cover an arbitrary tool surface crossing,
multiple activity planes or general large sliding.
The local energy derivative is a *structural residual*: the force on the
structure is its negative, while the reaction on the tool is its positive.
The current global `ContactTraceAssembly` uses the former for prescribed
tool-work evidence. This sign mapping is explicit in the private response
and must be preserved before any Operator/Procedure admission; a local
virtual-work test is not an accepted path-work ledger.
`contact_face_operator_research` now aggregates separately certified partial
and wholly active faces into shared global nodal rows and a COO tangent,
without pretending that integrated *face* identities are quadrature-point
samples. A two-face patch with shared nodes matches the gradient of its total
potential and the derivative of its total force; an illustrative MPC matrix
also satisfies `Tᵀ K T = d(Tᵀ R)/dq`. Distinct tool fingerprints, duplicate
physical faces, noninteger identities and raw uncertified distance integrals
are rejected. Each local certificate now carries immutable reference and
current face vertices; the bridge refuses a stale Newton trial or a response
swapped onto another mesh face. Face IDs, partial/whole status, finite feature
kinds, reference-active areas and energies remain per-face evidence; none is
renamed as a quadrature sample. This is an **Operator bridge**, not a global contact provider:
it does not own Newton trials, accepted states, tool path work, restart,
field export or a serial/MPI admission contract.
The same bridge now accepts a third, exactly zero face outcome only after the
existing strict whole-face separation proof. It records that face's positive
gap lower bound; a failed or budget-exhausted proof is never treated as
inactive. A batch mixing partial, whole and separated faces retains the
complete face list while allocating tangent entries only for active faces;
an all-separated batch has exactly zero residual, potential and tangent.
Under a common arbitrary rigid change of spatial frame, the two-active-face
patch's potential is invariant and residual, tangent action and moment about
a transformed reference point rotate covariantly. These are *fixed-trial*
operator tests, not a finite-strain material objectivity claim.
The DOLFINx bridge recovers complete physical face IDs and node triples from
the existing default positive CG1 tetrahedral contact trace, checking all
three point IDs, shape weights and reference area before discarding the
quadrature representation. A real unit-cube DOLFINx boundary lowered to two
faces and a distant closed tool gave an exactly zero two-face Operator. It
rejects a missing point, lumped rule or wrong area rather than making up face
coverage. This is an input-identity test, not a loaded global contact solve.
One bounded dispatcher now proposes the three proved face outcomes without
user-selected master facet numbers. Its nearest-point samples only rank
candidates; a complete geometry proof remains mandatory for every accepted
face. A short reentrant translation path exposed a missed *proposal*, not a
failed physical certificate: sparse points missed the second finite feature
on three partly active states. The dispatcher now tries sampled candidates
first and then the remaining pairs within its declared 64-master-facet cap;
the full finite-feature/sign proof still gates every acceptance. That path
now traverses separated, partially active and wholly active face states, and
each accepted response agrees with the corresponding independently selected
two-feature proof. This exhaustive fallback may be expensive on an
unsupported mesh and is not a production search-complexity claim.
Re-evaluating a mixed partial/whole/separated patch after a trial nodal
perturbation succeeds with fresh certificates. A face crossing two tool
activation planes is refused. This avoids case-owned facet-ID glue while
keeping the present small-facet-budget and one-cut limitations visible.
The pure complete-boundary trial function composes that dispatcher with
verified face extraction and additive Operator assembly. If any face fails,
it returns no partial Operator and reports its physical `slave_face_id`.
A real DOLFINx tetrahedron with the two reviewed reentrant-tool boundary
faces lowered through this path and reproduced the independently assembled
local contact potential; the active faces were classified partial and whole.
The companion distant-tool DOLFINx test classified both selected faces as
separated. Neither test solves a global J2 equilibrium problem.
For each finite-facet competitor, a vectorized slave/master AABB distance
lower bound is now checked before the exact center-to-triangle projection.
The minimum distance of the two boxes bounds every slave/master point pair
from below, while the chosen affine feature's distance over a convex P1 face
is bounded above by its maximum at face vertices. A strict gap between those
bounds excludes that competitor without an exact query; inconclusive boxes
retain the existing KKT/Bernstein proof or refusal. The far-competitor and
near-true-competitor tests distinguish safe pruning from false acceptance.
No full-solve speedup is inferred from this component change.
Fixed-body prescribed-tool paths now re-certify the mixed two-face patch at
each quadrature station. Translational, rotational and simultaneous
translation-plus-rotation generalized work integrate to the change in contact
potential; the simultaneous test takes the moment about the *moving* tool
reference. This confirms the local force/moment sign and reference-point
mapping. It is still not a path of accepted J2 equilibrium increments, so
neither a total-system first-law claim nor checkpoint/work promotion follows.
A real DOLFINx P1 tetrahedral elastic matrix and the certified two-face
contact Operator now also enter the same manufactured Newton residual and
tangent rows, with strongly fixed degrees of freedom. The converged free
displacements match the manufactured solution. This checks FE row ordering
and additive local linearization; it is not the finite-J2 Procedure, a
line-search/rollback test or evidence that the face law is admitted in
`model.step(...)`.
The existing point-sampled contact adapter and the private face-integrated
Operator now share one narrow serial PETSc insertion routine for already
computed nodal rows and COO tangent entries. It preserves the full tangent
for prescribed-correction lifting while omitting constrained rows and
columns from the inserted matrix; it rejects malformed/nonfinite inputs
before mutation. A real PETSc matrix/vector test checks the constrained
rows and free block against independent dense assembly, while 122 existing
serial finite-J2/contact Procedure regressions passed after the refactor.
This unifies the numerical insertion mechanism, **not** their evidence or
accepted-State contracts: the face Operator retains physical face IDs,
active reference areas and separate per-face extensive forces/moments,
without fabricating contact-point IDs or a smoothed pressure field.
For prescribed 3D tool motion, a separate private evaluator now maps the
current slave trial into the unchanged tool's reference frame, repeats the
complete face proof there and pushes nodal residuals, every 9-by-9 face
tangent block, resultant and moment back into the spatial frame. The trial
retains the actual motion factor and moving moment reference. Independent
reconstruction of the moved triangulated tool agrees at three pose stations;
finite differences, combined translation/rotation work and a large rigid
frame change give separate checks. Reusing immutable tool topology avoids a
per-trial mesh rebuild in principle, but no measured full-Step speedup or
accepted-state motion ledger is claimed yet.
A narrow private DOLFINx trial adapter now binds one named `RigidContactPair`,
its complete P1 boundary trace and current displacement to that pure face
Operator. It refuses friction, nonserial layouts and an initial-overlap limit
whose signed-distance face bound cannot be proved within the declared
refinement budget. It reuses the existing closed-solid 1-Lipschitz initial
gap certificate, rather than interpreting three Gauss samples as the maximum
initial penetration. A 0.1 limit is refused and a 0.5 limit is proved on the
partially active reentrant fixture. The adapter checks exact pair,
displacement and rigid-pose identities again before either PETSc insertion;
changing the Newton displacement or tool factor after evaluation causes an
addressable failure with no matrix/vector mutation. This creates a real
Model-asset-to-Operator-to-PETSc path.
The private adapter now also exposes a face-location diagnostics record and
the ordinary constraint-dual sign/resultant, including the moving tool's
moment about its declared reference point. A real DOLFINx trace checks the
fixed and moved duals and pose rollback; these are *trial observations*, not
an accepted-path work ledger or a point-pressure field. For the partially
active reentrant fixture, a separate physical-distance quadrature splits
the barycentric triangle at zero gap and closest-feature switches. Its
potential and nine nodal force components agree with the certified local
integral under independent 64/128-point convergence checks. This validates
that bounded fixture, not every possible reentrant corner or Step result.
The explicit `certified_closed_two_feature_research` mode now also lowers
**one** named pair into the existing serial standard finite-J2
`model.step(...)` Procedure. A small real tetrahedral two-face calculation
accepts two increments, emits face-location Result metadata (never invented
point pressure), and produces a nonzero displacement. Interrupted serial
checkpoint/restart reproduces the continuous solution and contact potential.
A prescribed translating tool accumulates accepted work and reproduces it
after restart; forced failure after energy recording atomically restores the
prior displacement, committed material, tool factor and work history. The
same checks have now been repeated with **nonzero accumulated J2 plastic
strain**, so material history is not inferred from an elastic-limit test.
At each accepted station the Procedure records whether every selected
physical face was certified, maximum active reference area and active/mixed
face counts. That accepted-station record survives serial restart and rolls
back on a failed increment; a corrupted station count is rejected before
checkpoint mutation. It does not prove collision-free motion *between*
accepted stations or provide a physical point pressure at a sharp feature.
The normal residual/tangent, material and strong-boundary rows share the
ordinary Newton path; no second solver was introduced. Two named certified-face tool
pairs now compose in that same Step, with separate face evidence, additive
nodal force and potential, moving-tool work and interrupted restart. A stale
pose on just one pair refuses the whole trial before PETSc mutation. One
certified-face pair can now compose with one ordinary frictionless point pair
in the same serial Step. The mixed wrapper first checks both displacement and
tool poses, then adds separate certified-face and point contributions to the
one Newton residual/tangent; a stale mixed trial is rejected before PETSc
mutation. A directional difference checks the *combined* contact tangent.
The result preserves distinct face and point diagnostics, two named duals and
work records; moving-tool work and interrupted restart match the continuous
run. A reviewed smooth point tool can additionally carry its own sliding
Coulomb State beside a *frictionless* certified face. A fixture with active
point contact has positive sliding dissipation; serial restart reproduces the
displacement and dissipation, and a forced late acceptance failure restores
plastic State, point-friction State, pose, work and both path ledgers. A damaged
point-ID friction checkpoint is rejected without altering the receiving Step.
Friction *on the certified face* and MPI explicitly refuse this research mode. These tests
establish a bounded *software lifecycle*, not mesh/penalty convergence, an independent global
reference, general finite sliding across sharp features, or product release.
The accepted point-sample exporter rejects a mixed face/point Step before
creating files: a point-only archive would silently omit the face-integrated
part of the contact law. Face and point diagnostics remain separately available
in the Step result until a face-location scientific archive is defined.

The affine/MPC Procedure now admits the same opt-in face mode for a serial,
single-facet **smooth** closed-tool subset. A larger P1 face that no single
interior ball covers can be proved by a bounded longest-edge subdivision:
every accepted child lies in a winding-classified interior ball under the
1-Lipschitz distance bound; an unresolved or boundary-touching child refuses
the entire face. Eight subdivision levels and a 512-child budget are explicit
research limits, not a guarantee for all mesh sizes. On a wide triangular
prism, the periodic finite-J2 Step reduces the complete contact/material
residual, its reduced contact tangent matches independent directional
differences, and its displacement agrees with the corresponding analytical
plane. Interrupted restart preserves the accepted face-path ledger; a
corrupted ledger is atomically refused. A front-face triangulation seam in a
different otherwise smooth tool still fails the stricter two-feature
certificate. A normalized prescribed translation of that same smooth face now
uses the affine Procedure's ordinary load-coordinate method, rather than a
test-owned motion loop. The fixture has nonzero committed J2 plastic strain;
its reduced total residual closes, moving-tool work
is nonzero, displacement agrees with a translated analytical plane, and
serial interruption/restart preserves the solution and work. This does not
admit finite-sliding sharp-feature MPC or an arbitrary rigid-motion schedule.
Two independently named moving smooth faces also reproduce the single-tool
solution when their penalties sum to the same value; per-tool accepted work
adds to the single-tool work and survives restart.
A second smooth-periodic test combines one certified closed
face with one analytical-plane point pair, verifies the total MPC-reduced
residual, agrees with the two-plane analytical representation and checks
interrupted restart. A distinct periodic test holds the certified face fixed
while a separate moving smooth plane slides with nonzero Coulomb dissipation;
its restart matches the continuous result. Thus this is **not** admission of
moving-sharp-tool MPC, general
coplanar seam coverage or arbitrary industrial mesh contact.

For promotion beyond this research Step, the boundary adapter must keep its
complete-face enumeration and current-Newton-trial proof for every selected
face. Any unsupported feature set, second activation boundary, ambiguity or
proof-budget exhaustion must still reject the trial atomically. The present
`ContactTraceAssembly` requires quadrature point IDs and pointwise
contributions; the new face-integral contribution therefore retains a
separate face-level evidence record and the existing force/work sign
convention. The standard Procedure now adds its total residual and tangent
before strong-BC lifting and recomputes at line-search trials, then jointly
commits or rolls back material, tool path and contact state. The DOLFINx
nonlinear callback contract requires exactly this current-trial assembly
[DOLFINx nonlinear assembly][dolfinx-nonlinear]. No face integral is disguised
as three sampled trace points. Sharp-feature MPC, friction *on* research faces,
MPI, between-station path certification, independent global
comparison and mesh/penalty/increment convergence remain open. The opt-in
research string must not be presented as a released general nonconvex solver;
local virtual work is not by itself whole-path first-law closure.
The same sphere fixture runs to completion on 1, 2 and 4 subdivisions per
cube edge (6, 48 and 384 tetrahedra). The maximum displacement is
`1.75747e-5`, `2.49116e-5`, `4.06294e-5`; the final contact potential is
`4.94207e-7`, `4.84893e-7`, `4.69293e-7`, respectively. These are a
**mesh-sensitivity warning**, not convergence or a recommended physical
penalty: the pointwise maximum displacement changes materially. The exported
accepted contact samples explicitly mark their IDs as snapshot-only and
their pressure as force per reference area, not Cauchy stress.
The initial-overlap bound and trial partition were also checked with the
same face scaled from `1e-9` to `1e3` length units. A former absolute
`max(1, radius)` roundoff allowance falsely rejected the small-unit initial
limit; it now scales with the analytical radius and coordinate precision.
Conversely, a unit-radius sphere translated to a `1e12` world-coordinate
offset is refused because the geometry cannot be resolved to the declared
integration tolerance in binary64. This is a precision gate, not an
automatic rescaling or alteration of user geometry.
The commercial [analytical-rigid-surface guidance][abaqus-rigid-smoothing] also
notes that curved analytical geometry can reduce contact noise/search cost,
while discontinuous master normals impair implicit finite sliding. That
motivates an *explicit model geometry choice*, never hidden rounding of the
user's specified sharp tool. The contact-boundary quadrature issue is also
documented in [Duong--De Lorenzis--Sauer's refined-boundary work][duong];
their isogeometric/mortar method motivates treating the active boundary,
but does not validate this separate P1 radial rule or its paired-order screen.
The runnable closed-tool example now has a separate one-axis-at-a-time
mesh/increment/penalty sensitivity driver. Its eleven serial synthetic runs
all completed with certified smooth branches, but their terminal tool work
remains mesh- and penalty-sensitive, while the pointwise maximum PEEQ is not
monotone under mesh refinement (details in
`evidence/contact/2026-10-19-closed-planar-branch-ladder.md`). This is
evidence **against** promoting the present tiny fixture as a converged
hard-contact or external-solver benchmark. The driver is a reproducible
diagnostic, not a release threshold or a performance promise.
Both ordinary J2 and affine/MPC acceptance now validate the same typed
polyhedral diagnostic contract: a missing mixed-feature count, a non-Boolean
integration claim or inconsistent active face/edge/vertex totals rejects the
accepted step and restores its previous State. A visually plausible but
incomplete observation cannot silently enter the historical evidence.

Ownership for the next implementation is deliberately narrow: surface/search
admission belongs under `boundary_models/`; feature integration and the
fixed-multiplier law are Operators there, not new solvers; the outer AL loop,
Newton/MPC ordering and atomic acceptance belong to the existing finite-J2
Procedure; accepted histories and checkpoint identity belong to State; force,
work, field-location and trust claims belong to Result/Verification. The
external comparator and convergence ladder are test/evidence assets, not
hidden runtime heuristics. This keeps the public Model focused on declaring
the contact pair and its physical/numerical policy.

## Nine ordered gates and reading map

Each bibliography entry motivates a design question; none certifies AgentFEM.
The gate stays open until its listed evidence exists at the proposed scope.

1. **Frozen contract and external comparator.** Freeze geometry, mesh family,
   units, initial clearance, rigid pose history, material parameters, solver
   controls and observables. Prefer a reproducible external Abaqus/Standard
   input/results pair supplied or reviewed independently; record contact
   formulation and enforcement differences. Predetermine observable-specific
   tolerances and a comparison station set. Current state: internal examples
   and local regression exist, but no same-input external benchmark for the
   nonconvex finite-J2 fixture. Reading: [Abaqus' published finite-sliding
   rigid-surface verification/input list][abaqus-verification],
   its published [C3D8 solid against analytical rigid cylinder input][abaqus-c3d8-input]
   and [finite-sliding C3D4/C3D8 surface-to-surface input list][abaqus-surface-verification],
   [its small-sliding stress/displacement verification cases, including
   augmented-Lagrangian examples][abaqus-al-verification],
   [COMSOL's analytical/NAFEMS verification collection][comsol-verification]
   and [NAFEMS advanced contact benchmark catalog][nafems]. A published
   verification case is a comparator protocol, not an external result for
   our own geometry. In particular, that published three-dimensional
   meshed-rigid case deforms a **shell** around a cylinder, so its numerical
   output cannot be relabeled as a solid finite-J2 same-input comparator.
   The C3D8 rigid-cylinder deck is an **elastic contact protocol**: it
   specifies an analytical cylinder, a solid contact face, normal loading
   and later sliding, but changes friction and has distinct linear and
   geometrically nonlinear steps. It supplies an input, not numerical output
   for our finite-J2 same-input case. The C3D4/C3D8 surface-to-surface blocks
   use two deformable bodies rather than our prescribed rigid tool. Thus
   neither source closes the external J2 gate; the elastic rigid-cylinder
   portion can serve as an earlier interoperability checkpoint with matched
   material, constraints and observables.
   A more ambitious later comparator is [NAFEMS Contact
   Benchmark 3][nafems-contact-3], which includes elastoplastic contact and
   springback; its detailed material, geometry and published observables must
   be obtained and frozen before claiming a comparison. Keep this separate
   from the smaller same-input AgentFEM/Abaqus-Standard J2 plane/convex-tool
   pair needed for the bounded 0.4.4 claim.
   Additional comparator screening rejects two attractive but mismatched
   shortcuts: the [public NAFEMS rounded-punch reproduction by SimScale][simscale-r0094]
   uses *linear elastic* punch and foundation, although it provides useful
   geometry, mesh and contact observables; the open [Frohne--Heister--Bangerth
   elastoplastic-contact study][frohne-contact] and deal.II step-42 provide an
   excellent semismooth/active-set algorithm reference but formulate the
   obstacle inequality for *small displacements*. Neither is the missing
   finite-strain J2 same-input result. NAFEMS also explicitly warns that mesh,
   element, contact and load-step choices must be matched before cross-code
   differences can be attributed to a solver. Until an independent deck and
   its accepted-station results are frozen, Gate 1 remains open rather than
   borrowing an unrelated published force curve.

2. **Surface and search admission.** Validate closed/open topology, outward
   orientation, self-intersection, initial clearance/overclosure and allowed
   free features; search and project in the current configuration on every
   trial. A local candidate cache may accelerate search only with a
   conservative coverage/invalidation proof, never by reusing old forces or
   normals. Test large sliding, seam crossing, geometry-scale changes and an
   intentionally missed-candidate trap. Current state: smooth projectors,
   triangulated BVH and reviewed closed-solid classifiers exist. Implicit
   rigid pairs now have an optional `initial_penetration_limit` in length
   units. On a first-order analytical-plane trace, the extremal nodal signed
   gap is an exact face-wise bound; this catches a corner penetration missed
   by quadrature. On a first-order triangular trace against a reviewed closed
   convex or nonconvex rigid solid, a bounded adaptive subdivision uses the
   true signed distance's 1-Lipschitz inequality to prove a conservative
   face-wise upper bound. It fails closed on ambiguous projections or if the
   declared depth/sample budget cannot prove the requested limit. For the
   convex subset, a support plane that
   dominates at every affine slave-face vertex dominates on the whole face;
   its nodal minimum directly bounds the initial penetration. Mixed-feature
   faces still use the adaptive certificate. A near-touching convex ordinary
   Step now passes a 1e-9 declared limit without refinement. This is only an
   initial-overlap result, not a later sliding/integration certificate. A
   reviewed **open** triangle sheet instead needs the explicitly selected
   `reviewed_planar_branch_partition`: complete coplanar coverage and
   nearest-branch dominance at the reference pose certify one oriented
   affine gap per slave face; its nodal minimum gives the initial limit.
   A rim/competing feature fails closed. The open sheet never inherits the
   closed-solid 1-Lipschitz signed-distance argument, and an arbitrary
   curved projector inherits neither proof. The
   current smooth-plane branch certificate has a separate conservative
   refinement: when the whole slave face fails only its coarse competitor
   radius test, it bisects the reference triangle once and admits the parent
   only if all four children independently certify the *same oriented tool
   plane*. This changes no quadrature or contact law. On the reviewed
   reentrant-tool fixture a wider face formerly rejected by the coarse
   `2r` screen is now certified with positive child clearance. The same
   test succeeds on an open triangle sheet with a nearby noncoplanar
   competing patch; moving its slave face nearer the competitor still fails.
   The corresponding six-increment global finite-J2 open-sheet Step completes
   and matches the otherwise identical analytical-plane displacement and
   prescribed-tool work; the unrefined whole-face certificate rejects its
   initial trace. This tests the whole Model-to-Result path, not only a
   standalone geometric helper.
   Disabling refinement reproduces the old refusal, and a true closed-solid
   edge-transition face remains rejected. The depth and total subface budget
   are bounded; child planes that differ never become an approximate
   sharp-feature rule. The
   ordinary Step capability query reports unsupported geometry/order before
   Step construction. The explicit pair still rejects this policy. No branch
   silently moves slave nodes or the tool. Arbitrary large-CAD and ambiguous
   sharp features are not admitted. Reading:
   [Abaqus finite-sliding/general-contact formulation][abaqus],
   [Abaqus meshed-rigid-surface verification][meshed-rigid] and
   [Abaqus initial-overclosure choices][initial]. The conservative subdivision
   follows the classical 1-Lipschitz signed-distance property, distinct from
   any particular neural approximation; see the distance-function discussion
   in [Cavalletti--Mondino][distance].

3. **Contact-boundary/feature integration.** Make the active integration
   partition a trial-geometry Operator input. A zero-gap boundary and a
   nearest-feature switch are different discontinuities: the latter can
   contribute a line term to the Newton tangent. Keep positive weights and
   one consistent residual/tangent partition; refresh a changed partition
   before accepting a Newton trial. Verify manufactured narrow-strip,
   two-plane and curved-transition oracles, then the nonconvex L-tool response
   matrix. Also distinguish a *surface quadrature crossing a nearest-feature
   switch* from actual edge--surface, edge--edge or point contact. A corrected
   surface integral alone does not establish the latter's nonpenetration
   constraints or pressure units. Either admit those lower-dimensional
   branches with independent force/moment and tangent evidence, or reject
   them explicitly at the current geometry; do not silently assign an
   arbitrary averaged surface normal. Abaqus/Standard itself distinguishes
   the bounded small-sliding surface formulation from finite-sliding general
   contact: its [general-contact controls][abaqus-general-controls] state
   that small-sliding assignments deactivate edge and vertex contact. This
   is evidence that a surface-only approximation needs an explicit scope,
   **not** evidence that our finite-sliding route may silently do the same.
   Farah--Wall--Popp use separate point,
   line and surface multiplier families for non-smooth finite-deformation
   contact [point-line-surface], whereas Farah--Popp--Wall compare segment,
   element and hybrid mortar integration schemes [mortar-integration]. These
   are architecture comparisons, not drop-in algorithms or evidence that
   our one-sided P1 operator already has their coverage. Haikal--Hjelmstad's
   oriented-volume formulation is a third non-smooth contact direction
   [oriented-volume], but is developed for quadrilateral/hexahedral body
   elements; it does not convert our tetrahedral slave and triangular rigid
   tool into a validated volume-contact operator by renaming the gap.
   [Laursen--Govindjee's non-smooth-surface treatment][laursen-corner]
   makes the physical-corner issue an explicit unilateral constraint problem,
   analogous to a non-smooth yield surface. Separately, a [3D mortar
   derivation][mortar-segmentation] tracks not only clipped integration cells
   but also the derivatives of their moving vertices. These clarify the
   implementation split: first certify which positive-area slave regions
   belong to face, edge or vertex gap branches; then vary both each branch
   integral and its moving region boundary. A feature-aware quadrature
   sampler alone is neither the corner law nor a consistent Newton tangent.
   Current state: analytical planar clipper, research two-plane oracle
   and opt-in <=4-plane convex partition pass; the result now flags *sampled*
   mixed-feature slave faces for reviewed closed solids in both standard and
   periodic J2 Results, without claiming a complete transition integral or
   changing the Newton law. Nonconvex feature
   transitions remain open. Reading: [Duong--De Lorenzis--Sauer refined-boundary
   quadrature][duong] and [COMSOL contact quadrature guidance][comsol-settings].
   The convex partition now admits only the actual default three-point P1
   reference rule and its facet-ID grouping. Three arbitrary points whose
   shape values merely sum to one are rejected before their measure can be
   silently reinterpreted as that rule; this is an input-contract safeguard,
   not a new geometry or friction capability.
   An independent four-feature rectangular-window integral now matches the
   exact quadratic penalty energy, including the reference-area map; its
   perturbed directional derivatives and a real FEniCSx assembled tangent
   agree with the moving pairwise-line tangent. A small finite-J2 Step also
   completes and resumes from an interrupted checkpoint with the same
   displacement and tool work. A five-feature prism still fails closed.
   This only widens the reviewed **convex** slice; it does not establish
   nonconvex nearest-feature switching or physical edge/vertex contact.
   The cited isogeometric extended-mortar formulation is a comparison design,
   not a drop-in P1 facet implementation: its enriched pressure interpolation
   and two-half-pass force construction differ from our one-sided trace.

   The next implementation must distinguish *sampling* from *consistent
   differentiation*. For a slave face partitioned into current nearest-feature
   regions `A_i(u)`, write the scalar contact functional as
   `Pi(u) = sum_i integral_{A_i(u)} phi_i(u,x) dA`. At a feature-switch line,
   the adjacent `phi_i` agree because the selected distance is continuous, so
   the first shape-boundary term cancels. Their displacement derivatives need
   not agree when the selected normals differ; differentiating the assembled
   residual then has a moving-line contribution. This is a derivation for our
   piecewise-smooth penalty law, not a formula imported from another element.
   There is an additional geometry boundary: on an affine P1 slave triangle,
   squared distance to an admissible master face interior, edge interior or
   vertex is a quadratic in the slave coordinates. Equality of two such
   branches is generally a **conic**, not the straight line used by the
   existing convex support-plane partition. The latter is exact because its
   selected *signed support gaps* are affine; it must not be renamed as a
   generic nonconvex nearest-feature partition. The next candidate needs
   either verified conic/feature-region segmentation with its moving-boundary
   tangent or an explicitly approximate, error-controlled alternative with
   separate release criteria. This conclusion follows from our branch
   formulas and the cited 3D segmentation/linearization work; it is not a
   claim that the cited method already handles our closed-solid sign policy.
   There is a directly relevant mathematical construction in
   [polynomial integration over a triangle clipped by a conic][conic-integral]:
   it subdivides until conic/triangle intersections have tractable cases and
   integrates polynomial quantities over the selected region. It is *not* a
   contact algorithm: its own floating-point discussion warns of delicate
   classification and cancellation, and it supplies neither our signed
   rigid-solid geometry nor the moving-feature derivative. A realistic
   bounded route is therefore BVH candidate localization, admissible
   face/edge/vertex quadratic branches, verified conic-region integration,
   moving-boundary linearization, and a conservative unsupported fallback.
   Adding high-order Gauss points globally is neither a geometric proof nor
   a substitute for this partition.
   The local oracle now also checks a cheap *preflight*, not an integrator:
   six values of a known quadratic branch difference recover its degree-two
   Bernstein coefficients on a P1 triangle. Their extrema bound the whole
   face, catching a circle-like conic hidden inside a triangle whose three
   vertices all report the opposite branch. This can reject unsafe
   vertex-only admission or certify a single branch after subdivision; it
   does not decide the sign of a closed solid, prove which face/edge/vertex
   is geometrically admissible, or provide interval-certified floating-point
   bounds by itself.
   The candidate must reproduce both the integral and its directional
   derivative and pass the total Newton/MPC tangent check. A fixed quadrature
   rule, a frozen feature partition or a smaller Newton residual alone is
   insufficient.
   A new independent *two-feature* research oracle makes the conic problem
   explicit: squared distance to a horizontal edge and to a separate vertex
   has equality curve `y=((x-3)^2+1)/2` on a P1 reference triangle. Its
   straight chord misses the curve by `0.027864` at the midpoint. Split
   positive Gauss integration gives the P1 residual; differentiating its
   moving parabola gives a negative semidefinite boundary term. The oracle
   now perturbs **all six in-plane P1 nodal freedoms**, including horizontal
   motion that changes the curve itself, at both zero and nonzero trial
   displacement. The full tangent's maximum error against independently
   re-partitioned central differences at the zero state is `4.0e-11` with
   that term, versus `0.1415` without it; the potential's independent first
   difference also matches the residual. At the nonzero trial the full
   tangent errors are `4.6e-11` versus `0.1412`, and the potential-gradient
   difference is `7.3e-11`
   (`tools/verify_quadratic_contact_feature_transition.py`). This is a
   local candidate-feature calculation, not a signed-gap law for a complete
   closed tool, an exact-arithmetic proof or a public Step.
   `tools/verify_contact_feature_transition.py` is already an independent
   two-plane unit-triangle counterexample: its moving-line term exactly
   cancels a nonzero fixed-branch bulk tangent. It proves why the term is
   necessary in that manufactured setting, not that the closed nonconvex
   three-dimensional implementation has been supplied.
   A second independent oracle, `tools/verify_reentrant_contact_integral.py`,
   computes signed distances directly to the six segments of an extruded
   L boundary, without calling the AgentFEM projector. One affine slave
   triangle has **all three ordinary face points outside** the rigid solid,
   so its sampled penalty potential is zero, yet positive-area material near
   its corners penetrates: 128/256/512 positive-subtriangle refinements give
   potentials 0.000348765/0.000349677/0.000349913 at unit penalty. The last
   two differ by about 0.07%. This is a numerical refinement reference, not
   an exact-arithmetic certified integral, and supplies a concrete target
   for the nonconvex feature-partition work rather than an excuse to turn on
   a sampled full Step as though it were complete.
   The independent oracle now additionally subdivides only unresolved
   subfaces. The signed distance's 1-Lipschitz bound encloses their energy;
   fully inactive subfaces contribute exactly zero, while fully interior
   subfaces with one provably nearest straight segment use an exact quadratic
   three-point integral. At depth 10 it processes 24,777 subtriangles and
   brackets this potential in `[0.000348447, 0.000351592]` (unit penalty,
   per unit extrusion), containing the 512-grid reference while excluding
   the default rule's zero. This is a floating-point conservative-margin
   **energy reference**, not formally verified interval arithmetic. It also
   bounds each component of the *total translation* force in
   `[-0.005760, -0.003644]`; a separate 512-grid central-difference screen
   gives x-force `-0.005333` inside that wide interval. Carrying the parent
   P1 shape functions through subdivision also gives componentwise bounds
   on the three reference-area nodal traction integrals; a separate 512-grid
   centroid estimate for the x components is approximately
   `[-0.0038923, -0.0001697, -0.0012706]`, all inside their corresponding
   intervals. A separate P1 nodal-direction difference of the fixed-area
   potential also falls inside all six component intervals. These broad
   reference bounds are not a production residual, pressure field or
   Newton-tangent certificate. The unresolved cells still require
   feature-region geometry and moving-boundary differentiation for Step use.
   A new opt-in
   `boundary_models.contact_energy_enclosure.bound_closed_contact_penalty_energy`
   transfers the 1-Lipschitz signed-distance argument from that independent
   2D oracle to any reviewed, closed 3D rigid tool and a current affine P1
   slave face. It splits the largest uncertain reference-area cell first,
   retains a scalar distance at nonunique nearest-point queries without
   inventing a normal, and falls back to a finite tool-vertex upper bound
   if even the winding sign is unresolved. The ordinary implicit contact
   provider exposes the read-only diagnostic as
   `closed_penalty_energy_enclosure()`, with explicit depth, cell and face
   budgets; it is never part of the default Newton residual. On the same
   extruded L fixture, the default three-point rule reports zero while the
   new generic depth-6 interval is approximately
   `[0.0001480, 0.0006654]`, containing the independent 512-subtriangle
   value `0.0003499`. At depth 8 with 49,999 leaves, it narrows to
   `[0.0002897, 0.0004174]` in about six seconds on the development Mac.
   Those are floating-point conservative-margin bounds, **not** formally
   directed-rounded intervals; neither a positive lower bound nor a tight
   energy interval supplies a nodal force or consistent tangent. A trial
   combination with exact smooth-cell proofs only reduced the depth-8 width
   about 4% while increasing runtime from about six to ten seconds, so it
   was not retained. The subdivision-only experiment also remains negative:
   after depth 6, 254 child faces were still unsupported by the existing
   complete local Operator proofs. Gate 3 therefore remains open.
   [Hiermeier--Wall--Popp's variationally consistent mortar study][variational-mortar]
   is a useful independent standard for varying *all* active contributions,
   but its symmetric two-body mortar formulation is not our algorithm.

   A 2026-10-18 fixed-input screen used
   `examples/finite_j2_closed_concave_contact.py` with 12 increments,
   penalty 20, direct linear solve, backtracking, one MPI rank and one
   OpenBLAS/OpenMP thread. The table reports the accepted end state, not a
   timing comparison. `q2` is the default positive face rule; `q5/q7` are
   explicit quadrature-degree requests.

   | Cells/axis | Rule | Tool force x | Active reference area | Max PEEQ | Tool work |
   | ---: | --- | ---: | ---: | ---: | ---: |
   | 2 | q2 | -0.307580 | 0.416667 | 0.00671873 | 0.00806207 |
   | 2 | q5 | -0.334865 | 0.447799 | 0.00670389 | 0.00895618 |
   | 2 | q7 | -0.336957 | 0.441456 | 0.00726967 | 0.00876881 |
   | 3 | q5 | -0.328318 | 0.445520 | 0.00812605 | 0.00888055 |
   | 3 | q7 | -0.319265 | 0.446202 | 0.00820966 | 0.00869530 |
   | 4 | q5 | -0.337238 | 0.428587 | 0.00841117 | 0.00874221 |
   | 4 | q7 | -0.320473 | 0.452688 | 0.00852799 | 0.00867533 |

   A further same-input q9/q11 check on the 2-cell mesh found tool-force x
   -0.336930/-0.321905 and active area 0.428296/0.440369; on the 3-cell
   mesh it found force -0.317957/-0.324226 and area 0.453882/0.450798.
   The higher positive Basix rules are now available as explicit study
   controls, not defaults. The q5/q7/q9/q11 force and active-area differences
   remain nonmonotone across this small mesh ladder. Completion, a small
   energy residual and a higher rule
   therefore do **not** establish a converged nonconvex result. Do not
   promote default q2 or q11 as a silent accuracy fix. The discriminating next
   experiment is a genuinely changing trial feature partition with verified
   residual/tangent, followed by a new mesh/increment/penalty matrix. This
   is a Gate 3 failure record, not a scientific benchmark result.
   An opt-in `DolfinxImplicitRigidContact.quadrature_sensitivity()` diagnostic
   now evaluates positive rules at the *same current* displacement and rigid
   pose, without another nonlinear solve or State mutation. On the accepted
   2-cell q9 end configuration the q9/q11 structure-force x values were
   -0.336930/-0.320336 and active reference areas 0.428296/0.440369. This
   isolates a roughly 5% force difference due to the rule at one fixed
   configuration; q11 is not a reference truth. It is frictionless and
   scalar-penalty-only, and reports
   `certifies_feature_transition_integral=false`. The next algorithmic
   candidate is trial-geometry segmentation or a variationally consistent
   mortar/feature treatment, with line-tangent and coverage tests.
   [Pinto Carvalho et al.'s rigid/deformable dual-mortar study][dual-mortar]
   motivates a distinct longer-term formulation, but its efficiency claims
   cannot be imported into this one-sided penalty implementation.
   A second opt-in fixed-configuration diagnostic now uniformly subdivides
   each affine P1 reference face into 4^level positive three-point subfaces.
   It preserves reference area and interpolation and is explicitly **not** a
   moving transition partition or line-tangent implementation. On that same
   2-cell accepted q9 state, level 3 (1,536 samples) gave potential
   0.008144 versus q9's 0.008140, but structure-force components
   (-0.334995, 0.010878, 0.020429) versus q9's
   (-0.336930, 0.016867, 0.012983). Close potential alone did not imply a
   reliable force vector. The level-0/1/2/3 active-area sequence was
   0.416667/0.437500/0.416667/0.430990, also not a convergence certificate.
   Level 4 (6,144 samples) changed the area again to 0.436523 and the
   force to (-0.325105, 0.017034, 0.024163), while potential remained near
   0.008136. Uniformly spending more points therefore did not close the
   feature-transition force question either.
   The production design now branches deliberately: a user-declared physical
   fillet/smooth tool can use the existing external `SmoothRigidSurface`
   extension and must pass the gap/Hessian checker and independent geometry
   comparison; the exact sharp nonconvex tool must wait for its own
   trial-geometry segmentation/line-tangent evidence. An implicit hidden
   normal average or silent fillet would change the model. This distinction
   follows [Abaqus' explicit advice to smooth analytical rigid corners for
   finite-sliding Standard contact][abaqus-rigid-smoothing], not an assumption
   that AgentFEM's present mesh already has those fillets.
   [Wolff--Bucher's assumed-gradient distance field][distance-field] is a
   different promising route for complex geometry, but its smooth interpolated
   gradient is a declared gap/geometry approximation. It cannot be advertised
   as the exact sharp-feature answer without an error study.
   A new algorithm may enter the Step only after both integral and
   differentiation checks, not merely a finer screenshot or work balance.
   The accepted closed-solid point diagnostic already distinguishes physical
   face, edge and vertex projections (not merely triangulation seams).
   Engineering verification now fails if a sampled *active* point belongs to
   a physical edge or vertex, even if no mixed-feature slave face was sampled;
   a corrected surface integral would not by itself define that separate
   lower-dimensional law. The diagnostic also rejects counts inconsistent
   with the active-point total. Exploratory results retain the observation
   without a trust upgrade; release still requires a certified integral. A
   real finite-J2 reentrant-edge Step now supplies the negative engineering
   test, not just synthetic metadata. Its Result exposes named face, edge and
   vertex active-point counts at the final accepted configuration beside the
   underlying per-pair metadata, so a user can inspect the reason without
   decoding a private trace. The runnable reentrant-tool example also prints
   the failed engineering feature claim beside `status=completed`; nonlinear
   completion is not silently presented as scientific acceptance. These final
   counts alone do not summarize earlier
   accepted increments. The ordinary serial finite-J2 Step now keeps a bounded
   per-pair path summary: maximum sampled mixed-feature faces and active
   physical edge and vertex points, plus whether every observed station used
   a certified transition integral. The initial and every accepted station
   contribute; failed increments roll back the summary and serial checkpoints
   restore it with count and pair-identity checks. Older checkpoints lacking
   it remain readable but mark the path incomplete. The summary is independent
   of output frames and the progress-event ring buffer, both of which may
   omit earlier states. It is still only *sampled* path evidence, not a
   whole-face coverage proof. The serial periodic/MPC path now records the
   same sampled maxima through its existing accepted-observer transaction:
   the initial and each accepted station are counted, late failures restore
   the prior record, and the identified recorder is checkpointed with the
   material and contact Operators. A damaged record is atomically refused.
   Older affine penalty checkpoints lacking the recorder load only through
   an explicit identity-compatible migration, marking historical coverage
   **incomplete**; subsequent observations cannot invent the missing past.
   An entirely observed flat convex path can now pass the narrow engineering
   claim of *no observed* uncertified feature transition, while its release
   integral-certificate claim still fails under plain Gauss quadrature.
   Thus a complete observation ledger is not a whole-face integration proof.
   For the separate *convex* route, accepted point diagnostics now say whether
   the reviewed changing-trial support-plane partition supplied its bounded
   feature integral. The default Gauss rule explicitly reports that no such
   certificate exists, even when no mixed-feature face happened to be sampled.
   Release-level verification now requires this certificate for every
   polyhedral pair; zero *observed* mixed-feature faces or active points no
   longer suffices. An empty or unattached accepted-path record cannot pass
   that claim vacuously; it must agree with current per-pair diagnostics. A
   future whole-face separation certificate may justify
   a separate no-contact release path, but point samples cannot.
   The narrow-strip tests in `test_contact_active_partition.py` and the
   missed-interior-overlap test in `test_initial_gap_bound.py` are concrete
   counterexamples to equating zero active quadrature points with separation.
   Engineering-level verification continues to report the narrower sampled
   diagnostic. This release guard does not itself supply a scientific reference
   or make the nonconvex operator accurate.

4. **Normal enforcement as an explicit law.** Retain penalty as one declared
   approximation. Prototype a separate opt-in augmented-Lagrangian law first
   on a fixed-quadrature analytical plane. With positive-outside gap `g`,
   fixed nonnegative multiplier `lambda`, and `k>0`, use
   `p=max(0,lambda-k*g)` during each Newton solve. Update the multiplier only
   after a converged solve; require separately specified global equilibrium,
   penetration and complementarity checks before the whole increment commits.
   Store stable point IDs, units, law and policy in State/checkpoint identity.
   Reject trial-changing contact quadrature until a conservative multiplier
   transfer is designed and tested. Current state: the ordinary Step remains
   penalty-only. A solver-neutral fixed-reference-trace AL candidate now has
   point-law, residual/tangent, spring-plane, independent scalar
   elastoplastic-bar indent/release and rollback tests. The scalar bar freezes
   old plastic history through each equilibrium solve, updates the normal
   multiplier only between converged solves, and commits both histories only
   at an accepted tool station; it is **not** an ordinary finite-element Step.
   A real serial DOLFINx tetrahedral trace also matches an independently
   UFL-differentiated
   plane/sphere functional, residual and tangent. There is still no global
   Step, MPI, or installed-package acceptance. A high-penalty L-tool trial
   failed Newton, so neither hard-contact convergence nor an AL benefit may
   be claimed. Reading: [Abaqus contact enforcement controls][acl],
   [COMSOL penalty/augmentation trade-off][comsol-method], and
   [Burman--Hansbo--Larson's augmented-Lagrangian FE analysis][burman]. The
   latter warns that a discontinuous multiplier approximation can need
   stabilization; [Hansbo--Rashid--Salomonsson's stabilized obstacle
   formulation][hansbo-stabilized] likewise explains that too many discrete
   multiplier constraints can make a system singular. Our independent point
   multipliers inherit neither proof and require a spatial-pressure/mesh
   check. A bounded opt-in fixed-trace rank diagnostic now builds the active
   weighted normal-constraint Jacobian and exposes duplicate point constraints:
   two active multipliers with the same nodal support and normal have row
   nullity one, whereas the three independent P1 triangle Gauss rows have
   full local row rank. This is a *negative screen*, not an inf-sup proof: it
   caps dense problem size and never certifies spatial stability. It can now
   also inspect `J T` when supplied the *actual* dense displacement reduction
   `u = T q + u_bar`; a three-point triangle that has full unreduced row rank
   drops to rank two when one normal DOF is fixed or tied by an MPC. This
   catches a different false positive than the duplicate-point screen, but
   the production Step still needs to pass its real reduction and a mesh
   ladder before using the result as an admission criterion.
   The global AL Step still needs a reduced-constraint rank/mesh-pressure
   study. The [exact-penalty alternative][exact-penalty] computes a continuous
   displacement-dependent multiplier estimate and avoids the outer AL loop;
   its published examples also show conditioning sensitivity, even before
   contact activates, and parameter-dependent step refinements. It does not
   repair our unresolved nonconvex integration or silently replace the
   declared AL policy. If compared, count its local multiplier-recovery and
   tangent work and use the same mesh, J2 path and acceptance observables.
   Nitsche is a serious *primal* alternative:
   [Chouly--Hild][nitsche-small] analyze small-strain unilateral contact
   without a separate multiplier or discrete inf-sup condition, and
   [Mlika--Renard--Chouly][nitsche-large] extend an unbiased formulation to
   large-deformation elastic frictional contact. Neither result is a proof for
   our finite-J2 material, one-sided rigid tool or sharp nonconvex integration.
   A fair method comparison would use the same analytical-plane J2 path,
   penetration, reaction, Newton work and mesh ladder, while counting the
   additional boundary traction and its consistent tangent that Nitsche
   requires. Do not add a second public enforcement choice on the basis of a
   1D penalty comparison alone.
   The [ASiMoV DOLFINx contact implementation][asimov-contact] is useful
   concrete evidence that custom contact assembly and parallel surface
   workflows can be built on the same FE ecosystem. Its maintainers describe
   it as experimental and it uses Nitsche formulations; it is therefore an
   architecture/comparator lead, **not** a drop-in backend for AgentFEM's
   finite-J2 penalty State, rigid-tool work ledger or feature certificates.

   A private *global* feasibility probe now substitutes the fixed-trace AL
   Operator into the existing serial J2 Newton assembly, leaving its
   material update provisional while manually iterating the multiplier. A
   moving planar tool compressed a plastic cube; with declared penalty 3000
   and penetration/complementarity tolerances 1e-6 it required eight total
   Newton equilibria before one joint **test-owned** commit, with final PEEQ
   above 0.005. At penalty 300, another case still missed the tolerance after
   30 equilibria. The repeatable regression locks only this bounded
   feasibility observation; it is not a public Step, a robust penalty policy,
   atomic checkpoint, or a general convergence proof. The slow outer
   convergence is consistent with the linear-rate cost identified in
   [exact-penalty] and motivates comparing a simultaneous mixed semismooth
   route such as [Alart--Curnier][alart-curnier], rather than automatically
   nesting dozens of solves in the product. PETSc's bound-constrained
   [variational-inequality Newton options][petsc-vi] are another algorithmic
   reference, but a bound on displacement DOFs does not directly represent
   a nonmatching rigid-surface gap or cure multiplier-space instability.
   The next *research* comparison should hold the same fixed trace and J2
   path while solving the Alart--Curnier-style semismooth mixed residual
   `R_u(u,lambda)=R_bulk(u)-B(u)^T[lambda-k g(u)]_+` and
   `R_lambda(u,lambda)=lambda-[lambda-k g(u)]_+`. On an active point the
   latter has displacement derivative `k B` and zero multiplier diagonal;
   on an inactive point it has unit multiplier diagonal. This exposes why a
   rank-deficient *reduced* `B T` can make a simultaneous solve singular,
   even when a scalar example converges in one Newton step. It is a design
   candidate, **not** the implemented global law; a mesh-stable multiplier
   space or reviewed stabilization, consistent geometry derivatives, and
   joint accepted State remain mandatory before a public option exists.
   Two open implementations sharpen the comparison: [GetFEM's contact
   bricks][getfem-contact] distinguish nodal rigid-obstacle and integral
   nonmatching-contact AL variants, while [deal.II step-42][dealii-step42]
   couples a primal-dual active set, plastic constitutive iteration and line
   search in a 3D obstacle example. The latter uses a small-strain law and
   nodal obstacle constraints, so it is an *algorithmic lifecycle* reference,
   not a same-input finite-strain J2/triangulated-tool validation. For our
   candidate, reduced-rank and active-set changes must be tested before
   comparing outer AL against a simultaneous mixed solve; neither package's
   documented convergence can be transferred to our multiplier space.

   A real DOLFINx P1 planar-trace rank screen now makes the spatial issue
   concrete. With 1/2/3 tetrahedral cells along each side, the boundary has
   4/9/16 normal displacement nodes but the three-point-per-triangle trace
   carries 6/24/54 independent point multipliers; its active Jacobian row
   nullity is 2/15/38. Reusing the existing positive vertex-lumped trace
   with **coalesced shared nodes** gives 4/9/16 multiplier points and zero
   row nullity in the same *unreduced* screens. Both discretizations can drive
   the private finite-J2 outer-loop feasibility probe to an accepted plastic
   state, but they give different PEEQ on that coarse mesh. Thus coalescing
   repairs this particular algebraic redundancy, not inf-sup stability,
   pressure accuracy, complementarity across faces or global product
   acceptance. The one-cell plastic feasibility fixture keeps the same
   rank result after its actual strong-boundary reduction; a mesh/pressure
   ladder remains a separate gate. This is
   consistent with the
   [dual-mortar multiplier-space analysis][dual-mortar-space] and the
   [reported boundary-contact inconsistency of unmodified dual bases][dual-mortar-boundary];
   those papers motivate the space design rather than validating our nodal
   lumping as an equivalent mortar method. An actual periodic-J2 fixture is
   stricter still: its 24 Gaussian point constraints have unreduced row rank
   22 and only rank 9 after the real 81-by-21 MPC transformation. The same
   surface's 9 coalesced nodal constraints have unreduced rank 9 but reduced
   rank only 5. Thus even the nodal candidate is **not** admitted under
   periodic MPC merely because its bare surface matrix is full row rank.
   The controlled all-active screen is an algebraic counterexample, not a
   physical pressure field: at the fixture's initial pose with zero accepted
   multiplier, just one point actually penetrates and its reduced active row
   has rank one. The all-active nullity therefore must not be reported as a
   failure of that particular physical increment. A multiplier basis
   compatible with the *actual* primal reduction and changing active
   inequalities remains research work.
   The bounded rank diagnostic now also reports active point IDs with a
   clearly negative gap but zero row in the *current reduced linearization*.
   In a deliberately shifted analytical-plane probe on that fixture, all
   nine nodal points are active, the reduced row rank is three, and four
   penetrated points have no first-order reduced displacement direction.
   This is useful preflight evidence for a proposed hard-contact law, not an
   automatic infeasibility claim: the probe does not establish that its
   current trial satisfies the affine MPC offset, and a curved gap could
   change at second order. No public AL Step is enabled by this diagnostic.

   A second research-only screen used the same finite-J2 Newton owner with
   coalesced nodal multipliers against an independent homogeneous hard-plane
   J2 reference. For one and two tetrahedra per side, the outer law required
   four equilibrium solves, reached penetration below 3.4e-7 for a prescribed
   0.002 indentation, and its tool force differed from the scalar reference
   by about 0.016%. This narrow positive result verifies a complete accepted
   increment on those meshes; it does **not** validate arbitrary pressure
   patterns, dual-space stability, changing contact sets, or a public AL Step.

   The homogeneous hard-plane research probe now covers 1, 2, 3, 4 and 6
   tetrahedral cells per axis. Each required four frozen-multiplier global
   equilibria; the accepted tool-force magnitudes stayed within
   `0.8147538507–0.8147538544` against the independent homogeneous target
   `0.8148807630`, and PEEQ stayed near `0.001188541` on this prescribed
   uniaxial path. This is an unusually *easy* spatial case: the exact field
   is homogeneous, so nearly identical mesh results do not establish
   nonuniform contact-pressure stability, feature-switch convergence or a
   public AL Procedure. The trace and test-owned state remain research only.

   A second, deliberately nonuniform oblique-plane J2 screen now runs the
   same test-owned nodal AL outer loop on 2/3/4/6 tetrahedra per axis. All
   four reach their declared penetration/complementarity tests, retain
   nonzero plastic history, have an independent active-row Jacobian after
   the *actual strong-BC reduction*, and show higher mean pressure where
   the prescribed tool compression is larger. This is a more discriminating
   path than homogeneous compression: the frozen 2/3/4/6-cell screen took
   6/7/8/9 frozen-multiplier equilibria, and tool-force magnitudes were
   2.712/2.711/2.719/2.725. Yet maximum PEEQ increased from 0.00956 to
   0.01054, while the far-side nodal mean pressure fell from 0.246 to zero.
   These are **not** a monotone field-convergence or spatial-stability proof;
   there is no independent same-input contact benchmark. It
   therefore screens feasibility only; it does not promote the multiplier
   space or enable the public AL Step. In line with
   [Burman--Hansbo--Larson][burman] and
   [Hansbo--Rashid--Salomonsson][hansbo-stabilized], row rank at four finite
   meshes cannot replace a mesh-uniform inf-sup or reviewed stabilization
   argument.
   [MOOSE's mortar contact design][moose-weighted-gap] makes the missing
   contract more explicit: a multiplier couples to a *weighted gap* and its
   basis/normalization, not merely to a convenient list of Gauss-point
   pressures. Its documentation even distinguishes stored scaled multipliers
   from physical pressure and rejects incompatible restart settings. Its
   [Petrov--Galerkin note][moose-petrov] reports that a dual multiplier basis
   may produce a negative weighted nodal gap; changing the test space is a
   deliberate method choice. These are reasons **not** to relabel our
   coalesced nodal feasibility probe as a stable mortar method. The next
   implementation should specify the primal/dual pair, coverage treatment,
   pressure recovery and restart identity together, then test a nonuniform
   contact-pressure field under mesh refinement and MPC reduction.

   On the same 2-cell/axis, 12-increment, q7 L-prism diagnostic above,
   increasing the *penalty* from 20 to 200, 1,000, 10,000 and 100,000
   completed the nonlinear solves but changed the final active reference
   area from 0.4415 to 0.4415, 0.4415, 0.1914 and 0.0967, respectively;
   maximum sampled penetration fell from 0.0470 to 0.0319, 0.0170, 0.00690
   and 0.00174. Tool-force x changed from -0.337 to -1.927, -3.487, -5.099
   and -5.705. This is **not** a hard-contact or penalty-independent limit:
   both the contact footprint and force change substantially. The older
   high-penalty Newton failure was a different trial and remains valid at its
   own inputs; neither failure nor completion generalizes to every setting.
   Adding an outer multiplier loop before fixing or bounding the nonconvex
   integral would hide, not resolve, that spatial uncertainty.

5. **Friction and moving-feature State.** Verify normal-pressure coupling,
   stick/slip return mapping, tangent, finite sliding, force/moment and
   dissipated work independently for each supported geometry. Reproject the
   tangential basis and carry accepted history across a facet seam without
   inventing a sharp-edge traction law. If an active noncoplanar edge or vertex
   is unsupported, reject and roll back. Current state: analytical and locally
   planar faceted Coulomb routes exist. Accepted tangential elastic slip now
   follows the rigid master's incremental rotation before the minimal
   old-normal-to-new-normal transport; exact co-rotation of a contacting plane
   has zero relative slip, unchanged recoverable energy and no spurious
   dissipation in both implicit and explicit FEniCSx checks. This fixes spin
   about an unchanged normal, which normal-only transport cannot detect.
   Sharp-edge friction is not admitted.
   Reading: [Abaqus Coulomb law and finite-sliding term][friction],
   [Abaqus' local tangential slip/traction convention][vfric],
   [COMSOL frictional-contact solver advice][comsol-solver], and
   [Duong--Sauer's surface-potential finite-sliding formulation][duong-friction].
   The last is a thermodynamic and tangent-design reference, not evidence
   for an unverified sharp-edge friction law.

6. **One nonlinear Procedure and State transaction.** Assemble the total
   material, contact and load residual/Jacobian before strong-boundary or
   periodic-MPC reduction. Newton, line search, cutback and convergence checks
   use that same total system. A rejected trial restores displacement, J2,
   normal multiplier, tangential State, projection, rigid pose and work;
   an accepted increment commits them atomically. Check interrupted versus
   continuous paths and corrupted-checkpoint rejection. Current state: these
   contracts pass for existing bounded penalty pairs; new enforcement and
   changing integration identities each need their own tests. Reading:
   [PETSc nonlinear line-search/trust-region methods][petsc-snes] and
   [COMSOL contact solver sequencing][comsol-solver].

7. **Results and mechanical evidence.** Publish per-pair raw gap, pressure,
   normal/tangential force, moment, active area and slip with field location,
   interpolation and units. Keep rigid prescribed-motion work, frictional
   dissipation, material work/storage and numerical penalty or augmented
   functional distinct. An algorithmic Lagrangian is not physical stored
   energy. Current state: named force/moment/work and raw contact-point output
   exist on selected routes; the AL research candidate separately exposes
   physical tool reaction and generalized tool force/moment. Prescribed-plane
   translation/rotation finite differences confirm their work-conjugate signs. A
   general first-law closure and cross-method pressure convention are not
   established. Reading: [Abaqus contact output
   semantics][abaqus-output] and [COMSOL important contact variables][comsol-vars].

8. **Independent science gates.** Run a small ladder: spring/plane exact
   solution; open/closed and unloading; smooth Hertz/contact footprint;
   planar sliding friction; convex corner; reviewed nonconvex transition; J2
   loading/unloading; MPC and moving multi-tool composition. For each relevant
   rung check finite-difference total tangent, action/reaction and moment,
   mesh x increment x enforcement sensitivity, regional plasticity, failure
   rollback and restart. Distinguish integration-point fields from smoothed
   display fields and compare equivalent observables with the frozen external
   case. The first independent global scalar oracle now checks a nearly
   elastic finite-J2 unit bar and rigid-plane penalty against the series
   compliance limit `F=delta/(1/E+1/k)` at two mesh levels and two penalties;
   all four satisfy 5e-4 relative tolerance for delta=1e-4. This small-strain
   limit checks the total force path, not plastic flow, curved geometry or
   Abaqus equivalence. A second independent scalar oracle now constrains the
   *plastic* global path: a laterally fixed unit bar has
   `F=diag(lambda,1,1)`, `q_trial=-E log(lambda)` in compression,
   `dp=max(0,(q_trial-Y)/(3G+H))`, and
   `Pxx=E(log(lambda)+dp)/lambda` for `nu=0`. Solving
   `Pxx+k(lambda-x_tool)=0` predicts right displacement, contact force and
   PEEQ at two tetrahedral mesh levels; the tests compare all three without
   calling the AgentFEM material update as their reference. This checks a
   restricted homogeneous J2/contact state, not general forming. The
   same independent scalar return map also follows an eight-increment
   indent--release--reindent path. It checks displacement, contact force and
   PEEQ at loaded, detached and reloaded stations; the accepted mid-path
   checkpoint resumes to the same displacement, plastic history and tool
   work as the uninterrupted run. Under lateral constraint, release can
   reverse-yield before axial force vanishes, so a first-load PEEQ cannot be
   substituted for residual displacement. This is a bounded transaction and
   material-path reference, not nonconvex tool validation. The audit also
   found that automatic J2 incrementation could skip an interior tabular
   reversal, thereby changing the prescribed path while reporting Newton
   convergence. Interior table knots are now required accepted stations:
   fixed controls missing one fail before solving; automatic controls land
   on each knot without forcing an extra output frame. Tabular inputs are
   copied and their metadata identity frozen at creation, closing the
   caller-array mutation gap between runtime path and restart fingerprint. The
   denominator follows the [MOOSE radial-return documentation][moose-return].
   Requiring accepted stations at every interior tabular knot is AgentFEM's
   conservative stateful-path policy, not a claim about Abaqus' default
   increment scheduling. Abaqus documents explicit time points for evaluating
   a cyclic response at known load maxima/minima, and treats exact output
   times as a separate incrementation choice; these distinguish physical
   path resolution from output-frame frequency [Abaqus direct-cyclic time
   points][abaqus-time-points], [Abaqus output timing][abaqus-output-times].
   A separate **diagnostic, not acceptance** now runs
   `examples/finite_j2_hertz_contact.py` on a quarter finite block with a
   spherical tool. Holding `--quarter --grading 2 --indentation 0.002
   --radius 20 --increments 4` and the example's other defaults fixed gives:

   | cells/side | penalty | full tool force | active reference area | max penetration |
   | ---: | ---: | ---: | ---: | ---: |
   | 7 | 100,000 | 0.712466 | 0.142459 | 0.00007267 |
   | 9 | 100,000 | 0.696875 | 0.143011 | 0.00007231 |
   | 11 | 100,000 | 0.692319 | 0.138502 | 0.00007205 |
   | 7 | 300,000 | 0.726972 | 0.134057 | 0.00002812 |
   | 9 | 300,000 | 0.710146 | 0.135631 | 0.00002568 |
   | 11 | 300,000 | 0.706214 | 0.126903 | 0.00002499 |

   Reproduce one row with `python examples/finite_j2_hertz_contact.py
   --quarter --cells 11 --grading 2 --penalty 100000 --indentation 0.002
   --radius 20 --increments 4` in the FEniCSx environment. At 11 cells,
   tripling penalty still changes force by about 2.0% and active area by about
   8.4%; the area also changes nonmonotonically under refinement. Two versus
   four increments were nearly identical at the coarse elastic 2/3-cell
   levels, but that does **not** establish plastic-path or general time-step
   independence. On the 11-cell, 300,000-penalty row, increasing contact
   quadrature degree from 3 (4 points/face) to 5 (9) and 7 (16) gives tool
   forces 0.706214, 0.705531 and 0.705975 respectively, while active areas
   are 0.126903, 0.129983 and 0.127824. This makes force look much less
   sensitive than the active footprint; no quadrature degree is promoted on
   that observation alone. The half-space Hertz force printed by the example
   is only a scale reference: the solved problem is a finite, bottom-clamped block with
   finite penalty, finite-strain kinematics and sampled contact. Neither its
   Hertz mismatch nor completion alone certifies or refutes this formulation.
   The next gate needs a denser, independently checked finite-block reference
   and a declared pressure/area discretization tolerance, not a selectively
   chosen force match.
   An additional higher-order screen used the same quarter block,
   `--displacement-degree 2 --bulk-quadrature-degree 4
   --contact-quadrature-degree 7 --linear-policy gmres_gamg` and four
   increments. At penalty 100,000, 5/7 cells per side gave full tool forces
   0.685484/0.683987, active areas 0.149148/0.144965 and maximum
   penetrations 7.20347e-5/7.18637e-5. At 7 cells and penalty 300,000,
   these were 0.697151, 0.136021 and 2.4576e-5. The higher-order force
   varies less with this mesh change, but penalty and footprint sensitivity
   persist. This is a separate discretization screen, not a converged
   benchmark or a same-DOF performance comparison with P1.
   The nonconvex penalty sequence is not converged.
   Reading: [NAFEMS benchmark
   catalog][nafems], [Abaqus contact verification cases][abaqus-verification]
   and [COMSOL contact meshing advice][comsol-mesh].

9. **Performance and installed release.** Measure complete accepted-increment
   time, nonlinear iterations, search, assembly, linear solve, output and
   memory on frozen inputs. Keep robust direct solve as default; promote an
   optional iterative solve or cache only with same-path results and repeated
   platform-scoped evidence. Run ordinary installed `model.step` cases, early
   unsupported-combination checks, serial checkpoint/export, Linux/macOS CI
   and explicit platform availability report. Current state: opt-in GMRES/GAMG
   shows a same-host small-screen gain, not a universal speedup. A second,
   larger serial screen fixed the quarter finite-block sphere case at 15
   Hex cells per side (3,375 cells), `--grading 2 --penalty 100000
   --indentation 0.002 --radius 20 --increments 4`, with one OpenMP/BLAS
   thread. Only `--linear-policy direct` versus `gmres_gamg` changed.
   Reversed-order direct/GAMG/GAMG/direct runs took 28.071/11.675/11.326/
   28.084 seconds end to end (within-process case wall time), a 2.44x median
   ratio **on this host and input only**. All runs used 20 linear solves and
   returned tool force 0.68951518028 within 2e-12, active area 0.1453987193
   and maximum penetration 7.18696e-5; GMRES counted 540 Krylov iterations.
   The direct/iterative linear phases were about 18.37/1.61 seconds, so the
   observed gain has an identified source. This case remained elastic,
   finite-block Hertz is not an exact scientific reference, and the screen
   does not establish the same speed or robustness for plastic loading,
   friction, sharp features, another mesh or another machine. A separate
   4-cell/axis, 12-increment plastic J2 L-tool solve returned the same tool
   force, PEEQ, active area and **failed** engineering feature claim under
   direct and opt-in GAMG (differences below 1e-12 in reported scalar
   observables); this checks solver-policy consistency, not validation or a
   timed speedup on nonconvex contact. Keep the
   iterative method explicit opt-in pending those checks; distribution and
   independent scientific acceptance remain separate. Reading:
   [PETSc GAMG near-nullspace requirement][gamg], [PETSc profiling][profiling]
   and [Hoefler--Belli scientific benchmarking][benchmarking].

   A separate **plastic-contact runtime** screen on the same host explains
   why changing KSP alone is not the dominant lever here. With an 8-cell/axis
   tetrahedral reentrant L-tool, 12 increments, penalty 20, degree-5 contact
   quadrature, direct solve and one OpenMP/BLAS thread, the pre-change accepted
   increment ledger attributed about 9.65 s to backtracking, 6.38 s to
   material updates, 4.90 s to residual assembly, 0.56 s to tangent assembly
   and only 0.80 s to 60 linear solves. Opt-in GMRES/GAMG reduced linear time
   to 0.42 s but the complete case remained about 28 s in both policies.
   Backtracking had already evaluated the accepted next-state material and
   total residual; the following Newton iteration was evaluating both again.
   The standard J2 Procedure now carries that *accepted* trial response and
   PETSc residual into the next iteration. Rejected trials still destroy their
   residual; basic/no-line-search keeps its former path. This is the same
   function-value handoff described in [PETSc's nonlinear-solver iteration
   model][petsc-snes], adapted to AgentFEM's material State transaction.

   On the exact same source and machine, a temporary A/B harness restored the
   old discard-and-recompute behavior without changing tolerances or input.
   Bracketed direct runs took 28.59/28.51 s with discard versus 19.02/19.01 s
   with accepted-trial reuse (roughly 1.50x, or 33% less complete-case wall
   time on this input). The selected scientific JSON fields (quantities,
   histories, contact duals, mechanical ledger, feature path and verification)
   had identical SHA-256 in the old-behavior and new-behavior runs; both
   `fields.h5` and `fields.integration-points.h5` were byte-identical too.
   This includes the **same failed** engineering feature claim: the runtime gain
   does not validate the nonconvex contact law. It is not a cross-platform
   speed guarantee, and the finite-J2 affine/MPC Procedure needs its own
   ownership review before sharing this execution optimization.

## Promotion rule

Gates 1--4 and 6--8 are mandatory for any claim involving nonconvex rigid
features. Gate 5 determines the friction/geometry combinations allowed in the
claim; a locally planar Coulomb test never licenses sharp-edge friction. Gate
9 is mandatory for a release. If the nonconvex gate remains open, publish only
the actually verified smaller subset, still marked experimental where the
independent physical evidence is absent. Full Abaqus general-contact parity
is not a 0.4.4 claim.

## First implementation evidence (bounded)

The opt-in research module `boundary_models.normal_augmentation` separates
the signed algorithmic AL functional from physical contact energy. It binds
accepted multipliers to a contact-pair identity, fixed reference trace (point
IDs, node support, shape values, integration weights and reference query
geometry), rigid-surface fingerprint, units and explicit
acceptance/outer-iteration policy. It requires
a converged *total* residual
at each frozen-multiplier outer iterate, and atomically refuses mismatched
restart State. Its spring/plane exact solution and smooth plane/sphere
finite-difference tangent tests and the independent DOLFINx/UFL plane/sphere
operator oracle are local mathematical evidence only. They do not establish
the J2+contact+MPC product workflow. The next admission is a
single supported analytical plane in the full nonlinear Procedure, with the
same total-system and accepted-increment transaction as penalty contact; an
adaptive integration partition must continue to reject AL until multiplier
transfer is separately verified.

### Whole-increment augmentation design, not a second solver

For that analytical-plane admission, keep `FiniteStrainJ2StandardProblem` as
the only Newton/line-search/linear-solve owner. The contact Operator reads a
fixed outer-iterate multiplier and contributes its total residual and tangent
before the existing MPC reduction. The Procedure snapshots displacement,
material State, tool pose/work and accepted multipliers *once* at the start of
the increment, then repeats: solve total Newton with a frozen multiplier;
test total equilibrium; test penetration and complementarity; if needed,
update only the trial multiplier and solve the total system again. Only a
converged final outer iterate permits one joint commit and one checkpoint.
An exhausted outer loop or Newton/cutback failure restores the complete
pre-increment snapshot. In particular, a converged inner Newton iterate is
not a valid checkpoint, and its algorithmic AL functional is not physical
stored contact energy. Result must reuse the ordinary constraint dual and
accepted-path work evidence, but mark physical contact-energy closure
unavailable until the finite-penetration enforcement residual is separately
bounded; it must not insert the AL functional into `contact_potential` merely
to make an energy equation appear closed. The current penalty
`ContactTraceAssembly` explicitly requires a nonnegative physical potential,
so reusing it as an AL wrapper would violate its contract. The new Operator
needs a contribution type that exposes the same residual/tangent and
force/moment protocol while keeping algorithmic and physical energy distinct.
Admission requires the existing penalty Step to remain unchanged by default
and the AL path to pass a fixed-trace restart identity,
one plastic-history path and one MPC-reduced total-tangent comparison. A
changing feature partition cannot use this State until a conservative,
identified multiplier-transfer rule has its own tests.

### 2026-10-19: bounded fixed-plane AL enters the ordinary Step

`steps.normal_augmentation(...)` now explicitly selects a bounded experimental
outer Procedure inside the existing standard finite-J2 Step. It is limited
to one fixed frictionless analytical plane, serial P1 tetrahedra, a scalar
penalty and ordinary strong constraints. Coalesced nodal reference weights
remove the previously demonstrated duplicate Gaussian multiplier rows. An
individual normal row with no free displacement after strong constraints is
rejected. This algebraic check is not a mesh-uniform inf-sup theorem: no MPC,
moving/curved tool, mixed space, changing trace or friction promotion follows.

The Operator keeps the multiplier fixed throughout every inner Newton and
line-search trial. The Procedure reconfirms the **total** equilibrium residual
before augmenting. The material always starts from the same pre-increment
committed history. Only the final outer acceptance jointly commits material,
displacement and multiplier; inner success cannot produce a checkpoint.
Failed Newton, exhausted augmentation and an injected post-multiplier-commit
failure all restore the preceding accepted boundary in targeted tests.

Complementarity now bounds both `abs(lambda*g)` and `abs(p*g)`, where
`p=max(0,lambda-rho*g)`. Testing only the old multiplier could accept a large
effective pressure with zero initial multiplier. The research multiplier
checkpoint schema advances to v2, so old acceptance semantics are not silently
restored. Pressure is force per reference area, and the algorithmic functional
is not exported as physical contact energy. The mechanical-energy ledger is
explicitly unavailable for this admission pending a bounded enforcement-work
residual. Fixed-tool coordinate work remains zero, without a closure claim.

Evidence is in `tests/test_finite_j2_normal_augmentation.py`: actual plastic
loading/unloading, independent homogeneous nominal-stress limits at two mesh
sizes and two penalties, plastic displacement-controlled limit, joint
restart, corrupted multiplier rejection, and public capability/execution
scope checks. The independent prescribed-displacement comparison exposed a
separate standard-J2 backtracking issue: provisional iterates must first
satisfy the current nonzero strong boundary values. The repair is covered by
a test that inspects **every** trial and compares with full Newton. The 202
existing standard/contact/periodic regression tests passed after this repair.

Additional independent discrete-limit tests now exercise **nonuniform**
oblique-plane pressure (elastic and plastic) and mixed open/closed nodes.
The partial-contact oracle repeatedly solves contact-free equality-constrained
elastic-limit J2 problems, selecting active nodes from assembled reactions and
gaps; it does not use the AL or penalty operator. At two and three cells per
edge, displacement, active nodes and normal nodal forces agree with the AL
Step within the declared test tolerances. This is a same-discretization
enforcement check, not an independent continuum pressure-convergence result.

Two integration defects were found and repaired before promotion. First,
accepted pressure export must use `max(0,lambda-rho*g)`, not the inherited
penalty-only response. Exported pressure times reference weights and normals
now reproduces the assembled force; metadata states augmented enforcement and
coalesced nodal reference location. Penalty-only sensitivity/enclosure methods
reject AL rather than silently changing its law or multiplier trace. Second,
automatic load adaptation must not treat the sum of all inner Newton counts
as a single Newton difficulty. That earlier implementation exhausted 50
increments in an automatic-cutback test. The controller now uses the maximum
inner Newton count; total iterations/timings and outer augmentation counts
remain separately reported, and genuine outer failure still cuts back.

The new public `examples/finite_j2_augmented_plane.py` uses normal Model and
Step APIs. On its two-cell-per-edge plastic block the half-path checkpoint
continued through all four increments, final right displacement
`0.00500004 m`, with 198 augmentations in total. A 60-augmentation budget
initially rejected the second increment (`2.64e-7 m` penetration versus the
declared `1e-7 m` limit); the example now explicitly permits 120 augmentations
per increment without loosening the acceptance tolerance. This records actual
cost and a bounded failure, not a speedup or universal convergence claim.

Next: retain the public fixed-plane slice as experimental; establish
nonuniform pressure and active-set convergence evidence before selecting a
general multiplier space. MPC needs a compatible reduced space, moving tools
need accepted enforcement-work accounting, and changing feature partitions
need explicit conservative transfer. The nonconvex geometry and independent
same-input finite-J2 comparator gates remain open.

### Fixed-plane follow-up: cost, reversed loading and integration checks

The runtime rollback token now uses immutable numerical arrays and owner/
policy identity. Disk checkpoint serialization remains a separate operation;
a regression forbids checkpoint-list serialization during a normal solve.
The global assembled fixed-multiplier tangent also passes a free-DOF
directional difference, in addition to the local contact-law checks. Accepted
events report outer augmentation count, maximum inner Newton count and actual
line-search residual evaluations, rather than conflating them with load steps.

`tools/benchmark_normal_augmentation.py` records warmups, shuffled repeats,
failed candidates, source hashes and every accepted field/material snapshot
on the public four-increment example. An initial screen with the unchanged
`1/128` line-search floor measured median solve/result times of
`1.377/0.641/0.344 s` for penalties `500/1500/5000 Pa/m` at two cells per
edge, and `3.319/1.494/0.752 s` at three. This is roughly 75--77% less time
on these **small example problems by explicit numerical-policy selection**,
not a general software speedup or an explicit-composite-case claim. The
example default remains unchanged. Geometry, material, load stations and
acceptance tolerances were held fixed; all compared accepted fields met the
script's declared screen.

Penalties `50000/500000` failed at the original line-search floor. A diagnostic
with the explicitly smaller `1/8192` floor completed at two cells (median
`0.261/0.425 s`, versus `1.404 s` at penalty 500 in that same run), but both
still failed at three cells. Thus neither increasing penalty nor lowering
the search floor is a general convergence remedy. A separate stricter
componentwise comparison initially found off-diagonal stress differences
of order `1e-4 Pa` with the original contact acceptance. Tightening both runs'
penetration/complementarity tolerances from `1e-7/1e-6` to `1e-9/1e-8`
passes the unchanged componentwise comparison. Do not hide that accuracy/cost
dependency behind endpoint displacement agreement. The numerical role and
robustness tradeoff of AL penalty are consistent with [COMSOL's contact
settings](https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_modeling.05.147.html).

An independent scalar reference now covers sixteen accepted stations of
loading, separation, reverse plastic flow and recontact, at one and two cells
per edge, with and without restart after reverse flow. For homogeneous
`F=diag(exp(e),1,1)`, zero Poisson ratio, signed plastic logarithmic strain `a`
and PEEQ `p`, it uses
`q_trial=E*abs(e-1.5*a)`,
`dgamma=max(q_trial-Y-H*p,0)/(1.5*E+H)` and
`P11=E*(e-a_new)/exp(e)`, followed by a scalar equilibrium root or the rigid
stop stretch. No FE contact/material routine is used in that reference.
Displacement, Fp, PEEQ, plastic dissipation and reaction match at every station.
This is a homogeneous same-law verification, not spatial convergence or an
independent commercial-code comparator, and does not close Gate 1 or 8.

That test exposed a real input-semantic bug: a signed loading amplitude was
incorrectly rejected as a normalized progress coordinate. Constant/field
strong values now accept finite signed scaling; fixed tools do not consume
a motion coordinate. Independent boundary histories retain normalized input
checks, and moving-tool schedules retain their admitted range. No moving-AL
capability was added.

The first remote complete suite at `366824b0` reported 4301 passes, 359 skips
and 13 failures. Eleven failures shared one builder defect: the default
`contact_integration='gauss'` was being injected into mixed, no-contact J2
capability checks. That default is no longer misrepresented as an explicit
contact option. A rejection test now uses the private line-search tuple
contract, and an absolute-residual mesh test explicitly requests tighter
Newton tolerances instead of relying on a mesh-dependent initial relative
norm. Its original field and `1e-6` absolute residual assertions remain.
The two affected files plus the new cyclic reference passed 52 local tests;
the preceding runtime/policy/standard-J2 check passed 108. Remote integration
must be rechecked at the final revision before promotion.

### Call-local material linearization reuse

A whole-Step profile of the four-increment public augmented-plane example
identified duplicate elastic-trial SVDs and matrix inverses in the batched J2
stress return and its consistent tangent. The material now retains those
already-computed factors only within that update call. Nothing is cached on
the material instance, committed State or checkpoint; scalar and independent
derivative checks retain a recomputation path. No material equation, tangent,
penalty, acceptance tolerance or load station changed. Three consecutive
changed/retried batch updates use exactly three SVD calls, and reproduce the
original accepted state and derivative.

The pinned comparison is `bd9b3118`'s analytic tangent. The script
`tools/benchmark_j2_spectral_reuse.py` measures complete columnar responses and
the ordinary four-increment `solve_result`, excluding imports, JIT warmup,
model construction and disk export. After a seven-pair screen, a fresh
five-pair confirmation used three different mesh sizes. Every accepted
displacement/field, final material state, reaction and iteration count was
identical in this environment. On this Apple Silicon host:

| Workload | Recompute median | Reuse median | Elapsed reduction |
| --- | ---: | ---: | ---: |
| 4096-point nonproportional material batch | 0.02678 s | 0.02162 s | 19.2% |
| Whole Step, 3 cells per edge | 0.7473 s | 0.6612 s | 11.5% |
| Whole Step, 5 cells per edge | 2.4861 s | 2.1581 s | 13.2% |
| Whole Step, 8 cells per edge | 10.6094 s | 9.1895 s | 13.4% |

Raw timings, source hashes, ordering and declared thread environment are in
`evidence/contact/2026-10-19-j2-spectral-reuse-confirmation.json`; the earlier
screen is retained beside it. These are specific material and implicit-Step
measurements, **not** a full explicit-composite speed claim or an independent
physical-model validation. The environment variables request one BLAS thread;
they are not a measured guarantee about every native runtime's worker count.

The related integration screen passed 402 tests, including material histories,
analytic/numerical tangents, strong/mixed/periodic J2, AL, and finite-Hex energy.
The preceding installed-wheel check at `bd9b3118` reproduced full versus
resumed displacement, plastic variables and contact checkpoint data exactly.
A real two-rank negative test also rejects the unsupported AL combination
before trace assembly; this is not distributed AL support. An output audit
additionally requires the current accepted configuration before publishing an
AL reaction dual, even when an earlier increment had already converged.

### 2026-10-19: nonuniform refinement and authoritative contact coordinates

`tools/verify_normal_augmentation_spatial.py` supplies an independent
nonuniform manufactured elastic-limit solution, not another homogeneous patch:
`u_x = A*x*(1-x)`, `A=0.02`, `E=1000`, `nu=0`, with the analytical body force
and nominal right-face reaction derived from `P11=E*log(F11)/F11`.
The fixed plane constrains the right face; no plastic flow occurs. Three
tetrahedral meshes use 2, 4 and 8 cells per edge and the same explicit AL
policy. The retained report is
`evidence/contact/2026-10-19-normal-augmentation-spatial.json`.

| Cells per edge | Displacement L2 error | Gradient seminorm error | Contact-force absolute error | Maximum nodal pressure error |
| --- | ---: | ---: | ---: | ---: |
| 2 | 9.132e-4 | 5.775e-3 | 4.998e-2 | 5.067 |
| 4 | 2.289e-4 | 2.887e-3 | 1.250e-2 | 2.533 |
| 8 | 5.741e-5 | 1.443e-3 | 3.124e-3 | 1.281 |

Displacement orders are 1.996/1.996 and gradient orders 1.000/1.000.
The exact contact-force magnitude is 20.6150; the finest nodal pressure error
is still about 6.2%, so neither exact pressure nor general multiplier stability
is claimed. Penetration is below 2.73e-10 for the declared 1e-9 tolerance.
Elapsed times include measurement overhead and are not a performance comparison.

This refinement found that DOLFINx tabulated displacement coordinates can map
an original mesh vertex at `x=1` to `1+2.22e-16`. A strict zero-gap audit then
incorrectly reports overlap. After the existing adjacent-cell vertex/block
identity match, linear traces now copy the authoritative mesh coordinates
into their private reference buffer. There is no tool snapping, mesh mutation
or tolerance relaxation. Real tetrahedral/hexahedral tests with Gauss/nodal
rules retain rejection of a genuine `2**-45` overlap. Trace/high-order/implicit
contact regression passed 168 tests (8 skipped); a bounded two-rank trace
regression passed 36 tests per rank. Distributed trace evidence is not
distributed AL admission.

Compatibility is explicit: loading a `7d2f4652` four-cell checkpoint under the
corrected source rejects with `Normal multiplier checkpoint trace basis
differs`. An affected pre-correction run must retain its original candidate;
there is no inferred migration of multiplier history. The earlier successful
`bd9b3118` to `7d2f4652` restart comparison predates this geometry correction.
The corrected source was then built with `AGENTFEM_BUILD_NATIVE=0` and installed
into an isolated target (no change to the user's installation). The public
four-cell, penalty-5000 example passed continuous loading and half-load restart.
Displacement, accepted displacement, plastic deformation gradient, equivalent
plastic strain, plastic dissipation, contact State and checkpoint identity
were exactly equal. Accepted increment numerics/counts also matched; timing
and restart-event records are expected to differ. The wheel SHA-256 was
`813bcbc66d640e0e5955ba359db493cf5a50d027638e1195f733a7e97f0ebafc`.
This is installed implicit-DOLFINx evidence, not optional native-accelerator
validation. Two focused corrupted-multiplier/trace-identity tests confirm
atomic restore refusal; the nonuniform two-mesh and independent cyclic
reference follow-up passed five tests, with nine two-rank-only tests skipped
in that serial invocation.

## Fixed-plane accepted normal-work gate (2026-10-20)

This supersedes the earlier unavailable-energy statement **only** for the
admitted serial fixed-plane AL path with complete accepted history. It does
not close the broader Gate 4 or Gate 7 matrices above.

The existing contact-work module now integrates matched accepted pressure/gap
endpoints. With positive gap on the admissible side and reference weights,
`Wn=sum(w*(p0+p1)/2*(g1-g0))` is positive work into the structure;
`Bn=sum(w*(p0+p1)/2*abs(g1-g0))` is its non-cancelling discrete bound.
Positive-gap work is reported separately. Exact endpoint complementarity
does not eliminate closure/opening quadrature work. The
[Abaqus contact-energy example][abaqus-contact-energy] explicitly distinguishes
signed contact discontinuity work from physical stored energy; this is the
reason for separate channels, not a claim of identical algorithms.

The Procedure uses the existing accepted energy frames and joint transaction;
there is no new work State owner or solver. Augmentation/Newton iterations
are not path stations. AL has no physical spring-energy history. The raw
`Wext-U` residual is unchanged; the additional `Wext+Wn-U` diagnostic retains
material/path integration error rather than being forced to zero. A bound on
discrete work is not a bound on continuous-path error. No automatic energy
verification is added. Legacy incomplete ledgers stay unavailable, and
corrupt/missing/nonfinite/inconsistent new histories reject before restore.

Independent evidence uses the existing scalar finite-J2 stop reference,
including load, unload, reverse plasticity and recontact. The reference uses
no AgentFEM material or contact solver; stored Hencky/hardening energy,
plastic dissipation and endpoint work are reconstructed independently.

| Increments | Raw residual | Contact-aware residual | Discrete absolute normal-work bound |
| --- | ---: | ---: | ---: |
| 16 | -7.80259e-5 | -3.14497e-3 | 4.22724e-3 |
| 32 | -1.25948e-4 | -6.44190e-4 | 9.13273e-4 |
| 64 | 2.53142e-5 | -6.84374e-5 | 1.06248e-4 |
| 128 | 9.08018e-6 | -2.74590e-5 | 4.26926e-5 |

Units are joules for the unit-cube SI reference. The coarse raw residual is
misleadingly small due to cancellation. The corrected residual and absolute
work decline substantially under refinement, not identically to zero. This
is a bounded cyclic numerical check, not industrial validation or a universal
convergence rate. Reproduce with
`pytest tests/test_normal_contact_energy.py -q -s` in `fenicsx-env`.
The same suite compares contact work to an independent nodal-force dot
displacement path and verifies reverse-plastic restart and post-recording
failed-commit rollback. Test tolerances propagate the independently checked
stretch accuracy into accumulated external work rather than borrowing a
single-state energy tolerance.

Acceptance: 203 focused tests passed from an isolated installed wheel in
55.82 seconds, with the package import path asserted outside the checkout.
This includes the new numerical-work checks and existing penalty, multiple-pair,
mixed-J2 energy and contact-work transaction tests. The default public example
also completed from that installation and exported fields, contact samples,
checkpoint and result manifest. Its deliberately coarse four increments have
contact-aware residual `-0.00248248 J`; completion is not energy verification.
Correctness lint, generated-document consistency and strict documentation
build passed. Native acceleration was disabled for this implicit-DOLFINx
wheel; neither MPI AL nor accelerator performance is claimed. Wheel SHA-256:
`af0e97c59eb2f40bde7f8b25da565ae812a4e21017f9a485f8e8ec86d7dc45c1`.

## Prescribed-translation normal-augmentation gate (2026-10-20)

Decision: admit translation of the same analytical plane, not arbitrary
moving geometry. The multiplier basis is attached to the fixed material-side
reference nodes; rigid translation changes current gaps but not pressure
identities or normals. Every trial projects current positions against the
current tool pose. Rotations (including about a plane normal), curved tools,
changing traces, friction, MPC and MPI augmentation still reject.

The established segregated AL method holds pressure fixed inside equilibrium
iterations and updates it outside; see [COMSOL normal contact][normal-al].
We retain that architecture and the existing ordinary finite-J2 Procedure.
This is not a new coupled solver or a claim of mortar equivalence. Motion uses
the same normalized rigid schedule and resolved Step amplitude as the existing
penalty path; unloading/recontact does not erase plastic history.

For constant normal `n`, `dg=n·(du-dq)`, so endpoint structural contact work
splits into prescribed-tool work plus relative-gap work. The former already
enters `external_work`; the latter uses the existing signed normal-work
channel. Do not add total structural work to external work again. Tangential
plane translation produces no work. AL still has no physical spring energy.

State: a converged endpoint is tied to both displacement and resolved tool
factor. Changing either invalidates augmentation/commit evidence. Moving
checkpoints use a v2 contact payload containing the JSON-native motion identity
and accepted factor; restore validates it against the Step amplitude before
joint publication. Fixed v1 contact payloads retain their format. In-memory
rollback uses numerical snapshots, not per-increment JSON serialization.
After-contact-commit failure restores displacement, plastic state, multipliers,
tool pose and work together.

Independent checks:

- Scalar finite-J2 compression reference on two meshes: full accepted material
  state/pressure path, not just final force balance.
- Oblique translation: added tangential motion leaves displacement, plastic
  history and work unchanged.
- Global directional tangent with a tilted translating plane, fixed old
  material state and frozen multiplier.
- Load/unload/recontact with restart and atomic failed joint commit.
- Corrupt motion identity/factor refuses restoration; old optional `None`
  energy fields compare as unavailable, never coerced to numerical zeros.
- Current-point samples and ordinary Result retain the actual augmented
  pressure and separate tool/relative-gap histories.

The cyclic scalar reference is independent of the AgentFEM material update
and contact solver. It solves the traction-free one-dimensional branch when
open and imposes the translated stop when closed. The increment screen gives:

| Increments | Tool work | Raw residual | Contact-aware residual | Discrete relative-work bound |
| --- | ---: | ---: | ---: | ---: |
| 16 | 0.0716047 | -1.73303e-3 | -2.85968e-3 | 3.08773e-3 |
| 32 | 0.0730628 | -2.74924e-4 | -7.57166e-4 | 1.06100e-3 |
| 64 | 0.0731485 | -1.89257e-4 | -2.56478e-4 | 1.47999e-4 |
| 128 | 0.0733219 | -1.58346e-5 | -4.29142e-5 | 5.96754e-5 |

All energies are J for this deliberately soft unit-cube reference.
The corrected residual decreases by about 67 times; it is not forced to zero
and the discrete-work bound is not a bound on total energy error. Reproduce
with `pytest tests/test_translating_normal_augmentation.py -q -s`.
The source-focused contact/J2/restart set passed 205 tests in 67.44 seconds.
The public `examples/finite_j2_translating_augmented_plane.py` is the ordinary
Model/Step entrypoint for the same compression/unloading/recontact workflow.

The same installed candidate passed 205 focused tests in 66.88 seconds.
Public full/half/restarted 32-increment runs agree exactly in displacement,
plastic deformation gradient, equivalent plastic strain, plastic dissipation,
contact State, accepted energy history and checkpoint identity. Timing and
restart events appropriately differ. A subsequent guard also reserves
converged-but-uncommitted reaction duals for Procedure-owned work recording;
a previous successful solve does not license public trial output.

`tools/verify_normal_augmentation_spatial.py --tool-travel 0.01` independently
checks the final nonuniform elastic-limit field
`u=(0.02*x*(1-x)-0.01*x,0,0)`, with manufactured reference body force. On
2/4/8 cells per edge, observed displacement orders are 1.996/1.995 and
gradient orders 1.000/1.000. Contact force errors decrease from 0.0518575 to
0.0129639 to 0.00324105 N; maximum nodal pressure errors decrease from
5.22111 to 2.60956 to 1.31982 Pa. This is an elastic spatial benchmark, not
a manufactured intermediate load path or plastic spatial-convergence proof.
Reconstructable evidence is
`evidence/contact/2026-10-20-translating-normal-augmentation-spatial.json`.
Two-rank fixed and translating negative-admission tests both reject before
distributed assembly; no MPI augmentation capability is promoted.

An additional nonuniform **plastic** gate uses a clamped block with Poisson
ratio 0.3, free lateral motion and a translating plane at zero initial gap.
The independent enforcement reference has no contact Operator: it prescribes
the same normal displacement on the right face and obtains normal reactions
from the full finite-J2 residual. Positive pressure is checked at every
station, because a displacement constraint could otherwise pull a body that
should separate. Eight accepted stations on 1/2/3-cell meshes are compared at
penalties 1000 and 5000 Pa/m.

The exploratory `1e-8 m` penetration / `1e-7 Pa m` complementarity settings
passed displacement/state checks but missed the unchanged point-pressure
criterion on the three-cell mesh: maximum discrepancies at its first station
were about `9.44e-5` and `5.50e-5 Pa`. Tightening those contact tolerances by
100, without changing the mesh, material, pressure criterion or solver method,
made all six comparisons pass. The final maximum pressure discrepancies over
the full eight-station paths were:

| Cells per edge | Penalty 1000 | Penalty 5000 |
| --- | ---: | ---: |
| 1 | 1.58e-7 Pa | 9.88e-8 Pa |
| 2 | 5.93e-7 Pa | 3.27e-7 Pa |
| 3 | 7.71e-7 Pa | 6.72e-7 Pa |

This demonstrates tolerance sensitivity of a point-pressure observable, not
a reason to silently tighten every user's defaults or a proof of continuum
pressure convergence. The reference shares the constitutive kernel, so it is
independent enforcement, not independent material verification. Reproduce
with `tests/test_translating_augmentation_plastic_reference.py`.

The candidate including that provisional-dual guard passed **251**
focused tests from an isolated installed wheel in 76.74 seconds. Package
imports were asserted outside the checkout. Native accelerators were disabled
for this implicit-DOLFINx wheel; no native acceleration result follows.
Wheel SHA-256:
`b4094ebfa5d67a87d09d2bdd54629b19ca9053e3661f69b0e015ddaa5b720dd8`.
Correctness lint, generated-document consistency and the external-directory
strict documentation build passed.

### Cyclic accepted-output follow-up

Installed CLI use exposed a common output defect, not a contact-law failure:
at normalized Step coordinate 0.5 the cyclic tool amplitude is 1.0, and at
coordinate 0.75 it is 0.0. The old exporter compared the accepted tool pose
to the coordinate instead of the resolved amplitude. Both ordinary penalty
contact and augmentation reproduced `AFM-CONTACT-EXPORT-POSE` after a valid
solve. The shared Result exporter now checks the Procedure's resolved amplitude,
records coordinate and amplitude separately, and continues to reject stale or
nonfinite pose samples before creating output. This is not a bypass of pose
validation and does not change the solver or loading.

The follow-up wheel passed **331** targeted tests in 94.13 seconds from an
isolated installation, including ordinary friction and active/convex contact
partitions. SHA-256:
`c93795ccb5c35c5a6d6c962d7db6e2b146abb97113fed7779af82b9caf4df544`.
The public example was then run from outside the checkout: one continuous
32-increment run versus save at 0.5, restart to 0.75, and restart to 1.0.
Displacement, plastic/contact State, complete energy history and checkpoint
identity are exactly equal. Increment diagnostics also agree excluding elapsed
timings; execution logs correctly retain the two restart events. Both
intermediate stations now export their accepted raw contact arrays and VTP
view with the correct tool factor. This remains serial Mac evidence, not a
cross-platform or performance claim.

Compatibility was also checked against an actual fixed-plane v1 checkpoint
produced before this translation change (`f226d34c` candidate): the final
state and four energy frames load without altering the numerical solution.
That check is a completed-checkpoint restore, not a legacy partial-run
continuation claim.

Next: finish remote CI acceptance, then address the broader
mesh-by-penalty and changing-feature gates. Translation alone does not close
the nonconvex L-tool, multiplier transfer or commercial local-parity matrix.

### 2026-10-20 design: optional multiplier-iteration acceleration

The next bounded experiment targets repeated converged Newton solves, not a
new material algorithm or a relaxed acceptance criterion. A private weighted
dynamic-Aitken probe reduced outer updates on the nonuniform plastic press
and the unloading/recontact example while keeping the existing equilibrium,
penetration and complementarity checks. This is screening evidence only.

Design before promotion:

- Procedure owns increment-local secant history. State still owns the frozen,
  trial and accepted multiplier; no accelerator can commit material or contact.
- Apply the projected map `F(lambda) = max(0, lambda - penalty*gap)`. Use the
  fixed reference-trace area metric for the vector secant, then project the
  relaxed proposal onto nonnegative pressures. The penalty and Newton tangent
  are unchanged. The inner solve always sees a frozen multiplier.
- Reset history at each new increment/attempt and active-set change. Degenerate
  secants use the ordinary update; relaxation is bounded to `[0.1, 10]`.
  This safeguard is not a general convergence theorem or a promise that every
  problem is faster. Failed inner solves retain normal rollback/cutback.
- Add an explicit experimental `acceleration="aitken"` option; default `"none"`
  preserves the old path and checkpoint encoding. Accelerated checkpoints
  identify the numerical policy. No secant history crosses accepted boundaries.
- Validate scalar/degenerate maps, state witness and atomic failure, independent
  prescribed-displacement pressure, cyclic unloading/recontact, restart and
  work histories. Record actual outer/Newton work and alternate timing order.
- Do not extend AL admission to MPC, curved or changing traces in this change.
  The periodic reduced normal rows require a separately designed multiplier
  space and joint Procedure integration; deleting the guard is not a solution.

The research basis is the projected dual fixed-point formulation and dynamic
Aitken/Crossed-Secant comparison in Koliesnikova and Ramière,
[arXiv:2603.12205, sections 2 and 4](https://arxiv.org/html/2603.12205v1).
Their constant linear stiffness and reported performance are not assumed for
our finite-J2 Newton solves. We use a bounded, area-weighted variant with
active-set resets, not their claim of parameter-unbounded performance.

Implementation and bounded acceptance:

- The option is now a core Procedure control, not a benchmark monkeypatch.
  Its State update remains witnessed by the current converged total residual.
  Fixed/moving default checkpoint v1/v2 encodings remain unchanged; opt-in
  accelerated provider checkpoints use v3 and refuse a changed policy.
- A 121-test focused pass covered the initial implementation, independent
  prescribed-displacement pressure on three meshes/two penalties, scalar cyclic
  reference/refinement, injected cutback, joint failure rollback and restart.
  Further closing checks are recorded in the task report, not added to this
  count when they overlap.
- The committed benchmark warms both policies and alternates three repetitions
  of the public 32-increment translating-plane example on a 3×3×3 cube mesh.
  Median solve times: default 2.058291 s; Aitken 1.403542 s, **31.81% less
  solve time**. Outer updates: 106 → 45; Newton/linear solves: 209 → 140.
  All accepted displacement snapshots differ by at most 1.11e-10 m,
  final material state by 3.34e-10, final pressure by 5.02e-8 Pa and energy
  histories by 2.30e-8 J (the latter comparison also includes load coordinates).
  Setup was measured separately. This is a small same-machine implicit test,
  not a full composite run, universal speed guarantee or confidence interval.
- The first benchmark reader incorrectly treated the per-tool work dictionary
  as a float. It was corrected to flatten named tool channels and compare their
  availability; the solver and numerical tolerances were not changed.

No new MPC, rotation, curved trace, multiplier transfer or MPI admission was
added. Those gates remain explicit future work.

Closing verification: **376 targeted source tests passed in 118.47 s**, including
the nearby finite-J2, penalty/friction, active/convex partition and pressure/work
paths. Generated API consistency, correctness lint, diff checks and a strict
documentation build outside `site/` passed. A fresh isolated wheel ran the
public 32-increment example with `--acceleration aitken` from outside the
checkout, including accepted contact output, result manifest and checkpoint:
45 outer updates, final right-face displacement −0.015 m, tool work 0.0730628 J.
Wheel SHA-256:
`000f5fb3e096abe4cbff7eed2d00a16e07af01adcf7357c5c51334ddd3bee5de`.
This wheel smoke is not the same claim as running all 376 tests from that wheel.

Remote CI for the preceding `1e60e174` revision passed its unit/interface,
installed/native and distributed checks, but the overall FEniCSx job was
cancelled during the release-demo suite after about 30 minutes. It is not an
all-green release gate. The subsequent `a62cef23` revision passed
[Test run 39375343048](https://github.com/haoming-luo/agentfem/actions/runs/39375343048)
including Linux native distribution, FEniCSx and macOS installed-wheel
foundation acceptance, and
[Documentation run 39375343074](https://github.com/haoming-luo/agentfem/actions/runs/39375343074).
The FEniCSx job took about 22 minutes 48 seconds; no CI timeout or scientific
tolerance was changed to obtain this pass. PR #118 remains a draft, not a
merged/released 0.4.4 or an independently validated commercial-contact match.

### Partial-contact acceleration acceptance (2026-10-20)

The previous timing patch did not by itself establish accelerated behavior
on a spatially partial contact region. The follow-up tests add:

- Both default and Aitken updates against a separate equality-constrained,
  elastic-limit primal-dual active-set reference on two meshes. The reference
  assembles no contact or augmented residual. It verifies the same discrete
  nodal obstacle, not an independent constitutive law or continuum limit.
- A 16-increment finite-J2 path against a tilted fixed plane, with plastic
  loading, unloading and recontact. Every accepted displacement, material
  state, pressure, active set and energy record is compared between policies.
  The test requires a genuinely mixed active/open patch and changing activity.
- A checkpoint at peak load, with all subsequent stations compared after
  restoration; an injected failure after contact commit must restore pressure,
  material, displacement and the accepted ledger exactly before a valid retry.

The tests are `test_partial_contact_acceleration.py` and the extended independent
reference in `test_finite_j2_normal_augmentation.py`. This is correctness and
lifecycle coverage, not another timing claim. No runtime default, pressure
basis, numerical tolerance, geometry admission or MPI/MPC guard was changed.
The focused source run passed **102 tests in 56.40 seconds**, covering the
partial patch, accelerator controls, ordinary and translating augmentation,
independent plastic reference, rollback and restart. Correctness lint,
generated-document consistency and diff checks passed. This follow-up does
not substitute that source run for the preceding revision's installed CI.

### Serial affine execution and actual-offset diagnostic (2026-10-20)

The serial affine Procedure now delegates its fixed-factor Newton equilibrium
to `_affine_equilibrium.py`. The path owner still owns increment prediction,
cutback, physical acceptance, material/contact commit, output and restart.
The inner owner retains a full residual and its Operator contributions only
at the identical current trial, avoiding the former norm-evaluation followed
by a second residual/contact evaluation before tangent assembly. Changed
displacement or material trial reassembles; no trial cache crosses an increment
or checkpoint. PETSc resources are released on rejected trials and exceptions.
The distributed affine implementation and unsupported contact combinations
are unchanged.

`tools/benchmark_affine_trial_reuse.py` loads only the historical serial path
function from frozen commit `a5b5a8fc`, retaining the same material and Operator
implementations. It warms each path and alternates five measured repetitions,
checking accepted iteration/line-search paths, displacement, material vectors,
snapshot fields and contact nodal residual. The committed report is
`evidence/contact/2026-10-20-affine-trial-reuse.json`:

| Small serial path | Old median | Reuse median | Elapsed reduction |
| --- | --- | --- | --- |
| Fixed sphere | 0.10193 s | 0.09485 s | 6.95% |
| Moving sphere | 0.11023 s | 0.10133 s | 8.07% |
| Plane friction | 0.09730 s | 0.08907 s | 8.46% |
| Automatic plastic cutback, no contact | 0.07696 s | 0.07700 s | no demonstrated gain |

These tiny whole-path measurements identify avoided work, not a universal
speedup, statistical confidence interval, or a composite-run forecast.
The focused integrated source run passed 107 tests, with 10 MPI-only tests
skipped. A subsequent single bounded two-rank run passed 28 tests on each rank,
including those negative-admission checks and distributed affine tests.
Installed and cross-rank restart acceptance must be recorded separately.

Closing checks found a missing transfer of the residual threshold into the
fixed-increment failure report. It was repaired and a public-path regression
now requires the diagnostic and exact accepted displacement/material rollback
after an intentionally failed next increment. The expanded source run passed
218 tests in 63.60 seconds; correctness lint passed. An isolated native macOS
wheel, imported outside the checkout with the source-path override disabled,
passed 82 focused tests in 20.24 seconds. Its SHA-256 is
`524b653c2030140f5f38aaf2bac2801bbb8ef7f29e04b08fca28c5fea9436473`.
The installed ordinary periodic-J2 checkpoint driver passed 1→2 and 2→1 rank
restore. This checks preservation of the existing distributed no-contact
route; it does **not** admit distributed implicit contact.

The new private `contact_affine_gap_research.py` constructs **all supplied**
analytical-plane trace gaps as `g(q) = A q + b`, using actual reference
coordinates, transformation and affine offset. Exactly zero rows with
resolved negative offsets are algebraic incompatibility witnesses for this
floating-point discrete system; merely small rows are not treated as zero.
For a real periodic fixture at macro stretch 1.02, a plane limiting x to
1.01 conflicts at four immovable trace nodes, while x <= 1.03 has an explicit
feasible witness. Direct projection at randomized reduced coordinates agrees
with the affine gaps. This closes the earlier missing-offset diagnostic, not
the AL/MPC solver or multiplier-space gate.

A bounded optional SciPy/HiGHS search uses **free** displacement variables and
rechecks every gap of any returned candidate. Optimizer infeasibility alone is
reported as unresolved, not a proof; the witness says nothing about material
invertibility or points outside this trace. Tests include conflicting
nonstationary inequalities, tiny nonzero rows and a false optimizer witness.
The focused diagnostic/normal-law/CI-contract run passed 95 tests. The
[HiGHS API](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html)
documents the variable bounds and numerical statuses used here. Multiplier
identifiability and primal feasibility remain different questions; neither
this screen nor removing duplicate rows proves spatial inf-sup stability.

CI operational repair: at `a5b5a8fc`, the FEniCSx job passed its unit/interface
and installed/distributed checks, then hit its 30-minute job budget during the
release demos. Its unit/interface stage alone took about 23 minutes 10 seconds.
The job now has a 45-minute bound and process-local single-thread BLAS/OpenMP
settings. No tests, scientific tolerances or solver policies were removed.
This edit is not itself a successful release acceptance.

### Bounded affine augmentation admission (implemented, 2026-10-21)

The actual-offset screen separates an incompatible imposed macro strain from
an algorithm failure. A compatible interior face patch now has a separate
hard-obstacle reference: eliminate the active reduced normal coordinates
exactly, solve the same frozen-old-state J2 equilibrium without contact forces,
then recover pressure from the body residual. Two penalties and two mesh sizes
agree in displacement, plastic State and pressure; this is a discrete
formulation comparison, not an independent constitutive model or an inf-sup proof.

The **experimental** public extension is deliberately narrower than
general full-row-rank MPC: one fixed/translating analytical plane, serial P1
tetrahedra, one coalesced nodal pressure per contact node, and nonzero
**disjoint reduced normal row supports**. Sparse row inspection can establish
independence without a dense transformation or a machine-dependent rank
tolerance. Overlapping or prescribed normal supports remain rejected, even
when a more general method might handle them. This restriction is not a
statement that those models are physically impossible.

The ordinary affine path now passes single material commit,
failed-outer/callback rollback, translating-tool work, accepted normal
work, restart and capability-negative tests. A pre-trial snapshot owns
Operator rollback; the outer Procedure never commits material. Signed normal
work uses accepted gap/pressure endpoints and remains distinct from recoverable
penalty energy and physical dissipation. Output thinning must not thin this
ledger. Ordinary serial and distributed paths retain their existing behavior.
No curved, rotating, frictional, mixed or distributed augmentation is included.

The public cyclic example uses 16 increments, explicitly requests accepted
displacement frames, and records loading, unloading and recontact. Separate
tests resume from both loaded and unloaded boundaries and recover the same
final material/pressure/work history. Automatic cutback discards a deliberately
rejected converged outer trial; corrupt pressure, scalar work bounds and
station counts are atomically rejected. Normal relative-gap work is stored
at every accepted increment regardless of displacement-frame thinning.

Stage checks on the source tree: 168 targeted contact/affine/normal-work tests
passed in 96.65 s; 180 workflow/result/solver and corruption checks passed in
45.24 s. Two MPI ranks each passed 30 tests in 1.97 s, including early rejection
of both standard and affine distributed augmentation. This proves the negative
capability boundary and preserves ordinary distributed MPC, **not** distributed
contact support.

An isolated native macOS arm64 wheel (SHA-256
`0abbe500410d56143b18db1c128db30f0db9f7390959ee851f5893a329be178d`)
passed 24 installed affine/rollback/restart tests in 45.66 s, plus the new
frame-thinning/work-ledger test in 7.27 s. Its import location was explicitly
outside the source tree. The public example completed 16 cyclic increments
and 50 outer updates; final maximum penetration was `2.32004e-9 m`.
Independent meshio/HDF5 inspection read 17 complete `U`/`UMAG` frames, 384
tetrahedra and 125 points, with unwarped reference coordinates at every frame.
The HDF5 file occupied 220169 bytes. The example now explicitly requests
`output_every=1`; its first trial had retained only the ordinary default
initial/final snapshots. These are small-example output facts, not a size
estimate for the private 255041-cell composite model.

The first low-penalty research attempt (`1000 Pa/m`) did not meet the declared
outer acceptance within 60 updates. Comparisons at `10000/40000 Pa/m` on four
cells per edge and `20000 Pa/m` on eight cells passed the unchanged penetration
and complementarity criteria against exact active-constraint elimination.
Penalty-independent equilibrium is an observed discrete result in this matrix,
not a universal convergence-rate promise or a continuum pressure certificate.

Remote CI for stage `8f0c52f8` reached 4436 passed, 361 skipped and one failed
unit/interface test. That sole failure was the newly added workflow-policy
test importing PyYAML in the deliberately minimal numerical environment; it
was not a finite-element failure. The test now checks the fixed workflow text
using the standard library. No numerical test, tolerance or stage was removed.

### 2026-10-21: explicit full-Newton policy is not an Armijo test

Audit of the common affine Newton loop found a real option-contract defect:
`line_search="basic"` (and explicit `None`/`"off"`) previously selected one
trial of length one but still required Armijo residual decrease. On an
assembled cubic residual, the mathematically valid first Newton step from
`u=0.1` to `u=33.4` temporarily increases the residual before converging to
`u=1`. The new regression first produced two failures (basic and None) and
one pass (backtracking). A one-trial Armijo rejection is not basic Newton.

The corrected serial and distributed affine loops accept a finite,
constitutively valid full trial in basic mode. An invalid full trial is
rejected and restored; it is not silently backtracked or committed. Explicit
no-search now also lowers to PETSc `basic`, rather than leaving SNES free to
select its default. Default backtracking and its checkpoint identity remain
unchanged. Corrected explicit basic/no-search identities reject legacy
checkpoints whose algorithm would otherwise change on continuation.
This follows [PETSc's basic implementation][petsc-basic], not a claim that
full Newton is universally faster or more robust.

Evidence: 153 focused solver/workflow/material/restart/CI-scope tests passed
with eight genuinely distributed tests skipped in the serial run. The final
policy/reuse subset passed 17 tests, including an invalid full-trial
restoration and moving augmented-contact restart comparisons against
backtracking. Two MPI ranks each passed the periodic cubic total-reduction
comparison (one test, five then-deselected tests per rank). This is ordinary
distributed MPC evidence, not admission of distributed augmented contact.

### 2026-10-21: periodic plastic-patch sensitivity is measurable

`tools/verify_affine_augmented_path.py` uses the public example and reports
displacement norm, reference-area pressure, resultant force, mean/max PEEQ,
normal relative-gap work and constraint mismatch at the same four physical
stations. It does not export full field archives or tune the physical problem
to obtain a pass. The example's inexpensive 4-cell/16-increment defaults are
a workflow demonstration, not a mesh/increment accuracy recommendation.

The 12-case matrix in
`evidence/contact/2026-10-21-affine-augmentation-sensitivity.json` spans
4/8 cells per edge, 16/32/64 increments and penalties 10000/40000 Pa/m.
Every case meets the unchanged penetration/complementarity tolerances.
At fixed mesh and increment count, the two penalties give essentially the
same discrete response. Increment and mesh sensitivity remain significant:
for example the 8-cell final contact-force magnitude changes from about
0.69147 N at 16 increments to 0.75451 N at 32 increments.

The frozen `cf1bfc36` installed wheel was then used for 128/256-increment
refinement, recorded in
`evidence/contact/2026-10-21-affine-augmentation-time-refinement.json`.
All four cases pass numerical acceptance. At 8 cells, the final force changes
from 0.8176322 N to 0.8273062 N (1.17% relative to the finer path);
mean PEEQ changes from 0.05743225 to 0.05743618. At 4 cells, the corresponding
force change remains 2.29%, and the 256-increment 4/8-cell forces differ by
about 0.48252 N. Thus a successful solve and penalty insensitivity do **not**
close the spatial accuracy gate. Point pressure/PEEQ maxima near the patch
edge are not used as a universal convergence criterion.

The signed relative-gap work tends toward zero along the 8-cell time sequence
(about -2.02e-4 J at 32 increments, -1.35e-5 J at 128 and -4.31e-6 J at 256).
It is still an accepted-endpoint path quadrature, not elastic storage,
irreversible dissipation or automatic first-law verification. These data do
not establish an asymptotic order. Recorded costs include setup and scalar
observation with lightweight development overlap; this study is not a paired
performance benchmark.

The corrected-Newton wheel at `8da7b93b` (SHA-256
`a819e2515437cb0e8c81ba3f800bec782d261f464716b331eb45dee4554e001e`)
passed 33 installed policy/reuse/affine-augmentation tests in 76.15 s.
The earlier `cf1bfc36` default-backtracking checkpoint restored all 16 accepted
increments with this candidate. A separate installed TCP-transport MPI check
passed the periodic cubic comparison on both ranks; this process-local setting
does not alter global MPI configuration or admit distributed contact.
The complete remote `cf1bfc36` workflow `39464864247` passed Linux numerical
and macOS installed-wheel foundation acceptance; that does not automatically
validate later commits or close the remaining scientific gates.

### 2026-10-21: distinguish a support-model defect from contact convergence

The original point-pinned periodic example was deliberately kept unchanged
while inspecting its full-space accepted residual. With 4 cells per edge,
32 increments and penalty 40000 Pa/m, the resultant balancing the one-sided
tool is about 1.18533 N in x. The eight nodes of the pinned periodic corner
class carry that entire resultant; the sum on the other nodes is below
1e-11 N. Thus the point constraint is not merely removing a harmless rigid
translation in a self-equilibrated cell: it supplies the net external reaction.
This observation gives a concrete reason to question a continuum refinement
claim for that fixture. It does not prove that all sensitivity comes from
one cause or that any new fixture is automatically converged.

The 12-cell, 128-increment original-fixture run is retained in
`evidence/contact/2026-10-21-affine-augmentation-spatial-refinement.json`.
It completed all increments and numerical contact tolerances, with final
contact-force magnitude 0.530237 N versus 0.817632 N on 8 cells.
The subsequently started 16-cell run was intentionally stopped after the
support diagnosis; it is explicitly marked incomplete, not a failed
constitutive solve or a passed refinement point.

The example now accepts `--affine-end-face-support`, an explicit *physical*
change: x-motion on the two z end faces follows the declared macroscopic
affine motion. It reuses existing linear equations and the same contact law,
material, Procedure and State. The contact patch remains free in its normal
direction. This is a supported periodic structure, **not** a homogenization
RVE or a numerical stabilization setting. The distinction matters:
[COMSOL's periodic-cell theory][cell-periodicity] states the homogenization
assumptions and its own rigid-motion handling; it does not justify adding
arbitrary supports to an RVE. Our support conclusion comes from the measured
force balance, not from borrowing a commercial capability claim.

Three installed-fixture regressions passed: the original corner-class
resultant, the supported finite-area reaction and prescribed displacement,
and rejection of a checkpoint belonging to the other fixture followed by
supported restart/continuous agreement. They check the engineering problem
being solved, not only whether Newton terminates. The sensitivity driver
now records the fixture, installed runtime identity and transport setting,
and atomically saves every common physical station so an interrupted
experiment cannot lose all its intermediate evidence.

The supported-fixture matrix
`evidence/contact/2026-10-21-supported-periodic-contact-refinement.json`
completed all six 4/8/12-cell and 32/64-increment candidates at penalty
40000 Pa/m. Every accepted increment meets the unchanged 1e-8 m penetration
and 1e-7 pressure-length complementarity bounds. At 64 increments the final
force magnitudes are 4.588710, 4.229122 and 3.960610 N. The 8-to-12-cell
force difference is still 6.78% relative to the finer mesh, while displacement
L2 and mean PEEQ differ by about 0.96% and 0.59%. At 12 cells, doubling
32 to 64 increments changes final force by 5.89%. Consequently the new
fixture has **not** passed a converged engineering-response claim either.
The smaller sensitivity is not credited as a solver speed or accuracy gain:
the physical support changed. Neither the sign nor the coarse-path relative
error of the small signed gap-work quadrature is used as a dissipation test.
The earlier two-case screen is retained separately. All costs here include
setup/observations and brief lightweight test overlap, not paired timing.

The completed post-fixture checks were three source regressions, 82 focused
project-execution/environment/CI-selection tests, lint, generated-document
consistency and a strict documentation build. An adjacent read-only
execution-environment improvement records allowlisted MPI transport/progress
requests and Accelerate/NumExpr thread settings. It does not choose a
transport, expose unrelated environment values or add restart restrictions.
The private composite replay retains the earlier frozen `8da7b93b` installation;
this later diagnostic change is not injected into the running candidate.

### Scoped linear-solver ownership (2026-10-22)

The frozen serial affine equilibrium now owns one KSP across its Newton
iterations and releases it on success or failure. Every iteration still
assembles and sets the current reduced tangent; neither numerical factors nor
an old preconditioner are deliberately frozen. The scope ends before an outer
augmentation, a new increment, or a changed constraint reduction. The general
one-shot linear-solve helper also releases its owned KSP after configuration,
operator, solve or diagnostic exceptions; borrowed KSPs remain caller-owned.

This follows [PETSc's operator lifetime rules](
https://petsc.org/release/manualpages/KSP/KSPSetOperators/), not the distinct
[preconditioner-lagging option](
https://petsc.org/release/manualpages/KSP/KSPSetReusePreconditioner/).
The same-model 8-cell/16-increment supported-fixture screen has two warmups
and alternating fresh/scoped/scoped/fresh measurements. Median solve costs
were 25.2611 versus 24.8704 seconds (1.55% lower); this small single-host
difference is **not a substantial performance claim**. KSP creations decreased
from 149 to 59 over the path. All measured displacement/material/contact/work
arrays had zero maximum difference and the accepted Newton paths agreed.
Evidence and the reproducible driver are
`evidence/contact/2026-10-22-affine-linear-reuse-corrected.json` and
`tools/benchmark_affine_linear_reuse.py`. The first driver's post-solve result
access error is retained separately; it was not a failed physical solve.
The value retained here is bounded ownership and exception safety, with a
small measured reduction in repeated setup, not a claim of 20% acceleration.

### Supported-fixture time refinement (2026-10-22)

The eight-cell supported periodic fixture was continued with 128 and 256
increments at unchanged penalty 40000 Pa/m, penetration tolerance 1e-8 m,
complementarity tolerance 1e-7, material and reversal stations. Evidence is
`evidence/contact/2026-10-22-supported-periodic-time-refinement.json`.
Both paths completed and every accepted increment met the numerical contact
criteria; this is not an independent material or continuum verification.

| Increments | Final force magnitude (N) | Displacement L2 | Mean PEEQ | Newton iterations |
| --- | ---: | ---: | ---: | ---: |
| 128 | 4.384499107 | 0.01100073837 | 0.06352729940 | 1113 |
| 256 | 4.469811614 | 0.01100023434 | 0.06355785533 | 2075 |

The final force difference is 1.91% relative to the finer path, whereas mean
PEEQ differs by about 0.048%. The signed relative-gap work approaches zero
(-7.27418e-5 versus -1.86169e-5 J); it is not relabelled plastic dissipation.
This narrows the temporal uncertainty without removing the separate spatial
sensitivity or external-reference gate. In particular, the earlier 8-to-12
mesh comparison at only 64 increments is not a temporally converged spatial
study. The next mesh comparison must use a sufficiently refined common path.
Recorded costs include setup and observation, with brief independent test
overlap; they are not a speed comparison. These runs loaded the pre-KSP-reuse
implementation, and their runtime source identity is retained in the report.

### Accepted-series output boundary audit (2026-10-21)

A secondary audit while the frozen private replay runs found two mutable
descriptor aliases: caller-owned nested field selectors and a provider-owned
state schema. Two tests failed before the fix; the series now owns independent
small descriptors and freezes the schema only after its first successful
frame. It does not add a whole-state copy or change the numerical trajectory.

Four additional failing counterexamples showed that the streaming adapter
accepted a scalar under `cauchy_stress`, a different declared stress measure,
a missing source location, or a location change between frames. It now
requires its documented per-cell finite real 3×3 Cauchy stress and retains
the source location/measure in metadata. A changed provider descriptor is
rejected before writing the next frame. Existing valid frames can still be
closed as an incomplete segment; no physical validation is inferred from
correct labels or shapes.

The final combined output/project/cell-transport/journal/topology subset
passed **85 tests in 3.73 s**. This is a short source regression run, not a
performance measurement. The ongoing private full run remains on its frozen
`8da7b93b` wheel: its provider already satisfies these contracts, and no
runtime files were replaced during execution.

### Interrupted output follow-up (2026-10-21)

A two-frame serial Hex8 counterexample exits the child process with `os._exit`
after `write_fields` returns, skipping context cleanup and destructors. Before
the fix, reopening the heavy-data file fails with `bad object header version
number`. Complete-frame HDF5 flushes eliminate this reproduced failure.
An atomically replaced `*.partial.xdmf` labels the last written view incomplete;
an independent meshio reader verifies both frames, the volume topology and
point/cell data after the child has stopped. Clean close atomically publishes
the ordinary index and removes only its own partial index. Existing partial
output is not overwritten. Flush, partial-index and final-index failures are
tested explicitly.

The combined output/layout/project/response-material/history subset passed
**158 tests in 6.74 s**; lint and diff whitespace checks passed. These are
correctness checks, not a performance experiment. The frozen private replay
was not modified. This fix does not recover its previously damaged artifact,
provide concurrent readers or guarantee mid-write/power-loss safety. The
current per-frame-group format cannot merely toggle SWMR: that protocol
requires predeclared appendable datasets and a separate compatibility design.
The final result metadata and verified checkpoints still own completion and
restart respectively; an incomplete visualization is neither.

[cell-periodicity]: https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_theory.06.080.html
[petsc-basic]: https://petsc.org/release/src/snes/linesearch/impls/basic/linesearchbasic.c.html
[normal-al]: https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_theory.06.087.html
[abaqus-contact-energy]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEEXARefMap/simaexa-c-contactenergy.htm
[abaqus]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-genlcontnumcontrolsstd.htm
[nafems]: https://www.nafems.org/publications/pubguide/list/
[nafems-contact-3]: https://www.nafems.org/downloads/FENet_Meetings/Glasgow_UK_Oct_2004/FENET_Glasgow_Oct2004_DLE_Becker.pdf
[simscale-r0094]: https://www.simscale.com/docs/validation-cases/3d-punch-rounded-edges-nafems-contact-benchmark-2/
[frohne-contact]: https://www.math.colostate.edu/~bangerth/publications/2013-plasticity.pdf
[abaqus-al-verification]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-contsmslcont.htm
[abaqus-general-controls]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-genlcontnumcontrolsstd.htm
[initial]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-adjustsurfaces.htm
[distance]: https://arxiv.org/abs/1803.09687
[meshed-rigid]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-finslcontdefmeshrig.htm
[comsol-verification]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/StructuralMechanicsVerificationExamples.pdf
[moose-return]: https://mooseframework.inl.gov/source/materials/RadialReturnStressUpdate.html
[moose-weighted-gap]: https://mooseframework.inl.gov/source/constraints/ComputeWeightedGapLMMechanicalContact.html
[moose-petrov]: https://mooseframework.inl.gov/syntax/Constraints/Petrov-Galerkin_approach_for_Lagrange_multipliers.html
[abaqus-time-points]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-directcyclic.htm
[abaqus-output-times]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAECAERefMap/simacae-t-simodbeditorfield.htm
[duong]: https://arxiv.org/abs/1712.01179
[dual-mortar]: https://arxiv.org/abs/2201.01165
[dual-mortar-space]: https://doi.org/10.1137/S0036142999350929
[dual-mortar-boundary]: https://www.sciencedirect.com/science/article/pii/S0045782510003087
[variational-mortar]: https://doi.org/10.1016/j.cma.2018.07.020
[point-line-surface]: https://doi.org/10.1002/nme.5743
[mortar-integration]: https://doi.org/10.1007/s00466-014-1093-2
[closest-point-domain]: https://publikationen.bibliothek.kit.edu/1000009437
[mortar-segmentation]: https://mediatum.ub.tum.de/doc/1109994/1109994.pdf
[laursen-corner]: https://doi.org/10.1002/cnm.1640101103
[conic-integral]: https://arxiv.org/abs/1005.0990
[getfem-contact]: https://getfem.readthedocs.io/en/latest/userdoc/model_contact_friction.html
[dealii-step42]: https://dealii.org/9.7.0/doxygen/deal.II/step_42.html
[polygon-clipping]: https://doi.org/10.1145/360767.360802
[oriented-volume]: https://doi.org/10.1016/j.cma.2007.06.002
[distance-field]: https://doi.org/10.1016/j.cma.2013.02.015
[abaqus-rigid-smoothing]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMODRefMap/simamod-c-rigidsurf.htm
[burman]: https://arxiv.org/abs/1609.03326
[hansbo-stabilized]: https://doi.org/10.1016/j.finel.2016.03.005
[exact-penalty]: https://www.sciencedirect.com/science/article/pii/S0045782520302929
[nitsche-small]: https://doi.org/10.1137/12088344X
[nitsche-large]: https://doi.org/10.1016/j.cma.2017.07.015
[alart-curnier]: https://doi.org/10.1016/0045-7825(91)90022-X
[petsc-vi]: https://petsc.org/main/manual/snes/#variational-inequalities
[duong-friction]: https://arxiv.org/abs/1808.10420
[comsol-settings]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.146.html
[acl]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-contactcontrolsstd.htm
[comsol-method]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.143.html
[friction]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-coulombfric.htm
[vfric]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAESUBRefMap/simasub-c-vfric.htm
[comsol-solver]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.152.html
[petsc-snes]: https://petsc.org/main/manual/snes/
[dolfinx-nonlinear]: https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.petsc.html
[asimov-contact]: https://github.com/Wells-Group/asimov-contact
[abaqus-output]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-contactgeneral.htm
[comsol-vars]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.155.html
[abaqus-verification]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-finslcontdefrigid.htm
[abaqus-c3d8-input]: https://docs.software.vt.edu/abaqusv2025/English/SIMAINPRefResources/ei34srsc_surf.inp
[abaqus-surface-verification]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-finslsurftosurf.htm
[comsol-mesh]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/StructuralMechanicsModuleUsersGuide.pdf
[gamg]: https://petsc.org/main/manualpages/PC/PCGAMG/
[profiling]: https://petsc.org/release/manual/profiling/
[benchmarking]: https://htor.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf
