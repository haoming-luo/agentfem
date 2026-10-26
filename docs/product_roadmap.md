# AgentFEM Product Roadmap

## Direction

AgentFEM is building a readable, dependable, and extensible finite-element
platform for people and AI agents. It does not replace the finite-element
kernel already provided by FEniCSx, Basix, PETSc, and MPI. AgentFEM owns the
engineering language, numerical procedure, state lifetime, result contract,
verification evidence, and extension boundary around that kernel.

The public workflow remains:

```text
Study -> Model -> scientific assets -> model.step(...) -> SimulationResult
```

Capability claims are promoted by executable evidence. A formula, an example,
or a successful run is not by itself a validated engineering capability.

## Current product phase: the 0.4 foundation

### Windows onboarding simplification (2026-10-26)

The new user field report identifies Ubuntu distribution metadata download,
not the numerical stack, as the main installation blocker. Promote the
already reported TUNA image/import route to the short onboarding path:
WSL platform → checked Ubuntu image → ordinary account → mirrored environment
and installed first solve. No checkout, optional helper or Windows Features
checklist is a prerequisite. Retain mirror choice in local handoff notes and
offer a one-action manual fallback only when the agent lacks permission.
Detailed installer recovery remains available separately. The report's root
installation and older package success are evidence, not recommended defaults
or proof of current-release/project compatibility. New non-root quickstart
needs clean Windows/WorkBuddy acceptance; local documentation checks and
PowerShell parser CI do not substitute for it. This work does not change
solver algorithms, contact maturity or simulation performance claims.

PR #125 is merged at `5e6cba5b` after all applicable checks passed for
`11accb3b`, including PowerShell 5.1/7, Linux/FEniCSx and macOS installed-wheel
acceptance. This is engineering integration, not a new Windows host field trial.

### Active sprint decision board (2026-10-26)

The numerical follow-up starts from merged main `a862e123` (including PR #123).
It prioritizes bounded diagnostic and execution improvements,
not another capability expansion. CI establishes source/installation
regressions, not nonuniform spatial accuracy or commercial parity.

| Order | Work | Decision gate |
| --- | --- | --- |
| P0 | Keep engineering integration separate from scientific maturity | The release contract names periodic J2/contact as experimental; no general contact or mesh-independence promotion |
| P0 | Separate path-work error from spatial contact error | Independent elastic assembly/MPC audit, per-increment work/energy accounting, then one graded plastic path refinement at common stations |
| P1 | Remove redundant observation work | Scalar energy/feature histories must not assemble unused global tangents or copy whole fields; retain current geometry, accepted-state transactions and legacy observers |
| P1 | Preserve ordinary-use acceptance | Reuse unaffected installed output/restart evidence explicitly; validate changed assembly paths without AI-provided executable glue |
| Exit | Commit tested improvements and retain open science gates | No release tag or commercial-parity claim on partial validation; reuse unaffected full composite evidence |

Current progress: the independent discrete elastic audit agrees with production
force/energy on uniform and graded 8/12-cell meshes. A primal affine J2 energy
ledger now distinguishes stored energy, plastic/friction loss, penalty energy
and accepted external work. Portable two-phase 1↔2-rank checkpoint checks retain
its complete scalar history; this does not admit distributed implicit contact.
Scalar contact-energy observation no longer needs full residual/tangent assembly.

The initial-yield diagnostic explains almost all first-station work imbalance
before contact in the tested graded fixture. A same-source small full-cycle
control reduces the initial increment alone, with 232 → 237 accepted increments:
final relative energy residual falls from 0.2106% to 0.0298%, while final force
changes about 0.0014%. This is an accuracy diagnostic, not a runtime-speed claim
or justification for a new universal default. The graded full cycle now also
completes: joint initial-step and plastic-cap refinement lowers the final
relative work/energy residual from 0.2046% to 0.00625%, with a 0.1137% final-force
change. This separates path accounting error from spatial sensitivity; it does
not close the spatial or independent same-input reference gate. Details,
raw evidence and exact limits are maintained in the
[contact sprint ledger](internal/contact_0_4_4_local_parity.md).

The observation optimization has a separate 64-increment ABBA confirmation:
4.73% less solve/observe time within the energy-enabled candidate. A net
comparison against merged main measures 3.68% less time, including the new
energy history, with identical physical arrays and common accepted histories.
These are bounded serial implicit-fixture results, not additive percentages
or explicit-composite speed claims. The final merged-main candidate also passes
163 installed regressions and the ordinary `init → check → run` workflow;
CLI discovery no longer initializes MPI merely to display help or version.

Windows onboarding is a separate bounded engineering change, merged in PR #125:
reuse WSL/Ubuntu, honor an explicitly chosen mirror, remove the mandatory
source checkout, and give one manual action when permission/download blocks a
stage. A fresh Windows/WorkBuddy machine remains a user acceptance check;
PowerShell contract CI alone is not such a test.

The fixed-width edge-belt diagnostic must also record its sampled area: a
discontinuous indicator has mesh-dependent quadrature error. Complement it
with a continuous polynomial edge/center partition of unity whose reference
areas are known analytically. This diagnoses pressure redistribution without
inventing an error estimator or changing the contact law.

Historical entries below remain evidence, not a second current backlog.

Current integration results: SVD reuse passed independent history/rotation/
conditioning and global-path checks, with 3.15% less solve-and-observe time in
the fresh 64-increment ABBA confirmation. It changes neither the material law
nor the accepted path. A native wheel from `51c4a911` was installed into an
isolated AgentFEM environment with shared numerical dependencies. Ordinary
`agentfem run` partial/restart/continuous paths agree in displacement, material
state, contact force and work history. Installed output, interrupted publication,
history/retention and independent contact-reference suites pass. This is not a
clean Windows installation or new full-composite performance proof. Details and
the reconstruction protocol are in
`evidence/contact/2026-10-25-installed-periodic-contact.json`.

PR #118 is merged at `897f1f74` after final source `d8392e98` passed all applicable
checks in run 40121006505, including Linux native distribution, FEniCSx and
macOS installed-wheel acceptance. The final serial suite reports 4988 passed
and 428 skipped; optional-native and rank-specific skips are not native coverage.
Separate native and distributed gates pass. This is engineering integration,
not a 0.4.4 release tag or promotion of experimental contact.
An intermediate Linux run exposed
one ill-conditioned first-Piola component comparison failure (4961 other tests
passed). The failure is preserved; an independent high-precision diagnosis and
per-point objective tensor comparison replace that componentwise test, without
changing the runtime or its tolerances. Fresh Linux confirmation at `e4207c56`
passed 4976 serial tests and subsequent distributed/release-smoke checks in
run 40115587874. The final-head confirmation above also covers the later
boundary-admission safety changes. See the precision follow-up in the sprint ledger. The workflow and
material guide also separate finite-strain contact from small-strain
J2/Chaboche and defer geometry/friction/MPC combinations to one current
capability matrix rather than stale one-tool restrictions.

The release-scope audit also found a correctness hazard: several small-strain
material providers advertised boundary Operators they did not assemble. Native
J2/Chaboche, power-law creep, quasistatic/harmonic Maxwell and neutral small-strain
providers now reject these combinations before Step construction, with one
addressable reason shared by capability inspection and compatibility builders.
This is a necessary safety correction, not a new contact capability.

The spatial gate is narrower, not closed: the controlled 4/8-cell plastic
first-station force gap is 6.52%, with different sampled contact activity.
On the same coarse mesh, supported contact quadrature 7 -> 11 changes force by
only 0.00557%; halving the coarse plastic increment cap changes force by 0.02025%.
Raising coarse bulk quadrature 6 -> 8 changes force by only 0.0008892%.
None explains the spatial gap on its own. Independent polynomial P2/Q2 trace
tests pass force, moment, virtual-work and tangent integrals. The bounded
12-cell continuation now completes the first station with 305 accepted
increments. Forces on 4/8/12 meshes are 4.48528146 / 4.19262706 / 4.10251562 N;
successive coarse-relative differences decrease from 6.52% to 2.15%.
The 8/12 sampled active areas agree, but peak pressure continues to grow and
edge-weighted redistribution exceeds the total force change. This is a useful
trend, not a continuum-error bound or full-cycle accuracy certificate.
Evidence: `evidence/contact/2026-10-25-periodic-plastic-pressure-moments12.json`.

**Next bounded science gate:** stop uniform refinement here. First compare
contact-edge resolution against the same physical patch and controls, using
integrated reaction and smooth pressure moments rather than peak pressure alone;
then obtain an independent same-input reference. Any claimed improvement needs
its own fine-path sensitivity check. Do not simultaneously open a new contact
law, penalty sweep, full-cycle sweep or larger uniform mesh. Promotion remains
separate from the engineering fixes already merged.

The diagnostic now supports an edge-graded mesh with unchanged cell count,
physical patch, periodic trace equations and material/solver controls. It is
research fixture geometry, not a new public mesh generator. Positive cell
quality, physical volume/patch areas and checkpoint mesh identity have focused
tests. Elastic 8/12-cell force differences decrease from 1.01% (uniform) to
0.60% (graded); this is a refinement comparison, not a continuum-error bound.
The graded plastic first station completes through an ordinary checkpoint
continuation. With 14,739 displacement DOFs its reaction differs from the
46,875-DOF uniform-12 comparison by 0.416%, versus 2.196% for uniform 8.
The finer uniform result is not independent truth; sampled peak pressure
continues to rise. An independent same-input reference and graded-path increment
sensitivity remain open. No larger uniform sweep is started. Full details and
the interrupted/continuation evidence are in the contact sprint ledger.

The contact execution candidate sums current quadrature matrices sharing an
ordered nodal stencil before generating scalar COO entries, then inserts the
coalesced rows through PETSc's subset-CSR interface. It changes neither contact
quadrature, projection, physical law nor state ownership, and introduces no
new user knob or dependency. A 32-increment cyclic screen reduces submitted
COO entries from 7,661,700 to 340,200 and solve/observe time by 5.73%; independent
64-increment confirmation measures 5.48% less time with the same 208 Newton
corrections and matching four-station fields, states, contact force and penalty
potential. See `evidence/performance/2026-10-25-contact-stencil-confirmation.json`.
Full-matrix storage reuse alone is withdrawn:
its small screen showed no meaningful gain. Do not add these percentages to
earlier optimizations or transfer implicit timing to the explicit composite.

The existing J2 tangent backend now packs its nine derivative directions into
small rectangular products and reuses their common transform, without another
public tuning switch. Fresh serial cyclic P2/J2/plane-penalty/MPC ABBA evidence
measures 8.32% less solve-and-observe time with the same 32 accepted increments,
113 Newton corrections and four-station field/state/contact comparisons.
The 4096-point tangent kernel improves by about 35.78%, not the whole solver.
The original near-zero-component comparison failure and its justified per-point
tensor-scale follow-up are both retained; see
`evidence/performance/2026-10-24-j2-tangent-block-confirmation.json` and the
contact sprint ledger. Do not add this result to prior percentages or transfer
it to the explicit-composite run. The subsequent SVD-reconstruction acceptance
is recorded above and is a separate measured improvement, not an added percentage.

The user's 2026-10-23 clarification keeps **solver capability and throughput
first**. A separate contributor is handling WSL installation: inspect that
work for later integration instead of starting a competing installation sprint.
PR #119 and its dependent #120 are now merged into main at `d87f1c0e` and
`a7f69a8a`. Same-version build identity, installation identity, cross-package
locking, bounded host probes and preserved restart paths have source-level
regressions. The final onboarding head `af36afde` passed Windows PowerShell
5.1/7, numerical and installed-wheel CI. WSL/elevation remain mocked in those
Windows tests: real clean Windows/WorkBuddy acceptance is still explicitly
pending, and no new installer bundle is released. This closes source integration,
not the customer-acceptance gate or solver sprint.
The joined body/viscosity policy now has a complete late-state ABBA comparison
(5.69% less mean advancement time, identical accepted fields and physical
history). This is not a full-run gain. Current smooth-interior Q1 sign evaluation
with two-ray fallback also preserves the complete late accepted state. Combined
ABBA screening shows 4.83% less mean advancement time, not the sum of component
improvements. Process-local worker screening did not select a faster budget;
a versioned contact
restart contract separates explicitly partition-invariant execution budgets
from physical identity. Legacy checkpoints remain strict until resaved through
their original runtime. No silent identity bypass is permitted.
Keep the graded finite-J2 spatial study separate from both timing experiments.
The immutable-array validation cache has been withdrawn: its isolated late-path
comparison cannot distinguish the small mean difference from timing drift.
Keep the snapshot-before-validation correctness fix without the cache.
A Procedure-owned linear workspace now keeps current PtAP/tangents across
frozen normal-augmentation equilibria and releases resources at attempt exit.
Its small allocation-only gain enables a separate opt-in adaptive GMRES
preconditioner policy. The isolated 12-cell/32-increment screen reduces mean
time by 13.39%, with accepted station arrays within declared tolerances; fresh
48-increment confirmation of the integrated public route measures 13.73%,
with 48 accepted increments and 533 Newton corrections in every run. This is
an implicit-solver result, not an explicit-composite or full-run speed claim.
The graded 16/256 study is now complete: contact tolerances pass, but final
reaction changes 7.2893% from graded 12/256. Grading alone has not closed the
spatial gate. Stop blind refinement/penalty tuning; the next discretization
study must address compatible volume, contact-trace and periodic spaces.
Do not silently use P2 volume interpolation with the current P1 nodal AL
contract. The commercial-parity and independent-reference gates stay open.

The combined serial implicit execution route has also completed an isolated
12-cell/48-increment ABBA comparison: 24.32% less mean solve-and-observe time
relative to fresh material trials, separate outer workspaces and direct solves.
All four paths retain 48 accepted increments and 533 Newton corrections, and
all four-station field/state/contact/work comparisons pass. This evidence
supports the composition, not addition of its individual percentages, and
does not change the explicit-composite timing claim. See
`evidence/performance/2026-10-23-combined-affine-execution.json`.

The eight-cell P2 diagnostic now separates contact and bulk integration:
contact quadrature 7 -> 11 changes final reaction by 0.0351%, then bulk
quadrature 4 -> 6 by 0.00887%. These do not explain the much larger spatial
gap, and defaults remain unchanged. A separate native worker-local nodal-gather
prototype preserves the late accepted path but its 1.41% mean timing change
is smaller than the 2.17% baseline drift; it is withdrawn rather than added as
another public performance switch.

Further diagnostics distinguish two remaining errors rather than attributing
everything to contact: at fixed eight-cell P2 discretization, 32 -> 64 -> 128
increments change the final plastic reaction by 6.41% then 3.16%. With plasticity
disabled by a high yield stress, common Gaussian-penalty 8 -> 12-cell controls
change reaction by 4.46% for P1 and 1.02% for P2. This supports investigating
compatible volume/trace approximation and path refinement separately; it does
not certify continuum accuracy, prove locking or authorize P2 nodal AL.

An output audit found a separate correctness issue: integration-point cell
recovery normalized reference-rule weights without the geometric Jacobian.
The physical-measure correction now passes analytical distorted Q1/Hex8 and
curved-triangle tests, scalar/vector/tensor recovery, and two-rank checks for
owned/ghost publication, an empty rank and collective invalid-measure rejection.
Raw constitutive states and solver updates are unchanged; metadata distinguishes
mesh-geometry weighting from any inferred deformation-volume weighting.

The next work-reduction mechanism is now an opt-in core increment policy,
`steps.automatic(inelastic_prediction_safety=...)`, using the last accepted
plastic increment to cap the ordinary Newton-based next proposal. It does not
relax the declared plastic-increment limit or change mandatory load knots.
The bounded serial global-J2 strong/affine routes share explicit admission,
accepted-state rollback and versioned restart identity. Regional/mixed/MPI
prediction remains rejected, without restricting their existing policies.
On a cyclic P2/J2/plane-penalty/MPC fixture, source-frozen ABBA means are
186.400 s ordinary and 114.687 s predictive (38.47% less), with rejected
attempts 140 -> 13 and Newton corrections 1300 -> 806. Common-station reaction
differs by at most 0.204% from a finite 512-increment reference; that reference
does not establish continuum accuracy. This is a new increment algorithm,
not another unchanged-path execution gain, and cannot be added to 24.32% or
applied to the explicit composite runtime. Compact evidence is in
`evidence/performance/2026-10-23-inelastic-increment-prediction.json`.
Prospective 1024-step-reference confirmation retains the same predictive
232-increment path. Halving the plastic limit gives 449 accepted increments
and reduces maximum reaction/displacement/mean-PEEQ differences to that
reference from 0.202%/0.0384%/0.0291% to 0.104%/0.0119%/0.00166%.
Fixed-reference 512 -> 1024 final reaction still changes by 0.405%: preserve
the remaining path-accuracy uncertainty rather than selecting the more
flattering reference. The next scientific gate remains compatible spatial
approximation at controlled path resolution, not another controller knob.

The eight-cell P2 controlled-path follow-up reached common factors 0.25 and
0.5, then stopped at its 1800-second runner limit. At the first station,
path refinement changes reaction by about 0.27% from fixed 128 increments,
while the four/eight-cell controlled paths still differ by about 6.52%.
Unloading opens contact correctly at 0.5. Preserve these partial observations
and the timeout; they do not close full-cycle accuracy or spatial convergence.
See `evidence/contact/2026-10-24-periodic-p2-controlled-path8.json`. No larger
blind mesh/parameter sweep follows from this result.

| Order | Work | Current decision / acceptance boundary |
| --- | --- | --- |
| 1 | Retain unchanged material trials and scoped linear work | Implemented for bounded serial J2/AL; combined 24.32% timing reduction with station parity and mutation/rollback/restart guards. Unsupported materials/mixed/MPI use ordinary refresh; no scope expansion inferred. |
| 2 | Isolate spatial/path error and reduce rejected work | Spatial gate remains open. Accepted-plastic prediction now reduces rejected work on a bounded cyclic fixture; it is not an error estimator. Hold the physical patch fixed and keep volume/trace approximation and path-refinement evidence separate. |
| 3 | Explicit body/contact critical-path reduction | Selected joined-viscosity/interior-sign route remains a late-path result, not a new full-run claim. Reuse the 25-frame baseline; the new nodal-gather prototype is withdrawn rather than added as an unproven switch. |
| 4 | Bounded release integration | Solver/output and main-integration head `197ba2c8` passes its full remote CI, including numerical and installed-wheel gates. Subsequent tangent changes require their own remote checks. Scientific claims remain limited to passed geometry/law combinations; PR #118 remains a development candidate while spatial/reference gates are open. |

The output/recovery review reproduced and fixed two additional failure paths:
a closed serial writer could reopen and truncate its previous HDF5 file, and
completion cleanup could prune earlier checkpoints before result publication
succeeded. Closed writers now reject re-entry and writes; failed closes remain
explicit, and repeated cleanup cannot promote an interrupted index. Completion
retention publishes only surviving checkpoint references before deleting owned
older files. Result-construction/publication failures preserve those files.
The affected output/archive/project set passes 124 tests (one MPI-only skip).
After interruption, the focused material reuse, linear workspace, increment
prediction and checkpoint review passes 116 tests (one MPI-only skip). These
are overlapping risk-based regressions, not new timing or scientific-validation
claims. Existing generated `site/` changes remain untouched.

The first retained-material source candidate is scoped to a single equilibrium
attempt and compares evaluated old/new gradients, committed/trial state, current
stress/tangent/energy buffers, material parameters and time. State retains no
second response array, and the transient guard is never checkpointed. The
contact residual and global Jacobian remain current; this is not Jacobian
lagging. See `tests/test_retained_material_trial.py` and
`tools/benchmark_retained_material_trial.py` for mutation and ABBA checks.

### 2026-10-22 implementation priorities

Keep the implicit-contact scientific gate and the explicit-composite throughput
gate separate. A faster explicit implementation does not close implicit contact
convergence; a completed composite run does not establish external-code parity.

Sprint evidence now includes a complete frozen 100,000-increment run in
3 h 24 min 56 s, with all 25 frames, accepted histories and checkpoint arrays
equal to the preceding baseline. Independent VTK reads every frame correctly.
The controlled late-path comparison supports 12.11% less advancement time;
do not attribute the entire larger whole-run difference to code changes.
Reusable changes cover sufficient cell admission, exact topology/coordinate
reuse, operational restart identity, linear-option transport, prepared-operator
failure cleanup and opt-in lossless history archives with a data-only reader.
The new independent confined cyclic plastic/contact reference passes, while
the inhomogeneous spatial convergence and external-comparator gates stay open.
The complete 16-cell/256-increment contact path meets the original nonlinear
acceptance bounds, but the 12→16 final reaction difference is still 5.2481%.
Mean PEEQ and displacement L2 change much less (0.3729% / 0.2490%); neither
is a substitute for reaction convergence. Preserve this unresolved distinction
when choosing the next discretization/contact study.
An explicitly separate elastic-limit cycle also retains a 6.5122% 8→12
reaction gap with zero plastic strain. Prioritize spatial discretization and
the contact constraint representation; do not assume the J2 update is its cause.
Exact active-normal elimination on those same elastic meshes agrees with
augmentation to below 3e-8 N in total reaction, while preserving the spatial
gap. Further penalty tuning is not the priority for that observed difference.

1. **Measure and remove redundant execution.** The completed 100,000-increment
   composite run still attributes approximately 89% of solver wall time to
   residual assembly. Profile the late damaged configuration, including the
   overlapping body/contact branches, before changing algorithms. First test
   whether a coherent body/contact execution change removes remaining work.
   The sufficient native whole-cell screen is now installed and verified;
   retain uncertain-cell fallback, the material law, increment, numerical mass
   and accepted-state contracts rather than repeatedly retesting this same gain.
2. **Make execution evidence explainable.** Distinguish inclusive body cost,
   geometry cost, owner-thread waiting and publication cost. These are concurrent
   or nested quantities, not additive percentages. Use the existing performance
   ledger; do not create a second authoritative solver/state or benchmark policy.
3. **Close the actual implicit-contact accuracy gap.** Continue the bounded
   finite-area-supported J2/plane fixture with common accepted load stations and
   independent spatial/increment refinement. Diagnose contact integration and
   material path errors separately before extending geometry or enforcement.
   Keep the independent comparator and unsupported sharp-feature boundaries
   explicit; do not promote the 0.4.4 parity target by test count.
4. **Preserve recovery and usable output.** Keep the verified 25-frame run and
   its final checkpoint unchanged. New profiling runs write to separate locations.
   A restart identity mismatch must be investigated, never bypassed to make a
   timing experiment work. The selected combined change has now passed frozen
   installed whole-path acceptance; another full run is deferred until a new
   measured change merits it. History encoding is an output choice, not a
   solver/physical policy or a substitute for saved field frames.

The execution decomposition follows the distinction between element restriction,
quadrature work and global reduction in [libCEED's interface concepts](
https://libceed.org/en/latest/libCEEDapi/). Displacement-bounded candidate reuse
is informed by [LAMMPS neighbor-list maintenance](
https://docs.lammps.org/Developer_par_neigh.html), but any Q1 contact reuse requires
its own complete-coverage and moving-geometry tests. Neither source supplies an
AgentFEM speedup number.

### 0.4.4 bounded contact-parity sprint (target, not a capability claim)

The next release target is **local, explicitly bounded parity** for serial
three-dimensional solid-versus-prescribed-rigid, quasistatic implicit contact
with finite-strain J2. This means a user can state and independently verify
the same supported physical problem through the ordinary Model/Step/Result
workflow; it does not mean all Abaqus/Standard general contact, identical
floating-point fields, or automatic admission of every triangulated tool.

The ordered nine-gate task and technical bibliography are maintained in
[the 0.4.4 contact sprint record](internal/contact_0_4_4_local_parity.md).
The October 26 independent elastic assembly audit now passes uniform and
graded 8/12-cell controls, including periodic virtual work. This narrows the
spatial investigation but does not certify the plastic full cycle. The next
bounded checks remain graded plastic increment sensitivity and loading/unloading
histories. The primal affine J2 result now records restartable accepted energy
components and combines existing affine/tool work, with explicit unavailable
status for incomplete historical evidence; this is diagnostics, not automatic
scientific promotion.
Promote only combinations that pass independent same-input comparison,
mesh/increment/enforcement convergence, residual--tangent and lifecycle
checks. A failed nonconvex sharp-feature gate narrows the release statement;
it must not be hidden by a visually plausible contour or another penalty
choice. A moving feature boundary in a surface integral and actual physical
edge/vertex contact are separate numerical questions: passing one does not
license the other. Distributed implicit contact, deformable--deformable/self
contact, shell-thickness contact and arbitrary CAD remain outside this bounded
target.

The current serial affine Newton implementation reuses the total residual and
Operator responses only within an unchanged trial, avoiding a second contact
evaluation before the tangent solve. Small frozen-old-Procedure comparisons
cover fixed/moving sphere contact, friction and automatic plastic cutback;
this is not an explicit-composite speed claim. A separate bounded analytical
plane diagnostic now includes the actual MPC affine offset, distinguishing
immovable penetration from mere multiplier redundancy. Neither improvement
admits AL under arbitrary MPC; compatible multiplier spaces and their
independent pressure/convergence evidence remain required.
An option-contract audit also corrected explicit basic/no-search Newton:
finite valid full steps may temporarily increase the residual, while invalid
material trials still reject. Backtracking remains the default; serial and
distributed affine paths share the meaning with PETSc, and affected legacy
basic-mode checkpoints cannot silently change algorithm.
The bounded periodic-plane route now has joint work/restart and explicit
mesh/increment/penalty sensitivity records. A reaction audit distinguishes its
point-pinned one-sided-load demonstration from a separately declared,
finite-area supported fixture; changing the fixture is not credited as a
solver improvement. Neither numerical acceptance nor penalty insensitivity
closes the independent spatial-accuracy gate.

An interrupted long output exposed buffered HDF5 metadata loss between saved
stations. The serial unified writer now flushes complete frames and atomically
maintains an explicitly incomplete index, removed after clean final publication.
Abrupt-process-exit and independent-reader checks cover that bounded recovery
case; they do not establish SWMR, mid-write/power-loss recovery, or repair of
older damaged files. Checkpoints remain the only restart contract.

### 0.4.3 released composition and next gate (2026-10-17)

Version 0.4.3 was promoted by one composed scientific workflow, not
by accumulating isolated feature names. The first bounded native slice is
three-dimensional finite-strain J2 with ordinary strong constraints and one
analytical rigid plane under frictionless penalty contact. The plane may be
fixed or have prescribed translation/rotation. Its contact potential, residual
and semismooth tangent enter the same Newton system; the accepted J2 state,
contact energy, generalized tool work, reactions and serial checkpoint share
one increment lifecycle. The elastic small-strain contact provider must not win
dispatch merely because a history material also exposes `young` and `poisson`.

This is an experimental serial composition, not a general forming solver.
The released bounded route adds one triangulated rigid tool with current-trial
BVH projection and normalized prescribed motion, but only while active contact
lies on planar facets or coplanar interior seams. Its force--moment--work and
joint restart evidence are recorded in the
[candidate gate](internal/release_0_4_3_candidate_gate_20261016.md).
Curved/creased die transitions, periodic MPC reduction, mixed
displacement-pressure elements, friction and distributed implicit assembly
remain explicit follow-ups. Promote them one at a time only with tangent,
force/work/energy, failed-attempt rollback, restart and independent refinement
evidence. For MPC, reduce the *total* residual and tangent including contact;
do not bolt an unreduced contact force onto a converged plastic solve.

The tagged Linux wheel and source archive passed the installed-use, scientific,
MPI and macOS same-source release ladder and were published as 0.4.3. The
[candidate record](internal/release_0_4_3_candidate_gate_20261016.md) retains
the exact evidence and the limits. Future promotions still require reviewed
supported/unsupported combinations, independent references, sensitivity
checks and installed-platform evidence. A passed unit patch or one successful
forming picture is not release certification.

The next experimental slice now has exact active-branch sphere/cylinder
curvature, compared against an independently differentiated UFL penalty
potential. A separate serial affine-J2 composition reduces the full
material-plus-fixed-contact residual and tangent through the same periodic
MPC on Newton and line-search trials, with fixed-tool restart and dual
evidence. These are source-level candidate results, not a new release claim;
see the [bounded candidate record](internal/smooth_implicit_contact_mpc_candidate_20261017.md).
Noncoplanar triangulated transitions, MPI implicit contact, friction and
mixed pressure retain separate gates. Measure complete
accepted-step cost before choosing an acceleration target.

The follow-on source candidate composes normalized moving rigid contact with
serial affine-periodic J2 using the same full-space Operator and accepted
increment lifecycle. It adds provider-owned tool force/coordinate work,
restartable dual history, pose rollback and checkpoint rejection of incomplete
work paths; see the [moving affine contact record](internal/moving_affine_contact_candidate_20261017.md).
This closes the bounded moving-tool/MPC composition gate only. Noncoplanar
triangle transitions, MPI implicit assembly, friction, mixed pressure and
independent curved-structure validation remain open. The source candidate is
not promoted to a released capability by these local tests.

The next serial source candidate removes an unnecessary contact/strong-BC
disjointness restriction by lifting the native tangent and retaining
constrained matrix rows. A full-versus-quarter indentation comparison passes
at equal resolution. Its Hertz half-space screen does **not** yet pass an
independent accuracy gate: finite support, mesh and penalty effects still need
separation. See the [Dirichlet and Hertz candidate record](internal/implicit_contact_dirichlet_hertz_candidate_20261018.md).

The same source branch now composes multiple named frictionless tools in both
the ordinary serial strong-BC J2 Step and the affine-periodic J2 route, with
the total residual and tangent reduced by the same MPC. Strong-BC J2 also admits fixed or prescribed-moving analytical-plane
Coulomb pairs with separate accepted stick/slip and pose states, a generally nonsymmetric tangent,
separate friction loss channels and serial restart. These are bounded source
candidates, not a release promotion. Open/nonconvex sharp features, curved or
faceted friction, friction with affine MPC, distributed implicit assembly and
an external friction benchmark remain separate gates. For rotating planes the
current-pose master transport is included in the fixed-old-state tangent;
focused FE directional checks, nonzero moment, tool work and restart pass.

A further narrow source candidate admits a **closed convex** outward-oriented
polyhedral rigid tool for serial frictionless finite-J2 contact. Its exact
interior support-plane gap gives a deterministic limiting normal and tangent
at a noncoplanar feature; it does not smooth a crease or relax the old open
triangle rejection. Local Operator and planar-limit global tests pass, but a
small structural load path across one noncoplanar feature and its interrupted
restart also pass. External sharp-feature comparison, broad mesh/penalty
evidence and scalable search remain open. See the
[closed-convex candidate record](internal/closed_convex_implicit_contact_candidate_20261018.md).

The bounded serial candidate now also carries a selected-branch Coulomb
tangent for that certified closed-convex tool, two independently moving
frictional pairs, a two-region J2 material map, atomic restart and raw
accepted contact-point export. CG1/CG2 hexahedral contact quadrature can be
explicitly refined for discrete convergence checks; defaults stay unchanged
because higher rules did not monotonically improve the Hertz force screen. The
source-targeted and isolated installed-wheel smoke pass on Apple Silicon,
but the closest small-strain Hertz screen still has a 1.4% force discrepancy
after a farther-boundary Q2 mesh refinement. Sampling the Hertz pressure on the
same trace exposes a much larger pressure-shape difference, and changing the
face rule does not improve force and pressure together. Sharp-friction observables
remain mesh/penalty sensitive,
and external same-input Abaqus results are absent. This work remains an
**experimental, limited serial contact capability**, not local Abaqus parity
or a new release. The next promotion needs a frozen external comparator and
controlled mesh, domain, penalty and increment limits before broader
geometry/material/MPI admission.

A later experimental geometry slice adds a **reviewed closed, possibly
nonconvex** triangular rigid solid for frictionless serial implicit contact.
It uses a unique nearest-feature signed distance and an exact local
face/edge/vertex derivative; it does not average triangle normals. The
separate [candidate record](internal/closed_general_implicit_contact_candidate_20261018.md)
states its explicit embedding attestation, O(facets) sign-classification cost,
local and ordinary-Step evidence, and unresolved global reentrant-edge,
same-input reference, MPI and friction gates. This does not expand the
released 0.4.3 contract or certify arbitrary STL/CAD.

An additional reviewed triangular rigid tool may now use the *same* serial
fixed-old-state Coulomb Operator on a current planar face branch. A coplanar
two-triangle tool matches the analytical plane under translation and rotation,
including tool work and interrupted restart; active free or noncoplanar edges
still reject and roll back. This extends useful tool representation without
promoting arbitrary sharp-edge friction, distributed contact, or an external
Abaqus parity claim.

The next source candidate removes the built-in sphere/cylinder type gate for
smooth rigid contact: geometry owns a checked signed-gap differential, while
the existing Operator and Procedure remain unchanged. A third-party smooth
plane enters the ordinary serial J2 Step and an analytical plane enters the
periodic J2 composition. This is an extension seam and local regression
evidence, not general CAD or sharp-feature contact validation; see the same
[moving affine contact record](internal/moving_affine_contact_candidate_20261017.md).

The separate performance lane follows `skills/agentfem-performance`: profile
whole accepted increments on a frozen case, change one body/contact/data-flow
mechanism at a time, and promote only parity-preserving measured wins.
Windows/WSL project UX should make project custody explicit and offer an
opt-in native-Linux-filesystem execution location when measured I/O warrants
it, without silently moving Windows-side projects. It should surface the
effective runtime/native-extension status and test installed-package behavior
there. Neither a platform switch
nor altered mass/penalty/time controls may be reported as an algorithmic speedup.

The 2026-10-23 Windows onboarding follow-up is a separate, bounded product
task: reuse WSL preparation for Ubuntu/mirror and Complete Runtime routes,
preserve explicit approval/reboot/account handoffs, verify build identity on
resume, and report machine-readable progress. Source and mocked Windows-shell
evidence are tracked in [the installation guide](windows_agent_installation.md).
The source checkpoint also bounds read-only installer queries, preserves literal
Windows paths and interruption evidence, and tests concurrent status publication
and failure-record storage faults on PowerShell 5.1/7. The
[review record](https://github.com/haoming-luo/agentfem/pull/120) distinguishes
these passing software contracts from real-machine acceptance.
Clean Windows/WorkBuddy installation and actual domestic download availability
remain promotion gates; this is not a new solver capability or a speed claim.

The 2026-10-26 follow-up closes the environment-to-first-solve gap in the
Ubuntu/mirror route: a source-usable standard-library entry plans first,
creates only missing environments through one selected channel, reuses existing
environments without implicit package changes, and verifies installed identity,
health, protected workspace, a small installed-template solve and result
provenance. Each attempt retains separate logs and structured failure evidence.
Human-readable Windows status views preserve the existing JSON contract and
distinguish approval, restart, private account input, interrupted work and saved
acceptance. Local contract tests and documentation checks are complete; the
Windows 5.1/7 and Linux installed-entry CI gates are required before integration.
These checks do not establish clean-machine WSL/UAC, WorkBuddy or domestic
network acceptance, and do not promote experimental contact physics.

Integration on 2026-10-26: PR #123 merged at `a862e123` after candidate
`67940f6f` passed Windows PowerShell 5.1/7 contracts, Linux installed-entry
and FEniCSx acceptance, and macOS installed-wheel acceptance. The development
branch was removed remotely; local generated website changes were preserved.
Clean Windows/WorkBuddy and domestic-network feedback remain separate gates.
Development now returns to bounded contact verification: separate normal and
in-plane edge resolution before another plastic mesh/path run, then obtain an
independent same-input reference. Call-local J2 spectral reuse is already
implemented and evidenced in the 2026-10-19 record; do not schedule it again
as a missing feature or transfer its speedup to the explicit composite case.

### Lean project boundary (2026-10-15)

Checked serial array-mesh import and parent-cell-based exterior regions now
remove application-owned NPZ validation and boundary MeshTags construction.
Input-node lookup reuses the existing identity-based P1/Q1 helper. Private
surface selection and bonding eligibility remain explicit scientific inputs;
the reader does not merge nodes or create internal contact surfaces. See
[array mesh inputs](guide/array_mesh_inputs.md).

Promote reusable execution, trusted native compilation/cache, guarded polar
increments and accepted response export into the installed package. Keep private
constitutive equations, state mappings and problem geometry in the project.
`project.execute_step` wraps the existing serial transient Step; it does not own
a second time loop or choose scientific controls. See [project execution](project_execution.md).
Admission requires installed-wheel parity and accepted-state restart/failure
checks. This migration alone makes no new full-run speed or external validation
claim. MPI orchestration and general multi-point response export remain outside
this bounded interface.

### Bounded usability sprint and return to physics (2026-10-15)

The usability scope is runtime-aware non-executing project checks, discoverable
result files, explicit observable comparison, concise terminal summaries and an
installed real-FEM heat campaign template. See [workflow usability](workflow_usability.md).
It reuses Project, SimulationResult and Campaign rather than introducing another
workflow. Field remapping, a GUI and automatic scientific certification are not
part of this sprint.

The next scientific lane is composability of contact, friction and history
materials: first audit the existing supported combinations and their accepted
state, work/dissipation and restart contracts; then close one demonstrated gap
with a small general benchmark. Coulomb state, moving tools and explicit friction
already exist below; do not rebuild them. Damage coupling must preserve material
energy semantics and failure rollback, not inherit validation from elastic
friction tests. Keep bounded thermoelastic coupling as a maintained capability,
not a new open-ended rewrite. Shell/solid-shell forming, self-contact and broad
industrial validation remain feedback-driven longer-term work.

### Installed explicit workflow and delivery gate (2026-10-14)

The bounded serial response-only Hex8 workflow has progressed from its private
startup prototype to ordinary `model.step`, reusable core execution, readable
case files and a completed frozen full-grid review path. Native body/auxiliary
arithmetic, compiled-material dispatch and Q1 contact acceleration now ship as
optional core modules; confidential equations and source geometry stay external.
See [native explicit execution](native_explicit_execution.md) for the supported
interfaces and [performance evidence](performance_evidence.md) for measured scope.

Frozen candidate `608c50bb` has now completed the same full path in
14,646.987 s (4 h 4 min 7 s), 5.29% less than the previous core baseline.
Final checkpoint, reporting journal and independent volume/Warp checks pass;
this does not close source-case or numerical-inertia validation. The next
architecture slices are recorded in
[portable explicit performance](internal/portable_explicit_performance.md):
separate scientific accuracy from execution configuration, target the compiled
accepted-increment path, and investigate the controlling time limit before
committing to general multirate or device execution.

The delivery priorities are now:

1. Preserve the completed installed candidate and its platform/CI, full-run,
   response and volume-output evidence. New algorithms are compared at a
   declared accuracy target, not forced to reproduce its exact step count.
2. Close independent source-case validation and inertia/time/penalty sensitivity
   without changing inputs to meet a timing target. Completion of the reviewed
   experiment does not establish equivalence to the source solver's contact,
   initial-bond or variable-mass behavior.
3. Use whole-increment evidence to choose the next bottleneck. Preserve one
   accepted State owner; optimize execution/data movement without a second
   solver or a case-specific core. Keep unsupported distributed response-only
   combinations rejected until their own lifecycle gates pass.

The follow-up adds a frozen-project preflight with separate integrity,
artifact-platform and runtime-readiness results; it must not turn this Mac's
binary/worker configuration into Model requirements. Ordered installed
comparisons rejected the new copy/scan execution candidate; no new solver
optimization was selected in that follow-up. A separate
five-times-larger time-step experiment was rejected after early energy growth;
material incremental admissibility alone is not global dynamic stability.
Next-method work must identify the controlling body/contact/damping bound
before selecting a multirate scheme. Do not promote that failed screen or a
small copy/scan benchmark as a new full-run speedup.

The older dated entries below document intermediate gates, not the current
claim that only a startup pilot exists. They remain useful as provenance for
scientific boundaries that have not yet closed.

### Corotational history-material gate (2026-10-10)

The source-material review now has a compiled local reference and an independent
private batch implementation. General polar-frame kinematics are implemented
as a private core candidate, with objective stress transport and independent
noncoaxial integration checks. This does not admit that material to finite-Hex
`model.step`: its work-only energy output and response-only stability need
explicit contracts, followed by single-element/global lifecycle verification.
The next local slice now supplies a genuine response-only work contract,
serial quadrature transactions and real Hex8 internal-force assembly without
inventing DPDF/SENER fields. Trial failures restore state and response fields.
The private response-only route now advances the existing central difference
Procedure, records internal work separately from artificial energy, and restores
accepted material/force state atomically. Two moving planes or bounded lateral
cylinders compose with the existing contact work ledger and disk restart.
Independent explicit stability, ordinary Step admission and MPI remain open.
The original 255,041-cell mesh now runs a bounded four-tool startup pilot and
passes independent continuous/restarted four-increment bitwise comparison.
That endpoint is only 4e-10 s, not the full 0.1 s loading. Original interface
initialization, variable mass scaling and commercial comparison remain open.
Current-configuration deformable Q1 penalty contact now has a private serial
face-interior composition with force/moment/virtual-work and time-refinement
checks. This does not admit edge/corner normal cones or source ALL EXTERIOR
hard contact. A following closed-solid candidate now classifies all 1,000
unbonded source candidates with two true-Q1 rays; they are outside the complete
patch, so old open-patch negative gaps were not evidence of penetration.
Concave-edge force/moment/work checks and serial two-body restart pass. The
original case can opt into this bounded initial-candidate contact; arbitrary
ALL EXTERIOR search, collision detection and mass-scaled full loading remain open.
Follow-up: private accepted artificial-viscosity work/restart now passes its
targeted composition gates; this does not close variable mass scaling or
vendor-equivalent viscosity/stability. Explicit reference-surface adjustment
proposals preserve the original mesh and require an independently selected
bond mask plus validation of affected bulk cells. The source's initial bond set
is still unknown. Its body/body hard-contact behavior is also not equivalent to
a compression spring. A reusable quintic amplitude and amplitude-driven rigid
schedule now address the source loading-history mismatch without changing the
existing half-cosine `smooth_step` API.
The ordered nine-gate plan is
[corotational material development](corotational_material_development.md).
Keep confidential source equations and parameters in external project providers.

2026-10-11 continuation: segment/BVH pruning reduced the private 1,000-point
closed-Q1 query median from 11.812 s to 0.641 s with identical signed gaps.
This is a search-component result, not whole-loading throughput. Fixed and
variable numerical inertia now have private serial state/identity contracts,
accepted acceleration adjustment, artificial energy/linear/angular-momentum
inputs, atomic rejection and disk restart. A selective element-bound planner
assembles nodal inertia without changing physical constitutive density.
The planner consumes supplied positive element bounds; it does not establish
an evolving damaged-material/contact/damping stability certificate.

Remaining source-reproduction gates, in execution order:

1. Admit a justified evolving increment/mass policy and quantify added-inertia
   sensitivity; retain separate negative-curvature/damping handling. The initial
   elastic screen alone is approximately 8.5e-9 s, not the requested source dt.
2. Extend bounded initial candidates to the required body/body contact coverage
   with missed-crossing protection; penalty convergence is not exact HARD contact.
3. Confirm source initial bond eligibility/adjustment and review stabilization/
   characteristic-length differences. Do not invent interface damage or deletion
   absent from the supplied source deck.
4. Run nonzero/full physical loading with force--displacement, artificial-energy,
   penetration, increment and mass-scaling convergence; then compare independent
   source-solver evidence. Startup/restart capacity is not a completed reproduction.
5. Promote only proven combinations into ordinary Step options and MPI; keep
   general mortar, relaxation stabilization and deletion as separate product
   gates rather than prerequisites invented for this one elastic interface.

### User-driven mesh/interface slice (2026-10-09)

2026-10-10 continuation: the bounded finite Hex8 route now has ordinary-Step
lowering and accepted-response
energy sampling, declared virgin response, prescribed-motion work, material
dissipation, isotropic reference-interface energy and energy-preserving
restart. Independent finite elastic/plastic stretch checks pass, with time-step
refinement of the balance residual. A bounded opt-in negative-curvature growth
resolution policy preserves signed spectrum evidence; it is not a general
nonlinear stability guarantee. Nonuniform plastic response and distributed
bulk ownership/failure tests now pass. The shared rank-local numeric checkpoint
encoding additionally covers same-partition finite MPI restart, including empty
ranks, corrupt or swapped payloads and failed publication. Independent private
and ordinary-Step plastic capacity runs now both complete 262,144 cells and
1,000 increments; the public run peaks at 3.20 GiB. A separate three-pair,
4,096-cell plastic comparison reduces median integration time by 38.2% with
identical compared response arrays; see [performance evidence](performance_evidence.md).
This is not an industrial-forming or universal speedup claim. Distributed interface
integration, cross-partition finite state recovery, damage/deletion and advanced
hourglass laws remain separate gates, not requirements satisfied by these tests.

The tester's Hex8/interface brief is assessed by capability, not by reproducing
one private input file:

| Requested part | Current evidence-backed boundary | Next missing gate |
| --- | --- | --- |
| A1 uniform-gradient elastic Hex8 | Ordinary Step; isotropic/rotated orthotropic, stiffness hourglass, lumped mass, bending/wave/patch checks and separate artificial energy | Application-specific accuracy and commercial cross-comparison are not inferred |
| A2 finite deformation | Bounded history-material Step, finite objectivity/tangents, independent proportional and nonuniform plastic references, owned-cell MPI and same-partition restart | Automatic material-owned stability estimates; viscous/relaxation stabilization; damage/deletion and mass scaling remain unimplemented in this path |
| B1 nonmatching bonding | Serial fixed coplanar P1 and affine planar Q1 traces, genuine elastic law, coverage/force/moment/work checks and accepted recovery in explicit dynamics | Warped/general surfaces, cross-rank interface ownership and general convected material frames |
| B2 integration quality | Common-refinement overlap integration retaining both original bases, side-swap and patch evidence | Not dual mortar, arbitrary sliding, re-pairing or post-failure contact |
| Combined industrial forming | Small finite-bulk/equal-stiffness elastic-bond Step verified, without case-specific solver code | Moving contact plus the actual orthotropic finite material, damage policies and independent industrial reference remain separate work |

Prioritize these genuine gaps instead of advertising all of A2 or the complete
industrial model as finished. Installed-use and performance evidence support
usability and efficiency, but do not promote scientific scope by themselves.

The next bounded stability slice should first expose material-owned evidence
at the initial and accepted states, with explicit configuration, units and
validity scope; do not infer an isotropic wave model from an arbitrary dP/dF
array. Compare the composed bulk/hourglass/interface estimate against the
existing fixed-step screen and independent time refinement before replacing
the caller ceiling. An independently validated stability response could then
permit response-only material updates where the explicit integrator needs no
Newton tangent; do not skip tangent generation while the current admission
screen still depends on it, and measure the actual whole-step benefit.
Automatic variable increments are a subsequent Procedure
change: actual accepted times, prescribed-motion work and restart identity
must remain consistent, and a rejected attempt must not advance material state.
Do not add an alternative explicit solver or silently enable mass scaling to
make this gate pass. The tester's actual finite orthotropic material and moving
contact combination require their own evidence after these shared contracts.

Continuation candidate: a provider-declared `FirstPiolaTangentEnvelope` now
lowers through the fixed-reference mass-gradient Gram and ordinary Step. The
bounded serial route checks its principal-stretch domain, signed tangent and
restart identity; independent element/assembled spectra and a Neo-Hookean
external provider exercise the contract. This removes hand-entered frequency
ceilings only for materials supplying a valid domain envelope. Native finite J2
still needs its caller ceiling; its history-dependent proof and new distributed
envelope tests remain open. Variable increments, response-only updates and
moving-contact composition are not implied by this candidate.

Contact follow-up: numeric nested material checkpoints and accepted-force
sampling now pass serial composition with a moving plane and a finite elastic
history provider. The common work ledger, summed spectral ceiling, time
refinement, restart equality and post-commit rollback are exercised together.
The next candidate lowers one registered frictionless translating plane through
ordinary finite-Hex `model.step`, preserving the separate constitutive owner.
It passes a scalar finite-compression ODE comparison and a native finite-J2
plastic work/restart test. Curved/rotating tools,
multiple pairs, friction and the tester's actual orthotropic material remain
open; do not equate this serial slice with industrial forming completion.

Distributed contact follow-up (without a bonded interface): the same ordinary
Step now passes five two-rank gates for serial equivalence, same-partition
restart, empty partitions, rank-inconsistent admission/corrupt outer-state
rejection and post-commit rollback. The combined contact/finite-Hex MPI
regression passes 62 tests per rank; serial contact/bond regressions pass 57
with eight MPI-only skips. These are overlapping selections, not 124 distinct
MPI tests. Cross-partition finite-material restart remains separate work.

The next prerequisite hardens the existing sparse cohesive communication owner:
collective schedule/payload rejection, idle consumers and transpose virtual-work
checks now pass with two/three processes. This preserves sparse numeric payloads;
the added readiness exchange carries only small status records.

The following bounded distributed nonmatching slice now owns quadrature points
once, uses remote trace schedules, assembles single-count interface energy and
the global owner-summed stability bound, and preserves joint same-partition
restart. Two-rank elastic/J2 bulk + bond + tool tests compare with serial values,
ordinary results and exact resumed trajectories; corrupted interface identity
and post-commit failure roll back collectively. Reference geometry is replicated
and point identity remains partition-bound: scalable geometry discovery and
cross-partition finite-material recovery are NOT completed by this slice.

The next serial combination now composes bulk, an equal-stiffness nonmatching
elastic bond and an external translating plane through the same Step. Contact
and bond must have disjoint trace nodes (even shared edge nodes are conservatively
rejected). All three stiffness contributions enter the existing summed stability
bound once. Independent four-mass axial dynamics, temporal refinement, separate
energy channels, native-J2 plastic response and exact interrupted recovery are
the acceptance tests. This is not post-failure contact or
the tester's unspecified finite orthotropic law. Obtain that law, orientation
update and parameters before claiming equivalence; do not substitute J2.
A separate, rotated orthotropic Green-strain test provider also passes tangent,
energy-derivative, objectivity and joint recovery checks without core changes.
It proves the extension seam, not equivalence to the private user material.

This slice implements bounded fixed-reference nonmatching cohesive trace
integration and uniform-strain Hex8 stabilization. These are general formulation
capabilities, not an automatic translation of a specific
Abaqus forming model. Reuse triangle search and existing cohesive laws; keep
pairing, material state, force assembly and Procedure separate. The bounded
design and scientific promotion gates are in
[nonmatching interfaces](nonmatching_interfaces.md). Each geometry and execution
route requires its own evidence; local trace tests alone do not enable migration,
finite rotations or distributed interface execution.
Do not introduce artificial damage thresholds for an elastic-only interface,
or describe unstabilized one-point quadrature as Hex8R support.

The fixed coplanar P1 route now has common-refinement integration, per-facet
coverage checks, independent nodal-force audits and serial two-block
series-compliance evidence. Its experimental elastic `model.step` provider
reuses PETSc solving and Result output. The affine planar Q1 extension retains
the original four-node basis and degree-four common-refinement integration;
global 1:3 and 2:3 compliance tests exercise independent Hex8 partitions.
Global nonmatching damage, distributed interface ownership and general
convected anisotropic interface frames remain separate gates; the isotropic
finite elastic special case above does not promote them.
The experimental [uniform-gradient Hex8 policy](hex8_uniform_strain.md) now
has compact batch execution, serial global affine/bending/wave evidence and
ordinary explicit `model.step()` integration. Isotropic/rotated anisotropic
elasticity, disjoint material regions, physical/artificial energy separation
and serial restart are exercised. Failed explicit increments and auxiliary
checkpoint rejection restore the accepted field state. General finite forming
beyond the bounded history-material route, damage and distributed interface
composition remain separate gates.
Natural-load and strong prescribed-motion
work reuse the existing dynamic ledger; independent acceleration/oscillator
tests exercise work and time refinement. Checkpoints capture accepted endpoint
history even between output frames, preventing cumulative-work loss. The elastic Q1
interface now composes with this explicit Step, including an additive spectral
stability bound, interface energy, serial restart and failed-commit retry.
Warped/nonaffine quadrilateral traces remain rejected. These are bounded
small-strain capabilities, not an industrial finite-deformation reproduction.
The bulk Hex8 path additionally has two-/four-rank assembly, ordinary-Step work,
output and same-partition restart evidence. Nonmatching interface communication
and cross-partition restart remain unsupported. Rank-local input failures and
work sampling on partitions without constrained nodes now have collective guards.

The Hex8 result path now exports optional constitutive DG0 stress, tensor strain,
von Mises stress and physical energy density through the existing transient
result lifecycle. Material jumps are retained, artificial stabilization stress
is not mixed into material stress, and field provenance records the generating
formulation. An installed two-rank candidate exercises this path independently
of source-tree imports. Bounded energy-only monitoring and local scatter maps
reduce measured run time by about 26% on one 8,192-cell elastic workload; this
is a workload-specific measurement, not a general performance guarantee.

Hex8 geometry admission now uses bounded Bernstein subdivision of the
triquadratic Jacobian determinant, rejecting hidden inversion and unresolved
near-degeneracy before assembly. This does not certify approximation quality or
inter-cell non-overlap. The installed candidate also runs the Gmsh independent
Q1 bonded-block and Hex8 wave examples; both result manifests pass provenance
verification. Scientific evidence remains the separate analytic/convergence
checks, not the provenance seal alone.

The 0.4 line is an architectural consolidation, not a feature-count release.

The next Hex8 prerequisite now has a private total-Lagrangian contribution with
large-rotation objectivity, energy/tangent and existing J2 batch-protocol checks.
The bounded route now lowers through the ordinary `model.step()` with explicit
`kinematics="finite_strain"`, a caller-declared complete-path spectral ceiling,
one full-domain history material and declared reference density. A DOLFINx
bridge exercises existing quadrature trial/commit/rollback, global force and
moment, fixed-old-state tangents and a four-increment J2 patch. An instantaneous
nonnegative symmetric-tangent spectral screen is tested separately; unsupported
tangents are rejected, not silently symmetrized into a stability claim.
The private serial path now reuses the existing explicit Procedure, jointly
restores nodal/material state after rejected increments, and passes interrupted
versus continuous execution. A nonlinear Hencky bar exceeds 10% extension and
shows second-order time refinement against an independent ODE. This does not
admit arbitrary J2 trajectories: the signed endpoint screen reports positive and
negative curvature separately and requires an explicit bounded-growth policy
for negative curvature. It is not a general nonlinear stability guarantee.
Accepted work, stored energy, artificial energy and material dissipation now
share the existing ledger; optional unsmoothed DG0 views expose the accepted
response without reintegration. Next are broader independent references and
general objective interface/contact composition. The finite contribution already
reuses the elastic nonmatching interface in the equal-three-stiffness special
case, with opened-interface objectivity, current moment, joint recovery and
post-commit failure checks. This does not admit anisotropic/damaging finite
interfaces, contact or distributed interface state.
The shared finite-material driver now offers optional validated columnar
batches, retaining scalar/ordered providers and the same atomic trial boundary.
Native J2 uses its existing constitutive equations through this transport;
no new material formulation or provider framework is introduced. Serial
numeric-array auxiliary checkpoints avoid JSON expansion and preserve legacy
reads. Rank-local version-7 payloads now support same-partition MPI recovery,
including empty ranks and collective rejection of corrupt/swapped payloads;
cross-partition history migration remains unsupported. Spatial refinement uses
independent manufactured Neo-Hookean and proportional finite-J2 problems.
The plastic reference includes nonuniform tension/compression and a moving yield
front, with a separate body-force derivation rather than the solver residual.
Large affine endurance runs are capacity evidence only;
they do not replace nonuniform spatial or industrial validation. The remaining
promotion gates above are unchanged.
Do not infer these gates from local element tests or reuse the small-strain
constant spectral bound for a finite-deformation trajectory.
Its stable middle layer is:

```text
Model -> Operator -> Procedure -> State -> Result / Verification
                         |
                      Backend
```

Each owner answers one question:

- **Model:** what engineering problem is being solved?
- **Operator:** what mathematical contribution is assembled or applied?
- **Procedure:** how is the problem advanced and solved?
- **State:** what accepted and trial quantities must survive?
- **Backend:** which numerical runtime executes the formulation?
- **Result / Verification:** what was computed, and why is it usable?

For 0.4.0 and subsequent releases, AgentFEM must keep these boundaries acyclic, preserve one
provider-owned lowering route, retain atomic state and result lifecycles, and
pass a candidate-bound release ladder. The executable audit is:

```bash
python promotion_gate.py --target 0.4-foundation
```

A complete 0.4 candidate additionally requires complete serial tests,
representative two-rank state/nonlinear/output/checkpoint tests, clean installed
wheel acceptance, unchanged public examples and compatibility imports, Linux
and macOS acceptance, and benchmark evidence for every maturity change.
Windows runtime acceptance remains a separate product gate.

### 0.4.0 release and the next bounded cycle

0.4.0 consolidates the foundation rather than promoting experimental physics.
The release pipeline must gate publication on the same immutable wheel's
Linux/macOS foundation evidence. See [release scope](release_0.4.0.md).

After release, prioritize in this order:

1. installation/upgrade, examples, result readability, and actionable user
   failures; protect projects and results independently of runtime replacement;
2. supported mesh--element--formulation interoperability and early quality
   diagnostics, with small representative engineering regressions;
3. measured assembly/solver/state/output efficiency under fixed scientific
   inputs; no speed claim obtained by relaxing tolerances or changing physics;
4. one bounded external-validation track, beginning with the existing
   finite-strain RVE gaps, without reclassifying incomplete evidence as passed.

Each item needs a reusable owner, a finite acceptance test, and a stop/review
condition. New materials, forming, additional backends, or benchmark-specific
public APIs are not prerequisites for this consolidation. Keep routine CI
targeted; reserve the full installed-wheel ladder for milestones and releases.

#### Agile maintenance slice before coupling work

Do not turn the installation/mesh/performance priorities into another broad
foundation programme. The current bounded slice shares transient operator
accounting, fixes evidence loss after prepared-system release/continuation,
and reduces allocation in exact boundary-identity checks. It changes neither
the public modelling API nor scientific maturity. Targeted serial/restart and
two-rank tests are its exit gate; see the reproducible measurement instructions
in [development testing](development_testing.md#transient-lifecycle-maintenance).

The next bounded implementation is **sequential thermal--mechanical
coupling on a shared mesh**, using existing thermal fields, eigenstrain,
EngineeringStep and ordinary providers, not a new coupling god object.
Audit the following before implementing a public convenience layer:

1. Field transfer now guards same-mesh/space/component identity and explicitly
   declared equal time. Copy is exact-space only; interpolation is explicit.
   Equal array length is not evidence for compatible field transfer.
2. A transferred temperature may change only eigenstrain/RHS, or also material
   stiffness. Preserve this distinction in the existing time-input contract.
3. Define ownership of accepted source and target states, failed-target
   recovery, and restart provenance without confusing one-way sequencing with
   converged two-way coupling.
4. Begin with uniform free thermal expansion and restrained thermal stress;
   then one spatial-temperature, multi-region regression and two-rank restart.

The same-mesh slice is implemented with nodal field transactions and uniform/
linear-temperature analytical response tests, including two-rank rollback and
transfer after thermal restart. Temperature/material dependence retains the
ordinary operator lowering; no new global coupling solver is introduced.
The multi-region handoff regression now combines two conductivities, regional
E(T) and thermal expansion with exact cell-average stress checks. The existing
hot-wall example restores accepted heat and recomputes only mechanics; a
rejected structural trial rolls back field and transfer evidence together.
This is durable upstream reuse, not a combined multiphysics checkpoint.
The next two-way slice now has a bounded 3D verification prototype with a
reversible heat-feedback Operator, unrelaxed residuals, in-memory joint rollback
and an independently written monolithic oracle. See [coupling design](coupling_design.md).
The private experiment now verifies inward flux, prescribed-temperature heat
reactions, affine prescribed-motion path work, and joint durable recovery.
Registered strong boundaries and a bounded ordinary Step provider now reuse
the physical Procedure and joint restart, with explicit SI/participant-load
contracts and final-snapshot output. Accepted-window progress/checkpoint cadence
now reuses the existing reporter and policy, with explicit retention. Field-series
output remains explicitly final-snapshot-only. Local installed-wheel serial/MPI
acceptance passes for the candidate recorded under `evidence/coupling/`;
public-provider cross-partition recovery now also passes `1 -> 2 -> 1` with
prescribed motion and accepted force/work history. PR #90 is merged at
`1ecf41e0` after final-head FEniCSx/macOS/docs checks passed. The 0.4.1
release candidate retains the immutable-wheel publication gate. A separate smooth
Fourier-mode oracle now shows approximately second-order spatial and first-order
temporal convergence of both fields through the ordinary Step; its 1D mode
embedded in 3D is not general multidimensional validation. See the coupling
design for parameters, errors and fixed acceptance thresholds;
no general coupled capability is declared stable.

Execution order: finish this bounded thermal--mechanical product route first,
then audit contact against existing moving-tool/work/search implementations
before selecting any missing capability. A post-release two-rank audit passes
34 tests per rank for the explicit contact provider, distributed search and
accepted work. The serial search microbenchmark confirms existing BVH pruning
against exhaustive projection without modifying solver physics. Next measure
representative solve-stage costs before optimizing; implicit friction, wider
topology and extreme-scale routing remain separate gates. Do not reimplement already tested
contact features based on an outdated roadmap. Mesh compatibility and measured
performance changes follow demonstrated application needs, not a parallel
architecture rewrite. Small releases do not rebuild complete installers.

#### Next bounded implicit-contact slice (0.4.3 candidate)

The already-verified explicit triangle search is not itself an implicit
contact solver. The analytical-plane finite-J2 composition first established
one Newton residual/tangent, force/work evidence, rollback, and serial restart.
The subsequent fixed-facet candidate reuses `RigidContactPair`, the reviewed
BVH, and the same J2 Procedure, adding a current-trial boundary-quadrature
residual and fixed-branch tangent. Its scope and local evidence are recorded in
[the fixed-facet note](internal/implicit_facet_contact_candidate_20261016.md).
Keep it experimental and serial; do not promote curved/creased tooling,
moving-body work, periodic MPC, friction, or MPI by association.

The stacked moving-facet slice adds one normalized rigid pose without
rebuilding the reference BVH, plus accepted force--moment work and pose
rollback. The follow-on smooth analytical surface and fixed-tool periodic-MPC
source candidate is described in the
[new bounded record](internal/smooth_implicit_contact_mpc_candidate_20261017.md).
Neither slice certifies noncoplanar triangulated edge transitions or MPI.

Promotion order: (1) close full installed-wheel CI and a candidate-bound
scientific comparison for the analytical plane; (2) independently check
faceted-contact geometry, force, moment, tangent and convergence under
nonuniform deformation; (3) verify prescribed tool motion with generalized
work and joint rollback beyond the local plane-limit evidence; (4) reduce
both contact residual and tangent through periodic MPC; (5) establish MPI
owner/ghost and restart evidence. A new
release may ship a clearly bounded experimental subset, but its version
number must not imply that the later gates passed. Performance is measured
on fixed physics and an immutable installed candidate, separately from
changing penalty, inertia, or time discretization.

The current bounded serial candidate has progressed beyond the original
single-facet slice: ordinary finite-J2 Steps now exercise prescribed-DOF
contact, named multi-tool motion/work, reviewed planar/closed-convex Coulomb
branches, restart and accepted-point output. The
[candidate acceptance map](internal/implicit_contact_dirichlet_hertz_candidate_20261018.md)
separates these completed source and installed-Mac checks from the remaining
external same-input comparison, general sharp-feature laws, distributed
implicit contact and cross-platform release gates. Its Hertz force, pressure,
mesh, penalty and energy screens are directional diagnostics, not an
industrial-contact accuracy certificate. Do not silently widen the public
capability merely because this candidate can complete a Newton solve.

The next closed-general candidate extends the **serial rigid-tool** branch to
a caller-reviewed embedded triangle solid. Its normal law retains
current-geometry nearest-feature/winding evidence and rejects ambiguous
projection; a small ordinary J2 Step now exercises an active reentrant edge,
plastic state, tangent check and restart. See the
[bounded closed-solid record](internal/closed_general_implicit_contact_candidate_20261018.md).
The Coulomb extension is restricted to active locally planar faces; it rejects
physical sharp edges or vertices and rolls back the failed trial. The serial
affine-MPC path now has a versioned accepted friction-State checkpoint and a
tested plane/triangle/closed-face tangent reduction, but no arbitrary sharp
friction law or distributed operator assembly. This is not
yet a same-input external comparison, a converged parameter study, frictional
sharp-feature law, large-CAD backend or distributed implicit contact.
The fixed L-tool screen now exposes a concrete numerical gate: changing only
the positive face rule changes which physical sharp-edge points are sampled,
and a six-cell penalty bracket remains far from a hard-contact limit. Even the
same-input degree-7 screen through twenty cell divisions still changes maximum
PEEQ and active edge samples; a nearly stable resultant is not enough. The next
frictionless method experiment should partition slave-face integration at
zero-gap and nearest-feature transitions while preserving a common
residual/tangent trial partition, then compare force, moment, penetration,
plastic history and tool work across mesh, penalty and increment refinement.
Do not elevate higher polynomial face quadrature or a faster sign classifier
as an accuracy repair. Stateful edge friction and distributed ownership follow
only after this frictionless integration gate is independently checked.
The first bounded slice now clips a current-trial affine P1 slave triangle
against an analytical rigid plane, with a positive common residual/tangent
rule and an independent narrow-strip oracle. It does **not** resolve the
L-tool's changing nearest feature or replace the still-open mesh/penalty
convergence and independent same-input acceptance gates.
An analytical two-plane counterexample now fixes the next design constraint:
at an active sharp-feature switch the moving internal boundary contributes a
nonzero line term to the contact tangent. A zero-gap clipper alone cannot be
rebranded as a consistent L-tool edge algorithm. A research-only positive
two-plane area-plus-feature-line oracle now matches analytic and directional
derivative checks, including oblique planes and a spring-supported global
fixture checked against independent integration. A one-tetrahedron linear
elastic fixture also equilibrates, with its hand-derived bulk stiffness
independently checked against DOLFINx assembly. The remaining nonconvex
L-tool gate needs its nonlinear nearest-feature transition law and convergence
evidence before any such Step promotion.
The bounded convex-support-plane subset is now an opt-in ordinary-Step and
serial affine-periodic J2
implementation: it partitions at most four dominant planes per affine slave
triangle and includes the compressed feature-line Newton term. A real
finite-J2 Step, full Jacobian including MPC reduction, moving-tool work and
serial restart pass. Fixed support planes are prepared once; every current
trial still recomputes its gap, active subcells and projection, and a projection
feature disagreement rejects the trial.
An independent symmetric convex-corner fixture now checks exact potential,
integrated force and five directional total tangents with three dominant
planes. A four-feature rectangular window now additionally passes an
independent exact-energy oracle, perturbed directional tangents, assembled
FEniCSx tangent and an interrupted finite-J2 restart; five positive-area
features still reject. This expands a bounded
convex subset, not the nonconvex reentrant-tool convergence claim.
This does not close the nonconvex L-tool's nonlinear nearest-feature
transition or the independent mesh/penalty benchmark.
The current same-source fourteen-cell L-tool penalty screen is an explicit
admission blocker: increasing its penalty from 200 through 1000 to 5000
continues to change contact resultant, penetration and fixed-volume plastic
response substantially. Agreement between two face quadrature degrees at
one penalty is not a substitute for this limit. Before promoting general
sharp-tool contact, establish a mesh-by-penalty response matrix with a
physical penetration tolerance and an independent same-input reference;
retain the bounded convex option as experimental in the meantime.
If a penalty-only choice cannot satisfy both penetration and conditioning,
evaluate an explicit augmented-Lagrangian normal law as a separate Operator
and accepted-multiplier State. Do not silently switch enforcement inside an
existing contact pair or reuse the penalty checkpoint identity.
The installed-candidate penalty-20000 diagnostic did not complete under
either basic or backtracking Newton, even after doubling fixed increments.
This is a bounded failure observation, not proof that augmentation will
converge or that a higher penalty is universally unusable. First prove the
multiplier update and penetration tolerance on a fixed-point planar trace;
the trial-partition route has changing quadrature points and needs an explicit
State transfer/identity rule before any multiplier history can follow it.
The next bounded slice is now implemented: explicit
`steps.normal_augmentation(...)` lowers through ordinary standard J2 with one
fixed or prescribed-translating analytical plane, serial P1 tetrahedra and coalesced nodal multipliers.
It reuses the existing Newton, material trial state and joint accepted
checkpoint. Independent homogeneous limits, plastic loading/unloading,
failure rollback and restart have targeted evidence. Redundant strong/contact
normal rows reject early. This is still experimental: general multiplier
stability, general MPC, rotating/curved tools and partition transfer remain separate
gates. The fixed-plane accepted-work gate now records signed numerical contact
work, a non-cancelling discrete work bound and opening/closure contributions.
The AL functional is never stored energy. Original and contact-aware energy
residuals remain visible, with independent cyclic-reference and increment
refinement evidence; no automatic energy-verification claim follows. Legacy
checkpoints without work history remain unavailable rather than fabricated.
The fixed-plane route additionally has a scalar, non-FE coaxial plastic-stop
reference through loading, unloading, reverse flow and recontact on two
meshes, including restart after reverse plasticity. That check exposed a
resolved-amplitude/normalized-coordinate confusion: signed load amplitudes
must scale ordinary strong values, and fixed tools must not validate them as
motion coordinates. Time-dependent boundary histories and moving-tool ranges
keep their own domain checks. This does not enlarge the AL geometry scope.
The subsequent translation-only gate retains the same material-side nodal
multiplier basis while recomputing gaps against the current tool pose.
Tool work is recorded once as external input; numerical relative-gap work
is separate. Scalar plastic compression/recontact, increment refinement,
frozen-state tangent, joint failure rollback and serial checkpoint checks
cover this bounded path. The new public translating-plane example uses the
ordinary Step, not a case-owned solver. Rotation, changing normals, MPI and
general MPC augmentation remain closed until independently established.
The subsequent bounded affine gate admits one serial P1 planar pair only
when every reduced normal row has nonzero, disjoint support. It reuses the
ordinary affine path with a pre-trial Operator snapshot and a nested outer
Procedure; it never commits plastic history during augmentation. A separate
hard-obstacle elimination reference on two meshes/two penalties, cyclic
open/recontact, successive restart and injected rollback tests establish this
discrete slice. Signed normal work is retained at every accepted station
independently of cloud-output cadence. Overlapping support, prescribed normal
rows, rotating/curved tools, mixed and MPI augmentation still reject; this does
not close general multiplier stability or the independent external reference.
The optional safeguarded multiplier accelerator is now implemented on that
same bounded route. Its reference-area-weighted secant history is owned by
the attempted-increment Procedure, never the material or checkpoint State.
Default controls are unchanged. Independent pressure, cyclic energy and
restart/rollback checks accompany a 32-increment serial example whose repeated
median solve time fell 31.8%; this is not evidence for the explicit composite
case, curved/MPC augmentation or general commercial-software speed parity.
The follow-up acceptance adds a spatially partial plastic contact patch through
unloading/recontact, all-station pressure/state/work comparisons, restart and
post-commit failure recovery. Both policies also match a separate
equality-constrained elastic active-set reference on two meshes. This narrows
the acceleration validation gap without admitting new geometry or constraint
families. The `a62cef23` candidate passed remote Linux tests and macOS
installed-wheel foundation acceptance; scientific parity gates remain open.
The moving-plane slice now also has a nonuniform Poisson-coupled plastic
compression comparison against independently enforced prescribed displacement
on three meshes and two penalties. Point-pressure accuracy is checked
separately from displacement. Cyclic accepted-point export records Step
coordinate and resolved tool amplitude separately; intermediate output and
two successive restarts are exercised from an isolated installed wheel.
Before any default penalty/globalization tuning, retain full accepted-path
field comparisons and the measured failure boundary. Outer augmentation and
inner Newton/search work are now separately inspectable; local rollback uses
numeric arrays rather than serializing a disk checkpoint every increment.
The material-side follow-up shares one call's elastic-trial spectral factors
between the J2 return and its consistent tangent, without persistent caches or
new State ownership. Fresh paired tests at three mesh sizes reduce the bounded
implicit example's solve/result time by 11.5--13.4% while retaining every
accepted field and iteration count. This is measured software reuse, separate
from numerical-penalty tuning, and is not an explicit-composite full-run claim.
Keep the public scalar contract and independent numerical tangent oracle.
The fixed-plane route now also has nonuniform manufactured-solution spatial
evidence: P1 displacement and gradient errors converge at approximately second
and first order, while contact force and nodal pressure errors decrease.
This check exposed and fixed a linear-trace coordinate-tabulation roundoff
that falsely reported initial overlap on a zero-gap mesh. Original mesh
vertices are authoritative; real overlap still rejects at the declared limit.
Pre-correction experimental checkpoints with changed trace identities must
continue under their original candidate, not bypass the restore guard.
The same manufactured implicit-J2 contact screen also found a distinct
scaling opportunity in the *linear solver*, not the contact law: explicit
GMRES/GAMG with a recorded rigid-body near-nullspace preserves the tested
fields and reduces accepted-Step time on two mesh sizes. Keep direct LU as
the robust default while comparing the opt-in policy on independent meshes,
materials, constraints and platforms; never silently select it solely from
one machine's timing.

Stop after this vertical route has explicit field/response and state evidence.
Nonmatching-mesh transfer, broader nonlinear coupling, monolithic Jacobians,
and additional physics remain separate subsequent decisions. Installation or
mesh defects reported by users still interrupt this ordering when they block
an otherwise supported workflow; no speculative installer rewrite is planned.

### 0.3.8: the 0.4 foundation candidate

0.3.8 is the deliberate consolidation release before 0.4.0. It is promoted by
stable ownership and executable contracts, not by adding another material or
solver family. Its six product gates are:

1. one mesh--element--function-space compatibility matrix shared by inspection,
   validation, lowering, result identity, and checkpoint identity;
2. mesh-quality evidence and early, addressable rejection of unsupported
   topology or formulation combinations;
3. typed time inputs that distinguish right-hand-side changes, operator
   changes, state changes, and output-only observations before a Procedure
   chooses reuse or rebuild;
4. provider-owned reaction, force, work, and energy closure for exact MPC,
   weak constraints, and the first bounded contact route;
5. one MPI result, state-identity, and checkpoint lifecycle across supported
   procedures, including deterministic cross-rank-count restoration where the
   capability declares portability;
6. one independently installed extension that adds a material or Procedure
   through the public provider boundary without modifying AgentFEM core.

Every gate must preserve the stable Model, Operator, Procedure, State, Backend,
and Result/Verification ownership inventory. Passing only unit tests or adding
new public names is not sufficient evidence for promotion.

Gate 6 now has an executable installed-wheel acceptance path. The reference
extension is built as a separate distribution, discovered lazily through the
standard entry-point group, and activated only by an ordinary project's
declared dependency. `extension_gate.py` installs the candidate core and
extension wheels into an isolated environment, executes the same Model →
Procedure → Result/Verification lifecycle used by users, retains the provider
identity in the result, verifies the sealed manifest, and proves by before/after
hash that the installed AgentFEM package was not patched. The reference material
is intentionally only an acceptance fixture; new scientific behavior remains
owned by companion or third-party packages. Release-tier CI now runs
`foundation_gate.py` only after the complete serial suite, representative MPI
state/nonlinear/output/checkpoint checks, installed-wheel examples,
compatibility imports, and external provider gate have passed. The resulting
foundation certificate binds those stages to the exact clean commit and wheel
digest. The same release workflow then exercises that exact Linux-tested wheel
on a hosted macOS runner and performs one final evidence fan-in; promotion can
pass only when both immutable platform records match the candidate. This
cross-platform job is release-only, so ordinary targeted and core changes do
not consume macOS capacity. `promotion_gate.py --target 0.4-foundation` no
longer waits for a record that no workflow can produce or audits one platform
before the other exists.

Fresh-agent acceptance remains a separate product gate rather than a seventh
foundation abstraction. Release-tier CI now emits an immutable trial bundle
bound to the exact wheel, source commit, task and review instructions. A
genuinely fresh AI agent must still create, run, verify and explain the model
without repair prompts. The acceptance recorder then binds the project source,
structured result, transcript and reviewed explanation to that candidate.
Deterministic CI may prepare and audit this evidence, but cannot impersonate
the unfamiliar agent whose behavior the gate is intended to test.

For 0.3.8, that engineering contract and candidate-bound evidence route are
accepted as the release gate; continued zero-intervention runs by unfamiliar
users and agents remain field evidence rather than a reason to hold back the
foundation release.

Gate 3 now has one typed contract across the supported transient and ordinary
incremental nonlinear Procedures. Built-in and custom time inputs retain
separate RHS/operator/state/output effects. Linear implicit dynamics and
linear implicit-Euler heat transfer use those effects to select safe operator
reuse or refresh; nonlinear first-order and load-path Procedures declare
per-step or per-attempt residual/tangent assembly and restore the accepted
coordinate on rejection; matrix-free Explicit declares per-increment residual
evaluation and reports whether a fixed preflight stability bound must remain
conservative over an operator-changing path. Transient checkpoint schema v5
and ordinary nonlinear checkpoints bind the complete time-input identity
before authorizing restart. The runtime architecture audit now exposes both
owned-module counts and deliberately unowned utility roots in addition to
cycles and forbidden cross-layer dependencies.

Gate 4 is closed for its declared bounded routes: exact homogeneous rectangular
MPC, exact affine-periodic paths, scalar/normal/matrix elastic foundations, and
frictionless contact with one analytical rigid plane, fixed or following a
prescribed translation/rotation path. Force and work no longer share
one overloaded endpoint record. Providers publish `ConstraintDualEvidence` for
force and physical resultants; nonlinear and non-proportional Procedures publish
typed `ConstraintWorkEvidence` only after integrating accepted stations. The
homogeneous MPC contract states exact zero constraint work instead of inventing
a unit endpoint coordinate, while affine-periodic work retains its full path and
integration rule. Engineering verification checks that serialized duals and
path work agree with the balance contract. The contact provider owns its
potential, residual, tangent, nodal dual distribution, resultant, penetration
diagnostics, and contact energy in serial and under two MPI ranks. Its atomic
portable restart envelope covers the solution, increment ledger, events,
provider-dual history, natural-load work, bulk strain energy, contact potential,
and prescribed rigid geometry, with 1-to-2 and 2-to-1 MPI rank-count acceptance.
Moving tools publish force--translation and moment--rotation path work. General
surface search now has reviewed serial and distributed BVH infrastructure,
stable accepted/trial projection State, collective Procedure acceptance, and a
geometry-neutral pointwise frictionless penalty law. The local law preserves
the same conservative potential and explicitly separates structural residual
traction from conjugate prescribed-surface traction. It does not claim
a geometry-consistent Newton linearization or assembled path-work evidence. A
backend-neutral `ContactTrace` contract now carries stable point identity,
interpolation and explicit reference/current quadrature measure into a checked
reference assembly of nodal residual, conservative potential, resultant and
tool moment. The assembly accepts only matching projection records and query
coordinates. The DOLFINx trace adapter is executable for tagged exterior
triangles of tetrahedral meshes and quadrilateral faces of hexahedral meshes
with continuous blocked vector CG1 or CG2 displacement. It restricts the cell
basis to positive facet quadrature and evaluates the surface Jacobian from
complete Lagrange first- or second-order coordinate geometry. Both routes use
partition-independent facet/point identity, synchronized ghost values, and
collective rejection of unsupported spaces. Standard Q2 hexahedra also pass
the positive row-sum mass gate and the complete Explicit contact Procedure.
Standard P2 tetrahedra do not: their row-sum lumped mass has non-positive
vertex entries, so Explicit fails closed until a reviewed positive mass-lumped
high-order simplex element is available. Prism/pyramid mixed-facet adaptation,
implicit linearization, deformable master surfaces, free rigid-body dynamics,
and undeclared weak formulations remain outside this gate rather than
inheriting its evidence.

Reviewed triangle files can enter the same route as scientific geometry
assets with canonical identity and source hashes. The explicit contact Backend
automatically selects the serial BVH or the partitioned routed MPI BVH; the
Model does not own or expose search acceleration. This closes the import-to-
search ownership boundary, not CAD healing or general deformable-to-deformable
contact.

The first explicit Procedure hand-off is now executable for that narrow
DOLFINx slave trace. A composable residual evaluates the current predicted
displacement, repeats exact projection, applies the local frictionless penalty
law, integrates the trace, and adds only the owner-reduced contact vector to
the already assembled bulk residual. Its projection and work trials follow the
same increment commit/rollback decision as central difference. MPI-global
potential, force/reaction, active-point count and tool moment remain
inspectable. One explicit physical-time schedule can drive proportional rigid
translation and rotation; its initial and accepted generalized stations produce
force--translation plus moment--rotation work and interval power. Checkpoints
bind the schedule identity and restore the accepted path while recomputing the
memoryless projection. A caller-supplied whole-system stability ceiling
remains available. For fixed-normal or piecewise-planar projectors, the same
route can instead assemble a conservative contact spectral contribution from
the trace, penalties, Coulomb pressure coupling, lumped mass, and MPI
ownership, then add it to a supplied unsafed non-contact spectral bound before
selecting the time increment. Curved-normal geometric stiffness still
requires a reviewed caller ceiling. The same
residual now consumes the reviewed serial and routed distributed triangle
BVHs. Stable slave-point IDs cross changing rigid facets under finite sliding
in serial and at two MPI ranks, while accepted work remains rank-canonical and
pure tangential tool motion adds no false normal work. This closes one bounded
moving triangulated-tool route, not a shared multi-tool broad phase, arbitrary
higher-order topology, deformable contact, or implicit Newton linearization.
Multiple bounded pairs already compose through the residual chain,
including time propagation, atomic trial decisions, summed energy/work, and
nested checkpoint State; the reviewed mixed pair combines one analytical and
one triangulated tool without a case-specific solver.

Explicit time-step composition is now a Procedure-owned contract rather than
a contact or fracture special case. Body/material, cohesive-interface and
contact screens publish named equivalent spectral upper bounds. The Procedure
adds all simultaneously active stiffness contributions, converts the sum to
one central-difference limit and applies one safety factor. The legacy
finite-strain cohesive screen now delegates to this contract and retains its
body/interface compatibility fields only as a result view. This closes the
known unsafe `min(body_dt, interface_dt)` composition; formulation-specific
element eigenvalue bounds and state-dependent re-estimation remain later
accuracy promotions.

Gate 5 now has an executable checkpoint capability contract. Procedures must
separately declare the durable payload scope, accepted save boundary, atomic
publication, scientific identity, and MPI rank-count portability; a portable
scalar sweep ledger can therefore no longer masquerade as a full-field
restart. Transient, ordinary and affine nonlinear, harmonic, J2, creep,
viscoelastic, user-material, finite-strain J2, and cyclic-fatigue owners expose
that contract in their summaries. Cyclic bulk fields have moved from
partition-local shards to coordinate-keyed schema v2 and pass both 1-to-2 and
2-to-1 rank-count restoration. The composed global cyclic lifecycle now passes
those same migrations while preserving bulk fields, physical-facet-keyed
cohesive history, and the cycle-jump ledger before continuing to the
uninterrupted reference. The consumer boundary is now implemented:
`model.step()` resolves the exact
procedure contract before numerical execution, rejects malformed or
unsupported policies with stable `AFM-CHECKPOINT-*` diagnostics, and rolls
back any provider-registered Step objects when lowering fails. The same typed
contract is a top-level `SimulationResult` record, and engineering/release
verification checks every emitted checkpoint schema and portability claim
against it. Gate 5 is therefore closed for the declared procedures and tested
rank-count migrations. Collective high-throughput storage remains a scale
optimization, not a missing lifecycle contract.

## What is usable today

The installed capability catalog is authoritative. It covers the supported and
experimental boundaries for:

- linear solid mechanics, heat transfer, modal and structural dynamics;
- viscoelasticity, J2 plasticity, Chaboche hardening, creep, cohesive response,
  finite-strain hyperelasticity, and experimental finite-strain J2 workflows;
- multi-material regions, eigenstrain/thermal-strain semantics, material-aware
  result projection, periodic constraints, and selected Abaqus migration paths;
- mesh import, quality inspection, results, histories, progress,
  checkpoint/restart, provenance, convergence, and ParaView-oriented output;
- campaigns, scientific datasets, surrogate workflows, external providers,
  and learned-constitutive contracts.

Inspect the exact installed truth with:

```bash
agentfem capabilities
agentfem capabilities --json
```

## Near-term priorities

### 1. Trusted mechanics

The next scientific promotions focus on depth rather than catalog size:

1. complete external DCB, ENF, and MMB cohesive validation, including unstable
   propagation control and closed force--work--energy evidence;
2. advance finite-strain J2 through the remaining external RVE comparison.
   Ordinary strong-boundary J2 now records accepted-path dead-load
   work, prescribed-motion reaction work, stored energy, cumulative plastic
   dissipation, and their mechanical residual in the same rollback and
   checkpoint lifecycle. A four-to-forty increment refinement reduces the
   reference patch residual from about `9.4%` to `9.3e-6`, so the ledger
   exposes path-integration error instead of manufacturing closure. Current-
   configuration follower pressure now enters the same residual through its
   UFL-derived external tangent, while its accepted work integrates the
   configuration-dependent load vectors at both increment ends. Three-
   dimensional mixed P2/DG0 state now also passes one-to-two and
   two-to-one MPI checkpoint/continue with exact split primary fields and
   quadrature state. The external-beam gate is accepted from one
   executable package identity without relaxing a tolerance: the external
   curve, mesh, serial/MPI, checkpoint/restart, and `45 -> 90 -> 180`
   increment contracts all pass. The 90-to-180 RMS and maximum differences
   are `0.04460%` and `0.21498%`, within the fixed `0.2%` and `0.5%`
   contracts. For the separate Zhang Q2/DPC1 RVE, content-bound candidate and
   multi-axis audit records now replace caller-declared convergence flags;
   the scientific runtime, benchmark fixture/driver, executed high-order mesh,
   region/boundary tags, and periodic equations are bound explicitly. Path and
   quadrature slices must retain the identical discretization rather than
   relying on a nominal mesh-size label. A separate Appendix-B oracle
   evolving \(C_p^{-1}\) matches the provider's stress, plastic strain,
   inverse plastic metric and elastic energy on a non-coaxial path, so the
   remaining external gap is no longer attributed to a J2 normalization or
   state-variable mismatch.
   A 2026-10-07 source review found a phase-placement discrepancy between
   prose and Figure 10(a) in the inspected author manuscript. Explicit
   `geometry_source` now prevents mixing those models. Figure-based geometry
   at 2859 cells gives stress/energy/tangent errors of `0.099% / 0.014% /
   0.507%`, within the unchanged comparison contracts, but the last mesh pair
   changes energy by `5.3442%`; complete promotion remains blocked by energy
   convergence and remaining figure-specific evidence. The fine candidate
   has now been reproduced from clean commit `597e8795`, with identical stress
   and energy. A same-mesh 4/6/8-degree quadrature slice passes. Energy-channel
   attribution assigns 99.8554% of the last mesh change to the pressure
   constraint defect, not to the condensed channel; this is diagnostic
   attribution, not a license to switch the acceptance observable.
   Offline raw-point reconstruction localizes 97.19% of the defect in the
   worst 5% of cells, almost entirely in the matrix. An explicit interface
   refinement diagnostic uses 2448 rather than 2859 cells and reduces the
   defect by 17.30%, retaining the three external comparison checks. This
   remains a development experiment, not a mesh-convergence certificate or
   general adaptive-solver capability.
   A clean same-policy 1755/2448/3543-cell interface-size sequence now keeps
   all three external comparisons within their contracts. Its final stress
   and tangent changes pass at `0.06847% / 0.25812%`, but primal-energy change
   remains `0.85369%`, above `0.5%`; local refinement does not promote global
   mesh convergence. All raw-point energy reconstructions agree. Result-envelope
   auditing additionally exposed a self-hashed manifest; the writer now excludes
   only its own publication path while retaining external artifact verification.
   Original diagnostic archives retain the old envelope failure explicitly.
   The 2026-10-08 follow-up separates local from globally scaled refinement:
   local primal-energy change reaches `0.46575%`, but global change remains
   `0.72236%` (above `0.5%`). Local stress and global tangent also fail the
   decreasing-change requirement. Five clean runs verify their output seals;
   neither complete spatial axis is promoted. See the
   [spatial follow-up](https://github.com/haoming-luo/agentfem/blob/main/evidence/zhang_2021/2026-10-08-spatial-followup.md).
   Figure-specific 285-cell lifecycle verification now passes direct serial/MPI
   response equivalence, both cross-rank restart directions, and a local
   macro-tangent perturbation audit. A larger `1e-3` perturbation failed and
   remains recorded; MPI tangent condensation is still unsupported. These
   coarse lifecycle checks do not promote the fine spatial/formulation gates.
   See the [bounded verification record](https://github.com/haoming-luo/agentfem/blob/main/evidence/zhang_2021/2026-10-08-lifecycle-verification.md).
   The earlier diagnostics
   below use the prose geometry and must not be interpreted as unresolved
   constitutive errors or transferred to the figure geometry automatically.
   Initial 20/40/80-increment and degree-4/6/8 quadrature diagnostics show that
   path refinement and over-integration do not explain the remaining Table 5
   stress, energy and tangent gap. A 315/459/804/1590/2859-cell spatial
   sequence remains unconverged: at 2859 cells, close to the paper's stated
   2823, the published-energy error falls from about `20.5%` to `4.21%`, while
   the first-Piola error remains about `5.59%`. Element count alone therefore
   does not reproduce unpublished connectivity or geometry approximation. The
   fixed-old-state final-gradient differences now agree with the condensed
   macro tangent to about `1.43e-8`, even though that coarse tangent remains
   about `21.68%` from Table 5. This rules out an internally inconsistent
   Jacobian as the leading explanation without turning a self-consistency check
   into external validation. A controlled independently remeshed 1x1/1x2/2x1/2x2
   curved-geometry family now passes its 1% stress and 5% energy diagnostic
   contracts (observed maxima 0.7404% and 4.1493%). An independent Q9/DPC1
   hand-polynomial and curved-patch oracle passes. Serial/two-rank response
   equivalence and bidirectional 1-to-2/2-to-1 checkpoint continuation also
   pass. Exact discrete topology replication and full mesh/formulation
   convergence remain distinct open questions. The clean-source,
   content-bound `1e-5/1e-6/1e-7` tangent study now passes: its coarse-to-
   middle observed order is `1.998`, and the two finest finite-difference
   tangents differ by only `1.36e-8`. Blind refinement, parameter
   tuning, and tolerance relaxation are explicitly excluded;
3. extend the closed bounded MPC/weak/contact evidence contract only through
   independently verified provider routes; general contact remains a separate
   scientific promotion rather than a foundation blocker;
4. scale the now executable partition-independent raw integration-point HDF5
   result and portable quadrature-checkpoint identity from the compact
   root-gathered route to independently partitioned extreme-scale output;
5. retain every material and fracture capability at its proven maturity until
   its independent benchmark and failure tests pass.

The general-contact expansion after 0.3.8 is deliberately ordered by
ownership rather than by one forming example. Model assets will describe
surfaces, rigid-body motion, contact pairs and laws; Operators will own gap,
projection, residual, tangent, force, moment and energy; Procedures will own
search/update cadence and acceptance; State will own active projections and
history; Result/Verification will own penetration, generalized work and energy
closure. The immutable `RigidBody` asset now binds scientific surface,
fixed/prescribed kinematics, reference point, and rank-independent identity
without owning search or enforcement. `RigidContactPair` now binds one slave
boundary, rigid body, and scalar frictionless law without taking ownership of
search or Procedure stability. Prescribed analytical-plane motion and its accepted
force--moment--work contract are now the first executable slice. Next come
the solver-neutral projection contract (now executable for analytical planes,
circles/spheres and infinite cylinders), reviewed tessellated surfaces (the
serial triangle reference projector and fail-closed geometry audit are now
executable, together with a deterministic process-local AABB/BVH accelerator
and per-query work evidence plus stable barycentric facet coordinates),
distributed ownership/search (the stable-ID facet partition and all-gather
correctness reference now reproduce the serial oracle under MPI; scalable
neighborhood routing now has an oracle-checked two-stage rank-AABB path with
aligned integer/float `MPI_Alltoallv` payloads; owned exterior triangles from
first-order DOLFINx tetrahedral meshes now enter the same stable-ID contract,
including empty local shards; imported MeshTags and named BoundaryRegion
subsets now preserve their physical selection, while higher-order tessellation
and measured extreme-scale routing remain),
accepted/trial projection State (stable 64-bit contact-point identity,
facet-changing finite-slide trials, rollback, and accepted-boundary snapshots
are now executable without importing solver or contact-law ownership), then a
Procedure-owned search/update cadence (now executable as an every-evaluation
lifecycle with collective MPI rejection and explicit increment
commit/rollback) and an Operator that consumes the trial projection. Candidate
warm starts may optimize broad-phase work later, but exact gaps and normals
remain evaluation-local. Accepted discrete-surface projection evidence is now
serialized by stable global point ID and restored onto the current MPI
partition, including master facet and barycentric coordinates. It remains
audit State, not a search cache: projection is deliberately recomputed after
restart from portable accepted displacement and schedule time. The explicit residual hand-off
now supports fixed or proportional prescribed motion, a mandatory declared
stability ceiling, rank-canonical force--moment--work evidence, accepted-path
restart, and finite sliding over one serial or routed distributed triangulated
tool. Bounded rigid pairs compose with independent identity and summed
energy evidence. The first friction foundation is now solver-neutral: a pair
can declare penalty Coulomb friction; stable point-keyed tangential State
supports objective normal-rotation transport, stick/slip return mapping,
trial/commit/rollback and accepted-boundary snapshots; and its response keeps
recoverable penalty energy, irreversible sliding dissipation, and separation
release distinct. The bounded DOLFINx explicit residual now consumes that
contract. It follows the same master material point through prescribed
translation/rotation, forms the accepted slave-minus-master increment,
assembles normal and tangential residuals, and publishes separate recoverable,
dissipated, and separation-release energy channels. Point history is
checkpointed by global ID and restored onto the current MPI partition,
including empty local shards. The mandatory caller-supplied stability ceiling
now has a conservative automatic alternative for fixed-normal and
piecewise-planar geometry. It includes normal/tangential penalty stiffness,
Coulomb pressure coupling, cross-component contact coupling, real lumped mass,
and ghost-to-owner accumulation. It publishes contact and combined spectral
evidence, and deliberately rejects the unsafe shortcut of taking the smaller
of separately derived body and contact time limits. The external
finite-sliding gate is fixed to the public Abaqus/Explicit
deformable-body/rigid-surface family rather than a forming-specific solver.
Its parameters, two-stage protocol, and force/friction assessor are
machine-readable, while the registry explicitly retains
`protocol_solid_bridge_refinement_accepted_exact_b31_gate_pending`. A real deformable
tetrahedral-solid bridge now applies the fixed 500-unit resultant, accepts the
frictionless preload, atomically transfers it into a new stage, activates
`mu=0.3`, and completes the 0.1-unit rigid slide. Serial and two-rank runs
recover 500 normal force, the 150 Coulomb force, exact action--reaction,
positive dissipation, valid projections, actual stable-facet crossings over a
bounded tessellated tool, and the declared energy tolerance.
This is explicitly protocol-level evidence, not a pointwise B31 reproduction.
The bridge now also owns separated refinement evidence: one run halves both
stage increments at fixed physical duration, then another increases in-plane
mesh resolution while retaining the refined increments and duration. The
accepted certificate checks dissipation stability, energy non-growth, and
facet migration without claiming an observed convergence order from only two
levels.
Registered rigid pairs enter the
ordinary finite-strain Explicit Procedure without user-authored residual
plumbing. The Procedure builds reviewed traces, composes body, cohesive, and
every contact spectral contribution once, and preserves per-pair force,
moment, tool work, potential, friction dissipation, State, and checkpoint
evidence. Imported triangle tools and curved CG2 slave geometry now enter this
bounded route; Q2 hexahedra complete the Explicit Procedure while standard P2
tetrahedra fail the positive-mass gate. The next independent gate is an
element-equivalent external comparison if a reviewed beam-contact formulation
becomes available. Distributed multi-tool candidate routing, measured
extreme-scale behavior, additional topology, and implicit friction
linearization follow that bounded route. STEP
remains an optional geometry-adapter input that is repaired and tessellated
before the contact core consumes it. Shell/solid-shell forming and self-contact
remain later scientific promotions, not hidden extensions of the bounded
fixed-plane provider.

The first imported-tool boundary is now executable for triangle-only formats
supported by the optional `meshio` adapter, including STL. Import requires an
explicit coordinate scale because STL does not encode a dependable length
unit, binds the file SHA-256 as provenance, canonicalizes exact geometry to
stable facet identity, and exposes normal reversal only as an explicit user
decision. Mixed topology, degenerate/duplicate facets, non-manifold edges and
inconsistent winding fail before search. Approximate repair and STEP-to-mesh
tessellation remain external preparation capabilities; they do not silently
alter the contact Model asset.

### 2. Mesh and element foundation

Mesh breadth advances by preserved semantics and executable evidence, not by
recognizing more connectivity names.

- Maintain verified P1/P2 simplex and tensor-product import, patch, tag,
  quality, output, and MPI routes.
- Promote prism and pyramid routes only when the active DOLFINx I/O and solve
  stack preserves the external topology end to end.
- Keep beam, shell, cohesive, reduced-integration, hybrid, and stabilized
  elements behind dedicated formulations rather than topology aliases.
- Admit mixed-topology solve domains only when regions, assembly, results,
  checkpointing, and MPI preserve every block.
- Diagnose quality without silently moving nodes or remeshing a scientific
  input.

### 3. Composites and manufacturing

Composite development proceeds through reusable mechanics:

1. stable orientation, ply, thickness, orthotropic material, laminate, and
   failure-quantity semantics;
2. a public fibrous-shell Step with patch, locking, boundary-moment, state, and
   restart evidence;
3. contact, friction, inter-ply slip, forming controls, and per-layer output;
4. verified mapping of forming orientation, thickness, and defect state into a
   structural model.

Paper-specific geometries and conclusions remain outside the core.

### 4. Engineering workflow

- Keep one Step and one result lifecycle across procedures.
- Improve run comparison, artifact opening, imported regions/surfaces/sets,
  quality diagnostics, and selected Abaqus migration.
- Preserve project data independently of replaceable runtimes.
- Make long solves observable through bounded progress, stability and energy
  diagnostics, and restartable checkpoints.
- Keep concise human output and complete structured output as separate views of
  the same scientific record.

### 5. AI, data, and extensions

- Keep the core free of PyTorch and model-specific neural terminology.
- Let external providers own runtimes, devices, weights, and framework details
  while AgentFEM owns scientific conventions, state transactions, evidence,
  and failure semantics.
- Make campaigns resumable, deterministic, auditable, and independent of the
  physics definition of one case.
- Treat visualizations as views; retain a stable machine-readable scientific
  result as the data source.
- Keep MCP and other agent entrypoints thin: they operate AgentFEM rather than
  becoming another solver.

## Promotion discipline

Every capability moves through explicit maturity levels:

```text
contract -> experimental -> verified -> validated
```

- **contract:** syntax, ownership, and failure behavior are defined;
- **experimental:** a bounded implementation and regression evidence exist;
- **verified:** analytical, manufactured, or independent numerical benchmarks
  demonstrate correctness within a stated applicability domain;
- **validated:** suitable experimental or field evidence supports the declared
  physical use.

Performance evidence, a successful example, or a comparison with AgentFEM's
own earlier output cannot raise maturity by itself. Unsupported geometry,
missing evidence, incompatible constraints, and stale checkpoints fail closed.

## Later, after the foundation

The following remain valuable but must not destabilize the current spine:

- broader contact, fracture growth, phase-field, and multiphysics families;
- adaptive refinement and explicit, auditable mesh repair;
- richer CAD and assembly ingestion;
- a second finite-element backend as an architectural pressure test;
- larger graphical workflows built on the same Model/Step/Result contracts.

AgentFEM will not rewrite element tabulation, quadrature, distributed assembly,
or linear algebra merely to appear independent of FEniCSx. Its moat is the
stable scientific boundary that tells humans and agents what should be
computed, how it is advanced, what state is owned, and why the result can be
trusted.
