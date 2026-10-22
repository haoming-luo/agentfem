# Portable explicit performance strategy

## Decision

Preserve the engineering Model and correctness obligations, not this machine's
thread counts or this review case's time grid. Separate execution optimization,
numerical-method comparison and hardware selection. The next major target is
the accepted-increment execution path and the number of full-domain updates,
not another isolated arithmetic kernel or output-format rewrite.

## 2026-10-22: late-path audit and bounded execution changes

Worker-budget screening initially stopped at checkpoint admission: the Q1
execution identity contained projection/exact-service worker counts. Preserve
that failure as evidence, not a reason to patch a checkpoint. Contact snapshots
now have a v2 compatibility digest, using the same explicit-provider-declaration
pattern as the body kernel. Native ordered independent queries declare their
partition invariance; all physical, arithmetic and search/reuse settings remain.
The old v1 path still requires the original execution identity, then an ordinary
save upgrades the record. 58 targeted geometry/contact/restart tests pass,
including both worker directions, strict legacy migration, changed-provider and
malformed-record rejection, and outer-commit rollback. This is operational
portability, not a performance result by itself.

The resumed 400-increment worker screen retains identical accepted fields and
residual-state/work snapshots. Joined execution with material/projection budgets
8/4, 6/2, 4/2 and 8/2 takes 47.682, 50.749, 53.872 and 48.615 s with the smooth
sign policy. Bracketing two-ray 8/4 runs take 47.967 and 48.886 s. No worker
change is selected; the sign-policy timing difference is inconclusive at this
scale. These are not recommendations for another computer. The next private
harness also fingerprints per-increment history records separately: the earlier
"physical history" fingerprints referred to the final accepted residual snapshot
and cumulative work, not every intermediate record. Public evidence now makes
that scope explicit.

Installed whole-path acceptance is now complete at `1ed3c1e1`: 12,295.686 s
including preparation, output and shutdown; 25 frames, final material and
checkpoint arrays, 10,000 energy records and 800 observations equal the frozen
baseline. Independent VTK verifies every frame and zero Warp error. See the
public performance evidence for timing-scope distinctions. Do not replace the
controlled 12.11% late-window finding with the larger, confounded whole-run
difference. Retain the prior complete run as reference.

The closing source audit also adds explicit exception cleanup to prepared
linear operators: constructor failures release already owned PETSc objects,
RHS vectors release on failed solves, stale convergence evidence is cleared,
and failed matrix refresh closes the unusable owner. Six fault-injection tests
fail on the frozen old implementation and pass on the corrected source.
This is a lifetime/failure-semantics correction, not an explicit-run speedup.

The complete-history encoding is separately opt-in through
`RunPolicy(history_storage="npz")` or result publication. It retains every
binary64 sample, deduplicates identical axes and publishes immutable companion
arrays before the manifest commit point. Data-only readers do not initialize
FEniCSx; malformed files and rank-inconsistent histories reject explicitly.
The combined source regressions pass 237 tests (one MPI-only skip), and the
two-rank history/prepared-solver subset passes 14 tests per rank. The synthetic
storage benchmark is recorded separately from solver timing.

A clean native wheel built from `0913ab9a` also passes the same 237 targeted
tests (one MPI-only skip), with repository pytest path injection disabled.
All three native binary hashes match the earlier complete-run wheel exactly.
The closing Python changes have separate regression coverage; this is not a
claim that a second full trajectory ran at the closing revision. Source,
wheel and local JUnit fingerprints are recorded in
`evidence/performance/2026-10-22-closing-installed-validation.json`.

The final source follow-up aligns the two prepared MPC paths with ordinary
linear solves: clear previous convergence evidence at the start of a new
attempt, clear old algebraic residual/periodicity diagnostics, and honor the
explicit nonconvergence policy. Diagnostic return retains a failed reason and
actual residual; it does not promote an unconverged field. The algebraic
summary now exposes its solver policy and latest solve record. Both stale-state
regressions fail on the preceding installed candidate; 45 source checks and
six targeted checks per MPI rank pass after the fix. These are separate from
the preceding installed-wheel and complete explicit-run evidence.

The completed 25-frame trajectory at `743f10b4` recorded 14,932.091 s of
process elapsed time (not an isolated benchmark); inclusive residual assembly
was approximately 89% of solver wall time. Field output is not the dominant
cost. Preserve that run rather than regenerate a delivery archive.

A separate 95,000-to-95,120 continuation with detailed profiling recorded
18.939 s of advancement, 14.853 s inclusive body time and 13.795 s inclusive
deformable-contact time. The latter two **overlap** and cannot be summed.
These profiled numbers locate work; they are not uninstrumented throughput.
The contact-thread profile revealed repeated connectivity sorting during
current-geometry refitting, in addition to projection and inside/outside work.

### Invariants before implementation

- Reuse only exact connectivity adjacency and verified coordinate-equivalence
  maps. Adjacency cache keys are full ordered connectivity bytes, with at most
  four inputs of at most 4 MiB. Returned metadata are independent over immutable
  byte storage. No cached normal, gap, contact force or geometry-validity result
  survives a new configuration.
- For Q1 refitting, reuse the first-occurrence weld map only if every existing
  class still has equal coordinates and distinct classes remain distinct.
  Splitting/merging classes rebuilds through the original welding algorithm;
  a shared physical node with inconsistent coordinates still rejects. Geometry,
  normals, degenerate triangles, bounds and exact projection remain current.
- Extend the native affine-remainder **sufficient** cell screen beyond the
  near-identity ball. A normalized, cancellation-filtered determinant establishes
  orientation independently of the Gram matrix. Gershgorin lower/upper bounds
  on `F.T F`, with roundoff margins, bound singular values for the existing
  remainder test. Uncertain orientation, nonpositive spectral lower bounds,
  overflow or an excessive nonaffine remainder retain reference SVD/Bernstein
  checks. Positive mean determinant alone never admits a cell. These floating
  point guards are not a formal interval-arithmetic proof.
- No material law, increment, numerical mass, contact tolerance, force summation
  order or accepted-state/checkpoint payload is changed by these optimizations.

The spectral bounds use the standard [Gershgorin inclusion theorem](
https://www.cs.cornell.edu/courses/cs6210/2025fa/lec/2025-10-22.html).
The full-cell fallback retains the Bernstein approach described by
[Johnen, Weill and Remacle](https://hdl.handle.net/2078.5/258622).
Neither reference is a performance result for AgentFEM.

### Performance ownership

The existing `PerformanceLedger` now records `body_residual_inclusive`,
`body_cell_gather`, `body_kinematics_admission`, `body_material_and_force`,
`body_bulk_viscosity`, `body_force_scatter`, `deformable_contact_inclusive`
and `contact_owner_wait` when those paths execute. Body subphases are nested;
contact may overlap the body. `contact_owner_wait` measures the owner's wait
at the join, not total contact cost and not CPU utilization. Native-call elapsed
time may include scheduler/interpreter contention. Attempted failed phases
remain timing evidence; timing does not mean physical state acceptance.
Workers return elapsed durations; only the owner thread mutates the shared
ledger. Timings are operational evidence, not restart identity or physical state.

Promotion requires an unprofiled alternating same-state continuation comparison,
including displacement, velocity, acceleration, deformation gradient, material
history and stress. The early profiled screens had equal numerical fields but
are insufficient to claim a whole-run percentage. Full-run, other-machine and
WSL throughput remain separate evidence obligations.

### Same-state alternating continuation evidence

The 95,000-to-95,600 continuation used the unchanged private review material
and checkpoint, with material/cell workers 8/4 and geometry/projection workers
2/4. Runs were sequential, with cProfile disabled. Setup, checkpoint loading and
the final comparison archive are outside the measured advancement; five small
elapsed-time wrappers are present in both routes. This is not a complete-run
benchmark or a change to the recommended worker count on other computers.

| Execution order | Route | Advancement seconds |
| --- | --- | ---: |
| A1 | Frozen installed `743f10b4` | 87.257603 |
| B1 | Native affine screen + topology/weld reuse + shared timing ledger | 76.804792 |
| B2 | Same candidate, including immutable node-identity storage | 75.001206 |
| A2 | Frozen installed `743f10b4` | 85.470891 |

Candidate mean is 75.902999 s versus 86.364247 s for the baseline: **12.11%
less elapsed time in this late-path window**, or 1.138 times throughput. B1
preceded the metadata-only node-identity immutability hardening; B2 includes it.
All six compared full arrays (displacement, velocity, acceleration, deformation
gradient, material history and Cauchy stress) are bitwise equal across all four
runs. Body summaries, including accumulated material internal work and bulk
viscosity work, are equal. This parity verifies unchanged execution for the
tested path, not the physical validity of the mass scaling or material model.

Mean inclusive contact time decreased from 64.391321 to 52.776002 s; mean
inclusive body time decreased from 69.464271 to 58.818001 s, while the directly
wrapped material/force call stayed near 30.3 s. The concurrent branches cannot
be added. Reduced contact-side sorting/interpreter work can also shorten body
waiting, so these are not independent speedup contributions.

The affected geometry, response-material, deformable-contact, rollback/restart,
viscosity and performance-ledger regressions passed **190 tests** in the real
FEniCSx environment. Lint and whitespace checks passed. Subsequent early-state
and installed whole-path checks close this execution-parity gate, not
independent physical verification. Private full arrays remain in local evidence
folders;
no confidential material or model assets are copied into this repository.

### Exact coordinate identity follow-up

An internal shared coordinate-identity helper now uses stable columnar
`numpy.lexsort` rather than structured-row `numpy.unique` for duplicate checks
and first-occurrence Q1 welding. This is exact binary64 equality, not a new
mesh-merging tolerance. Signed zero remains equal, adjacent representable
coordinates remain distinct, and the first input point owns its stored bits.
Nonfinite/complex/malformed inputs reject. The helper does not change imported
mesh numbering or the STL importer's separate lexicographic storage convention.
[NumPy documents the stable multikey ordering](
https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html).

On one synthetic 50,000-point input, ten duplicate checks took 0.128–0.132 s
with the former row sort and 0.083–0.085 s with column sorting. This is a
microbenchmark only. A fresh full-mesh 95,000-to-95,600 continuation took
76.183 s, within the earlier combined candidate range: **no distinct extra
whole-step gain is established**. All six arrays and the complete body/work
summary remain exactly equal to the frozen original baseline. The combined
geometry, contact, solver-lifecycle and nonlinear restart subset passed 187
tests. This inexpensive common helper is retained without inflating the
earlier 12.11% window-scoped result.

### Early-path confirmation

A separate from-start 600-increment pair took 93.194532 s on the frozen
`743f10b4` installation and 81.550461 s with the combined candidate through
`f500e484`: 12.49% less advancement time in this single pair. Both executed
614 residual attempts (including initial substeps), with bitwise-equal six
state arrays and equal complete body/work summaries. This extends execution
parity to the initial path; one pair does not establish a statistical speedup
or replace the late alternating comparison. Setup and final comparison archive
remain excluded. The later complete installed trajectory above supplies a
whole-run runtime, with its separate load caveats explicitly retained.

### Versioned operational restart and policy consistency

The native body now records scientific restart compatibility separately from
actual execution provenance. Only declared independent-point/cell scheduling
may change: material/cell workers and material tile scheduling. Code/law,
parameters, state encoding, numerical controls, mesh/mass and backend arithmetic
remain checked. Legacy v4 body archives retain exact execution identity and
must first be restored compatibly and saved to a new record; no identity is
removed or rewritten. Custom kernels default to exact identity.

The targeted material/response/contact/substep suite passed 194 tests. A
subsequent 136-test subset covers bidirectional worker/tile restart, legacy
resave, atomic mismatch/corruption rejection, built-in native worker changes,
NumPy/backend refusal, and solver policy/lifecycle checks. Counts overlap;
they are not added into one claimed independent suite. Both field-backed and
owned-array material paths retain the tested accepted-state equivalence.
This removes a portability restriction, not a measured runtime percentage.

## Completed baseline

Frozen core `608c50bb` completed the 255,041-cell, 100,000-increment review path
in 14,646.987 s, including preparation, export and shutdown. The preceding
generic-core baseline took 15,465.055 s: 5.29% less elapsed time. All 60 final
numeric checkpoint arrays are bitwise equal, all 374 checkpoint comparisons
pass, and 10,000 reporting records agree. An independent ParaView reader sees
one 3D volume grid and exact Warp coordinates. The first offline presentation
check was blocked by sandbox-local communication; the separately authorized
rerun passed without rerunning or altering the simulation.

These runs share the same reviewed numerical-inertia and contact assumptions;
they are implementation evidence, not independent source-solver validation.
Lightweight development and read-only diagnostics overlapped the full run.

## What is portable and what is not

| Concern | Current finding | Intended ownership |
| --- | --- | --- |
| Model and material equations | Common Model/Step workflow; law remains external | Scientific inputs, not hardware tuning |
| Performance records | `results/performance.py` stores elapsed cost; no four-hour correctness gate | Result evidence, separately from accuracy |
| Workers | Native pool and dispatchers use 16-entry arrays; Python rejects counts above 16 | Backend implementation limit, not FEM semantics; replace with checked allocation when extending tested capacity |
| Restart | Native-material body records now separate versioned restart compatibility from actual execution provenance; v4 retains its exact check | Worker/tile changes require a partition-invariant declaration; code/law/numerical settings stay checked. Restore old archives with their original configuration, then save a new compatible record; never strip checks |
| Time grid | The private provider applies a uniform per-point increment ceiling | Case policy requiring scientific evidence; not a universal material or platform limit |
| Fixed final counters | 100,000-step checks belong to this private delivery reviewer | Use declared termination criteria for other problems |
| Scalar precision and order | Bitwise replay is useful for unchanged arithmetic | Algorithm/backend changes use justified observable errors and convergence instead |
| Runtime package | Selected binary is macOS-specific; core has reference fallbacks | Record actual capabilities; a local binary is not a cross-platform artifact |

Three comparison modes prevent the reference case becoming a product constraint:

1. Same algorithm: keep scientific inputs and accepted states fixed while
   changing data movement or execution. Exact comparison is appropriate where
   the operation order is unchanged.
2. New method: measure time to a declared accuracy using common physical
   stations and suitable field mappings. A different step count, mesh or
   summation order is allowed; error and event/state checks replace byte parity.
3. New machine: select supported resources locally, retain the selected
   configuration as provenance, and separate performance changes from accuracy.

An optional tuning cache should key on backend/precision, hardware capabilities
and workload size/shape; stale or missing entries use conservative defaults.
Do not require calibration to run. Do not select mass, penalty, physics or
accuracy through a hardware tuner. Operational settings remain outside case
physics, and observed winning worker counts are not global defaults.

## Where the completed run spends time

Solver run: 14,619.008 s. Residual assembly accounts for 12,909.961 s (88.31%);
the encompassing explicit increment accounts for 90.89%. They are nested,
not additive. History snapshots account for 2.79%. Preparation and the
remaining non-run process work together account for only 27.979 s (0.19% of
complete wall time). The accepted integration count is 100,007 for 100,000
macro increments: cutback proliferation is not the current problem.

With all other work unchanged, even eliminating residual assembly would give
only about 8.55x run speedup. Accelerating that phase 4x would give about 2.96x,
not 4x. These are idealized ceilings, not forecasts. A tenfold gain therefore
needs a broader reduction in work per accepted step, fewer full-domain steps,
or an independently justified change in hardware/formulation.

## Prioritized development slices

### 1. A compiled increment path with explicit state ownership

The current body dispatcher already fuses material response with force work.
The gap is the surrounding gather, geometry admission, viscosity, publication,
contact orchestration and repeated Python/native crossings. Extend the existing
Procedure backend rather than introduce another Model or case-owned solver.

A bounded compiled execution plan should keep working buffers resident, join
body/contact branches before acceptance, and reuse scratch storage only when
no published reader owns it. Immutable result snapshots must remain valid;
leases or copy-on-publication are implementation choices, not new physics.
Start with one accepted increment. Multi-increment windows may stop at output,
checkpoint, callback or cutback events, but must still commit or reject each
increment correctly. Unsupported callbacks use the reference route.

Measure the critical path with and without overlap, not a sum of nested Python
profile rows. A native call's wall time can include GIL reacquisition or waiting
for another branch. Demonstrate reductions in allocations, crossings and
whole-increment time at early and damaged/late stations before promotion.

### 2. State-dependent work reduction and time accuracy

Expose which contribution limits advancement: material increment accuracy,
body stability, penalty/contact curvature, damping or an explicit user ceiling.
The current `dt="auto"` selects an initial fixed increment; it is not a general
adaptive nonlinear trajectory. A local observed tangent is not a trajectory
stability certificate. A damage-softened stress does not prove every relevant
stiffness or contact bound has decreased.

First test a separately declared time-grid experiment at unchanged inertia and
loading duration. Compare reaction curves, damage localization, work, contact
events and time refinement against a finer reference; replay from an existing
checkpoint alone cannot certify earlier errors. Keep inertia sensitivity a
separate axis. If the uniform material ceiling is replaced, preserve the
incremental-state admissibility check and identify the replacement's scope.

Selective local time stepping becomes attractive only when a small
cost-weighted region controls the clock. For restrictive fraction `q` and
coarse/fine step ratio `r`, ideal work is `q + (1-q)/r` of global fine stepping,
before interface/contact/communication overhead. No speedup is promised from
element counts alone. The current uniform material ceiling effectively makes
that fraction one if the same update must still be performed everywhere at
every fine step: subcycling then offers no such saving. Establish whether the
ceiling is a necessary global accuracy condition or a conservatively uniform
case policy before investing in local stepping. Shared nodes, contact impulses/work, constraints,
irreversible state and global checkpoint times need a reviewed synchronization
contract. Do not confuse the existing global cutback subdivision with this
missing multirate capability.

### 3. Contact temporal coherence, extending what already exists

BVH refit, analytical rollers, exterior separation bounds, swept checks and
native projection are already implemented. Reintroducing them under another
name is not progress. The full run still performs hundreds of millions of
point classifications; counts identify repeated work, not its critical-path
share, because contact can overlap body work.

Evaluate conservative candidate-set reuse and inside/outside classification
reuse with displacement/topology validity bounds. Reusing a classification
does not authorize freezing a normal, force or slip state. Reproject active
contacts on the current geometry. Topology changes, new crack faces, failed
motion bounds and rollback must invalidate or rebuild the affected cache.
Admit the optimization only when it reduces whole-step time, not only queries.

### 4. Larger opportunities with explicit applicability gates

- Device-resident CPU/GPU execution can reuse the same Operator/State contracts;
  it needs a compatible material implementation, precision policy, scatter and
  contact path. Moving only a tiny kernel across devices is not the design.
- Quasi-static implicit/explicit alternatives can reduce wave-resolution work,
  but this response-only history material lacks an admitted global tangent.
  Preserve rate-dependent history, stability and path selection; a static solve
  is not an equivalent replacement for a dynamic instability.
- Spatial adaptivity, shells or solid-shells may reduce degrees of freedom and
  restrictive small elements. They change approximation and damage
  regularization requirements; they are not a transparent Hex8 optimization.
- Surrogates and reduced models target repeated-query throughput after training
  and validation, not the cost of this first high-fidelity reference run.

## Decision gates

The immediate next experiment is the compiled accepted-increment critical path,
paired with a scoped investigation of the uniform time ceiling. Do not begin a
full GPU rewrite, general multirate contact, shell formulation and implicit
damage solver simultaneously. Preserve the completed reference; freeze only a
selected candidate after the relevant correctness and error tests. A tenfold
goal is a research target, not a release acceptance statement without a run.

## Bounded follow-up experiments (2026-10-14)

The two-hour follow-up keeps the completed project unchanged and uses separately
installed wheels. One numerical process runs at a time. Grouping finite checks
into one call was rejected after both 600-increment runs became slower. A
small-array interpreter-lock threshold was also not selected: its 100–600 span
was 83.287 s against reference spans of 82.768/82.564 s. Neither attractive
source structure nor a technique used by another library establishes a gain.

One-pass copy/finite admission was a smaller candidate: immutable transport
still owns detached bytes and validates the exact bits it publishes. Two early
spans measured 81.190/80.546 s, about 2.18% below the reference mean. All final
numeric arrays, every-increment histories and 61 journal rows agree exactly in
each pair. Three archived body trials at steps 20,000/60,000/90,000 preserve all
12 arrays but do not show faster body execution.

The subsequent fresh 2,000-increment ABBA comparison rejected that candidate:
reference 100–2,000 spans were 288.950/291.249 s; candidate spans were
292.301/293.359 s. Candidate mean was 0.94% slower, despite exact numerical
arrays, every-increment histories and 201 journal rows in both pairs. The
runtime edit and its experiment-only test were removed from the product tree;
source snapshots, installed wheels and evidence remain in the private sprint
archive. Lightweight code/metadata work overlapped these sequential runs; no
second numerical job did. This does not establish a new full-run timing.
Do not promote the misleading two-percent short-run gain into a release claim.

The material provider was rebuilt from pinned source with ordinary optimization
and with GCC link-time optimization, preserving floating-point contraction and
fast-math restrictions. Both variants reproduce all 12 body arrays at the same
three stations exactly. LTO did not improve their phase times and is not selected.
The reproducible build recipe is useful private delivery evidence, not a new
public material implementation or cross-platform test.

An independent new-method screen increased the macro time step from 1e-6 to
5e-6 s, with unchanged inertia, loading duration, material law, contact/bond
settings and incremental stretch admission. It was stopped after gross energy
growth: at 0.0003 s, the ledger reported about 46.578 N·mm of accounted energy
against 1.34e-7 N·mm of input work. The process archived its accepted failure state
and exited; it did not reach the first 0.005 s comparison station. This is a
rejected experiment, not a faster completed solve. A material strain-increment
check is not a coupled body/contact/damping stability certificate.

The next time-step task is therefore diagnostic: identify the active bound and
its spatial cost distribution, including anisotropic body response, hourglass,
bond/contact penalties and damping. Only then choose global refinement or a
local-time method. Existing elastic wave/variational contact results do not
automatically cover irreversible, rate-regularized history damage. Keep this
scientific-method track separate from the unchanged-equation execution track.

The public design should name these constraints separately: constitutive update
admissibility, operator stability estimate/certificate, requested accuracy and
user ceiling. Each carries its source, evaluated state and scope; the Procedure
selects their applicable minimum. This extends the existing increment-limit
and stability contracts rather than pretending a response-only material has an
implicit tangent. Damping belongs in the stability analysis: adding it is not
a justification for increasing an explicit step. The official Abaqus explicit
analysis documentation likewise distinguishes fixed stepping from updated
stability estimation and calls for energy/history inspection of fixed-step runs.

## Recipient admission, not a hardware-dependent Model

`tools/check_frozen_project.py` checks a declared frozen-project manifest and
requirements before loading the trusted project. Hash checks are standard-library
only; `--probe` additionally tests imports and declared native libraries in a
bounded child, without running `case.py`. Its report distinguishes intact files,
binary-platform compatibility, runtime readiness and scientific validation.
It never rewrites the Model or a checkpoint to fit the host.

The completed macOS package currently links private Fortran libraries against
Homebrew GCC runtime paths. Copying it to a different computer is not proof of
readiness; Linux/WSL require an appropriate rebuild, and another Mac requires
compatible dependencies. A passing probe on this host is not recipient acceptance.
Requirements describe that particular artifact, not a universal AgentFEM OS,
Python ABI, worker-count or four-hour requirement. Recipient tests must retain
both the model identity and the new runtime provenance.

Release order remains: preserve the passed reference; select only measured and
numerically admitted changes; freeze the chosen runtime; perform recipient
preflight and a bounded installed solve; then run a fresh full timing/response
acceptance. Do not relabel short-run evidence as completion of that final gate.

### Next implementation slice, after this delivery

Build one optional **single-increment execution plan**, not a second public
solver. Its prepared inputs are existing topology/geometry, declared native
material and contact capabilities, typed time inputs and the current accepted
state. Reuse the existing native workers. Keep two independent responsibilities:

- Procedure schedules body/contact work, joins it and applies acceptance rules.
- State retains the old accepted buffers until all trial branches succeed;
  observers receive valid published state, never recycled scratch storage.

The first proof must include current-gradient gathering, body/material/viscosity,
contact, nodal update and acceptance—not just the already-fused material call.
Unsupported Python material/time callbacks use the reference route. A later
multi-increment window may stop at declared reporting/checkpoint/callback events;
it may not change their semantics or roll back already accepted earlier steps.
Failure-injection, cutback, observer visibility and restart checks precede any
throughput claim. Measure identical early, damaged and late accepted increments
before expanding the scope. Only after a clear full-increment gain should this
candidate consume another four-hour acceptance run.

In parallel at the design level, specify a stability-attribution report using
the existing contribution contracts. It should distinguish measured/estimated
bounds from caller ceilings and name the controlling cells/pairs and state.
Do not implement a general adaptive or local-time driver until that report
shows which work can safely run less often. This sequencing concentrates the
next sprint on one executable boundary, rather than another set of noisy
microkernel substitutions or simultaneous CPU/GPU/multirate rewrites.

## October 2026 execution and recipient-UX slice

The current candidate preserves the time grid, material and numerical mass.
It moves admission checks into the joined native material/body pass, keeps
immutable outputs as the publication boundary, batches owner-cell matching in
contact preparation, and reuses a checked output field layout. A worker-local
rejection still receives reference repair before any batch is published.

Two further policies are opt-in, not portable defaults: dynamically scheduled
material tiles and anchored closed-surface separation bounds. The former may
improve cache locality or heterogeneous-core balance, but increases dispatch
work; the latter bounds displacement from a fixed accepted geometry rather
than accumulating travel from old oscillations. Neither changes constitutive
equations or reuses stale contact forces. Both retain rollback/restart checks.
Do not select them solely from a kernel microbenchmark or a static geometry.

The history adapter already calls the reviewed VUMAT once per batch, not once
per integration point. Its arithmetic pathway is already native and the body
matrix-free. Rebranding these existing properties is not an optimization.
Guarded inverse-Newton polar iteration is a separate algorithm investigation;
retain the original domain, residual checks and reference repair, then compare
early and late material histories before using it in a complete trajectory.

Recipient usability is independent of scientific tuning. Project preflight
can require public API keywords and explicitly selected native accelerators;
ordinary cases without those requirements retain their reference routes.
WSL storage/CPU/memory observations are read-only startup advisories, never
automatic file migration, environment changes, or proof of execution speed.
An available binary is not evidence that a particular Step selected it.

One accepted-response XDMF series can include the final state. Raw endpoint
NPZ/JSON remains available without writing a duplicate final visualization.
Transient checkpoint JSON is compact, retaining all values and schema; this
reduces storage rather than claiming a solver speedup. The primary artifact,
partial/completed status and phase timings belong to the run record. Old
completed artifacts are not rewritten to adopt this layout.

Acceptance remains staged: installed boundary tests and independent volume/
Warp reading, frozen full-run preservation, isolated policy/algorithm timing,
then a selected fresh full candidate when there is a decision-changing gain.
Record concurrent compilation/test interference in any observed wall time.
No macOS result substitutes for a WSL recipient's actual installation/run check.

## Two-front screening decision (2026-10-15)

Keep implementation speedups separate from changes to the numerical problem.
The isolated material tile experiment reduced kernel time by approximately
13–18% at the sampled early/late states, but the single profiled 200-step pair
reduced advance time only from 32.58555 s to 30.84569 s (5.34%). It is a
candidate, not a whole-run claim. The alternate polar iteration did not show
a decision-changing advantage and is not selected.

The next implementation target is repeated full-array work across kinematics,
material admission, force and bulk-viscosity preparation. Profiled state
commit itself was negligible; rewriting transaction ownership wholesale is
not justified by this measurement. Prototype a fused checked path only where
immutable ownership proves that checks can be shared. Preserve the generic
reference route and all failure/publication guarantees.

The external algorithm track has three ranked questions:

1. Can displacement-bounded contact candidate reuse avoid repeated search?
   The existing anchored route must be measured before adding another cache.
   Reuse candidate exclusion certificates, never stale forces or normals.
   Test approaching/separating surfaces, face crossings and rollback.
2. What actually limits each accepted increment: body, hourglass, interface,
   contact, or damping? Report controlling contributions before attempting
   a less conservative global estimate. Include changing geometry and damage;
   a material increment limit is not a dynamics stability certificate.
3. Are small controlling regions sufficiently isolated for subcycling or
   selective mass treatment? These are separate method studies requiring
   interface work, inertia and response convergence checks, not execution
   settings to silently enable in this case.

LAMMPS skin-distance reuse is a design analogy, not a ready-made proof for
deforming Q1 surfaces. Abaqus documents changing nonlinear stability limits
and the reduction of stable steps by damping. Variational mass-scaling
research offers a longer-term alternative, but changes the mass operator and
must not be presented as unchanged-physics acceleration.

Reject a candidate early when its measured whole-step ceiling is small or
its representative timing fails to improve. Do not launch another full run
just to promote a kernel percentage. Existing frozen runs remain unmodified.

### Geometry worker isolation and rejected viscosity fusion

A late-run sample found two reference-projection workers whose main threads
were waiting for tasks while their libfabric communication threads consumed
roughly two CPU cores per worker. The spawn executor reimported the case,
which imported the FEM/MPI runtime. This is an execution side effect, not
required geometry work. An apparently paused main process was insufficient
to isolate a benchmark: its child processes remained active. Comparative
windows must verify and pause the complete owned process group.

Replace case-reimporting workers with persistent data-only geometry services.
The service accepts bounded numeric buffers, runs the unchanged Q1 reference
projection, and refuses startup if MPI/PETSc/DOLFINx is loaded. The solver
retains all state ownership. Worker failure/timeout is propagated and owned
children are reaped. Local socket SIGPIPE suppression is necessary because
PETSc can install a fatal process signal handler; do not globally override
the solver's signal policy to hide transport errors.

The selected service change was then tested on an actual accepted nonlinear
state: restore the same hash-checked step-65,000 checkpoint and compiled
material artifacts, then advance steps 65,001–65,200 through the ordinary run
entry. Candidate first, reference second; the competing frozen simulation's
entire process group was stopped and restored in a finally block. No synthetic
geometry warm-up was used in this late-state comparison. Advance excluding
callbacks was **31.6864 s candidate versus 46.1779 s reference**, a **31.38%**
reduction for this measured segment. All saved material arrays, both saved
field frames, and all 20 common saved history stations were exactly equal.
This is not a complete-run speed claim or independent Abaqus validation.

The first attempt correctly refused a freshly recompiled material binary:
source/build identities matched but linked-library hashes differed. The
comparison reused the original verified compiled artifacts instead of
weakening checkpoint identity. Cross-platform fresh simulation and
cross-binary checkpoint migration remain different contracts.

An earlier 200-step screen with deliberately prestarted geometry services
also showed the contention disappearing; its candidate included the later
withdrawn viscosity experiment, so the clean late-state comparison above is
the selection evidence. Direct process observations changed from approximately
176–190% CPU per idle legacy worker to 0% per idle numeric service on this Mac.
Other MPI vendors and hardware may not exhibit the same spin behavior.

The anchored separation option was screened independently on a geometry-only
interpolation between the real step-65,000 and step-65,200 trace configurations.
Forty queries per sequence, forward/reverse policy order: incremental took
4.203/4.148 s; anchored took 4.443/4.334 s. Forces and potential were exact,
but all policies performed the same 311,625 classifications and 4,337 oracle
point recoveries. This path does not benefit from the extra anchor checks.
Keep the reviewed incremental default for this delivery. The interpolation
screen is not a simulated dynamic trajectory or a general verdict against
anchored certificates for oscillating separated surfaces.

The fused determinant/density/length/viscosity experiment passed its focused
numerical tests but failed the first isolated whole-step selection screen:
200-step advance time was 29.897 s for the reference and 30.940 s for the
candidate. This pair does not establish a statistically significant slowdown,
but supplies no reason to add the new path. It was withdrawn from the release
candidate; its private experiment and installed wheel remain for audit.
The independent native-header dependency fix is retained to prevent stale
incremental builds after included C headers change.

References for the process boundary (the measured CPU behavior is local evidence):

- [Python spawn main-module imports](https://docs.python.org/3/library/multiprocessing.html#safe-importing-of-main-module)
- [PETSc signal handler implementation](https://petsc.org/release/src/sys/error/signal.c.html)

### Algorithm sources

- https://docs.lammps.org/stable/neigh_modify.html
- https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-expdynamic.htm
- https://arxiv.org/abs/2201.10475

### Reference catalogue

- [LAMMPS measurement](https://docs.lammps.org/stable/Speed_measure.html): workload,
  hardware, memory and parallel balance affect optimal execution settings.
- [LAMMPS neighbor lists](https://docs.lammps.org/Developer_par_neigh.html):
  displacement-bounded reuse motivates a contact-cache design, not direct
  transfer of particle cutoffs to deforming finite-element surfaces.
- [Abaqus selective subcycling](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-selectivesubcycling.htm):
  small controlling zones can benefit, while shared constraints and contact
  restrict partitioning; it is not a universal switch for forming.
- [MFEM assembly levels](https://mfem.org/howto/assembly_levels/): operator
  execution/storage can vary behind one interface. AgentFEM's current explicit
  body is already matrix-free; that label alone is not a new acceleration.
- [NVIDIA CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/): grouping
  repeated operations reduces launch overhead. A resident CPU execution plan
  is an architectural inference, not a claim that CUDA runs on the current Mac.
- [Abaqus speeding-up methods](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEGSARefMap/simagsa-c-qsimetspeedanal.htm):
  mass and loading-rate scaling require response checks; rate dependence
  prevents treating the two as interchangeable in general.
- [GCC link-time optimization](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html):
  cross-file optimization is an execution experiment; compile/link compatibility
  and numerical acceptance still apply. The local experiment above did not win.
- [Abaqus explicit stability and time incrementation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-expdynamic.htm):
  nonlinear stiffness and damping affect the stable increment; material-update
  admissibility and a fixed requested step are not substitutes for that limit.
