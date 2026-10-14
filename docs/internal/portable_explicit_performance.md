# Portable explicit performance strategy

## Decision

Preserve the engineering Model and correctness obligations, not this machine's
thread counts or this review case's time grid. Separate execution optimization,
numerical-method comparison and hardware selection. The next major target is
the accepted-increment execution path and the number of full-domain updates,
not another isolated arithmetic kernel or output-format rewrite.

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
| Restart | `_step_response_hex.py` embeds the whole execution-kernel summary in material identity | Split model/state compatibility, numerical implementation and operational provenance in a versioned migration; do not strip old checks |
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

## Primary references

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
