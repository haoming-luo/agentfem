# History-material output control — 2026-10-31

## Decision and ownership

Keep one `model.step` contract, one accepted-state owner and the existing live
XDMF writer. Add the established `output_every` spelling to the generic finite-
strain history Step rather than a project callback or a second output scheduler.
The constructor accepts positive integers (default 1), or `None` for segment
endpoints only. The property is read-only. Booleans, floats and nonpositive
counts are rejected; integers are not obtained by truncating user input.

Count the complete accepted ledger, including restored increments. Rejected
attempts and Newton iterations never count. Force the requested pause/completion
endpoint and an already-accepted continuation boundary, including in-memory
continuation. Do not fabricate an initial constitutive response. Sampling does
not insert time knots, change physical time, thin scalar evidence or modify
checkpoint retention. A newly constructed continuation may change output cadence
without changing scientific restart identity. The final result retains full fields.

Unsampled increments skip both cell recovery and writer invocation. Published
frames still contain one mesh with all supported nodal/cell fields, in reference
geometry, with closed immutable HDF5 files and relative references. Unsaved past
fields remain unavailable without recomputation. Exact prescribed output times,
generic `OutputPlan`, MPI and broader material admission are not added here.

## External precedent, not borrowed validation

The [Abaqus OUTPUT reference](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-output.htm)
distinguishes increment-frequency output from exact time marks; exact marks may
adjust integration increments. That distinction matters for rate-dependent laws.
Our bounded policy deliberately samples existing accepted stations and preserves
the numerical path. This is not a promise of identical Abaqus option semantics.

## Verification and measurement plan

- Two local history laws: identical accepted/attempted histories, displacement,
  state and reaction with dense/sparse/end-only publication.
- Adaptive material rejection: only accepted increments count.
- Pause and same-layout restart: global count, shared-boundary publication,
  changed cadence, zero-copy join and endpoint-only recovery.
- Nonuniform plane stress: all common saved field arrays agree with dense output,
  including effective three-dimensional deformation gradient.
- Real VTK reading and Warp: one grid, unchanged cell count and preserved stress
  and history fields in 2D and 3D.
- Forced write failure: previous frame readable, accepted checkpoint recoverable,
  no numerical retry induced by an I/O error.
- Ordinary project CLI: example option, required-API preflight, manifest and
  output count agree; no user solver or output loop.

`tools/benchmark_history_output_cadence.py` compares dense and sparse output on
the nonuniform educational plate with both laws. It warms each path, alternates
order, records individual total/recovery/write durations, frame counts, bytes and
runtime/source identity, and checks exact same-host state/force/accepted ledgers.
Temporary field data are cleaned; the small JSON report is retained. Remaining
time includes equilibrium **and** other lifecycle/result work, not just a pure
solver timer. Construction, cold JIT, checkpoints and manifest writes are excluded.
Timing with output instrumentation is diagnostic; it does not establish full-job
or composite speedup. `--no-profile` supplies a separate confirmation without
per-call timers. `--compare-preparation` alternates fresh expression preparation
and reuse at each output cadence, so a same-frame comparison is not conflated
with requesting fewer frames.

## Equivalent preparation reuse

The first profile isolated a repeated preparation cost, not expensive geometry
evaluation: 315 `owned_physical_weights` calls spent 1.231 s inside expression
construction and 0.004 s inside expression evaluation (instrumented diagnostic,
including warm paths). This motivated one small reusable optimization in
`QuadratureField`, not a new output backend.

On serial meshes only, keep the prepared geometric Jacobian expression keyed by
mesh object and a detached copy of quadrature points. Evaluate against **current**
coordinates on every call and apply current reference weights/multipliers. A
changed point set or mesh object rebuilds preparation. Neither physical weights,
source values nor recovered output fields are cached. Previous returned fields
remain independent. MPI retains its previous collective preparation route;
rank-local cache invalidation must not skip a collective JIT call on another rank.

Analytical distorted Q1, curved quadratic triangle, embedded surface and moving
geometry cases test the actual measure, not merely agreement with old code.
Additional checks cover point/domain/reference-weight changes and detached outputs.

## Evidence status

The final affected serial regression passed **513 tests**, with three explicitly
two-rank tests skipped, in 81.94 s in the real FEniCSx environment. Those three
tests were then run separately through `agentfem mpi-run -n 2 --timeout 120`:
**3 passed on each rank** in 0.33 s. They cover partitioned physical recovery,
an empty rank and collective bad-measure rejection, not distributed admission of
the generic history-material Step. Correctness lint, generated documentation and
58 scientific function cards validate.

The expanded regression initially found an old in-memory continuation assertion
that expected no starting frame (and two rather than three closed frame files).
Both expectations were updated to the deliberate shared-boundary contract; no
numerical tolerance was relaxed. Fresh runs retain default every-increment
output. The newly included in-memory boundary is an intentional compatibility
change, documented and covered by the continuation-view tests.

An instrumented alternating comparison, with exactly 16 output frames in both
variants, measured mean short-path times of 1.4613 → 1.2526 s for the educational
viscoelastic law and 1.3423 → 1.1258 s for damage. Mean recovery time was
0.2320 → 0.0286 s and 0.2283 → 0.0248 s respectively. These are diagnostic
measurements, not a composite speed claim or a release-wide percentage.
The [uninstrumented confirmation](../../evidence/history_output_cadence/serial_confirmation.json)
contains three alternating observations per combination and records source hashes,
runtime identity and all individual timings. With the same 16 frames, mean total
times were **1.4613 → 1.2633 s (13.5% less)** for viscoelasticity and
**1.3662 → 1.1512 s (15.7% less)** for damage. Individual ranges were
1.4557–1.4673 vs 1.2492–1.2732 s, and 1.3330–1.3907 vs 1.1209–1.1842 s.
These are small, warm, 8×8 educational plate paths on this Mac, excluding
construction, cold JIT, checkpoints and final manifest publication. They are
not platform-independent confidence bounds or full-job performance guarantees.

The optional four-frame policy plus reuse measured 1.2072 and 1.1372 s, about
17% less than the corresponding dense/rebuilt baseline. That combined comparison
requests fewer visualization samples; it is **not** equivalent output. Output
sizes fall from 603,059 to 162,457 bytes and 572,200 to 155,328 bytes, roughly
73% less. Geometry is shared, so total bytes do not scale exactly with frame count.
All measured variants retain identical accepted ledgers, final displacement,
material state and reaction on this host. Independent field/geometry tests provide
the correctness gate, not a timing threshold. With already-sparse output the
remaining preparation benefit is smaller and noisier; no additional gain is promised.

No release tag, installation package or complete composite simulation has been
produced for this slice. The upstream material/checkpoint PR remains a separate
CI and merge boundary.
