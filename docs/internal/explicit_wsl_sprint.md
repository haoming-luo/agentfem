# Explicit execution and WSL usability sprint

Baseline: main 5370a6e3 (PR109 merged). No physical controls change.

## Ordered delivery gates

1. One visualization entrypoint: time series includes the final station;
   optional raw endpoint archive must not write a second XDMF/HDF5 pair.
   Reuse checked cell transport across many fields and frames without stale
   mesh/key mappings or mutable published snapshots.
2. Honest execution timing: separate initial frames/restart, observer output,
   integration (including scheduled checkpoints), terminal checkpoint/export,
   and publication. No micro-timing represented as complete solver speed.
3. WSL advisories in ordinary check/run: detect actual mounted Windows storage,
   report CPU/memory/thread environment, no automatic relocation, resource
   changes or override of workspace custody protection.
4. Profile the current installed body/contact route before selecting the next
   arithmetic optimization. Prior private-path profiles are context, not
   evidence that today's bottleneck is unchanged. Preserve fallback, ambiguity,
   damage, work and transaction semantics; compare identical accepted states.
5. Installed candidate tests, independent volume-reader checks and private
   full-grid comparison. Full-run speed is only established by a full run.

## Reference rationale

- https://learn.microsoft.com/en-us/windows/wsl/filesystems: Linux-native
  storage for Linux I/O; preserve backup/custody warning for distro removal.
- https://docs.h5py.org/en/stable/high/dataset.html: chunk/filter cost matters;
  never introduce lossy compression to improve timings.
- https://docs.lammps.org/stable/Speed_measure.html: phase measurements and
  neighbor rebuild/skin tradeoff. Such a search cache is only admissible with
  a conservative motion bound and explicit invalidation, not a fixed skip.

## Boundaries

No Windows hardware is attached. Mocked WSL diagnostics plus Linux CI are not
Windows end-to-end performance acceptance. No private mesh/law is committed.
No automation or background promise substitutes for active development time.

## First candidate evidence (2026-10-15)

- Installed candidate 2: 147 targeted tests passed; four two-rank checks skipped
  in serial and then exercised in one bounded MPI run (21 passed per rank).
  Pytest's repository `pythonpath=src` override was explicitly disabled for the
  installed gates. The earlier source-path gate is not installed-package proof.
- Native worker validation rejects nonfinite stress/history/work, asymmetric
  stress, unwritten output and force overflow before publication. Local repair
  and immutable old-state checks remain covered. No physics controls changed.
- Batched P1/Q1 contact preparation retains owner-local node matching, the same
  integration rules and point identities. Warped/high-order/empty-rank tests
  pass; higher-order evaluation remains on the existing path.
- Private full-grid 600-step comparison: baseline 127.392 s; worker-validation
  and output candidate 124.925 s; additional batched preparation 119.220 s.
  These are one short run each, including setup/export, **not a full-run speedup**.
  Preparation was approximately 18.2, 18.4 and 13.4 s respectively; advancing
  600 increments remained approximately 104–105 s including initial fields.
  Do not attribute the startup saving to every increment or extrapolate it.
- All archived cell keys, material states, Cauchy stresses, gradients and times
  at increment 600 were bitwise identical across these three runs.
- One visualization series plus a raw endpoint archive is tested. Native
  worker changes have a new execution revision; no checkpoint identity bypass.

Next gates: independent ParaView check of the single-series project; frozen
full-run comparison and late-state investigation. A large solver throughput
improvement has **not** yet been demonstrated. WSL hardware timing is still
unavailable; the mount/resource diagnostics are covered by fixture tests only.

## Opt-in candidates and second admission gate

- Dynamically queued material/body tiles reuse the instance-owned native pool.
  They change scheduling only; pointwise providers must be invariant to batch
  partitioning. The default remains static slices. Exact local repair, joined
  validation and immutable old state remain mandatory. No throughput benefit
  is established yet; small tiles can increase dispatch cost.
- Anchored closed-surface separation certificates avoid accumulating obsolete
  oscillatory travel. Both endpoints of the current linear increment are
  bounded against immutable accepted geometry; positive bounds certify the
  whole segment. Forces, normals and projections are never cached by this
  mechanism. Exhaustion, rollback and restart discard the derived cache.
  This is opt-in and is not a finite search-skip interval.
- Projects can explicitly require native accelerators at preflight. Ordinary
  projects do not acquire this requirement. Paused runs show accepted progress
  and their primary visualization rather than implying scientific completion.
- Candidate 6 installed-package gate: **226 passed** (native scheduling,
  contact bounds/lifecycle, response material step, project/CLI/environment).
  The earlier real-Fortran candidate gate passed **129 tests**. No WSL machine
  or full-run speedup is certified by these results.
- PR110 Linux source CI: 3597 passed, 270 skipped, one stale test failed because
  it patched the replaced scalar vertex matcher. The test now checks the batch
  distance candidate dimension (eight owner-cell vertices), preserving its
  locality assertion. Focused response/trace gate: 34 passed, three MPI skips.
- The frozen full run is still candidate 3; independent builds and small tests
  overlapped it. Its wall time must retain this interference qualification.

Further rationale: OpenMP scheduling and oneTBB automatic chunking document
the balance/overhead tradeoff; LAMMPS neighbor rebuilding motivates conservative
motion-based invalidation, not transfer of a particle cutoff rule to Q1 faces.

- https://openmp.org/spec-html/5.0/openmpsu41.html
- https://uxlfoundation.github.io/oneTBB/main/tbb_userguide/Automatic_Chunking.html
- https://docs.lammps.org/Developer_par_neigh.html

## Storage and next arithmetic probe

Transient manifests now use compact JSON, preserving the schema, every history
row and standard readers. The retained private final manifest is 107,175,430
bytes; a compact-JSON inspection is approximately 88,342,412 bytes. This is a
storage observation, not a solver speed claim. Recovery/retention/contact-work
gate: 57 passed. No old archive was rewritten or deleted.

A private inverse-Newton 3x3 polar candidate retains the existing near-rotation
admission and residual gates. In 300 rotated/stretched random points, maximum
strain error against the independent reference was 2.43e-15; all 300 points
passed native admission. This is not a material-history, damage or dynamics
acceptance and no timing gain is established. Keep it outside the package
until comparative arithmetic and material-history evidence justifies promotion.
The existing runtime and frozen full run are unchanged.

- https://eprints.maths.manchester.ac.uk/694/
- https://pages.cs.wisc.edu/~sifakis/papers/SVD_TR1690.pdf

The latter motivates SIMD-friendly small-matrix algorithms; its graphics/SIMD
performance is not transferable evidence for this double-precision workload.

## Geometry-service admission (2026-10-15)

The next installed candidate passes **122 targeted tests**, including malformed
transport, terminated children, bounded timeout, ordered chunking, unchanged
parent environment and whole accepted-increment rollback in serial/overlapped
contact execution. `geometry_timeout` is an optional operational deadline, not
a hardware qualification or scientific checkpoint identity. A timed-out child
cannot publish a partial contact update. The candidate source archive and its
installed native wheel were checked separately; no private material or mesh
is included in the software archive.

The selected same-checkpoint 200-step late comparison reduced advance time
from 46.178 to 31.686 s with exact saved fields, material and histories. This is
not a full-run speed claim. Extended late-state validation and the original
frozen full-run gate remain separate. Details and rejected alternatives are in
[portable explicit performance](portable_explicit_performance.md).

The process-launch audit also inspected Campaign's independent-case workers.
Those workers actually solve separate FEM problems and need their own runtime;
they are not data-only contact services. Their deliberate safe-spawn policy
was not replaced by unsafe post-MPI `fork`, nor by this specialized protocol.
The optimization follows ownership and workload, not a blanket ban on MPI or
multiprocessing throughout the software.

## Critical-path audit and parallel winding

After isolating the data-only geometry services, a 120-accepted-increment
installed replay at 65000 measured body assembly at 104 ms/call and contact
at 123 ms/call. These overlap: neither the two times nor their percentages
may be added. A separate instrumented profile located 38 ms/contact call in
serial two-ray classification, 26 ms in refitting/preparation and 44 ms in
closest-point projection/recovery. Profiling overhead is not speed evidence.

The existing `projection_workers` budget now also partitions independent
winding queries. Each query retains the same ordered rays, ambiguity rules,
native fallback and output position. Per-call C workspaces are local; shared
tree buffers are read-only. Failed native slices fall back as whole slices;
workers join before any exception or output publication. No new global thread
environment, scientific option or checkpoint identity was introduced.

Installed comparison, alternating serial/parallel/serial/parallel (120 accepted
increments, same checkpoint/physics/output): solver-loop times were
18.003 / 16.799 / 18.702 / 16.700 seconds. Pairwise reductions are 6.7% and
10.7%. All 65 saved arrays in the final serial/parallel comparison were exact.
The selected installed candidate passes 57 targeted tests, including disjoint
query ordering, failed-slice fallback, geometry-service failure and contact
rollback. This is a short-run benefit, not a three-hour full-run acceptance.

A separate outer-box ray-skip experiment certified only nine of about 924,000
classifications in this replay and did not improve timing. It was withdrawn;
the existing operator already excludes far exterior candidates.

The preceding geometry-service candidate's 5000-increment replay also has exact
saved final arrays. Its 500 scalar history stations differ only in contact
potential energy, by at most 2.220446049250313e-16 (reference maximum magnitude
0.7057051180628224). The strict exact-history gate therefore did **not** pass;
do not rewrite that finding as bitwise history equality or independent physical
validation. The earlier full run was stopped on user request at increment
73620 with a successful accepted-failure checkpoint; it is not running.

The three-hour target requires less than 108 ms/accepted increment for 100,000
increments before allowing setup/export. The new short samples are around
140 ms/increment. Both the body critical path and remaining contact work must
improve; optimizing reporting alone cannot close the gap.

External algorithm review: Abaqus documents local contact tracking between
global searches, while LAMMPS uses displacement-bounded neighbor-list reuse.
These motivate conservative temporal certificates, not blindly copying an
every-100-increments search schedule. Our current broad phase already refits a
BVH and reuses proven exterior separation. The unresolved high-value question
is reuse of **interior classification evidence**, with a proved boundary-motion
bound, without caching forces or ignoring damage/contact changes. It needs
explicit invalidation on exhausted clearance, failed trial and restart.

- https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-expcontactpairform.htm
- https://docs.lammps.org/Developer_par_neigh.html
