# Four-hour performance sprint — 2026-10-14

## Baseline and decision

Start from `a4e411d8`; its Linux and macOS installed acceptance passed in
run 38422430243. The measured prior complete core run is 15,465.05 seconds
(100,000 increments), compared with 13,529.51 seconds for the earlier private
implementation. Keep both frozen; no scientific input or checkpoint identity
will be changed to manufacture a speedup.

The previous diagnostic attributed 2.08 of 16.84 profiled seconds to repeated
finite-array scans. The native scan already uses a branchless binary64
reduction, so blindly adding SIMD is not the first implementation. Investigate
redundant traversals and ownership, then measure the full accepted increment.

## Design under test

Use an internal immutable, validated binary64 buffer as a transport owner.
Admission freezes external writable storage, validates the resulting bytes
once, and then allows contiguous whole-word subranges to reuse that evidence.
No global pointer cache, writable ndarray alias, material-specific state size,
or additional authoritative solver state is introduced. Shape/type checks,
stress symmetry, positive determinants, material limits and scientific
acceptance remain separate checks. Metadata remains fresh for every consumer.
Raw immutable bytes are not automatically trusted. Offset reinterpretation,
non-native dtype and writable memoryviews must not inherit admission.

If the resulting integrated benefit is small, remove or narrow the change;
correctness alone does not justify a new internal abstraction. Next investigate
the shared body/viscosity passes only if their measured cost warrants it.

## Gates and limits

1. Native/Python admission equivalence, invalid input, slices, dtype/metadata
   mutation, lifecycle and rollback tests.
2. Installed baseline/candidate comparisons on identical early, damaged and
   late accepted states, plus sequential whole-Step segments with exact array
   and history comparison. Profile runs are not timing runs.
3. Freeze the selected candidate, document actual performance scope, run
   affected installed release gates and keep the readable project unchanged
   except for its explicitly selected runtime identity.

Never claim a new complete-run time from replay or extrapolation. A complete
run is separate from verification of already frozen final states. Keep the
source-law, contact and numerical-inertia limitations visible.

## Primary references

- [LLVM vectorization and reduction constraints](https://llvm.org/docs/Vectorizers.html)
- [Python buffer ownership and read-only export](https://docs.python.org/3/c-api/buffer.html)
- [NumPy frombuffer views and external ownership](https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html)

These references support the implementation strategy, not a speedup claim.

## First decision: do not ship the finite-owner experiment

The installed candidate passed 189 targeted tests, including byte-order and
immutable-lifetime checks. However, the isolated ABBA 100→600-increment spans
were 85.524/86.524 seconds for the frozen baseline and 84.897/85.689 seconds
for the candidate: only 0.85% less in the mean, smaller than the between-run
variation. This does not justify a new native storage type. Its patch and
measurements remain in the private sprint evidence, but the implementation was
removed from the production diff. Earlier overlapping runs are excluded.

## Second design: instance-owned sleeping worker executor

The compiled body route currently creates and joins material/auxiliary threads
for every increment. Reuse sleeping native workers inside one kernel instance,
with explicit close/context-manager support and an automatic lifetime backstop.
The executor owns scheduling only, never accepted/trial state or retained
numerical buffers. Every batch joins before repair, publication or error;
concurrent/reentrant calls and close-while-busy are rejected. Fork inheritance
is rejected before accessing inherited locks. Keep the per-call implementation
as a reference. Single-worker routes must not start an idle worker.

Hardware inspection: 10 physical/logical cores, comprising 4 performance and
6 efficiency cores, 32 GiB RAM. Prior worker counts are therefore hypotheses,
not universal defaults. Recheck budgets on the changed execution route. A
process-local Python scheduling-interval experiment is separate from core
semantics and will not silently alter the user's interpreter.

## Third design: preserve contact sparsity to the residual boundary

Contact traces already own fixed interpolation and node identities. Precompute
the sorted unique nonzero-support nodes and the contribution-to-row map once.
An opt-in indexed assembly stores only those rows, retains the logical full-node
count, and exposes explicit dense materialization. Normal/friction contributions
are combined in their original order before scatter. No force threshold, search
frequency, quadrature, contact scope or physics parameter changes.

The serial DOLFINx adapter consumes indexed rows; MPI retains the existing dense
local-plus-ghost assembly. Randomized 2D/3D tests include empty traces, negative
interpolation coefficients, duplicate contributions and friction. They compare
the materialized dense field and all force/moment/energy evidence exactly.

## Experiments not selected

- Four material workers regressed the 100→600 span to 86.387 seconds. Keep eight
  for this machine/case, not as the library default.
- Dynamic 4096-cell work distribution completed 600 steps in 118.093 seconds,
  versus 118.091 seconds for fixed slices with the same compact assembly. The
  extra scheduling contract/atomic work distribution was removed; it did not
  establish a useful improvement.
- A 1 ms Python switch interval did not establish a decision-changing benefit.
  No interpreter-wide setting is changed by AgentFEM.

The first compact candidate passed the offline 600-step checkpoint comparison
against the frozen baseline (nodal fields, material history/damage, accepted
history and contact evidence). This is implementation equivalence, not an
independent Abaqus or physical validation. Timings and later-state comparisons
will be recorded separately from this design decision.

## Fourth experiment: fused volume-equivalent viscosity policy — not selected

Shared the existing guarded six-product determinant calculation and evaluated
the density/length policy inside native cell workers, avoiding expanded
coefficient arrays. The experiment passed 131 focused installed tests and
three late body replays were bitwise equal. However, the fresh 600-step check
failed its predeclared velocity/acceleration equivalence tolerance (maximum
absolute differences 6.66e-8 and 0.0342 respectively); damage, contact and work
checks passed. The integrated elapsed time, 118.123 seconds, did not improve on
the compact candidate. Do not relax the gate to promote this optimization.
It is removed from the production diff, with the experimental wheel, patch
and failed comparison retained privately. It has no effect on delivery runs.

## Execution scheduling observation

Turning off contact/body overlap through the ordinary `contact_execution`
option took 148.643 seconds over 600 fresh increments on the experimental
runtime, compared with 118.123 seconds with overlap. Keep overlap for this
case; the default serial mode and its failure semantics remain available.
This comparison changes execution order only, not physics. It does not promote
the rejected experimental runtime.

## Selected candidate evidence before full-run freeze

- The last complete-run runtime remains the 15,465.05-second reference.
- A fresh current-HEAD baseline took 123.051 seconds through 600 increments;
  the 100→600 accepted span was 86.036 seconds.
- Two selected pool/compact runs took 118.091 and 115.990 seconds; corresponding
  accepted spans were 81.678 and 80.525 seconds. This is about 5.7% less mean
  steady-span time than the current-HEAD sample, not a completed-run claim.
- At accepted steps 20,000, 60,000 and 90,000, all 12 body replay arrays are
  bitwise identical. Body-only times changed by -0.16%, -0.07% and +0.40%:
  worker reuse alone did not establish a meaningful late-stage body speedup.
- Retain compact contact assembly and bounded worker ownership together;
  do not attribute their combined fresh-Step difference to faster constitutive
  mathematics. The selected path changes no material law or numerical inputs.

The next gate is the exact installed source-archive candidate, followed by a
fresh frozen project. Any run still in progress must be reported as such;
neither the rejected experiments nor the late replay can stand in for it.

## Installed acceptance

The selected source archive built a macOS arm64 wheel with all three native
modules. Wheel SHA-256:
`6d3766f6be8b27b84793937c39d2b3469f91506b2ee88866fbe78344ae7d9414`.
With checkout-path injection disabled, 238 targeted tests passed; eight
two-rank-only checks were skipped in serial. The subsequent bounded two-rank
contact suite passed nine tests on each rank. Ordinary installed project
smokes passed for static solid, steady heat and structural dynamics.

The first platform-status command correctly refused promotion from the dirty
development worktree. Pre-existing generated `site/` files are preserved.
Clean-commit/source identity is a separate release check; the successful
numerical smoke must not be reported as that platform gate passing.

After committing the selected code as `608c50bb`, the same wheel passed the
complete macOS release gate from a clean detached checkout (`source_dirty=false`).
The original generated files were not deleted. The public standing-wave example
also completed 640 increments, with relative displacement error 0.000709679
and relative energy-balance residual 1.93266e-6 in its small-amplitude scope.

The new frozen private project keeps the preceding readable `case.py`, execution
settings, mesh, inertia and source law unchanged. Its sole provider edit closes
the core-owned executor explicitly. All 631 frozen files are hash-checked before
launch. The fresh 600-step project/export check took 120.982 s and passed the
checkpoint comparison. Actual ParaView 6.1.1 read one grid, 296,074 points and
255,041 three-dimensional Hex8 cells with complete displacement and zero Warp
error. A new full 100,000-step process started at approximately 18:22 UTC on
October 13. It runs against that frozen installation; subsequent documentation
edits do not change its runtime. No second numerical job will overlap it.

The first 5,000-step full-run checkpoint passed 156 compared entries against
the earlier completed core run (150 numeric arrays/histories bitwise equal;
the remaining entries are scalar/string evidence). This is progress evidence,
not completion. The performance commit also passed remote Linux native
packaging, FEniCSx and macOS installed-wheel checks in run 38445551355.

The subsequent result-contract audit reproduced eight failing acceptance tests:
empty arrays were verified vacuously, array broadcasting hid layout differences,
and complex NumPy inputs lost their imaginary component. The comparison helper
now requires real, nonempty observations and matching layouts, with an explicit
scalar constant-reference exception. It rejects masked input requiring a common
selection and preserves immutable compared evidence when callers reuse buffers.
Thirty-seven targeted verification/result/golden checks passed, without a
second finite-element solve. This separate acceptance-layer change does not
alter the frozen full-run runtime, equations or comparison tolerances.
