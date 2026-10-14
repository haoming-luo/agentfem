<!-- SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fast-path migration without performance regression

Use this reference when moving proven private execution into generic AgentFEM
interfaces or investigating a slowdown after that move. API parity is not
execution parity. Do not copy confidential geometry or material laws into core.

## Compare the effective routes

Inventory both frozen versions before optimizing another arithmetic kernel:

| Boundary | Inspect |
| --- | --- |
| Kinematics | Nodal gather, cell batches, geometry admission, fallback rate |
| Material | Actual worker budget, local repair, repeated validation/loading |
| Force | Stress transforms, viscosity, hourglass, scatter, temporary arrays |
| Contact | Search, projection, inside classification, rebuild policy, workers |
| State | Accepted owner, trial copies, callback visibility, commit/rollback |
| Output | Append versus rewrite, checkpoint cadence, exports, shutdown |

Record runtime identity and the capabilities actually selected. An option in a
manifest does not prove that the installed wheel executes that fast path.
Confirm optional native sources/headers survive sdist-to-wheel packaging.

## Keep equivalent computation equivalent

- Fuse existing admissibility checks with the cell pass when profiling supports
  it. Share the same implementation and thresholds with the reference screen;
  retain the exact fallback for unresolved cells. Skipping validation is not
  optimization.
- Prefer disjoint worker slices and immutable published outputs. Join all
  workers, including on failure, before releasing buffers or publishing state.
  Do not expose a solver's writable live arrays to an unrestricted callback.
- Separate material, auxiliary and contact worker budgets where their costs
  differ. Inspect physical cores, heterogeneous core layout and memory pressure;
  do not transplant a successful thread count into a global default.
- Count gathers, copies, allocations and fallback calls over the whole
  kinematics-to-force pipeline. A faster material kernel alone may not improve
  the Step. Avoid a second state owner solely to enable a fast path.

## Select measurements for the decision

These measurements answer different questions; they are not both mandatory
for every edit. Reuse a trustworthy baseline and stage fixtures when scientific
inputs and the relevant implementation are unchanged. A documentation-only
edit does not invalidate numerical evidence; record exactly what was tested.

For a local comparison, run the same problem sequentially in alternating order
(for example baseline, candidate, baseline, candidate, or reverse order). Keep warm-up and
measurement stations explicit. Compare equal accepted increments and report
both elapsed process time and the chosen steady segment. Do not select only
the fastest trial. Extend or reverse the sequence if variation is comparable
to the claimed improvement. Keep competing numerical work out of a comparative
timing window; disclose unavoidable background interference. Do not require
this full sequence merely to discard an obviously slower or incorrect idea.

Separately replay immutable accepted states from early, nonlinear/damaged and
late stages when investigating equivalent execution. Reuse the same displacement, velocity, material history and step
size for both implementations. Include gather, validation, constitutive update,
viscosity and scatter in a body-pipeline measurement. Such a replay does not
measure contact search, state publication or an entire accepted Step. Do not
bypass restart identity checks merely to load another implementation: use a
read-only, hash-checked replay harness instead.

Compare every relevant output, including stress, history, work, damage and
contact evidence. Unchanged operation ordering can justify bitwise parity;
otherwise state justified numerical tolerances and test sensitivity. Never
relax tolerances simply to pass a faster implementation.

This execution-parity route is not the admission rule for a new numerical
method. Adaptive stepping, spatial adaptivity or an alternative Procedure must
be evaluated at a common accuracy target and physical stations; inherited
identical-step or byte-equality gates can wrongly reject a valid faster method.
Timing evidence guides selection on the measured hardware, not correctness on
another machine. Keep one-case counters and reference tolerances in that case.

## Promote only supported conclusions

Keep a compact evidence record with input/runtime hashes, selected workers,
measurement stations, repeats, elapsed times, fallback counts where available,
scientific comparisons and the measurement's exclusions.

Report time reduction as `1 - candidate_time / baseline_time`; report speedup
as `baseline_time / candidate_time`. They are different quantities. A pipeline
accounting for fraction `f` of runtime and accelerated by factor `s` gives only
an idealized total factor `1 / ((1-f) + f/s)`; verify it in the real Step.

An early short segment cannot establish full-run time or late contact cost.
After equivalent-physics and relevant failure-path checks pass, freeze the installed
candidate and measure a complete run with startup, export and worker shutdown.
Retain the prior reference, including regressions. Treat mass scaling, larger
increments, contact simplification or less output as separate experiments,
not as evidence for an unchanged-physics implementation speedup.

Once the delivery gate passes, stop optimizing for this release unless a known
defect or agreed runtime target requires another change. Record the next
bottleneck separately. More benchmarks and abstraction layers are not a product
outcome by themselves.
