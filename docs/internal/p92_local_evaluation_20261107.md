# P92 local evaluation reuse: bounded trajectory evidence

## Mechanism

The 32-cell bending/hold profile made 14,336 local material updates and
153,613 rate/Jacobian evaluations. About 7.82 of 9.88 profiled seconds were
inside the material provider. Inclusive times overlap and are not additive.
The profile identifies the work; uninstrumented A/B runs below measure savings.

Residual, Jacobian and final tangent callbacks often request the identical
local Newton candidate. Reuse its joint rate/Jacobian evaluation only inside
one fixed-old-state material update. Keep an owned candidate copy because
MINPACK can reuse buffers. A changed candidate or a new update recomputes;
there is no approximate key, cross-increment cache, timestep change or hidden
material commit. This is the same shared-evaluation opportunity supported by
[SciPy's joint residual/Jacobian contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html).
The implementation retains the existing solver and explicitly checks the
candidate ownership needed by this local reuse.

This improves the repository P92 research provider, not all user materials or
the composite-contact production path. The framework already has joint
stress/tangent and optional batch contracts; no second solver API is needed.

## Comparison

Apple M5 arm64; Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1; one serial process
with BLAS/OpenMP threads limited to one. The independent baseline is the exact
local update from commit `8d77c6887cc6d7c0ceab8b3e3fa817df2e95c9e7`.
Both executions use the same published parameters, geometry, 20 accepted
increments and 4 s ramp plus 60 s hold. A small-mesh warm-up precedes timing.
Timing includes model construction, complete solve and endpoint observations;
it excludes startup, comparison snapshots and visualization-file writing.

Six runs per mesh use the order baseline/candidate/candidate/baseline/baseline/
candidate. All 23 displacement, stress, committed-state and selected-history
arrays compare exactly. The following are medians, not statistical confidence
limits or estimates for a larger industrial job.

| Cells | Baseline (s) | Candidate (s) | Wall-time reduction |
| --- | ---: | ---: | ---: |
| 32 | 6.2861 | 4.1628 | 33.78% |
| 256 | 57.0357 | 37.6905 | 33.92% |

Individual measurements and source hashes are retained in
`evidence/creep/2026-11-07-p92-local-evaluation.json`. The measurements used a
temporary harness; `tools/p92_local_evaluation_benchmark.py` retains its same
comparison, with portable repository discovery and restoration of the selected
provider on exit. Run it manually from a full checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m tools.p92_local_evaluation_benchmark --resolution 2 --output new-timing.json
```

The retained command verifies that AgentFEM is imported from this checkout and
records its commit; it will not silently time an unrelated installed package.
This is source-benchmark provenance, not a requirement for ordinary projects.

Correctness tests cover reused/mutated candidate buffers, fresh update lifetime,
independent local Jacobians, full material tangents, evolving nonuniform
structure and restart: 146 targeted tests passed. The retained benchmark
command also completed its smallest-mesh smoke test. A separate 32-cell
baseline-local-update checkpoint was loaded through the ordinary unchanged
identity checks and continued with the candidate update; all 23 snapshot
arrays exactly match the continuous baseline. This is execution-path
continuation evidence, not a claim about arbitrary historical package formats.

Numerical parity preserves the prior research model;
it is not additional experimental validation or P92 engineering-life admission.
