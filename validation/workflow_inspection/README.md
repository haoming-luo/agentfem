# Model inspection maintenance — 2026-11-22

Baseline: `2bbf160f12c16a195ed76c345bae4c08e717c1a3`, PR #198.
Raw paired observations, Python/platform identity and measured source digests:
[`timing.json`](timing.json). Reproduce from an installed numerical environment:

```bash
PYTHONPATH=src python tools/benchmark_model_inspection.py \
  --baseline-ref 2bbf160f12c16a195ed76c345bae4c08e717c1a3 \
  --output /tmp/inspection-timing.json
```

The tool compares recorded baseline owner functions with the candidate in
one process using shared current dependencies. For baseline snapshots it
also selects the baseline checkpoint geometry routine. Five alternating
baseline/candidate pairs follow warm-up on each grid. Mesh/model construction,
Python startup, first JIT compilation, solving and field output are excluded.
Resource readings vary; all other scientific review entries and the entire
settings snapshot must match exactly before timing.

| Cells | Snapshot baseline / candidate | Time reduction | Review baseline / candidate | Time reduction |
| --- | --- | --- | --- | --- |
| 2,048 | 18.96 / 9.01 ms | 52.5% | 25.89 / 18.94 ms | 26.8% |
| 8,192 | 70.04 / 33.42 ms | 52.3% | 25.00 / 18.63 ms | 25.5% |
| 32,768 | 277.20 / 134.39 ms | 51.5% | 29.62 / 23.13 ms | 21.9% |

This is metadata-owner performance on the recorded Mac environment, not a
whole-simulation speedup. The node-row reuse trades temporary Python memory
for fewer repeated conversions: at 32,768 cells, `tracemalloc` peak rises from
15.53 MB to 20.47 MB. These are Python-traced allocations, not process RSS or
native solver memory. Records live only for one snapshot; coordinate edits
are read again on the next call. Batched geometric quantization caps its
temporary coordinate blocks independently of total cell count.

## Changes and verification

- Reuse immutable source-node records across cells and regions within one
  snapshot. The v1 JSON fingerprints and source-label/seam distinctions stay
  unchanged.
- Batch checkpoint coordinate quantization and per-cell sorting. Independent
  legacy references cover 1D/2D/3D, float32/float64, repeated keys, chunk
  boundaries, curved Q1/Q2 meshes and real MPI distribution. Valid float64
  v2 checkpoint geometry identities remain byte-identical.
- Correct the float32 local-extrema/MPI receive-buffer type mismatch in the
  coordinate-key policy; actual float32 meshes now get the intended bounds.
  This does not authorize loading a checkpoint with a mismatched identity.
- Reuse region integration/range evidence within one model review. A shared
  cache-hit decision handles different Python aliasing across MPI ranks;
  returned entries are detached and no cache survives geometry updates.
- Validate one region communication plan and gather region fingerprint
  batches together. One-rank evaluation errors or ownership mismatches reach
  every rank before the next communication phase.
- Share full output kwargs between snapshot and Step in the dynamic and
  nonlinear templates; reject foreign-mesh budget targets; derive comparison
  coverage from recorded opaque/missing inputs as well as producer flags.

No Step numerical algorithm, constitutive update, convergence tolerance or
accepted-state lifecycle is changed. Local/installed/remote acceptance for the
final candidate is recorded on PR #198 rather than attributed to this timing
record's older implementations.
