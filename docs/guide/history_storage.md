# Complete histories without oversized JSON

`SimulationResult` separates scientific histories from their file encoding.
The existing default embeds history values in the project result JSON. For
long paths, opt into binary history storage without changing the samples:

```python
from agentfem import project, results

policy = project.RunPolicy(history_storage="npz")
report = project.execute_step(step, context=project.current_run(), policy=policy)
```

For an already constructed result:

```python
project.current_run().publish(result, history_storage="npz")
# Or without a project:
result.write_manifest("result.json", include_histories=True, history_storage="npz")
```

This creates one immutable `result-histories-<identity>.npz` companion and a
compact JSON inventory. Keep them together when moving a run. Arrays are
uncompressed binary64; identical axes share one stored array, while each
history retains its own units, axis semantics, shape and description. No
samples are dropped and no time values are rounded. This is a storage choice,
not an integration policy or a change to the numerical result.

The JSON is the publication commit point. A failed replacement leaves the
previous JSON and its previous companion readable. A failed publication can
leave an unreferenced new companion; do not delete old artifacts while a
reader or another saved manifest may still use them. This contract addresses
process interruption, not a filesystem or hardware durability guarantee.

Read histories without loading a finite-element runtime:

```python
from agentfem import results

histories = {item.name: item for item in results.read_histories("result.json")}
time = histories["strain_energy"].abscissa
energy = histories["strain_energy"].values
```

The reader accepts both inline and binary histories. It checks the manifest
content against its seal when present, and the companion SHA-256, member
inventory, shapes, finite values and strictly increasing
coordinates; it never unpickles objects or resamples data. Missing/corrupt
companions and metadata-only histories fail explicitly. A checksum detects
content changes; it does not authenticate the author or validate the physics.
Reading histories does not scan unrelated field artifacts: use
`provenance.verify_manifest` when the entire artifact inventory must be checked.

`agentfem compare ... --history NAME` reads either encoding. The in-memory
`results.compare_records` remains a pure record comparison: callers wishing
to compare binary histories must explicitly load their values first. Loading
does not rewrite a sealed manifest or promote its scientific trust level.

Collective `write_manifest(..., comm=comm)` requires matching full histories
on every rank; array fingerprints participate in that check. It is not a
gather operation for unrelated rank-local curves. Project `execute_step`
retains its existing serial scope.

Field frames (`results.xdmf` and companions), live accepted-history journals,
and restart checkpoints retain their separate responsibilities. Binary
histories neither replace restart state nor reconstruct unsaved field frames.
This option reduces history publication/storage cost; it is not a claimed
speedup of the contact or material solver.

The public `tools/benchmark_history_storage.py` records environment and source
fingerprints, warms both encodings, then compares an alternating four-trial
sequence. On the recorded synthetic 20-channel, 100,001-sample case, total
history/manifest size decreased from 99,469,125 to 16,828,163 bytes (83.1%).
Mean publication time was 1.968 s versus 0.093 s, and reading 1.041 s versus
0.018 s. Every binary64 sample round-tripped exactly. These are local storage
measurements, not whole-simulation speedups or other-platform guarantees;
see `evidence/performance/2026-10-22-history-storage.json` in the repository.

Implementation references: [NumPy NPZ](https://numpy.org/doc/stable/reference/generated/numpy.savez.html)
defines an uncompressed archive of named NPY arrays; the reader uses
[`allow_pickle=False`](https://numpy.org/doc/stable/reference/generated/numpy.load.html).
Result exports use the existing
[module attribute loading mechanism](https://peps.python.org/pep-0562/)
so data-only readers need not initialize the numerical backend.
