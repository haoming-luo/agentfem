# Performance evidence

## Bounded constitutive stress conversion measurement (2026-10-09)

The shared finite-strain quadrature driver converts Cauchy to first Piola stress
in NumPy blocks of 1,024 points, replacing a Python loop of individual 3-by-3
inversions. For 20,000 float64 points on the development Apple Silicon host,
five warm repetitions gave median 0.06990 s for the scalar expression and
0.00533 s for the chunked expression (13.1x for **conversion only**).
Inputs used NumPy seed 823, `F = I + normal(0, 0.03)` and random stress arrays;
OpenBLAS/OMP thread counts were one. The maximum absolute difference was
8.88e-16. Empty partitions, invalid determinants and nonfinite input/output
remain explicitly checked. This excludes constitutive updates, assembly, I/O
and the global solve; no whole-solver acceleration factor follows from it.

The private finite-Hex residual also separates array-based rollback snapshots
from JSON-ready durable snapshots. On a 4,096-cell (16x16x16) unit cube with
the finite-strain J2 state schema, five warm snapshots measured median
0.00379 s for list expansion versus 0.000156 s for copied arrays (24.2x for
snapshot creation only). `tracemalloc` peak allocations were 22.54 MB versus
4.36 MB. These are Python-tracked allocations, not process RSS. Real disk
checkpoints originally retained the JSON auxiliary contract. The subsequent
optional numeric-tree encoding below also removes that expansion from durable
serial material snapshots; neither measurement is a whole-solver speedup.

`tools/benchmark_checkpoint_arrays.py --size 16` separately compares durable
auxiliary serialization for 4,096 finite-J2 cells, including copied state,
encoding and atomic file writes, but excluding nodal archives and solving.
JSON uses 14,048,099 bytes; typed numeric arrays plus metadata use 4,365,404 bytes
(about 69% less). Python-traced peak allocations are 79,684,017 versus 8,730,193
bytes (about 89% less); these are not process RSS measurements. Every restored
gradient and material field is compared with its source. This measurement
overlapped a capacity run, so its wall times are not presented as a speedup.
The optional v6 format uses non-executable NumPy arrays and a small JSON tree;
existing JSON auxiliary checkpoints retain v5 and remain readable. This does
not extend serial material history to MPI-portable restart.

`tools/benchmark_finite_hex_trial.py --size 16 --repeats 7` measures a complete
private serial trial (material, geometry admission, force and spectral screen),
excluding mesh preparation, commit, restore and I/O. On the same host with
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`, caching immutable state-layout
sizes and replacing scalar NumPy energy
sum checks with the identical scalar tolerance reduced the warm median from
0.27001 s to 0.22358 s (17.2%). The force norm remained 0.7198301095792459.
Neither material equations nor rejection tolerances changed; boundary tests
compare both sides of the former NumPy tolerance. `--profile` is diagnostic
only and must not be enabled for comparable wall-clock measurements.

These environment settings do not prove that every runtime uses one thread.
A subsequent process sample found the old MPI launcher supervisor consuming
CPU in two libfabric sockets threads while waiting for its child. NumPy build
metadata alone does not establish the library actually loaded at runtime;
the October 10 environment loads conda OpenBLAS through its BLAS/LAPACK dylibs.
Treat the measurements
above as bounded same-environment observations, not certified single-thread
benchmarks. The lightweight installed MPI entry now probes the linked vendor
without initializing MPI in the supervisor; the numerical child retains its
normal runtime and environment. Timeout and interruption still terminate the
owned process group. The transport itself is not silently reconfigured.
The underlying mechanisms are documented in the
[mpi4py initialization controls](https://mpi4py.github.io/mpi4py/stable/html/mpi4py.html)
and [libfabric sockets provider](https://ofiwg.github.io/libfabric/v2.6.0/man/fi_sockets.7.html).
This supervisor saving is not an acceleration factor for the numerical kernel.

## Optional MPI idle-CPU diagnostic (2026-10-10)

`tools/diagnose_mpi_idle.py` measures bounded, isolated single-rank children.
It never initializes MPI in its parent or changes the user's environment:

```bash
python tools/diagnose_mpi_idle.py --compare-provider tcp --output /tmp/mpi-idle.json
```

On this macOS ARM64 host with MPICH 5.0.1 and libfabric 2.5.1, three paired
two-second observations measured 1.963–1.995 equivalent CPU cores while idle
with `FI_PROVIDER` unset, versus less than 0.000036 with `FI_PROVIDER=tcp`.
A process sample attributed two busy background threads to the sockets
connection listener and endpoint connection manager. These are MPI runtime
threads, not material integration or an unbounded Python loop.
Raw observations are archived in
`evidence/hex8/2026-10-10-mpi-idle.json`.

The same installed AgentFEM wheel passed the same 50 selected tests per rank
on two ranks with both the default provider and TCP. The selection covers
finite Hex state/energy, ordinary Step output, checkpoint restore, small-strain
Hex and transient rollback. This is local compatibility evidence, not a
multi-node scaling study or a Windows guarantee. Use a process-local
`FI_PROVIDER=tcp` only after checking the relevant installed MPI runtime;
AgentFEM does not set it automatically. The libfabric project documents the
[TCP provider](https://github.com/ofiwg/libfabric/blob/main/man/fi_tcp.7.md)
and marks the older sockets provider deprecated in its
[provider overview](https://github.com/ofiwg/libfabric).

Reducing idle CPU is useful independently of solve time, but is not a solver
speedup factor. Keep the provider identical on both sides of a performance
comparison. Thread-limit environment variables alone do not prove that MPI
has no background threads.

The serial 32,768-cell, 500-increment affine endurance diagnostic completed
with maximum displacement/stress/reference-energy absolute errors of
4.44e-16 / 7.00e-13 / 6.00e-14. Bounded provider batches reduced observed peak
process RSS from 1,054,834,688 to 724,598,784 bytes (about 31%). The older run
was a dirty development diagnostic; the newer run identifies clean commit
`f5991141`. Both had brief diagnostic overlap, so their 1,090.9 / 1,062.4 s
wall times are not offered as a controlled speedup. The check is an affine
path endurance test, not a spatial convergence or industrial validation result.

A further local trial optimization admits a frozen displacement copy once,
then reuses that same admitted geometry through material and force evaluation.
It does not cache admission across increments or skip standalone geometry
checks. Nine warmed 4,096-cell trials measured 0.21909 s before and 0.19580 s
after (about 10.6% less trial time), with identical force norm. Those runs used
OMP/OpenBLAS/vecLib limits of one and no concurrent numerical workload.
Input-mutation isolation, folded-cell rejection and downstream rollback remain
covered by targeted tests. This is again a trial measurement, not a whole-solve
or cross-software performance claim.

AgentFEM records execution cost as a first-class part of
`SimulationResult`. Performance evidence explains the cost of a computation;
it does not turn a completed solve into a scientifically verified result.

The public result contains a stable `agentfem.performance-evidence` record:

```python
result = model.step(target=displacement).solve_result()

print(result.performance["wall_seconds"])
print(result.performance["workload"]["global_dofs"])
print(result.performance["solver"]["convergence"])
```

The same record is written to `result.json` and travels into scientific
datasets through the ordinary result summary.

`wall_seconds` normally covers the measured public execution call, including
solve and result/output construction. A transient result instead accumulates
its recorded `run()` calls and the final result construction, so checkpointed
or deliberately segmented execution remains visible. Work completed before
the declared boundary is not retroactively counted. In particular, mesh/model
construction and a JIT compilation completed while creating the numerical
problem remain outside the boundary; compilation triggered lazily during a
measured stage remains inside it. Every manifest records the exact boundary so
a warm solve cannot be presented as an end-to-end cold-start measurement.

## What is recorded

The common contract separates four kinds of evidence:

| record | meaning |
| --- | --- |
| `stages` | wall-clock time and call count for solve, result/output, assembly, history, or provider-owned stages |
| `workload` | global degrees of freedom, global cells, dimensions, and finite-element identity |
| `solver` | declared solver policy and convergence/iteration evidence |
| `parallel` | MPI rank count and the timing reduction rule |

Runtime versions, operating system, machine architecture, scalar precision,
PETSc and MPI identity remain in the result's runtime provenance. They are
referenced rather than copied into the performance record, so the two sources
cannot silently disagree.

## MPI interpretation

Timing is measured on every rank. Before publication, AgentFEM reduces each
stage to minimum, mean, and maximum rank time. `seconds` is the maximum: the
parallel critical path that determines elapsed solve time. The spread between
minimum and maximum exposes load imbalance.

Stage timings may be nested and therefore must not be added. For example,
`residual_assembly` may be included inside `run_wall`. Call counts are retained
as a single value when every rank agrees and as a min/max range otherwise.

## Comparing two programs

A defensible performance comparison requires the same physical model and a
declared accuracy target. Record at least:

- equations, material laws, loads, constraints, and time interval;
- mesh, element order, global degrees of freedom, and time increments;
- nonlinear and linear tolerances, algorithms, and preconditioners;
- requested fields, output cadence, and checkpoint policy;
- hardware, rank/thread count, software versions, and timing boundary;
- field norms, quantities of interest, balances, and convergence evidence.

Startup, mesh generation, JIT compilation, solve, output, and verification
should be compared separately. A faster run with different physics or a looser
accuracy target is not an AgentFEM performance claim.

## Contact search measurement

`tools/benchmark_contact_search.py` measures the existing serial triangle BVH
against exhaustive projection on a deterministic planar grid. It checks exact
facet/status identity and tolerance-bounded points, normals, gaps and barycentric
coordinates before timing. Tree construction is reported separately; warmed
query batches alternate execution order and retain individual samples.
Geometry, query and search implementation hashes identify the comparison.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python tools/benchmark_contact_search.py --output /tmp/contact-search.json
```

The 2026-10-09 local 128/512/2048-facet, 16-query audit evaluates only 32 exact
facets per batch in each case, versus exhaustive counts of 2048/8192/32768.
All projections agree. This confirms existing pruning; it is **not** a new
algorithmic improvement, an end-to-end FEM speedup, or distributed scaling
evidence. Timings depend on the machine and are not CI acceptance thresholds.
Curved/ambiguous geometry and MPI correctness remain covered by separate
regressions, not by this planar performance fixture. Profile a representative
full solve before investing in search optimization; do not extrapolate this
microbenchmark to industrial forming. Raw local samples are retained under
`evidence/contact/2026-10-09-search-microbenchmark.json` in the repository.

## Finite-material transport comparison

The 2026-10-09 controlled comparison uses clean source `07ee074b`, macOS ARM64,
Python 3.11.15, DOLFINx 0.11.0, PETSc 3.25.3 and MPICH 5.0.1. Other numerical
jobs were finished first. Each pair alternates execution order after warmup;
raw times, runtime versions and thread environment are retained. The two routes
use the identical native J2 integration and tangent implementation: only the
ordered per-point object protocol versus optional columnar transport differs.

| Fixed workload | Ordered median | Columnar median | Time reduction |
| --- | ---: | ---: | ---: |
| 4,096-cell complete trial, nine measured pairs | 0.17674 s | 0.08102 s | 54.2% |
| 4,096 cells, 200 increments, three measured pairs | 38.8836 s | 17.9949 s | 53.7% |

The first boundary excludes commit, restore and I/O. The second includes time
integration, geometry/stability checks, transactions and scheduled monitoring,
but excludes setup and disk I/O. This is approximately 2.16 times faster for
the measured trajectory, **not** a general application or industrial-forming
speedup. Both routes retain the same equations, tolerances and acceptance checks.
All compared nodal/material fields agree within 2e-12 absolute/relative tolerance.
The affine trajectory remains in the elastic branch of finite-strain J2 and
independently matches displacement, stress and stored energy. It does not
measure plastic-path throughput or validate the private route for public use.

Reproduction tools are `tools/benchmark_material_transport.py` and
`tools/benchmark_finite_hex_transport_run.py`; raw records are
`evidence/hex8/2026-10-09-columnar-trial.json` and
`evidence/hex8/2026-10-09-columnar-trajectory.json`. Do not multiply these factors
by separate checkpoint or geometry microbenchmark factors.

## Scientific trust boundary

`SimulationResult.performance` is operational evidence. It does not modify
`trust_level`, satisfy a verification claim, or establish accuracy. A release
or publication performance claim must pair this record with benchmark and
convergence evidence under a named applicability boundary.
