# Performance evidence

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

## Scientific trust boundary

`SimulationResult.performance` is operational evidence. It does not modify
`trust_level`, satisfy a verification claim, or establish accuracy. A release
or publication performance claim must pair this record with benchmark and
convergence evidence under a named applicability boundary.
