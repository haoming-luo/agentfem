# Accepted-frame result output

Ordinary analyses continue to use `step.solve_result(output="fields.xdmf")`.
A custom finite-element residual can use the same output coordination without
implementing its own XDMF or VTK writer:

```python
from agentfem import results

result = results.SimulationResult(name="custom_analysis")
with results.output_series(
    result, "fields.xdmf", paraview=True,
    coordinate_name="load_factor", primary_name="U",
    primary_semantic_name="Displacement",
) as series:
    # Inside your solver's accepted-state loop, after convergence/commit:
    series.write_accepted(coordinate=factor, primary=u, fields=(stress, energy))
run.publish(result)
```

`examples/custom_energy_output` is an executable nonlinear-potential example.
It uses P2 displacement, DG0 stress and energy, five accepted load levels and
both output formats. A failed Newton trial must not call `write_accepted`;
the solver owns acceptance, rollback and material-state commitment.

## Roles and semantics

XDMF/HDF5 is the default field dataset. PVD/VTU is an optional display companion.
They do not replace raw integration-point state or checkpoint files. The latter
remain separate existing APIs. DG0 stress is a recovered field, not the full
integration-point history. `inspect_output(path)` checks dependencies and HDF5
shapes, not correctness of the physical model.

Do not label velocity, flux or another vector as displacement. Custom sessions
retain the primary field's own name unless explicitly overridden. Only declared
`Displacement` or `Mode shape` semantics recommend Warp. Two-dimensional vector
storage can have three components for display, without changing physical
component count. Reference geometry is the default; do not apply Warp again to
already-deformed coordinates.

## Serial and MPI

Serial sessions use the existing single-grid compressed XDMF backend. MPI
sessions use DOLFINx collective XDMF and optionally collective VTK. The primary
MPI field is interpolated to the geometry degree for XDMF compatibility; the
record explicitly describes that conversion. It is not a lossless checkpoint
of high-order degrees of freedom. Auxiliary fields must be compatible with the
selected existing backend (the verified combination is Lagrange primary plus
DG0 cell fields). Arbitrary mixed/discontinuous spaces are not promised.

Every rank enters and writes the session. Field schemas, paths and coordinates
must agree across ranks; rank-local invalid values are rejected before I/O.
After the session has closed, the usual project rank-zero publication may run.
This does not promise recovery from a killed MPI process or a failure inside a
third-party collective I/O call.

## Lifecycle and failures

- One session is one fixed-mesh segment. Coordinates must increase strictly.
- Start another file/session for restart or remeshing. Multiple segment artifacts
  are retained; an existing output is not overwritten without `overwrite=True`.
- Field arrays are written immediately. Reusing a live function for the next
  accepted state cannot change previous frames. Temporary field memory is
  per-frame; XML/index metadata still grows with frame count. Existing Step
  solvers may independently retain their own accepted-state histories.
- A `.incomplete` marker remains when a session fails. File existence alone does
  not indicate completion. The marker is removed only after successful field
  finalization and dependency checks.
- Optional display failure preserves a successfully written scientific dataset.
  The result remains `completed_with_output_errors`; strict mode raises, while
  `strict=False` permits a partial result with explicit diagnostics. Scientific
  writer errors always raise because continuing a damaged writer is unsafe.
- `run.publish(result)` checks declared output and preserves the result status.
  Scalar-only results need no XDMF. Publication never silently recomputes a field.
  CLI runs return a nonzero status for partial output failure while retaining
  the published partial result.

Existing low-level writers remain expert interfaces. Existing result fields and
checkpoint APIs are not removed. The standard static field exporter and serial
history exporter now share the new session coordinator; transient output also uses the session while retaining its existing accepted-state
scheduling. Explicit legacy PVD requests remain presentation-only.
