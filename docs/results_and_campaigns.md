# Results, Batch Simulation, and Learning Data

## One Result Contract

`results.SimulationResult` is the scientific result of one analysis. It can
hold:

- scalar or fixed-shape quantities of interest;
- live finite-element fields and field-artifact links;
- time/load/iteration histories;
- solver and model metadata;
- visualization, checkpoint, and report artifacts.
- a scientific verification report and explicit trust level.

XDMF, CSV, and NumPy files are artifacts, not the result abstraction itself.
This distinction lets one solve feed visualization, a campaign, a report, or a
training dataset without each consumer reverse-engineering output files.
`write_manifest(...)` records artifacts inside the manifest directory with
relative paths by default, so moving one complete result directory does not
break its links.

Linear problems, engineering `AnalysisStep` objects, and nonlinear problems
provide `solve_result()` while preserving the older `solve()` field-return
contract.

Linear `solve_result()` records PETSc KSP reason, iteration count, and residual
norm. It no longer infers convergence merely because the Python call returned.
The result can consume that evidence through a low-ceremony policy:

```python
solved = step.solve_result()
solved.add_quantity("maximum_displacement", max_u, unit="m")
solved.verify(
    "engineering",
    required_quantities=("maximum_displacement",),
).require()
```

## One execution evidence stream

Standard nonlinear, explicit, implicit-dynamic, first-order transient, and J2
steps emit the same JSON-safe `SolveEvent` contract. The terminal, optional
`.sta` status file, result histories, and complete manifest trace are views of
that one stream. `print_every` changes only human display cadence: every
accepted time increment remains in `result.metadata["execution"]`, while
cutbacks and failed attempts remain auditable rather than being filtered out.

This is important for both reproducibility and AI operation. An agent does not
infer convergence by scraping prose, and a human does not receive a simplified
story that differs from the machine record. Stateful J2 owns its quadrature
state, while heat, Explicit dynamics, and Standard dynamics share one transient
checkpoint envelope. `CheckpointRecord` makes every schema and portability
boundary visible without pretending a partition-bound checkpoint is portable.

Transient heat, Explicit dynamics, and Standard dynamics also share one-call
field output:

```python
result = step.solve_result(output="results.xdmf")
```

The returned result owns the accepted-time evidence and references both the
XDMF index and HDF5 heavy data. Heat writes temperature by default; structural
dynamics writes displacement, velocity, and acceleration. Supplying
`fields=(...)` replaces that default catalog. Request output on the first solve
or when resuming an incomplete checkpoint. A resumed output is marked
`continuation_segment` with its physical start time. AgentFEM refuses to
fabricate missing intermediate frames after a step has already completed.

The three transient procedures can also pause and resume with the same public
contract:

```python
step.run(until_step=50)
checkpoint = step.save_checkpoint("restart/step-50")

resumed = build_the_same_step()
resumed.load_checkpoint(checkpoint)
result = resumed.solve_result(output="restart/continuation.xdmf")
```

The checkpoint records current fields, accepted time, algorithm identity,
execution events, diagnostic histories, mesh/function layout, and one state
shard per MPI rank. Version 2 uses atomic publication, generation-specific
shards, and per-shard integrity checks. It intentionally requires the same mesh
partition and MPI size. This is useful restart today and a precise boundary
for the future global-cell identity needed by cross-partition restart.

Dynamics results carry kinetic energy and, when the step exposes a linear
stiffness operator, recoverable strain and total mechanical energy. Transient
heat carries `thermal_content = 1^T C T`. These are sampled from the same
engineering operators consumed by the solution procedure and survive restart.

## Field output versus history output

AgentFEM follows the established finite-element distinction:

- **field output** is a spatial field over a mesh at selected frames;
- **history output** is a frequently sampled scalar/vector/tensor at a small
  region, probe, control node, or global integral;
- **quantities of interest** are named values selected for reports, campaigns,
  verification, or learning datasets.

`SimulationResult.add_quantity` and `add_history` construct an already computed
scientific result. They are not output-request syntax. Batch forms
`add_quantities(mapping)` and `add_histories(axis, mapping)` avoid repetitive
calls. A field-output request is declared before export:

```python
request = results.field_output(
    "U", "S", "E", "EVOL",
    every=1,
    configuration="deformed",
    backend="xdmf",
)
artifacts = request.write_finite_strain(
    "output",
    domain=domain,
    snapshots=step.snapshots,
    material=material,
)
```

The field catalog owns aliases and physical meaning. In a finite-strain
context, `E` resolves to `LE`; `GREEN` requests Green--Lagrange strain
explicitly. This keeps short engineering vocabulary without allowing one name
to silently change tensor meaning.

## One compact scientific and presentation series

AgentFEM's default finite-strain backend writes one temporal `.xdmf` index and
one compressed `.h5` heavy-data store. Topology is stored once. Each frame
contains:

- its load/time value;
- coordinates `x + scale*u`;
- nodal `U` and `UMAG`;
- every requested cell field on that same grid.

ParaView therefore opens one source, finds the time axis, animates the actual
deformed geometry, and switches among all fields without a Warp filter or
multi-block selection. The HDF5 store also retains the reference coordinates,
so the directly viewable result does not discard finite-element provenance.
`deformation_scale=1` is the physical configuration; another value is
explicitly a presentation scale.

XDMF is the small XML description and HDF5 is the compact numerical payload.
Inlining millions of values into XML would produce a much larger and slower
file, so the pair should be treated as one logical result dataset. The optional
`backend="pvd"` and `"both"` modes remain available for tools that specifically
require PVD/VTU, but they are no longer the default.

## Quantities of Interest

`results.integral`, `results.average`, and `results.l2_norm` assemble physical
expressions with MPI reduction. `SimulationResult.add_dof_statistics(...)`
captures coefficient extrema and counts for quick inspection; those values are
explicitly not physical domain integrals.

```python
solved = step.solve_result()
solved.add_dof_statistics(u, prefix="u", unit="m")
solved.add_quantity(
    "mean_temperature",
    results.average(T, measure=dx),
    unit="K",
)
solved.add_quantities({"mean_temperature": mean_T, "heat_flux": total_flux})
solved.write_manifest("result.json")
```

## Point and path probes

Field values at physical coordinates use the same MPI-safe interface in serial
and distributed runs:

```python
tip_u = results.probe(U, at=(length, height / 2))
temperatures = results.sample_points(
    T,
    ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0)),
)
centerline = results.sample_path(
    T,
    start=(0.0, height / 2),
    end=(length, height / 2),
    count=101,
)
centerline.add_to(
    solved,
    name="centerline_temperature",
    unit="K",
    distance_unit="m",
)
```

AgentFEM locates owned cells, chooses one deterministic MPI owner for each
point, and returns the same ordered values on every rank. All ranks must request
identical coordinates. Missing points raise by default; `missing="nan"` is an
explicit alternative for exploratory sampling. For a discontinuous field, a
point exactly on an interelement boundary has a side ambiguity, so place the
probe inside the intended cell when a one-sided value is required.

## Standard stress and strain fields

Small-strain elasticity now uses one reusable projection path:

```python
stress, strain, mises, energy = results.small_strain_cell_fields(
    U,
    material,
    study=study,
)

with io.XDMFTimeSeries("solid.xdmf", domain) as writer:
    writer.write_fields(0.0, U, stress, strain, mises, energy)
```

The standard names are `S`, `E`, `MISES`, and `SENER`. Their default `DG0`
representation is the global L2 projection onto piecewise constants, hence a
cell average rather than an arbitrary centroid value. In plane strain, Mises
stress includes the constitutively implied out-of-plane stress. The lower-level
`results.project(...)` remains available for reviewed UFL expressions and
higher-order output spaces.

`results.reaction_resultant(problem)` reports the MPI-global residual
resultant for strong Dirichlet constraints. Affine MPC, weak, and contact
reactions deliberately require separate definitions. For proportional linear
loading, `diagnostics.linear_static_energy(...)` reports strain energy,
external work, and their closure; non-zero prescribed displacements require
reaction work in a displacement-control-specific history.

## Structured observation grids

A common physical grid separates a learned field representation from the FEM
mesh used by each case:

```python
grid = surrogates.regular_grid(
    bounds=((0.0, length), (0.0, height)),
    shape=(128, 64),
)
sample = datasets.fem_observation_sample(
    U,
    grid,
    name="displacement",
    unit="m",
    components=("U1", "U2"),
    outside="mask",
)
sample.write("displacement_grid.npz")
```

The export records physical axes, array order, component layout, units,
mesh-independence policy, and an optional inside-geometry mask. It is MPI-safe
and uses the same coordinates for every case. This is a real data-preparation
path for structured-grid neural operators; model architecture and training
remain external responsibilities.

## Campaign to Dataset

A campaign evaluator may return:

- a mapping of declared outputs;
- `campaigns.CaseOutcome` with provenance/artifacts;
- or a `SimulationResult`.

In the last case AgentFEM extracts only the declared QoIs into the scientific
dataset. Live DOLFINx fields are not serialized into table columns; their
artifacts remain linked.

Campaign configuration can be written as safe JSON for parameter definitions,
sampling, output contracts, and execution policy. Python still supplies the
trusted model builder/evaluator. The JSON file never imports or evaluates
arbitrary code.

The current runner is serial across cases and MPI-aware within each FEM case.
Use deterministic plan shards for schedulers or separate MPI jobs rather than
Python threads.

Before training, require reviewed campaign evidence:

```python
dataset = report.require_dataset(
    minimum_samples=4,
    quality="engineering",
)
training = surrogates.train(
    dataset,
    estimator=surrogates.RidgeSurrogate(),
    validation_fraction=0.2,
    seed=2026,
)
guarded = training.guard(fallback=run_high_fidelity_case)
```

`require_dataset()` rejects a partial campaign by default. A quality preset
requires each sample to carry an accepted result assessment and records the
dataset admission decision. The lower-level `minimum_trust_level` remains
available when a project deliberately supplies its own evidence policy. Failed simulations
are not silently discarded before learning; the caller must review them and
set `allow_partial=True` deliberately. That acceptance and the failed case IDs
are retained in the returned dataset metadata. `surrogates.train(...)` keeps the
reproducible split, trained estimator, independent validation, and
applicability guard together. The same protocol accepts the built-in
transparent baselines or the optional PyTorch MLP adapter.

For direct PyTorch training without an AgentFEM trainer:

```python
bundle = dataset.to_torch()
loader = bundle.loader(batch_size=64, shuffle=True)
```

AgentFEM owns names, units, shapes, provenance, splitting evidence, validation,
and applicability. PyTorch owns tensors, modules, automatic differentiation,
optimizers, and training loops.

## Next Result Priorities

1. named reaction force/resultant extractors beyond the current strong-BC and
   J2 residual fields;
2. nodal smoothing and higher-order stress recovery beyond implemented DG
   projection;
3. natural-load, weak-constraint, affine-MPC, and broader transient work/energy
   histories beyond the current strong-displacement J2 history;
4. cross-partition MPI restart identity and checkpoint retention policies;
5. graph and basis-coefficient field encodings for unstructured neural operators;
6. scheduler executors that preserve identical case records.
