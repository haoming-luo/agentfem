# Results, Batch Simulation, and Learning Data

## One Result Contract

`results.SimulationResult` is the scientific result of one analysis. It can
hold:

- scalar or fixed-shape quantities of interest;
- live finite-element fields and field-artifact links;
- time/load/iteration histories;
- solver and model metadata;
- visualization, checkpoint, and report artifacts.

XDMF, CSV, and NumPy files are artifacts, not the result abstraction itself.
This distinction lets one solve feed visualization, a campaign, a report, or a
training dataset without each consumer reverse-engineering output files.
`write_manifest(...)` records artifacts inside the manifest directory with
relative paths by default, so moving one complete result directory does not
break its links.

Linear problems, engineering `AnalysisStep` objects, and nonlinear problems
provide `solve_result()` while preserving the older `solve()` field-return
contract.

## One execution evidence stream

Standard nonlinear, explicit, implicit-dynamic, first-order transient, and J2
steps emit the same JSON-safe `SolveEvent` contract. The terminal, optional
`.sta` status file, result histories, and complete manifest trace are views of
that one stream. `print_every` changes only human display cadence: every
accepted time increment remains in `result.metadata["execution"]`, while
cutbacks and failed attempts remain auditable rather than being filtered out.

This is important for both reproducibility and AI operation. An agent does not
infer convergence by scraping prose, and a human does not receive a simplified
story that differs from the machine record. Checkpoint state is still owned by
each procedure; `CheckpointRecord` makes its schema and portability boundary
visible without pretending every checkpoint is MPI-portable.

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
dataset = report.require_dataset(minimum_samples=4)
training = surrogates.train(
    dataset,
    estimator=surrogates.RidgeSurrogate(),
    validation_fraction=0.2,
    seed=2026,
)
guarded = training.guard(fallback=run_high_fidelity_case)
```

`require_dataset()` rejects a partial campaign by default. Failed simulations
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
2. named point and path probes;
3. stress/strain projection for visualization;
4. external-work and energy-balance histories beyond the current linear and J2
   internal-energy diagnostics;
5. transient checkpoint writers and MPI-portable restart identity;
6. mesh-independent field sampling for neural operators;
7. scheduler executors that preserve identical case records.
