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
break its links. It also adds a deterministic provenance seal over the result
record and every registered artifact. Run `agentfem verify result.json` to
detect a partial copy, replaced HDF5 payload, or edited manifest. This integrity
check is deliberately separate from scientific verification; see
`docs/provenance_seal.md`.

For frozen runtime locks, serializable loading bases, campaign-backed response
operators, and threshold-event localization, continue to
`docs/scientific_experiments.md`.

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

AgentFEM's serial static-solid and finite-strain result paths write one temporal
`.xdmf` index and one compressed `.h5` heavy-data store. Topology is stored
once. Each frame
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

Ordinary static output retains reference coordinates and exposes `U` as point
data, so **Warp By Vector** produces one deformed geometry with every cell and
point attribute attached. Finite-strain presentation output can instead store
`x + scale*u` directly when requested. `SimulationResult.metadata.field_output`
records the backend, layout, and geometry convention. For a two-dimensional
model, the finite-element unknown remains the physical vector `(Ux, Uy)`, while
the visualization dataset stores `U=(Ux, Uy, 0)` on XYZ geometry. ParaView can
therefore select `U` directly as a three-component Warp vector. The result
contract records `physical_components=2`, `stored_components=3`, both model
and storage geometry dimensions, and the semantic alias `Displacement -> U`;
no duplicate displacement array is written.

The low-level `io.XDMFTimeSeries` mirrors DOLFINx and may expose one XDMF Grid
per Function. It is an expert compatibility API, not the normal result path.
Prefer `step.solve_result(output=...)`: serial static and transient analyses
write the compact single-grid XDMF/HDF5 layout. Under MPI, AgentFEM retains the
collective XDMF/HDF5 scientific record and also registers a
`fields_paraview` PVD artifact whose time steps each contain one geometry with
all point and cell fields. The normal ParaView workflow therefore does not
require **Extract Block** or **Append Attributes**. The machine-readable
`field_output.recommended_visualization_artifact` points to the compact XDMF
in serial and to this PVD in MPI; `visualization_requires_extract_block` is
therefore false even though the collective scientific XDMF retains DOLFINx's
grid-per-field organization.

Static balance evidence also fails closed. Strong Dirichlet reactions use the
unconstrained assembled residual, but MPC, weak, contact, projection, and
multiplier constraints require provider-owned dual forces. When any declared
constraint lacks that channel, `SimulationResult` records
`static_equilibrium.status = unavailable` and a
`constraint_balance_contract` instead of publishing a partial reaction sum as
a complete equilibrium check. A provider closes force balance only by supplying
both its generalized dual and the corresponding physical-space resultant; it
closes work only by also supplying the accepted work-conjugate coordinate.
`AnalysisStep.constraint_dual_provider` is the narrow post-convergence seam for
that evidence. The common ledger consumes the returned records; a provider is
not allowed to promote completeness by returning a Boolean flag alone.

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

For cell or constitutive data, coefficient extrema are not enough. Pass the
resolved scalar values together with their physical cell or quadrature weights
to obtain an exact volume-weighted distribution:

```python
statistics = results.weighted_field_statistics(
    mises,
    integration_weights,
    quantiles=(0.05, 0.5, 0.95),
    thresholds=(yield_stress,),
    location="quadrature_points",
    representation="raw_constitutive_values",
    comm=domain.comm,
)
```

The returned record distinguishes raw integration-point values from cell
averages or recovered nodal fields and states the quantile and threshold
definitions. MPI ranks receive the same summary. The weights are explicit:
AgentFEM does not infer a physical volume distribution from arbitrary field
coefficients.

`QuadratureField.weighted_statistics(...)` supplies those weights from the
reference quadrature rule and absolute geometric Jacobian. Only owned cells
contribute under MPI; ghost cells are excluded. Tensor fields require an
explicitly selected component or invariant, so the API never invents a scalar
meaning.

The full definitions, measure convention and references are collected in
[RVE homogenization and physical field statistics](reference/rve_homogenization_and_statistics.md).

## Periodic-cell evidence at every accepted increment

`results.periodic_cell_history(periodicity)` is an online scientific request,
not merely a postprocessor over saved XDMF frames. It records a lightweight
macroscopic state after every accepted affine increment while spatial fields
may be saved less often:

```python
output = results.output_plan(
    "output/rve",
    field=results.field_output("U", "S", "E", every=5),
    requests=(results.periodic_cell_history(periodicity),),
)
```

If twenty increments are accepted, the CSV and NPZ contain the initial state
plus all twenty accepted states even though the spatial result contains only
the requested sparse frames. The recorder retains macroscopic tensors and one
preceding microscopic state, rather than accumulating all finite-element
fields in memory.

Each accepted macro row also carries its actual increment size, Newton
iteration count, final residual, periodic-equation mismatch and accepted
attempt number. This places cutback and convergence evidence beside the state
it qualifies instead of requiring a later join against terminal logs.

For Cauchy stress \(\boldsymbol\sigma\), AgentFEM uses

\[
\eta = \frac{\sigma_m}{\sigma_{\mathrm{vM}}},\qquad
\bar\theta = 1-\frac{2}{\pi}
\cos^{-1}\!\left(\frac{27J_3}{2\sigma_{\mathrm{vM}}^3}\right).
\]

The normalized Lode parameter is `+1` in axisymmetric tension, `0` in pure
shear and `-1` in axisymmetric compression. Both quantities are undefined
when the deviatoric stress vanishes. Structured results therefore carry
`homogenized_stress_state_defined`; a numerical placeholder is never evidence
that a hydrostatic state's triaxiality or Lode angle exists.

For two consecutive accepted compatible states, the finite-strain
macrohomogeneity audit compares the same trapezoidal first-Piola work at both
scales:

\[
\Delta w_\mu = \frac{1}{|\Omega_0|}
\int_{\Omega_0}\frac{\mathbf P_n+\mathbf P_{n+1}}{2}:
(\mathbf F_{n+1}-\mathbf F_n)\,\mathrm dV,
\]

\[
\Delta w_M = \frac{\overline{\mathbf P}_n+
\overline{\mathbf P}_{n+1}}{2}:
(\overline{\mathbf F}_{n+1}-\overline{\mathbf F}_n).
\]

The result records both work densities, their signed residual and relative
error. This is a quasistatic periodic/affine Hill--Mandel contract; it does not
silently omit body-force or inertia power from a problem where those terms are
present. The macro stress remains normalized by the complete reference-cell
volume, so void volume carries zero stress rather than changing the result to
a matrix-phase average.

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

Model-generated static elasticity now provides its standard fields in one call:

```python
result = step.solve_result(output="solid.xdmf")
# result.fields contains Displacement, S, E, and MISES

# Opt in to a diagnostic field set when the analysis needs it.
energy_result = step.solve_result(
    output="solid_with_energy.xdmf",
    field_variables=("S", "E", "MISES", "SENER"),
)
```

Small-strain elasticity uses one reusable projection path. Application code
normally requests it through the step so field creation and single-grid output
remain one transaction:

```python
solid_result = step.solve_result(
    output="solid.xdmf",
    field_variables=("S", "E", "MISES", "SENER"),
)
```

The available standard names are `S`, `E`, `MISES`, and `SENER`. The
engineering default is `U/S/E/MISES`: `U` is the primary unknown, `S/E` are
standard mechanics fields, and `MISES` is materialized for immediate plotting
even though it is derived from `S`. `SENER` is opt-in because a per-cell energy
density is primarily diagnostic; total strain energy and energy closure belong
in compact histories or verification quantities.

The default derived-field representation is a global L2 projection onto
discontinuous `DG0`, hence a cell average rather than an arbitrary centroid
value. It is not an integration-point dump, nodal extrapolation, or smoothed
nodal field. No values are averaged across neighboring cells or material
interfaces. Each `FieldResult.processing` record preserves this distinction.
In plane strain, Mises stress includes the constitutively implied out-of-plane
stress. The lower-level `results.project(...)` remains available for reviewed
UFL expressions and higher-order discontinuous output spaces.

For regional materials, `results.small_strain_partition_fields(...)` assembles
one global projection from `(material, region)` contributions. It preserves a
single field over the original mesh rather than writing one incomplete field
per material.

`results.reaction_resultant(problem, on=support, component=0)` reports the
MPI-global residual resultant on one named strong-Dirichlet boundary. Omitting
`on` retains the whole-field sum. Affine MPC, weak, and contact reactions
deliberately require separate definitions. For proportional linear
loading, `diagnostics.linear_static_energy(...)` reports strain energy,
external work, and their closure. `results.static_work_balance(...)` extends
that contract to non-zero strong prescribed motion by integrating the
conjugate reaction along a proportional path. Ordinary model-generated static
solid results record natural-load work, prescribed-motion work, total external
work, strain energy, and their closure automatically.

Affine MPC reactions must be defined from the dual of the reduced constraint
or from a declared macro-motion mode; summing eliminated slave residuals is
not invariant to the chosen elimination graph. Weak-constraint reactions are
consistent boundary tractions/fluxes from the weak form. Until those dual
consumers are implemented, AgentFEM marks their work balance unavailable
instead of applying the strong-Dirichlet formula.

A model-generated linear static solid also records force equilibrium without
extra application code:

```python
result = model.step(target=displacement).solve_result()
result.quantities["external_force_resultant"]
result.quantities["reaction_force_resultant"]
result.quantities["force_balance_residual"]
result.quantities["relative_force_balance_error"]
```

The external resultant is assembled from the same complete right-hand side
used by the solver; it therefore includes all body and boundary contributions
in `F`. The reaction is the unconstrained residual at strong Dirichlet dofs.
The metadata states this scope explicitly rather than implying support for MPC,
contact, weak-constraint, or multiplier reactions.

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

An external solver protocol can request the same geometric operation without
constructing a learning dataset:

```python
sample = results.sample_rectilinear_grid(
    U,
    bbox=(0.0, length, 0.0, height),
    shape=(128, 64),
    reduction="magnitude",
)

image = sample.values   # array order: (ny, nx)
inside = sample.inside  # false for points outside a non-rectangular mesh
```

For three dimensions, `shape=(nx, ny, nz)` produces `(nz, ny, nx)`. Domain
exterior values are `NaN` and the explicit mask is authoritative. The routine
is MPI-safe and uses the same ownership rules as point probes.

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

The runner supports serial execution, spawned local processes across
independent cases, and MPI participation within one FEM case. Use
`campaigns.local_processes(workers=...)` for a local parameter sweep. It does
not use Python threads or fork an initialized numerical process, and it cannot
be nested inside a within-case MPI communicator. Use deterministic plan shards
for schedulers or separate MPI jobs.

Campaigns may receive `scientific_inputs={...}`. Source meshes supplied as
`Path` objects are hashed by content; materials, loads, procedures, and
observer plans use their public IR/summary contracts. Opaque objects make
coverage incomplete. After a refinement campaign, use `convergence.audit(...)`
to issue an observable-aware multi-axis certificate rather than inferring
convergence from a successful solve or a plotted curve.

For a single analysis, call
`simulation.add_scientific_inputs(mesh=Path(...), material=..., loading=...)`
before `write_manifest(...)`. This stores the same content-addressed input
record directly in the `SimulationResult`; it does not replace runtime,
artifact-integrity, convergence, or validation evidence.

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

1. affine-MPC, weak, and contact reaction force/resultant definitions beyond
   the current named strong-boundary and J2 residual fields;
2. nodal smoothing and higher-order stress recovery beyond implemented DG
   projection;
3. natural-load, weak-constraint, affine-MPC, and broader transient work/energy
   histories beyond the current strong-displacement J2 history;
4. cross-partition MPI restart identity and checkpoint retention policies;
5. graph and basis-coefficient field encodings for unstructured neural operators;
6. scheduler executors that preserve identical case records.
