# Benchmark modeling and reusable platform capabilities

A benchmark should show **what physical problem is solved and what is compared**.
It should not implement a second mesh engine, state manager or solver. Small
functions that express a physical map, loading history or observable are useful;
replacing the whole case with one opaque call is not the objective.

## Module audit (2026-11-06)

All 17 Python modules in `agentfem.benchmarks` were reviewed. The following
classification covers the executable providers and the supporting modules.

| Modules | Physical content kept in the benchmark | Platform implementation / decision |
| --- | --- | --- |
| `delamination_fem`, `enf_fem`, `mmb_fem` | Arm dimensions, material/interface laws, precrack, opening/bending/lever loading, damage-front criteria and mode partition | Shared split rectangle, named traction loads, monotonic cohesive-state/work bookkeeping. See [delamination workflow](delamination_benchmark_workflow.md). |
| `elasticity_3d` | LE10 ellipse equations, thickness, supports, pressure and stress recovery at D | `mesh.mapped_grid` owns Q1/Q2 connectivity and MPI distribution; solve stays `model.step`. |
| `inelastic` | Cylinder geometry, plane-strain constraints, pressure/history, Lamé and creep references, explicit axisymmetric `2*pi*r` factor | Reuses mapped grid and `QuadratureField.owned_physical_points/weights`; no hand-written backend sampling. |
| `ratcheting_structure` | Published specimen profile, cycle history and 3 mm gauge window | Public centroid selection replaces backend midpoint access. Gmsh geometry construction remains explicit because it defines this specimen. |
| `dynamic_fracture` | Wave excitation, cohesive laws, loading histories, preload compatibility and crack-speed measurements | All six legacy step calls become `model.step`; tensor/energy averaging uses `results.average`. |
| `contact_sliding` | Contact surfaces, preload/sliding stages, friction and facet-crossing observations | All four legacy step calls become `model.step`; existing contact operators remain the numerical implementation. |
| `forced_vibration` | Supports, frequency axis, damping, K/M/C/F composition, independent response references | Already uses public operators and result APIs; retained. |
| `cyclic_plasticity` | Published material coefficients, stress/strain histories and material-point comparisons | Already uses public material/path workflows; retained. |
| `lefm` | Refined crack mesh, shared tips/split faces, loading and independent interaction-integral evidence | Public array-mesh constructor replaces backend mesh creation. A single explicit `ufl.grad(u)` remains to identify the displacement gradient being measured. |
| `thermoelastic_coupling` | Independently written mixed UFL reference and coupled energy balances | Deliberately retained as a bounded formulation experiment, not an introductory case or a production solver provider. Reusing the tested solver for its reference would remove independence. |
| `mixed_mode` | Beam references, compliance/energy-release formulas and comparison conventions | Analytical oracle; retained independently of the numerical providers. |
| `registry`, `evidence`, `golden`, `__init__` | Inventory, evidence rules, tolerances and exports | Supporting metadata, not numerical solver code; retained. |

Only the two explicitly identified formulation/evidence modules still import
UFL directly. `test_benchmark_public_workflow.py` checks that ordinary providers
do not regain backend imports or legacy structural step entry points. New
independent formulations can be added with an explicit justification; this is
not a ban on mathematical expressions.

## New reusable geometry vocabulary

A curved domain remains readable because its physical map stays in the case:

```python
def cylinder(parameters):
    radius, angle, height = parameters
    return np.asarray((radius*np.cos(angle), radius*np.sin(angle), height))

domain = mesh.mapped_grid(
    axes=(radii, angles, elevations), mapping=cylinder,
    cell_type="hexahedron", geometry_degree=1, comm=comm,
)
```

The axes contain **geometry nodes**, so each direction has `degree*cells+1`
entries. Q2 quads/hexes use Basix's actual local node ordering. Q1 triangles and
tetrahedra use a fixed subdivision. Axis 0 varies fastest. The routine handles
root-owned input and distributes the mesh; it does not infer materials,
constraints, cracks or units. The map must be nonfolding; successful mesh
creation is not a mesh-quality certificate.

For explicit topology, existing callers can continue to pass
`coordinate_element` to `mesh.from_arrays`. Ordinary cases may instead specify
`cell_type` and `geometry_degree`; these alternatives cannot be combined.
Connectivity remains in Basix order, including higher-order nodes.

Sampling is also explicit:

```python
gauge_cells = mesh.locate_cells(
    domain, lambda x: (x[1] >= 35.9) & (x[1] <= 38.9), by="centroid",
)
coordinates = stress.owned_physical_points
weights = stress.owned_physical_weights()
```

Centroid selection returns owned cells only. Default vertex-based selection
is unchanged. The benchmark still decides the averaging window and whether a
physical measure requires an additional axisymmetric factor.

## Validation record

Starting point: `origin/main` at `2c9ddbb7`, followed by the delamination commit
`cdcf4316` on `refactor/delamination-benchmark-platform`.

- The first pass compared six delamination paths / 72 sampled point records:
  every recorded field was exactly equal before and after refactoring.
- The second pass independently recorded LE10, J2 cylinder, axisymmetric creep
  cylinder, center-crack LEFM, thin-3D plane-stress patch, wave arrival,
  cohesive energy, classical cohesive crack and finite sliding: all nine
  serialized evidence records were exactly equal before and after changes.
- Final serial suites: **223 passed, 4 skipped** across three nonoverlapping
  groups. The opt-in real shouldered-specimen tests were explicitly enabled.
- Skips: a two-rank contact test, a distinct-rank thermoelastic test, the
  unavailable optional SIMULIA deck and the opt-in full three-level forced
  vibration study. These are not counted as passes.
- Independent new geometry tests cover affine area/volume, Q1/Q2 connectivity,
  invalid maps/inputs, array-mesh element width and centroid sampling.
- A two-rank check completed all assertions for Q1/Q2 global volume, shared
  boundary measures, owned-cell selection and collective invalid-input errors.
  This machine then hung in `MPI.Finalize`; those processes were stopped.
  Numerical/collective assertions passed, but clean MPI shutdown is **not**
  claimed from that run. Serial test workers recorded pytest's exit status and
  bypassed the same local teardown problem without changing test execution.

Reproduction groups:

1. `test_mapped_grid`, `test_external_elasticity_3d_benchmark`,
   `test_external_inelastic_benchmark`, `test_fracture_fem`, `test_array_mesh_archive`:
   47 passed.
2. `test_dynamic_fracture_benchmarks`, `test_external_contact_benchmark`,
   `test_ratcheting_structure`, `test_external_forced_vibration_benchmark`,
   `test_external_cyclic_plasticity`, with `AGENTFEM_RUN_EXTERNAL_STRUCTURE=1`:
   49 passed, 3 skipped.
3. `test_benchmark_public_workflow`, the three delamination suites,
   `test_cohesive_path`, `test_split_rectangle`, `test_boundary_partition_assembly`,
   `test_interfaces`, `test_sprint_interfaces`,
   `test_thermoelastic_coupling_prototype`, `test_thermoelastic_procedure`,
   `test_release_goldens`: 127 passed, 1 skipped.

No analytical references, acceptance tolerances, loading histories or material
parameters were relaxed. This is a benchmark/platform refactor, not a change to
the mechanical models or a claim of new numerical accuracy.

### Mainline integration follow-up (2026-11-13)

The current `fenicsx-env` run with process-local `FI_PROVIDER=tcp` and single
BLAS/OpenMP threads completes normally; no teardown bypass is used. Four
two-rank Q1/Q2 mapped-grid tests verify root-owned input, global area/volume,
boundary measures, owned centroid selection and collective invalid-input
rejection. Both ranks report four passes and the bounded `agentfem mpi-run`
launcher exits with status zero. These tests are in the existing distributed
mesh CI suite, so clean exit is now an executable regression rather than an
uncommitted local assertion. This does not certify every MPI backend.

The affected serial geometry, cohesive-path and benchmark suite passes 143
tests, with three explicit skips: the separate two-rank contact case, the
unavailable optional SIMULIA deck and the opt-in full three-level vibration
study. Public API generation, focused Ruff and diff checks also pass. Existing
before/after reference evidence above is retained; no new full-composite speed
or scientific accuracy claim is made. Remote macOS acceptance remains required
before merge.
