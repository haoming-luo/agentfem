# Response-only materials in an ordinary Step (experimental)

`constitutive.explicit_user_material` adapts a trusted user callback without
inventing a tangent or fixing the number of internal variables. The callback
receives a `MaterialPointArrayBatchInput` and returns an
`ExplicitMaterialBatchOutput`. Core validates the response, publishes trial
fields, and commits only after the complete increment succeeds. The callback
must not mutate accepted history or commit global state.

```python
material = constitutive.explicit_user_material(
    state_schema=schema, update=user_update,
    identity={"law": "my_law", "revision": "pinned-source-hash"},
    density=physical_density,
)
model.material(material)
step = model.step(
    target=u,
    element_policy=elements.uniform_strain_hex8(
        kinematics="finite_strain", hourglass_modulus=modulus,
    ),
    dt=dt, steps=steps, time_increment_ceiling=reviewed_ceiling,
    stress_free_initial_state=True,
)
result = step.solve_result()
```

The runnable `examples/response_only_material.py`
example supplies a small external elastic callback and moving tools, then saves
an accepted checkpoint and material arrays. It requires no confidential assets.
Run it with the installed package and an explicit fresh `--output` directory.

Checkpoint cadence belongs to the Procedure, not a case-specific stepping loop.
Pass `checkpoint=checkpointing.every(5000, directory="run/checkpoints")` to
`model.step(...)`. The policy saves accepted states at absolute increment
multiples and at the final increment. After `step.load_checkpoint(path)`, call
`step.run()` or `step.solve_result()` normally: a restart between stations does
not shift the schedule. A partial `run(until_step=...)` does not create an
unscheduled checkpoint; call `save_checkpoint(...)` explicitly when that pause
must be retained. The response-only provider retains its serial-only boundary;
scheduled saving does not add MPI portability.

For append-only scalar reporting, use the existing accepted event stream rather
than repeatedly calling `run(until_step=...)` just to inspect the last record:

```python
journal = results.AcceptedHistoryJournal("run/history.jsonl")
result = step.solve_result(progress=journal.reporter(step, every=10))
result.metadata["accepted_history_journal"] = journal.summary()
result.add_artifact("accepted_history", journal.path)
```

The observer copies an already evaluated history frame; it does not evaluate
the material, own state, or create a checkpoint. It samples absolute increment
multiples and the final increment, regardless of terminal display cadence.
`every` must be a multiple of `history_every`; stale frames are rejected, not
relabeled with the current time. Use the normal `history=(results.history(...),)`
solve argument for additional scalar channels. Start a new journal after
restart. This observer is serial-only and also works with ordinary explicit,
implicit-dynamic and heat Steps. An output failure is raised after numerical
acceptance: stop the run, save the accepted state if possible, and resume with a
new journal; do not retry a poisoned writer. It does not silently roll back an
already accepted physical increment.

The current admission is serial, three-dimensional, one full-domain material,
and finite uniform-strain Hex8. It starts from a stress-free, virgin state.
Multiple `RigidContactPair` model assets may use frictionless planes or cylinders
with prescribed translation. Cylinder projection retains the existing backend's
geometric limitations; this does not add end-cap, self-contact, or general
ALL EXTERIOR enforcement. Rotation, external loads, constraints, eigenstrains,
regional material dispatch and MPI are refused by this provider.

One `boundary_models.deformable_contact_pair` is admitted with explicit
bilinear Q1 traces, independent node maps and candidate selection. It reuses
the existing one-sided reference-area penalty operator, including conservative
path rejection. It is not symmetric ALL EXTERIOR, self-contact or hard contact.

`interfaces.reference_trace(..., geometry="bilinear")` preserves warped Q1
faces. `interfaces.initial_nodal_bond` binds an explicitly eligible, coincident
node set once; `fracture.nonmatching_cohesive_force` passes that pairing to the
Step as `cohesive_force`. Only isotropic elastic bonds are admitted here. The
default `bond_contact_policy="disjoint"` rejects shared trace nodes;
`"independent"` explicitly adds the declared contributions without switching,
rebonding or silently disabling either law. One-sided integration still needs
refinement and side-swap assessment.

Optional `boundary_models.RigidContactTracePolicy` selects Gauss or nodal
integration. Only uniform frictionless nodal laws may coalesce coincident
quadrature nodes by summing their reference areas. A cylinder transverse-strip
bound requires an explicit unit direction, displacement budget, and compatible
nonrotating motion. The budget is checked over every displacement node before
each evaluation; exceeding it rejects the trial rather than omitting contact.

`bulk_viscosity=elements.BulkViscosityPolicy(...)` selects the existing
volume-equivalent length rule. Its reference density must equal the physical
material density. Artificial viscosity work is not constitutive dissipation.

Optional `mass_scaling_factor` is a finite scalar or a nodal array of factors
at least one, in the current serial displacement-node order. It changes only
numerical inertia, not physical material density. Restart verifies inertia
identity; changing nodal order or scaling is not a portable continuation.
The summary records added mass. A successful scaled run does not establish
quasi-static accuracy or mass sensitivity.

For new cases prefer the explicit numerical policy:

```python
mass_scaling = time.FixedMassScaling(factor=2.0, maximum_added_mass_ratio=1.1)
# Pass mass_scaling=mass_scaling to model.step(...).
```

The numbers above illustrate the interface, not a universal recommended mass.
For the admitted zero-displacement/zero-velocity initial state, the built-in
`time.InitialMassScaling(target_increment=..., reference_modulus=...,
maximum_added_mass_ratio=...)` generates a fixed numerical mass directly from
the current mesh. `reference_modulus` is a stress-unit bound on the declared
reference `dP/dF`, **not automatically Young's modulus**. The software combines
the uniform-Hex body/hourglass reference estimate, isotropic elastic bond,
each rigid penalty trace, each initial deformable-contact candidate neighborhood,
and initial linear bulk viscosity. It reuses the actual force trace adapters;
no prepared inertia file or case-written nodal assembler is needed.

Its `mass_scaling_preparation` report identifies all contributions and candidate
coverage. The deformable estimate bounds nonnegative frozen Q1 interpolation
on initial candidate facets; shared vertices are counted once per secondary
point. It does not bound the derivative of moving closest-point projections,
later contact neighborhoods, curved negative geometric stiffness, damage
softening or nonzero-velocity quadratic viscosity. This is explicitly a
**reference-model initial estimate**, not an all-state tangent inferred from
an arbitrary user material. It leaves the explicit step ceiling, cutback,
energy ledger and sensitivity obligations in place. Its mass remains fixed;
use the provider policy below for reviewed evolving bounds.

`time.TargetIncrementMassScaling(target_increment=..., bounds=provider,
identity={...}, maximum_added_mass_ratio=...)` instead computes nodal factors
from `MassScalingBounds(stiffness={...}, damping={...})`. Its immutable
`MassScalingRequest` provides physical nodal masses, reference coordinates,
cell-to-node layout, accepted displacement/velocity, current factors, time and
the exact required contribution names. There is no precomputed-file dependency.
The provider is trusted deterministic code and must supply all named body,
hourglass, cohesive, contact and damping contributions applicable to the Step.
The code validates coverage and arrays, not the mathematical correctness of a
user's estimate. Bounds are unscaled diagonal majorants for symmetric PSD
stiffness and lagged damping, not previously mass-scaled eigenvalues. A user
material's initial elastic stiffness need not bound a damaged nonlinear path.

`update_every=None` plans once before execution. A positive integer updates at
macro increments `1, 1+n, ...` in the existing transaction; substep retries do
not shift this schedule. The provider must cover evolution between evaluations.
Mass never decreases, and a budget breach rejects rather than silently clips
the proposal. Velocity remains unchanged and the artificial energy/momentum
inputs are recorded separately from external work. Restart checks provider,
policy and layout identity. This is serial and not partition-portable.
`dt` remains explicit and cannot exceed the target used for planning. A policy
neither selects the time step nor proves physical accuracy or full-path
stability. Do not combine `mass_scaling` with legacy `mass_scaling_factor`.

Ordinary completion now accepts
`model.step(..., output="fields.xdmf", history=(request,)).solve_result()`.
The existing writer owns connectivity and attaches U/V/A to one grid per time;
additional material fields still require explicitly named accepted-state data.
Substep retry remains limited to the built-in energy ledger; custom history
requests with substeps are rejected. Final-only output does not reconstruct an
animation of increments that were not stored.

Inspect the accepted loading history, not only the final kinetic-energy ratio:
a small endpoint value can hide substantial intermediate inertia. Numerical
kinetic energy belongs in the dynamic balance; its physical-mass and added-
inertia components are separate diagnostics, not interchangeable measures.
State the denominator and a near-zero rule for any ratio. In particular, do
not label internal work as recoverable energy or infer quasi-static validity
from an exactly closed numerical ledger. Compare the response under controlled
inertia/time-grid changes before making that scientific claim.

The caller's increment ceiling is **not a stability certificate**, including
when contact and damping are present. This provider does not select an automatic
stable step. Internal work is reported as work, not relabelled stored energy.
`completed` does not imply independent physical verification.

`operator_increment_ceilings` may explicitly distinguish `body`, `rigid_contact`
and `deformable_contact` limits; omitted roles inherit `time_increment_ceiling`.
Every declared limit must be finite and at least the actual `dt`, and a contact
override requires that asset. Body damping reduces the declared undamped body
ceiling, so setting the body ceiling equal to `dt` can correctly reject a damped
increment. This option preserves distinct operator assumptions rather than
silencing damping checks; it does not certify their chosen numerical values.

`time.ExplicitSubstepPolicy` enables bounded dyadic retries for typed contact
path failures and optionally material cutback requests. Intermediate work is
recorded along accepted subincrements; a terminal failure rolls back the whole
macro increment. This serial path requires `history_every=1`. Ordinary unknown
exceptions remain failures, not reasons to keep trying. Retry counts describe
the current process and are not a cumulative restart history.

`elements.ExplicitHexBodyKernel` is an optional trusted execution extension.
It receives immutable arrays and fixed reference geometry, not finite-element
Fields or commit operations. Core retains increment checks, output validation,
trial publication and acceptance. The kernel identity participates in restart
admission. This removes the need to replace global methods for fused body
arithmetic; it does not itself prove a native kernel mathematically equivalent
or faster. Private constitutive constants and native binaries remain external.
An optional `prepare_kinematics` callback fuses the mean gradient and the
reference-perturbation sufficient screen. It must not certify cell validity
from the mean determinant alone. Uncertified cells retain core affine and
Bernstein checks, and all mean gradients retain positive-determinant admission.
`evaluate_viscosity` optionally replaces only the admitted bulk-damping
arithmetic. It receives immutable physical coefficients and a reference
fallback; core validates force, power and damping-bound outputs and keeps
artificial damping separate from constitutive dissipation.

The tests exercise ordinary Step selection, response contracts, moving-tool
work, damping/scaling combinations and exact checkpoint continuation on tiny
meshes. They do not establish full industrial-case equivalence or speed.

`boundary_models.Q1ContactExecution` supplies instance-local trusted closest
point, closed-solid classification, conservative distance-bound and swept-path
callbacks. Core still selects candidates,
resolves ambiguities, constructs forces and owns transactions. Callback results
are structurally checked; this is not a proof that a plugin found the global
minimum or classified a solid correctly. Identity is included in contact
restart admission. Reference fallbacks and refitted geometry keep the same
instance configuration; unrelated models are unchanged.
Distance and swept certificates are trusted numerical algorithms, not arbitrary
heuristics: uncertain geometry must use the reference proof or remain
unresolved. A false separation certificate can skip a physical contact and is
not made safe merely by returning a well-shaped boolean array.
An optional `project_batch` callback receives validated query arrays, search
options and an instance-bound reference fallback. It may accelerate complete,
strictly ordered candidate sets, but must preserve candidate budgets, ties,
trace/normal ambiguity and query-indexed errors. Core checks returned facet
identities, bilinear weights, reconstructed points, distances and unit normals
in batches. This transport check does not prove global minimum ordering. No
callback may replace global geometry methods or modify another model's search.

`interfaces.project_reference_points` and `propose_reference_adjustment`
support explicit pre-model preparation. Adjustment returns a proposal, never
mutates a live bulk mesh, and does not infer bond eligibility. Apply through
original node identities, validate affected bulk cells and rebuild pairing.

`step.accepted_material_fields()` returns detached read-only cell records for
export: state schema, internal variables, Cauchy stress, deformation gradient
and cell identity. It rejects an active trial. Private state-variable names and
their interpretation remain the external material provider's responsibility.

Use the reusable cell-data adapter instead of private dof-indexing loops:

```python
accepted = step.accepted_material_fields()
stress = results.cell_field_from_values(
    mesh, accepted["cauchy_stress"], name="cauchy_stress",
    cell_keys=accepted["cell_keys"],
)
with io.ResultWriter("fields.xdmf", mesh, (u, stress)) as writer:
    writer.write(accepted["time"])
```

`cell_field_from_values` requires one finite scalar/vector/tensor per cell in
this serial workflow. Optional keys are original mesh input cell indices, not
external element labels; missing, duplicate or foreign keys are rejected.
It writes detached DG0 values through the actual cell-to-dof map in one array
operation. It does not average material points, smooth interfaces, infer units
or interpret damage columns. Add units and processing evidence to
`SimulationResult.add_field` when publishing the field as a scientific result.
The writer retains reference coordinates and all attributes on one grid, so
ParaView Warp acts on the complete dataset.

For delivery of a 3D solid result, the independent reader check can also
require volume cells:

```sh
pvpython tools/check_paraview_warp.py fields.xdmf --vector Displacement --cell-dimension 3
```

Use the actual exported vector name. This checks one mesh, cell types/dimensions,
complete displacement and Warp coordinates; it does not certify mesh Jacobian
quality or physical accuracy. Two-dimensional models should request dimension
2, not be forced into the solid check.

`material_storage="owned_arrays"` is an explicit serial execution option for
this bounded Procedure. Immutable accepted arrays are the single material-state
owner; quadrature Functions are internal publication buffers. User material
callbacks must consume the supplied request only, not capture/read internal
quadrature Functions. Use `accepted_material_fields()` for observation and the
ordinary checkpoint API for persistence. Those boundaries publish accepted
data; rollback restores the accepted owner. Arbitrary internal-field observers
are not supported in this mode. The default `"fields"` keeps ordinary field
ownership. Both modes retain material validation and finite checks; this
choice does not change the law, time step or inertia. Multiple named state
variables are supported, not a fixed material-specific history length.

The optional `agentfem[exact-geometry]` extra uses GMP rational arithmetic for
exact Q1 fallback predicates when installed. Without it, standard-library
`Fraction` remains available. Selection happens once on import; no case replaces
global geometry functions. `boundary_models.exact_geometry_arithmetic()` reports
the selected library/version. Both paths interpret binary64 inputs exactly;
this is not a tolerance relaxation or a geometric approximation. See the
[gmpy2 rational documentation](https://gmpy2.readthedocs.io/en/stable/mpq.html).

## Public material-frame utilities

User providers may import `constitutive.polar_strain_increment` and
`constitutive.push_corotational_stress` without depending on private module
paths. The first returns a `CorotationalIncrement` with tensor strain (including
tensor shear), endpoint material-to-spatial frames and an explicit `method`.
Inputs are finite real batches of shape `(points, 3, 3)`; total deformation
gradients must preserve orientation and be well-conditioned, and optional
`reference_frame` values must be proper orthonormal axes in the same batch
layout. Masked components are rejected, not silently unmasked. Returned arrays
are detached and read-only; the helper does not own material state.

```python
kinematics = constitutive.polar_strain_increment(
    request.deformation_gradient_old, request.deformation_gradient_new,
    integration_rule="exact",
)
# The user's constitutive update consumes kinematics.strain_increment and
# accepted material history, then returns local symmetric Cauchy stress.
spatial_stress = constitutive.push_corotational_stress(
    local_stress, kinematics.frame_new,
)
```

The declared path is linear in right stretch. `exact` integrates that path;
it does not reconstruct an unknown history, equal `log(U_new)-log(U_old)` for
general noncommuting stretches, or certify a vendor's VUMAT increment. The
opt-in `midpoint` rule has a separate method identity and second-order
subdivision evidence. Include the chosen rule and material axes in provider
identity. Stress rotation does not change Cauchy stress into Piola stress;
core owns the latter conversion in the element route. These helpers imply no
constitutive energy decomposition, tangent, or automatic stable increment.

The separation follows the general distinction between kinematics and material
updates described by [MOOSE finite-strain kinematics](https://mooseframework.inl.gov/source/materials/ComputeFiniteStrain.html)
and [VUMAT local-coordinate conventions](https://docs.software.vt.edu/abaqusv2025/English/SIMACAESUBRefMap/simasub-c-vumat.htm).
Their different increment algorithms are not presented as equivalent.

## Acceptance and remaining promotion gates

### Accepted fields, pair evidence and configured histories

`step.accepted_material_fields()` returns detached, read-only cell arrays and
`state_variables`, a dictionary named by the registered `MaterialStateSchema`.
The flattened `state` array remains available for compatibility. No stress or
SDV smoothing is implicit. Export through `results.cell_field_from_values` and
`io.ResultWriter`, keeping displacement on the same volumetric grid.

`step.accepted_contact_evidence(name=None)` returns JSON-compatible records from
the contact owners without rerunning search or material updates. Rigid records
include accepted forces, motion coordinates, generalized moment, contact
potential and path work. The moment is the surface-motion generalized force,
not an ambiguously labelled tool reaction. Deformable records include opposite
physical forces, current-coordinate moments about a common origin, active
reference area and explicit selected-candidate coverage. Certified separation
does not invent a closest-point projection. These observations reject live
trial/unaccepted displacement and never promote scientific trust.

`step.accepted_interface_evidence()` observes the bounded elastic bond at the
same accepted station. It evaluates that interface at fixed history, without
reintegrating the body, and records separate side forces, current-coordinate
moments, force/moment balance, interface energy and reference-normal opening /
reference-tangential slip. These are not finite-rotation fracture mode measures.
Pairing metadata reports integration area and both trace areas, explicitly not
a geometric overlap proof. With no bond it returns `None`.

Declare custom histories in `model.step(history=(results.history(...),))` or
pass them when first calling `run`. The ordinary configured route works for
both `run` and `solve_result`. With automatic substeps, pure custom callbacks
observe accepted macro stations only; the physical work ledger still consumes
each accepted subincrement. Checkpoint recovery retains stored histories;
changing their schema mid-run is not an implicit migration.

For imported serial P1/Q1 node identities use `spaces.locate_input_nodes`.
For finite-Hex boundary geometry use a tagged boundary and
`interfaces.reference_trace_from_boundary(..., geometry="bilinear")` where
appropriate. Neither helper identifies coincident nodes by coordinate equality.

### Scope of existing full-run evidence

The migrated installed workflow has completed a separately frozen full-case
run, including owned-array state, batch Q1 search, body/viscosity acceleration,
scheduled checkpoints and accepted reporting. Final nodal/material/contact
evidence and the complete reporting history match the retained prior core
implementation; independent ParaView reading verifies the new single-grid
output. The readable-case refactor additionally passed short-run parity and
final checkpoint recovery. See the bounded
[candidate evidence](internal/release_0.4.2_candidate.md).

This is implementation and delivery acceptance, not independent physical
validation. General stability, discretization/inertia sensitivity and external
solver agreement remain model-specific obligations. The bounded serial scope
and unsupported-combination checks above remain unchanged. Confidential source
laws, meshes and detailed case results remain external.

## Independent geometry scheduling

`contact_execution="overlap"` is an opt-in execution choice for a registered
deformable pair. It evaluates copied contact kinematics in one run-scoped worker
while the owner evaluates the body. Only the owner accesses PETSc, adds forces,
publishes trial state and commits. A rejected branch joins the other branch
before contact rollback; serial contact-before-body error precedence is retained.
Worker shutdown occurs on every `run` exit, including failure. The default is
`"serial"`; MPI and concurrent use of the same Step are not supported.

This requires the geometry operator and its trusted callbacks to have exclusive
state and no access to live FEM fields or the body material. It is not generic
thread safety for arbitrary callbacks. Scheduling does not change the contact
law, assembly order, time grid or checkpoint schema. Small serial/overlap and
restart comparisons are regression evidence, not a full-case speed claim.

## Optional stateless native arithmetic

Source builds may opt into `AGENTFEM_BUILD_NATIVE=1 python -m pip wheel .`.
This compiles a small core extension, without NumPy C headers, for conservative
affine/orientation screens, guarded small determinants, finite-storage scans,
ordered cell-to-node force reduction and bounded pairwise work contraction. Ordinary
pure-Python builds remain supported; a missing compiler or unavailable extension
retains NumPy, LU and Bernstein fallbacks. No compilation occurs during solve.
Native compilation disables floating-point contraction and refuses fast-math
builds; finite-value admission must not be optimized away. CI imports the
installed extension explicitly before its targeted suite, so a silent optional
build fallback cannot count as native-kernel acceptance.

The affine screen uses a near-identity homotopy for orientation and a guarded
Gram-matrix bound for the residual deformation. Uncertified cells still use the
existing affine/SVD/Bernstein admission; a positive mean determinant alone is
never sufficient. It does not change the element formulation or stable step.
Force reduction preserves the input order for every nodal component, validates
indices and finite outputs, and returns detached storage before PETSc publication.
Those stateless arithmetic kernels contain no material law or accepted state.
The optional fused material dispatcher described below owns only temporary,
joined worker batches; accepted-state ownership remains in the Procedure.
The work contraction still uses force times nodal displacement, not a power
substitute. Its signed and absolute products use bounded pairwise reduction;
floating-point reduction grouping may differ from NumPy and is checked against
the same absolute-product roundoff budget. Small determinants retain the exact
six-product order and the LU fallback for cancellation, overflow and tiny inputs.
Finite-storage scans preserve NaN/Inf rejection, including unaligned buffers.

The response Step summary reports `affine_geometry_screen`. Checkpoint physics
identity remains unchanged by this implementation-only optimization. Builds and
runtime artifacts must still be recorded in performance/reproducibility evidence.
Native and reference results must be compared at equal accepted stations before
making a throughput claim; native availability alone proves neither speed nor
independent physical validity.

For packaged material-independent body and Q1 geometry acceleration, including
the trusted compiled-material ABI and ordinary factories, see
[built-in explicit execution](native_explicit_execution.md).
