# Numerical controls and input/output contracts

These controls extend the existing workflow. They do not silently change the
physical problem or replace the declared numerical method.

## Mesh and field inputs

`mesh.from_arrays` requires a 2D integer connectivity array, nonnegative global
indices and finite real coordinates shaped `(points, dimension)`. Integer
coordinates become float64; explicit float32/float64 are retained. The supplied
coordinate element is preserved, including high-order geometry. Input errors
are exchanged across MPI ranks before collective mesh construction.

`expressions.interpolate` restores original field values if interpolation fails
or produces non-finite output. Numerical interpolation failure on any rank
rolls back values on every rank. Expression structure and parameters should be
consistent across ranks. Known-field interpolation is distinct from symbolic
unknowns used in a nonlinear residual.

## Time updates and matrix reuse

The fixed-step transient field procedure snapshots fields before updating loads.
A failed load update or solve restores the fields and re-applies loads at the
last accepted time. Load-update callbacks should set values from time, rather
than accumulate irreversible side effects. If load restoration fails, the
original exception retains a note explaining that failure.

`PreparedLinearProblem` reuses its matrix for RHS-only or prescribed-value
changes. After a bilinear coefficient changes, call `refresh_matrix()` before
solving again. Mesh, space or constrained-dof changes require a new prepared
problem. `summary()` reports matrix and RHS assembly counts. Refresh uses the
existing solver configuration; no unconditional preconditioner-reuse flag is
introduced.

The generalized-Maxwell adaptive path already uses step doubling and rollback.
Its next accepted step is now limited by a bounded proportional error controller
as well as the existing iteration controller. `time.error_step_factor` computes
`safety*(tolerance/error)**(1/(order+1))`, bounded by the supplied factors. It
proposes a step size; it does not itself accept a solution. This is a local error
estimate, not a guarantee on accumulated global error or speedup for every case.

## Transport and pressure

`operators.intrinsic_time_scale` accepts optional `time_step`, adding `(2/dt)^2`
to the inverse-squared advective/diffusive scale. Stagnation points avoid zero
denominators; the SUPG contribution still depends on the streamline test term.
This scale does not guarantee monotonicity, and the discrete transient residual
must remain consistent with the chosen time scheme.

`solvers.attach_nullspace(A, modes, rhs=b)` accepts explicit physical null modes
in the assembled matrix layout, orthonormalizes copies and tests them against A.
For symmetric systems the optional RHS check detects incompatible load/flux.
It does not pin pressure or silently project a physical imbalance away. The
caller destroys the returned PETSc NullSpace. A constant pressure mode is only
appropriate when boundary conditions leave that reference free; pressure or
traction conditions may remove it. Nonsymmetric problems require separate
left-nullspace treatment, not the optional symmetric-RHS check.

## Nonlinear convergence and failure information

In standard finite-strain J2, the normalized path coordinate remains in
`[0, 1]`, while the resolved Step amplitude may reverse sign or exceed one.
Ordinary strong reference values and natural loads use that amplitude. A
fixed contact tool has no motion coordinate to update. Prescribed moving tools
still enforce the admitted range of their motion definition; supporting a
signed traction amplitude does not authorize extrapolating a tool schedule.
`PrescribedValuePath` accepts finite signed scaling for constant/field values.
When independent boundary histories are present, its input remains their
normalized coordinate. Standard finite-J2 Steps continue to reject those
independent histories, requiring one explicit Step amplitude.

Affine Newton backtracking now requires the stated sufficient residual decrease
(or meeting the absolute residual tolerance). A merely smaller residual no
longer bypasses that condition. Existing load cutback, material trial/commit and
restart paths remain in use.

Linear nonconvergence raises `diagnostics.ComputationalFailure`, retaining
RuntimeError compatibility. `exception.as_dict()` includes a stable code,
solver stage, PETSc reason, iterations, residual and suggested next checks.
These are diagnostics, not a claim that one root cause has been proven.

### Experimental constant-normal plane augmentation

Finite-strain J2 accepts an opt-in outer augmented-Lagrangian iteration through
the ordinary Step. The default remains penalty contact:

```python
step = model.step(
    target=displacement, material=material,
    incrementation=steps.fixed(4),
    normal_augmentation=steps.normal_augmentation(
        penetration_tolerance=1e-7,
        complementarity_tolerance=1e-6,
        length_unit="m", pressure_unit="Pa",
        maximum_augmentations=120,
    ),
)
```

Declare a named `rigid_contact_pair` with one fixed or translating analytical plane and a
scalar penalty. The bounded implementation uses coalesced nodal reference-area
weights on serial P1 tetrahedra and ordinary strong boundaries. A multiplier
whose normal motion is already fixed by a strong boundary is rejected; the
software does not arbitrarily divide reactions between redundant constraints.
Rotating or curved tools, friction, general MPC, mixed fields, higher-order/hex cells, custom
contact quadrature and changing contact partitions are not admitted here.
This is not a mesh-independent mortar stability or general contact claim.

One narrow affine/MPC route also uses the same `normal_augmentation` option.
It requires one exact periodic constraint and **disjoint nonzero reduced normal
row supports** after `u = T q + u_bar`. The sparse admission rejects contact
nodes whose normal motion is prescribed, as well as overlapping reduced
supports; it does not delete rows or arbitrarily distribute contact pressure.
Overlap may be mathematically solvable, but is outside this sufficient bounded
admission. The condition proves discrete row independence, not mesh-uniform
inf-sup stability. A plane imposed against immovable periodic corners therefore
does not become admissible just because a smaller increment is used.

`examples/finite_j2_periodic_augmented_plane.py` gives an ordinary Step example
with an interior contact patch, plastic loading/unloading/recontact, accepted
displacement frames and restart. The same affine Procedure performs Newton,
cutback, joint State acceptance and checkpointing. Its `checks` record outer
updates and the largest inner Newton count separately. Signed relative normal
work, its discrete absolute bound and open-gap contribution appear under
`<pair>_normal_*` histories; they are not `contact_potential`, dissipation or
automatic energy verification. Ordinary MPC/tool generalized work remains in
the existing constraint histories.

A prescribed translation uses the existing `prescribed_rigid_motion_schedule`
on the rigid body, with normalized endpoints 0 and 1 and no separate amplitude.
For strong boundaries the ordinary Step amplitude owns the tool coordinate, including unloading and
recontact. Multipliers stay on the same material-side reference nodes; every
trial projects the current positions against the current tool pose. This does
not require transferring history between tool facets, nor license such transfer.
Motion along the plane is frictionless and contributes no work.
The affine route instead uses its normalized macro-path coordinate for the
tool schedule and the declared deformation-gradient path for the macro motion;
it does not accept a second independent Step amplitude.

For gap `g` (negative in penetration), the inner solve uses fixed multiplier
`lambda` and pressure `p = max(0, lambda - penalty*g)`. Only after total Newton
equilibrium may the outer iteration update `lambda`. Acceptance requires both
penetration and complementarity: `max(abs(lambda*g), abs(p*g))`. The latter
tolerance has pressure-times-length units (Pa m in the example), not force
units. Unit strings declare the model's coherent units; they do not convert
values. The penalty retains pressure-per-length units. A small penalty can
require many outer iterations; a large one can worsen linear conditioning.
Neither a faster solve nor universal penalty independence is promised.
Do not confuse the load increment with Newton's `minimum_step_length`: the
latter limits residual backtracking along one Newton correction. A very stiff
contact onset can require a shorter correction even when the load step is
adequate. The public `NewtonSolverOptions` exposes that floor explicitly; it
is never reduced silently. A smaller floor permits more trial evaluations,
not a weaker convergence or penetration tolerance.
For automatic incrementation, Newton difficulty is the maximum inner Newton
iteration count, not their sum over all augmentations. The latter remains in
the total-work report. This prevents repeated easy inner solves from
spuriously shrinking the load step. An exhausted outer budget still rejects
the increment and follows the ordinary cutback policy.

Materials always start their trial update from the same accepted state during
the entire outer loop. One accepted increment jointly commits displacement,
plastic history, multiplier and prescribed tool pose. Newton/outer failures restore that boundary;
an inner Newton success cannot be checkpointed. The ordinary serial
`save_checkpoint`/`load_checkpoint` stores the multiplier basis, policy and
accepted configuration and rejects incompatible restoration.

The October 19 candidate now uses the original mesh vertex coordinates for
linear P1/Q1 contact traces, after identifying each vertex's displacement
block. This removes coordinate-tabulation roundoff without moving the mesh,
snapping to a tool or loosening an overlap limit. Experimental checkpoints
created before this correction can have a different trace fingerprint and
are intentionally rejected. Continue those runs with their original frozen
candidate, or start a new run; there is no automatic multiplier transfer.

Result reports the actual contact force and augmentation counts. Reference-area
pressure is not Cauchy pressure. The signed AL functional is **not** physical
contact stored energy. The accepted ledger now reports signed numerical
`normal_contact_work` separately, together with a non-cancelling
`normal_contact_work_bound` and the positive-gap component
`normal_contact_open_gap_work`. For matched constant-normal plane samples these are
`sum(w * (p_old+p_new)/2 * delta_gap)` and, for the bound, the same sum with
`abs(delta_gap)`. For a fixed tool positive work enters the structure. For a
translating tool this is **relative-gap work**: structural contact work equals
tool work plus relative-gap work. Tool work is already part of external work,
so it must not be added twice. Opening/closure can
produce nonzero endpoint work even with exact complementarity at both ends.
The bound applies only to this discrete work, **not** to the continuous-path
integration error or general contact accuracy.

The original `mechanical_energy_residual` remains external work minus stored
material energy and plastic dissipation. The additional
`normal_contact_corrected_energy_residual` includes signed normal work;
neither residual is forced to zero. There is no AL `contact_potential`
history. Fixed-tool coordinate work is zero, which alone does not prove energy
closure. Compare both residuals, the absolute-work bound, and increment
refinement; cancellation can make an inaccurate coarse path look balanced.
These histories do not automatically promote the result to verified.

Only jointly accepted endpoints enter this ledger. Failed attempts restore it
with displacement, plastic state and multipliers. New checkpoints retain and
check its complete accepted-path coverage; legacy checkpoints without prior
work remain energy-unavailable after continuation. No missing history is
reconstructed. Moving-plane checkpoints additionally validate the motion
identity and resolved amplitude against the Step coordinate before publishing
the restored state. Fixed-plane v1 snapshots retain their format. These
semantics do not enable rotating/curved AL or MPI.
`results.export_accepted_contact_points(...)` exports the same augmented pressure
used in equilibrium, with its nodal reference-area location and enforcement
recorded. It does not substitute the old `penalty*penetration` pressure.
Increment records expose total Newton work, the maximum inner Newton count,
outer augmentations and line-search residual evaluations. A rejected search
reports its declared minimum step and trial count; it does not diagnose every
such failure as a tangent error or automatically relax the search floor.
In-memory rollback snapshots use immutable numerical arrays. JSON conversion
belongs to the on-disk checkpoint boundary, not each attempted increment.

Run `examples/finite_j2_augmented_plane.py` from the source checkout
for loading, output and restart. Tests compare a homogeneous elastic limit to
an independent nominal-stress formula, a plastic limit to a prescribed-motion
solution, and interrupted to uninterrupted loading. These are bounded numerical
checks, not validation of arbitrary industrial forming.

`examples/finite_j2_translating_augmented_plane.py` adds prescribed compression,
unloading and recontact through the same ordinary API. It exports accepted
pressure, tool work, relative-gap work, raw/contact-aware residuals and a
restart checkpoint. `tests/test_translating_normal_augmentation.py` compares
this path with a scalar finite-J2 reference and 16/32/64/128 increments.
Both pressure and plastic history are compared; balancing total force alone
is insufficient. The independently checked moving-tool tangent freezes the
old material state and multiplier, not the current geometry.

The nonuniform Poisson-coupled plastic reference additionally compares normal
pressure with independently imposed displacement reactions on each mesh.
A displacement tolerance is not a pressure-error tolerance: pressure accuracy
must be checked separately against contact-tolerance and mesh refinement.
See `tests/test_translating_augmentation_plastic_reference.py`; its tighter
benchmark tolerances do not change the software defaults.

For this bounded plane route, `steps.normal_augmentation(...,
acceleration="aitken")` optionally accelerates the outer multiplier updates.
It uses reference-area-weighted dynamic relaxation, bounded to `[0.1, 10]`,
with resets on active-set changes and at every attempted increment. Degenerate
or invalid secants revert to the ordinary update. The inner Newton method,
penalty, loading and equilibrium/penetration/complementarity tolerances do not
change. Failure still invokes the ordinary rollback/cutback path; there is no
universal convergence or speed guarantee.

The default remains `acceleration="none"`. Accelerated checkpoints carry a
distinct numerical-policy identity and do not silently load into a different
policy. Secant history is not physical State and never persists across accepted
boundaries. Increment diagnostics record the actual relaxation factors and
reset count. The public translating-plane example accepts
`--acceleration aitken`; use the same option when resuming its checkpoint.

The partial-contact regression also exercises a tilted fixed plane under
plastic loading, unloading and recontact. It compares every accepted
displacement, material state, pressure, active set and energy record, restores
a loaded checkpoint, and injects failure after pressure commit to check atomic
rollback and retry. An independent equality-constrained elastic active-set
reference checks both acceleration policies on two meshes. These tests cover
a changing subset of fixed material-side nodes, not multiplier transfer to a
new trace or curved-tool geometry.

`tools/benchmark_multiplier_acceleration.py` compares the two policies on the
32-increment cyclic example after warming both paths and alternating run order.
The recorded serial Mac test reduced median solve time from 2.058 s to 1.404 s
(31.8%), with outer updates 106 → 45 and Newton/linear solves 209 → 140.
Accepted displacement history, final material/pressure and energy histories
passed the declared comparison thresholds. This small implicit-plane result
does not establish a speedup for explicit composite damage, general curved
contact, MPI or other machines. The benchmark evidence is retained under
`evidence/contact/2026-10-20-multiplier-acceleration.json`.

Accepted contact-point exports preserve the normalized Step coordinate as
`coordinate_value` and record `resolved_load_amplitude` separately, with
`tool_motion_factor` for each moving pair. For a cyclic path, coordinate 0.5
may mean tool factor 1 and coordinate 0.75 may mean factor 0. Comparing these
different quantities would falsely reject valid intermediate frames.
Stale or nonfinite tool factors still reject before output creation.

The split outer-multiplier/inner-equilibrium design follows the established
[segregated normal-contact formulation](https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_theory.06.087.html).
This is methodological context, not an implementation equivalence claim.

The independent cyclic finite-J2 stop reference additionally checks stored
energy, plastic dissipation and accepted work at 16/32/64/128 increments.
See `tests/test_normal_contact_energy.py`. Numerical contact discontinuity work
is also distinguished from physical stored energy in the
[Abaqus contact-energy example](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEEXARefMap/simaexa-c-contactenergy.htm).
This motivates separate reporting, not equivalence with Abaqus algorithms or
its output variables.

`tools/verify_normal_augmentation_spatial.py` adds a nonuniform manufactured
finite-Hencky elastic-limit solution on three refined tetrahedral meshes.
It measures displacement and gradient errors, contact force and nodal pressure
errors separately. The observed displacement/gradient orders are approximately
two/one; the pressure is not exact and requires its own refinement evidence.
This benchmark does not establish general multiplier-space stability or
plastic-contact spatial accuracy.

`tools/benchmark_normal_augmentation.py` compares explicit penalty candidates
on this example using warmups, shuffled repeats, the complete four-increment
Step and all accepted material/field snapshots. Failed candidates remain in
the report. It is a reproducible numerical-control experiment, not automatic
tuning or a general acceleration benchmark. Both `--penalty` and
`--minimum-step-length` are visible example options. Compare the complete
accepted path, not only the final displacement, before adopting different
controls for a history-dependent material.

This distinction is also made in the
[COMSOL contact settings](https://doc.comsol.com/6.4/doc/com.comsol.help.sme/sme_ug_modeling.05.147.html):
AL penalty affects outer convergence and can compromise robustness when too
large. The software does not infer a universal value from one successful run.

The separation of nodal and integral multiplier spaces follows the caution in
the [GetFEM contact documentation](https://getfem.org/userdoc/model_contact_friction.html):
choosing a multiplier space requires stability analysis, not merely adding an
outer loop.

## Point coverage

```python
sample = results.sample_points(field, points, missing="nan", return_info=True)
print(sample.summary())
values = sample.values
```

`sample.located` records mesh coverage; `sample.finite` records finite values.
The summary separates missing points from non-finite values inside the mesh.
Empty requests have undefined coverage (`None`), not a misleading 100% score.
The default still returns the original NumPy array and raises on missing points.
For discontinuous fields, the existing deterministic cell-owner convention is
unchanged; sample either side of an interface for one-sided values.

Pressure reference selection:

| Situation | Treatment |
|---|---|
| Full velocity Dirichlet conditions leave constant pressure free | Attach the constant-pressure null mode; check boundary-flux compatibility. |
| Reporting requires a specified mean pressure | After a gauge-free solve, shift pressure by the difference between the target mean and its volume-weighted mean. This does not change velocity. |
| A physical pressure/traction condition already fixes the reference | Do not add a constant-pressure nullspace or an extra pressure pin. |
| Using a single pressure dof as a numerical reference | Record the dof/location and check velocity sensitivity; it must not replace missing physical boundary conditions. |

A volume-weighted mean is the pressure integral divided by the domain measure,
not the arithmetic mean of nodal coefficients. Nullspace treatment does not
repair an unstable velocity/pressure element pair or inconsistent flux data.
