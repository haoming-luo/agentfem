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

### Experimental fixed-plane normal augmentation

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

Declare a named `rigid_contact_pair` with one fixed analytical plane and a
scalar penalty. The bounded implementation uses coalesced nodal reference-area
weights on serial P1 tetrahedra and ordinary strong boundaries. A multiplier
whose normal motion is already fixed by a strong boundary is rejected; the
software does not arbitrarily divide reactions between redundant constraints.
Moving tools, friction, MPC, mixed fields, higher-order/hex cells, custom
contact quadrature and changing contact partitions are not admitted here.
This is not a mesh-independent mortar stability or general contact claim.

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
plastic history and multiplier. Newton/outer failures restore that boundary;
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
`normal_contact_open_gap_work`. For matched fixed-plane samples these are
`sum(w * (p_old+p_new)/2 * delta_gap)` and, for the bound, the same sum with
`abs(delta_gap)`. Positive work enters the structure. Opening/closure can
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
reconstructed. These semantics do not enable moving/curved AL or MPI.
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
