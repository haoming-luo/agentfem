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

Result reports the actual contact force and augmentation counts. Reference-area
pressure is not Cauchy pressure. The signed AL functional is **not** physical
contact stored energy: the full mechanical-energy ledger is marked unavailable
until the finite-penetration enforcement-work residual has an explicit bound.
Fixed-tool coordinate work is zero, which alone does not prove energy closure.
`results.export_accepted_contact_points(...)` exports the same augmented pressure
used in equilibrium, with its nodal reference-area location and enforcement
recorded. It does not substitute the old `penalty*penetration` pressure.

Run `examples/finite_j2_augmented_plane.py` from the source checkout
for loading, output and restart. Tests compare a homogeneous elastic limit to
an independent nominal-stress formula, a plastic limit to a prescribed-motion
solution, and interrupted to uninterrupted loading. These are bounded numerical
checks, not validation of arbitrary industrial forming.

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
