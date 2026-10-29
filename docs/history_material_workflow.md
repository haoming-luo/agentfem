---
title: "From local history materials to structural simulation"
description: "Use one bounded finite-strain structural workflow with replaceable local material updates, physical time, atomic restart and readable multi-field output."
---

# Local history materials in a structural model

The experimental finite-strain history route lets a material author supply a
local update, not another global solver. An ordinary case still uses
`Study → Model → material/boundaries → model.step → SimulationResult`.
The current admission is serial, one homogeneous material, strong prescribed
displacements and reference dead loads, in 3D or explicitly declared 2D plane
strain/limited plane stress. It is not a hydrogel-specific solver or automatic multiaxial inference
from a uniaxial calibration.
Displacement must be a standalone H1-conforming vector field with one component
per spatial dimension. DG fields require interface terms, and a mixed-field
subspace cannot silently discard its pressure equation. Both are rejected.

## Start with the complete example

[`examples/history_material`](https://github.com/haoming-luo/agentfem/tree/main/examples/history_material)
contains four files: `case.py`, `material.py`, `agentfem.toml`, and `README.md`.
With an installation exposing `mechanics.FiniteStrainMaterialStep` and
`constitutive.PlaneStressReduction` (checked by the project manifest), run:

```sh
agentfem check
agentfem run --name viscous -- --material viscoelastic
agentfem run --name damage -- --material damage
agentfem run --name plane-stress -- --material viscoelastic --assumption plane_stress
```

Only the local material changes. Both use the same plane-strain structure,
hold/reversal schedule, quadrature, Newton procedure and output path. The
example material constants are synthetic; neither example is a calibrated
hydrogel law, robust rubber formulation or mesh-objective fracture model.

The model-level entry is:

```python
model.material(material)
step = model.step(
    target=u,
    duration=2.0,
    time_unit="s",
    amplitude=cycle,
    incrementation=steps.fixed(16),
    output=run.artifact("results.xdmf"),
)
result = step.solve_result()
run.publish(result)
```

`time_unit` records the chosen consistent unit; it does not convert material
parameters. `duration` is required and positive. The amplitude argument is a
normalized coordinate, not physical seconds. At coordinate `a`, material time
is `duration*a`; an increment from `a0` to `a1` has physical
`dt=duration*(a1-a0)`. Tabulated reversal/hold knots are visited exactly, even
when a requested fixed or automatic increment would cross a knot.

## The material contract

Reuse `constitutive.UserMaterial`, `MaterialStateSchema`, `MaterialPointInput`
and `MaterialPointOutput`; there is no second neural-material interface.

| Author supplies | Meaning |
| --- | --- |
| `name`, `state_schema` | Named scalar/tensor internal variables, initial values, optional units and output aliases |
| `tangent_convention` | Exactly `MaterialTangentConvention.first_piola_deformation_gradient()` for this structural route |
| `summary()` | Stable JSON identity including implementation version, material parameters and scientific assumptions; not mutable counters |
| `update(point)` | Local stress, algorithmic tangent, trial state, optional stored energy and suggested time scale |
| Optional `update_batch(request)` | Same ordered point responses; no hidden commit or shared mutable history |

Inputs contain committed `F_n`, trial `F`, committed state, endpoint physical
time, positive `dt` and the Step's optional material-property vector. The local
contract also has optional thermal/external fields, but this initial structural
route does **not** supply their histories; requests are rejected rather than
silently ignored.

The existing neutral output contract returns **Cauchy stress** and an explicitly
declared **9×9 algorithmic `dP/dF`**, in row-major `iJ,kL` order. The quadrature
driver converts stress to first Piola with `P=det(F)*sigma*F^(-T)`. The tangent
must differentiate the complete discrete material update while holding old
state fixed, not just its elastic predictor. A spatial UMAT Jacobian is not
this tangent. A material returning `P` in the Cauchy slot is incorrect.

The Procedure assembles `∫ P:grad(v) dV - external_work` and the matching
reference tangent. Plane strain embeds the in-plane gradient in a 3×3 `F`
with `F33=1` and zero out-of-plane shear. It neither solves plane stress nor
guesses missing transverse constitutive behavior. Initial displacement is zero;
the material author must choose a compatible initial reference state.

## Explicit plane-stress reduction

For a compatible **three-dimensional** law, use a 2D `plane_stress` Study and
add this option to the same `model.step(...)`:

```python
plane_stress=constitutive.PlaneStressReduction(symmetry="reflection_symmetric")
```

The declaration asserts that the material and its accepted history preserve
reflection symmetry about the model plane. It is not inferred from one stress
sample or from a uniaxial calibration. The current bounded reduction admits
block-diagonal `F` with positive `F33`; at every quadrature point it solves
`P33=0` using a safeguarded local Newton iteration on `log(F33)`. Every candidate
reuses exactly the same old state, old full `F`, endpoint time and physical `dt`.
Transverse shear stress and tangent coupling are checked. A positive transverse
algorithmic stiffness is required; failure rejects the global attempt and the
usual increment policy controls cutback.

For in-plane components `a=(11,12,21,22)`, the derivative supplied to structural
assembly is `A_aa - A_a33 * A_33a / A_3333`, using the **full discrete update's**
`dP/dF`. The original material output is not modified: Cauchy stress is converted
using the actual 3D determinant and effective `F`, not an `F33=1` surrogate.
`constitutive.reduce_plane_stress(material, point, reduction=policy)` exposes
the same uncommitted material-point calculation, returning the original 3D
response and a separate 4×4 reduced tangent. Scalar local iterations currently
call `update`; they do not imply a vectorized material-batch speedup.

Use `constitutive.check_material_tangent(material, point)` for the complete
3D law, then `constitutive.check_plane_stress_tangent(material, point,
reduction=policy)` for the reduced discrete derivative. The latter independently
resolves transverse equilibrium for each positive/negative in-plane
perturbation and returns explicit 4×4 component-order evidence. It does not
replace missing tangents, commit history, switch to one-sided differences or
claim experimental validation. For nonsmooth laws, check each smooth branch
separately; choose perturbations resolvable above the local solve tolerance.

Effective 3D `F`, including accepted thickness stretch, belongs to the shared
trial/commit/rollback and checkpoint lifecycle. The result includes raw `F`
and recovered `F_CELL`; its `(2,2)` component is the thickness stretch. Default
visualization remains a 2D reference surface, not an extruded thickness mesh.
2D forces/energies are per unit reference thickness. Local relative tolerance
uses a physical stress scale (not an arbitrary unit-sized floor); optional
absolute tolerance has the material's stress unit and is part of restart identity.

This is **not** general anisotropic plane stress, finite-rotation shells,
membrane directors, mixed incompressibility, or automatic multiaxial physics.
Out-of-plane rotation/shear and singular or negative transverse stiffness are
rejected. A compatible initial reference state is still the author's
responsibility. A 1D calibrated law still needs explicit transverse/shear/
volume assumptions before it can be used here.

## Who owns history and failure

- The material computes a candidate response from the input old state. It must
  not mutate an independent, hidden global history.
- The framework evaluates every Newton/line-search candidate from the same
  committed state and `F_n`. A successful global increment commits once.
- Local failure, failed equilibrium or material-requested cutback restores
  displacement, material state, stress, tangent and loading values together.
- Automatic cutback subdivides physical time as well as the normalized path.
  It does not imply a material-specific error estimator or automatic local
  substepping. A provider implementing substeps must return the derivative of
  the **whole** substepped update.
- Initialization and restart do not integrate a fictitious tiny time step to
  reconstruct stress. Accepted response arrays are part of restart state.

`step.solve(until=0.5)` pauses at normalized coordinate 0.5.
`step.solve_result(until=0.5, output="first.xdmf")` also publishes that accepted
segment through the standard result interface. A paused result has
`status="partial"`, `equilibrium_converged=True` and `path_complete=False` in
its solve metadata. The existing `converged` flag remains a full-path flag.
The physical time and requested/accepted coordinates are explicit. At an
already accepted coordinate, this method can publish current state without
reintegration; it cannot synthesize past fields. Values before the accepted
coordinate, zero, nonfinite values and booleans are rejected before output is
opened. Continuation writes to a new path; preserve or join both segments.
Choose an existing path station to preserve the original increment partition;
an extra pause station is a different time discretization for history laws.
`step.save_checkpoint(path)` requires an accepted state; `load_checkpoint(path)`
checks mesh/space, schema, parameters, loading, duration, response arrays and
the accepted/attempted time ledger before changing live state. It also checks
displacement/gradient consistency and the prescribed boundary values at the
saved time. Restore failure is atomic. This first checkpoint format is serial
and same-layout only.
Changing rank count or remeshing is not supported by this Step.

Accepted fields are observations, not editable initial conditions. Do not modify
displacement, committed history, stress/tangent, physical-time position or live
loading constants through their array views. The Step detects such changes at
continuation, reaction/output and checkpoint boundaries, including after an
accepted-state observer. Prescribed targets and the increment policy must also
remain consistent with the Step's construction identity. To run a different
problem, construct another Step; do not reuse an old checkpoint identity.
If external code has already corrupted live state, restore a valid checkpoint
into a fresh compatible Step. Detection does not reconstruct overwritten arrays
or authenticate arbitrary archive edits. Ordinary numerical failure still uses
the existing atomic trial rollback, without this user intervention.

## Results that can be viewed directly

`solve_result(output="results.xdmf")` writes accepted frames through the shared
`io.ResultWriter`. Each time has one reference mesh with three-component
visualization displacement and compatible cell fields. In ParaView, use
**Warp By Vector**, then color by `S_CELL` or a recovered state. No hand-edited
XML, Extract Block operation, node merge or connectivity code belongs in the
case.

This route uses immutable closed frame files under `results-data/` and updates
the small `results.xdmf` index atomically. The index is readable during the run;
refresh the reader to see new frames. Reference geometry and topology are
shared, not repeated in each frame. Preserve the index and its data directory.
The writer's `artifacts()` inventories only files referenced by its current
published view, including a partial view after interruption.

Raw `S`, `P` and state remain integration-point fields in `SimulationResult`.
`*_CELL` is a reference-geometry-weighted DG0 average, not nodal extrapolation,
interelement smoothing or a raw quadrature plot. Stress measure, recovery and
state units are recorded. `RF` is the full nodal residual; select constrained
DOFs when extracting reactions. Optional `SENER` is included in the final
result only when defined at every point there; the live visualization schema
does not fabricate zero energy for earlier points where a provider omitted it.
Stored energy is not total internal work or proof of energy balance.
State aliases and energy-component names must not collide with displacement,
stress, reaction or recovered `*_CELL` names. Collisions fail during Step
construction rather than silently replacing fields during publication.

A restored Step writes a new visualization segment and records its restart
source. It cannot invent fields that were not saved in the old run. Compatible
saved segments can be joined with the existing zero-copy continuation view.
Numerical acceptance precedes output: an I/O failure preserves the last
accepted restartable state and the already published frames.
Progress-reporting failures are not material rejections and do not trigger
numerical cutback; a secondary reporting error never hides the original failure.
Completed-result and reaction publication also check accepted displacement and
material identity. Changing a solved field or material parameter externally
cannot silently turn stale stresses into a new accepted result. Recursive
result publication during an active solve is rejected before opening a writer.

## Evidence and limits

`tests/test_finite_strain_material_step.py` checks fixed-old-state material
tangents, superposed rotations, degree-one/two 2D/3D patches against material points,
nonuniform traction equilibrium and assembled tangent directions, physical-time
sensitivity, cutback, restart equivalence, corrupt restore rejection and real
VTK reading/warping. `tests/test_live_xdmf.py` also checks process interruption
and concurrent reading of closed published frames. These verify the bounded
workflow, not arbitrary user constitutive physics.

`tests/test_history_plane_stress.py` additionally checks independent scalar
force-controlled history solutions, a separate Neo-Hookean thickness solution,
the condensed local and assembled structural tangents on both smooth history
branches, degree-one/two patches, unit rescaling, in-plane rotation, cutback,
thickness-aware restart, readable interrupted/continued frames, nonuniform
damage spatial refinement, viscous time refinement, independent uniaxial J2
local reduction and real VTK/Warp reading of stress and `F_CELL`.
None of these tests calibrates a user material or promotes arbitrary laws.

Unsupported combinations fail explicitly: MPI, contact/weak boundaries, MPC,
regional materials, mixed pressure, undeclared/general plane stress, temperature histories,
undeclared/spatial tangents, arbitrary UMAT conversion and declarative
`OutputPlan`. The required Newton policy raises on exhausted increments;
non-raising failure mode is not admitted yet. Nearly incompressible materials
and softening require appropriate formulation/regularization and mesh/time
studies; this displacement route does not remove locking or localization.

Material authors should first run `constitutive.check_material_tangent` and
independent loading, unloading, hold, cycle, stress-measure and rotation checks.
For a nonsmooth constitutive switch, test each smooth branch and the switch
separately; a centered difference across the switch is not its one-sided
algorithmic derivative. Solver completion remains distinct from verification
and experimental validation.
The full 3D tangent checker reports `one_sided_components` if a negative
perturbation would invert `F`; such evidence is labeled mixed difference, not
central difference. The reduced checker instead rejects inadmissible
perturbations. Both normalize error arithmetic before forming norms, so stress
unit scaling does not turn a wrong tangent into a passing zero-error check.
Perturbations too small to change the stored gradient are rejected explicitly.

## Extension authors

Use a Step provider only when introducing a genuinely different global
procedure. A local material in the admitted scope does not need one. The
upgrade checker recognizes an explicit module-level
`StepProvider(lower=function_name, ...)` and does not recommend replacing that
lowerer's necessary `add_step` by recursive `model.step`. Custom output inside
an extension is directed to `io.ResultWriter`, not its enclosing
`solve_result`. There is no blanket exemption for a directory named `providers`;
indirect/dynamic registration still needs manual review.

## Design references

The separation follows established external-material practice, not a new
constitutive algorithm: [MGIS/FEniCS state update and revert](https://thelfer.github.io/mgis/web/FEniCSBindings.html)
and [DOLFINx external constitutive operators](https://a-latyshev.github.io/dolfinx-external-operator/demo/05_demo_plasticity_mohr_coulomb.html)
keep local integration separate from global residual/tangent assembly.
AgentFEM reuses its own existing material and state contracts; it does not add
an MGIS or JAX runtime dependency for this workflow.

The transverse solve and tangent condensation follow the established local
plane-stress strategy discussed in [Ahmadi et al., plane stress in finite
deformation hyperelasticity](https://arxiv.org/html/2410.22562v1). Here it is
applied to the declared algorithmic tangent of a fixed-old-state history update;
the paper's hyperelastic evidence alone does not validate history materials.
