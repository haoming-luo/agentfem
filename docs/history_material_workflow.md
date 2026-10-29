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
strain. It is not a hydrogel-specific solver or automatic multiaxial inference
from a uniaxial calibration.
Displacement must be a standalone H1-conforming vector field with one component
per spatial dimension. DG fields require interface terms, and a mixed-field
subspace cannot silently discard its pressure equation. Both are rejected.

## Start with the complete example

[`examples/history_material`](https://github.com/haoming-luo/agentfem/tree/main/examples/history_material)
contains four files: `case.py`, `material.py`, `agentfem.toml`, and `README.md`.
With an installation exposing `mechanics.FiniteStrainMaterialStep`, run:

```sh
agentfem check
agentfem run --name viscous -- --material viscoelastic
agentfem run --name damage -- --material damage
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
`step.save_checkpoint(path)` requires an accepted state; `load_checkpoint(path)`
checks mesh/space, schema, parameters, loading, duration, response arrays and
the accepted/attempted time ledger before changing live state. It also checks
displacement/gradient consistency and the prescribed boundary values at the
saved time. Restore failure is atomic. This first checkpoint format is serial
and same-layout only.
Changing rank count or remeshing is not supported by this Step.

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

## Evidence and limits

`tests/test_finite_strain_material_step.py` checks fixed-old-state material
tangents, superposed rotations, degree-one/two 2D/3D patches against material points,
nonuniform traction equilibrium and assembled tangent directions, physical-time
sensitivity, cutback, restart equivalence, corrupt restore rejection and real
VTK reading/warping. `tests/test_live_xdmf.py` also checks process interruption
and concurrent reading of closed published frames. These verify the bounded
workflow, not arbitrary user constitutive physics.

Unsupported combinations fail explicitly: MPI, contact/weak boundaries, MPC,
regional materials, mixed pressure, plane stress, temperature histories,
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
