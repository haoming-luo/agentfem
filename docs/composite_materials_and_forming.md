# Composite materials and forming

AgentFEM treats composite mechanics as three independent assets:

```text
constitutive material + assignment orientation + section/ply placement
```

The Study still declares the physical analysis and `model.step(...)` remains
the only ordinary solution entry. A carbon/epoxy lamina is therefore not
duplicated for every angle, and a laminate section is not disguised as a new
material family.

## Oriented elastic solids

Two-dimensional plane-stress orthotropy and general reduced anisotropy are
available together with general and engineering-constant three-dimensional
anisotropy. Voigt conventions are explicit:

- 2D strain: `(epsilon_11, epsilon_22, gamma_12)`;
- 2D stress: `(sigma_11, sigma_22, sigma_12)`;
- 3D strain: `(epsilon_11, epsilon_22, epsilon_33, gamma_23, gamma_13, gamma_12)`;
- 3D stress uses the same component order with tensor shear stresses.

Orientation is an assignment rather than a constitutive constant:

```python
from agentfem import constitutive, materials

lamina = constitutive.orthotropic_plane_stress_2d(
    ex=135e9, ey=10e9, nuxy=0.30, gxy=5e9, density=1600.0,
)
frame = materials.MaterialFrame.from_angle(45.0, name="ply_45")
model.material(lamina, orientation=frame)
```

For reviewed component output, request `S_MATERIAL` and `E_MATERIAL` through
the ordinary small-strain result-field selection. Global `S`/`E` remain the
default so visualization and balance conventions do not change silently.

`MaterialFrame` rejects non-orthonormal and left-handed bases. Constant 2D and
3D frames are FEM-integrated through the standard elasticity operator. A
spatial orientation field, convected finite-strain orthotropy, and anisotropic
thermal expansion remain later capabilities and must not be inferred from this
route.

## Plies and laminate sections

`Ply` stores a stable name, material, thickness, degree-valued angle, and the
requested number of through-thickness integration points. `LaminateSection`
owns ordered placement, reference-surface offset, the classical laminate
`A`, `B`, and `D` matrices, generalized resultants, and per-section-point
strain/stress recovery:

```python
section = materials.laminate(
    [
        materials.ply(lamina, 0.125e-3, angle=0, name="bottom_0"),
        materials.ply(lamina, 0.125e-3, angle=90, name="lower_90"),
        materials.ply(lamina, 0.125e-3, angle=90, name="upper_90"),
        materials.ply(lamina, 0.125e-3, angle=0, name="top_0"),
    ],
    name="cross_ply",
)
response = section.evaluate(
    membrane_strain=(1e-3, 0.0, 0.0),
    curvature=(0.0, 0.0, 2.0),
)
```

This is a reusable, locally verified classical-laminate section asset. It is
not yet a shell finite element. The separation is intentional: a future shell
provider will consume the same section without moving section mechanics into
`Model` or inventing a second public workflow.

The ply vocabulary follows established engineering practice: Abaqus composite
layups likewise keep ply material, thickness, orientation, and integration
points as distinct inputs. AgentFEM does not attempt to copy the Abaqus input
language; it preserves the scientific concepts needed for reviewed migration.
After `inspect-abaqus` has resolved a composite section's references,
`materials.laminate_from_abaqus_section(...)` can lower the common composite
solid/continuum and shell row layouts. It requires `reviewed_by`, retains the
source location, and rejects unknown materials or ambiguous rows instead of
guessing.

## Woven reinforcement surface response

Textile forming is not well represented by simply rotating one orthotropic
solid matrix. Warp and weft directions convect, become non-orthogonal, and can
have markedly different tensile, trellising-shear, and bending response.
AgentFEM therefore exposes two separate frame types:

- `MaterialFrame`: an orthonormal basis for conventional anisotropy;
- `FiberFrame`: two independent structural directions that may become
  non-orthogonal.

The first surface constitutive contract provides independent tabulated yarn
tension, odd trellising shear, and bending channels:

```python
frame = materials.fiber_frame((1, 0), (0, 1), name="plain_weave")
tension = constitutive.tabulated_response(
    (0.0, 0.05, 0.10), (0.0, 50.0, 120.0), extrapolation="linear"
)
shear = constitutive.tabulated_response(
    (0.0, 0.20, 0.50), (0.0, 10.0, 80.0),
    symmetry="odd", extrapolation="linear",
)
fabric = constitutive.decoupled_fabric_surface(
    frame=frame,
    warp_tension=tension,
    weft_tension=tension,
    shear=shear,
    bending_stiffness=((2, 0, 0), (0, 2, 0), (0, 0, 1)),
)
local = fabric.evaluate(((1.05, 0.20), (0.0, 1.0)))
```

The result retains current fiber directions, both yarn strains, current and
shear angles, generalized resultants, tangent, bending moments, and stored
energy.

For a finite-kinematics **in-plane membrane** solve, use the same public
workflow and a zero bending matrix:

```python
from agentfem import fields, models, studies

study = studies.static_membrane()
model = models.create(study=study, mesh=domain)
u = model.field(fields.displacement(domain))
fabric = constitutive.decoupled_fabric_surface(
    frame=frame,
    warp_tension=tension,
    weft_tension=tension,
    shear=shear,
    bending_stiffness=((0, 0, 0), (0, 0, 0), (0, 0, 0)),
)
model.material(fabric)
# Add ordinary strong displacement constraints and/or natural loads.
result = model.step(target=u, increments=10).solve_result()
```

The provider derives both residual and Jacobian from the tabulated stored
energy, drives accepted load increments through the standard nonlinear
lifecycle, and rejects non-positive deformation Jacobians. A symbolic field
cannot raise an extrapolation error only at selected quadrature points, so
every globally used response curve must choose `extrapolation="constant"` or
`"linear"` explicitly.

`solve_result()` adds catalogued cell fields `FABRIC_GENERALIZED_STRAIN`
(warp strain, weft strain, trellising angle),
`FABRIC_GENERALIZED_RESULTANT` (the conjugate generalized resultants), current
`FABRIC_WARP_DIRECTION`/`FABRIC_WEFT_DIRECTION`, and `SENER`. Their component
order is fixed in the capability card rather than inferred by a plotting
script. The earlier concise names remain accepted only as input aliases, so
saved results stay explicit.

This is an **experimental FEM-integrated membrane**, not a shell. It consumes
yarn tension and trellising shear and deliberately refuses a nonzero bending
matrix instead of discarding it. `mechanics.director_shell_kinematics(...)`
provides objective local membrane, transverse-shear, and curvature measures
for the next provider, but it does not claim an element interpolation,
locking treatment, or global shell solve. Tool contact, friction, inter-ply
slip, explicit quasi-static controls, and forming observables remain separate
promotion gates.

## Multilayer and fibre-curve foundations

Stacks with different reinforcement directions must not be collapsed into one
fictitious orthotropic frame. `fabric_layer(...)` and `fabric_stack(...)`
therefore retain a stable identity, frame, and response for every
layer while sharing the same surface deformation:

```python
stack = constitutive.fabric_stack(
    [
        constitutive.fabric_layer(woven_0, name="ply_0"),
        constitutive.fabric_layer(
            woven_45,
            name="family_45",
            physical_layer_ids=("ply_45_bottom", "ply_45_top"),
        ),
    ],
    name="forming_stack",
)
local = stack.evaluate(F_surface)
family_45 = local.by_name("family_45")
```

Only stored energy is safely additive without choosing a common generalized
frame. Warp/weft/trellising resultants remain attached to their own layers.
This avoids a plausible-looking but physically ambiguous aggregate vector and
provides the constitutive object needed by a later multilayer shell provider.
One computational layer normally represents one physical layer. Identical
layers with the same material and orientation may instead be grouped by
listing every `physical_layer_id`. Energy, generalized resultants and tangent
then scale by the explicit number of listed layers. The identity list and
aggregation semantics enter the result manifest; AgentFEM never infers a
hidden multiplicity from one scalar.
The same stack can enter the current in-plane membrane Step when every layer
has zero bending stiffness. Generic fabric output requests then expand into
stable per-layer names such as
`FABRIC_LAYER_000_PLY_0__FABRIC_GENERALIZED_STRAIN`; aggregate `SENER` remains
available and the exact layer-name/prefix/frame mapping is recorded in the
result's field manifest. This is shared-kinematics membrane equilibrium, not a
multilayer shell: transverse slip, independent material normals, and layer
contact remain later gates.

Fibre bending also has two meanings that must not be conflated.
`mechanics.fiber_curve_kinematics(...)` evaluates the directional derivative
of a unit fibre curve and separates:

- in-plane curvature, along the tangent direction normal to the fibre; and
- normal curvature, along the surface normal.

Both are objective under a superposed rigid rotation and are reported as
changes from the reference surface. The required unit-direction gradients are
explicit inputs. A future global implementation must obtain them from a
verified second-gradient, rotation-free, or mixed-director discretization; a
standard displacement membrane cannot manufacture them after the fact.

`decoupled_fibrous_shell(...)` now defines the constitutive contract that such
an element must consume. It keeps four independently calibrated energy blocks:

- warp/weft tension and trellising shear;
- transverse director shear;
- warp/weft in-plane fibre bending;
- warp/weft normal fibre bending.

Its response exposes each generalized resultant, each energy channel, and one
consistent block tangent in a documented order. The membrane law must carry a
zero legacy bending matrix so bending cannot be counted twice. This remains a
local law, not a shell element: interpolation, neighbouring-element curvature,
locking control, boundary moments, contact and nonlinear evolution still
belong to the future provider and procedure.

`law.generalized_expressions_ufl(q)` is the narrow bridge from that local law
to a future global operator. It accepts the documented nine objective measures
and returns one symbolic stored energy plus the nine conjugate resultants.
UFL differentiation of that same potential supplies the residual and
consistent Jacobian. The method deliberately does not construct `q`: choosing
mixed fields, interpolation, compatibility constraints, quadrature and
stabilization remains the shell operator's responsibility.

For an embedded three-dimensional surface,
`mechanics.surface_deformation_gradient(...)` supplies the objective lift used
to convect the reference fibre frame. It maps the two reference tangents to
their current counterparts and completes the surface map by sending reference
normal to current normal. The completion is a constitutive convenience and
does not claim a physical thickness stretch.

Finally, `fabric_forming_limits(...)` provides a small fail-closed screening
contract for user-declared yarn strain, trellising angle, and curvature
limits. The result reports dimensionless utilization and the governing mode.
It is deliberately called an *assessment*, not a wrinkle or defect predictor:
contact, boundary forces, bending equilibrium, and geometric instability must
come from the forming solve and its validation evidence.

That boundary reflects the literature. Boisse and co-workers identify yarn
tension, in-plane shear, and bending as separate contributors to deformation
and wrinkling, with bending mechanics not safely inferred from membrane
response. The fibrous-shell work of Liang, Colmars, Boisse and later Bai and
co-workers adds material-director/slip kinematics and in-plane bending rather
than treating the reinforcement as a conventional shell.

## Evidence and limits

The machine-readable benchmark
`agentfem.benchmark.fibrous_shell_foundation` binds the following checks to
their references, execution command, acceptance criteria, and explicit
limitations. It verifies the reusable foundation; it does not certify a global
shell or forming procedure.

The automated local evidence currently checks:

- 3D orthotropic elasticity recovers the isotropic limit;
- rotated planar elasticity enters the standard `Model`/UFL operator;
- material-frame stress and strain use explicit `S_MATERIAL`/`E_MATERIAL`
  result identities;
- tensor rotation preserves elastic energy;
- a symmetric cross-ply laminate has a vanishing `B` matrix;
- section-point identities and per-ply recovery are stable;
- identity deformation has zero fabric response;
- tension-only yarns do not generate artificial compressive force;
- tension, trellising shear, and bending remain independent energy channels;
- director-shell measures vanish under a superposed rigid rotation and detect
  a constant-curvature patch;
- fibre-curve measures distinguish in-plane and normal bending and remain
  objective under a superposed rigid rotation;
- the three-dimensional surface-deformation lift reproduces both current
  tangents and the current normal and transforms objectively;
- the local fibrous-shell law keeps membrane, transverse shear, in-plane
  bending and normal bending as independent positive-semidefinite energy and
  tangent blocks, and rejects overlapping bending ownership;
- multilayer stacks retain varying layer frames and stable layer identities
  while adding only compatible energy scalars;
- a multilayer membrane patch enters the standard Step, preserves its exact
  layer-field manifest, and writes per-layer result fields without combining
  incompatible local vectors;
- forming limits report explicit utilization and refuse curvature assessment
  when curvature kinematics are absent;
- the woven membrane is selected by the standard Step provider, solves a
  nonzero traction patch, and reports positive Jacobian and stored energy;
- membrane lowering rejects a nonzero bending law rather than hiding it.

No shell element patch test, contact benchmark, or drape experiment has yet
promoted this membrane foundation to a forming-capable fibrous shell.

## Promotion roadmap

1. **Fibrous shell kernel:** mixed displacement/director or independently
   justified rotation-free interpolation; membrane, transverse-shear,
   in-plane-bending and normal-bending patch tests; documented locking control.
2. **Forming procedure:** tool geometry, unilateral contact, friction,
   blank-holder loads, inter-ply slip, quasi-static explicit energy controls,
   checkpoint/restart, and per-layer result fields.
3. **Verification ladder:** bias-extension and picture-frame shear, cantilever
   bending, in-plane bending/virtual-fibre calibration, hemisphere or double-
   dome draping, and multilayer varying-orientation cases. Mesh, time-step,
   penalty/contact, and imperfection sensitivity remain separate axes.
4. **Manufacturing-to-structure handoff:** transfer fibre directions,
   thickness, shear history and declared defects into a cured laminate or
   solid model with a conservative mapping audit.
5. **Later research:** irreversible shear/bending, compaction and permeability,
   mesoscopic yarn/contact models, data-assisted calibration, and uncertainty.

## References

- P. Boisse et al., “The bias-extension test for the analysis of in-plane
  shear properties of textile composite reinforcements and prepregs: a
  review,” *International Journal of Material Forming* (2022),
  <https://doi.org/10.1007/s12289-022-01682-8>.
- B. Liang, F. Colmars, and P. Boisse, “A shell formulation for fibrous
  reinforcement forming simulations,” *Composites Part A* 100 (2017),
  <https://doi.org/10.1016/j.compositesa.2017.04.024>.
- X. Peng and J. Cao, “A continuum mechanics-based non-orthogonal constitutive
  model for woven composite fabrics,” *Composites Part A* 36 (2005),
  <https://doi.org/10.1016/j.compositesa.2004.08.008>.
- “A Shell Formulation for Textile Composite Forming Simulations,”
  *Procedia Manufacturing* 47 (2020),
  <https://doi.org/10.1016/j.promfg.2020.04.125>.
- R. Bai et al., “A specific 3D shell approach for textile composite
  reinforcements under large deformation,” *Composites Part A* 139 (2020),
  <https://doi.org/10.1016/j.compositesa.2020.106135>.
- R. Bai et al., “A multilayer shell approach for simulating composite
  preforming with varying fibre orientations,” *Composite Structures* 372
  (2025), <https://doi.org/10.1016/j.compstruct.2025.119593>.
- R. Zheng et al., “Numerical prediction of the in-plane bending properties of
  fibrous reinforcements using a mesoscopic virtual fiber finite element
  approach,” *Composites Part B* 322 (2026),
  <https://doi.org/10.1016/j.compositesb.2026.113771>.
- P. Boisse et al., “Bending and wrinkling of composite fiber preforms and
  prepregs,” *Composites Part B* 141 (2018),
  <https://doi.org/10.1016/j.compositesb.2017.12.061>.
- T. X. Duong, M. Itskov, and R. A. Sauer, “A general isogeometric finite
  element formulation for rotation-free shells with in-plane bending of
  embedded fibers,” *International Journal for Numerical Methods in
  Engineering* 123 (2022), <https://doi.org/10.1002/nme.6937>.
- Q. Steer et al., “Modeling and analysis of in-plane bending in fibrous
  reinforcements with rotation-free shell finite elements,” *International
  Journal of Solids and Structures* 222–223 (2021),
  <https://doi.org/10.1016/j.ijsolstr.2021.03.001>.
- Dassault Systèmes, “Defining composite plies,” Abaqus 2025 documentation,
  <https://docs.software.vt.edu/abaqusv2025/English/SIMACAECAERefMap/simacae-t-prpcompositesshellcontinuumplies.htm>.
- Dassault Systèmes, “Fabric material,” Abaqus 2025 documentation,
  <https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-fabric.htm>.
