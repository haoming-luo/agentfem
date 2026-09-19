# Composite materials and forming

AgentFEM treats composite mechanics as independent assets:

```text
material + orientation + section/ply placement + strength assessment
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

## Ply-strength assessment

Elastic constants, strength allowables, and the chosen failure surface are not
the same object. `CompositeStrengths2D` therefore stores the five common
plane-stress allowables independently from the lamina material. Built-in
maximum-stress, Hashin, and Tsai--Wu criteria consume an explicit material-axis
stress order `(sigma_11, sigma_22, tau_12)`:

```python
strengths = constitutive.composite_strengths_2d(
    xt=1500e6, xc=1000e6, yt=50e6, yc=200e6, s12=100e6,
)
failure = constitutive.assess_ply_failure(
    (750e6, 25e6, 50e6), strengths, criterion="hashin_2d",
)
```

The result contains every mode index, the governing mode, and the proportional
load factor at which the first index reaches one. The load factor is solved on
the actual criterion, not approximated as `1/sqrt(index)` for the nonhomogeneous
Hashin matrix-compression branch. A user criterion can implement the small
`PlyFailureCriterion` protocol without modifying AgentFEM.
Tsai--Wu is created as `TsaiWu2D(interaction=...)`: the normalized interaction
coefficient is mandatory and must preserve a convex quadratic surface.
AgentFEM does not silently substitute the common empirical approximation when
biaxial strength data are unavailable.

For a `LaminateSection`, `assess_laminate_failure(section, response, ...)`
rotates every recovered section-axis stress into its named ply material axes
and retains the stable section-point identity. This is first-ply initiation
screening only. It does not reduce stiffness, redistribute load, evolve
fracture energy, or claim final laminate failure; those belong to a separately
verified progressive-damage procedure.

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

`mechanics.fibrous_shell_kinematics_ufl(...)` defines the matching operator
side of that hand-off. Surface deformation supplies yarn stretches and
trellising, an independent material director supplies transverse shear, and
independent unit-fibre fields supply in-plane and normal curvature. The
function is objective and exposes one fixed nine-component order. It still
does not enforce unit length, surface tangency or convection: those are the
mixed operator's constraint equations and must pass their own patch and
locking tests before a global shell Step is made public.

For the first no-slip layer,
`mechanics.fibrous_shell_compatibility_ufl(...)` exposes the minimal exact
residual: one unit-director equation and two three-component fibre-convection
equations. Unit length and surface tangency of each fibre are reported as
diagnostics because they already follow from exact convection; adding them as
extra multipliers would make the system redundantly constrained. The contract
chooses no multiplier, augmentation, condensation or penalty strategy.

The first global discretization track is now explicitly rotation-free and
neighbouring-element based. `mesh.cell_neighborhood(...)` supplies its first
shared primitive: every owned interior facet is paired with both adjacent
cells and both cell-local facet positions, including ghost-cell adjacency at
MPI partition interfaces. This topology is also reusable by DG, estimator and
fracture algorithms. `mesh.cell_neighborhood_geometry(...)` adds the two cell
centroids, facet midpoint, center vector, distance and direction in embedding
coordinates without choosing a finite-difference rule. These quantities are
translation invariant and give a future curvature reconstruction an explicit,
auditable length scale. `mesh.cell_pair_directional_difference(...)` adds the
corresponding two-cell value difference per center distance and exactly
recovers affine scalar/vector cell-center fields, including across a verified
MPI partition. It remains one directional difference, not a reconstructed
gradient, fibre curvature, or shell Step.
`mesh.reconstruct_cell_gradient(...)` then combines the locally visible
owned/ghost stencil by weighted least squares in an SVD-derived tangent basis.
It exactly recovers affine scalar and vector fields on triangle and
quadrilateral meshes in serial and two-rank tests, while recording neighbor
counts, rank, and condition number for every owned cell. Rank-deficient or
ill-conditioned boundary stencils fail instead of returning artificial zero
curvature. The geometry decomposition and compact neighbor weights live in a
reusable `CellGradientOperator`; repeated fields or nonlinear iterations do
not repeat the SVD, and arbitrary constant offsets are annihilated exactly.
The cached operator also exposes its exact transpose. Scalar and vector
inner-product identities are verified in serial and under two MPI ranks, so a
future bending energy can map gradient duals back to cell residuals without
inventing a second discretization. On a distributed mesh the transpose
deliberately returns local and ghost-cell contributions; reverse scattering
to the owning ranks remains an explicit backend assembly responsibility.
`operators.cell_gradient_energy(...)` closes the corresponding linear
operator identity for scalar or vector cell fields: one cached `G` evaluates
`0.5 sum(w k |Gv|^2)`, the exact residual `G.T w k Gv`, and the matching
matrix-free tangent action. Finite-difference derivatives, tangent symmetry,
positive semidefiniteness, and the constant-field nullspace are verified;
rank-local energy and ghost residual contributions retain explicit MPI
assembly semantics. `mesh.owned_cell_measures(...)` supplies physical DG0
integration weights aligned with owned cells and excludes ghosts, so global
energy counts every cell once in serial or MPI. This is a reusable nonlocal
operator foundation, not yet the nonlinear fibre-bending virtual work of a
forming shell.
An energy operator additionally requires a partition-complete stencil. With
the current shared-facet ghost layer this means a one-ring reconstruction
under MPI. A wider stencil remains useful for serial reconstruction studies,
but AgentFEM refuses to use it as distributed stored energy until an expanded
cell halo is available; a locally visible but incomplete two-ring graph is not
silently treated as partition independent.
`mechanics.reconstruct_fiber_curvature(...)` consumes that gradient
with three-dimensional fibre directions and 3x2 current tangents, then returns
the signed in-plane and normal curvature channels already used by the local
constitutive law. For the smooth field
`a=(cos(0.8x), sin(0.8x), 0)`, relative in-plane-curvature errors on 4x4, 8x8,
16x16 and 32x32 quadrilateral meshes are approximately 1.50e-2, 3.86e-3,
9.75e-4 and 2.45e-4. A superposed three-dimensional rigid rotation leaves both
channels invariant. Boundary moments and complete shell equilibrium remain
separate promotion gates.
`operators.fiber_direction_bending(...)` now takes the next, deliberately
bounded step for a mixed independent-direction formulation. It normalizes the
cell directions before the same cached reconstruction and evaluates separate
in-plane and normal curvature energies. Its analytical first variation
contains both the reconstructed-gradient transpose and the local changes of
the fibre coordinate/projection frame. The residual matches finite-difference
energy derivatives, is insensitive to positive pointwise rescaling, transforms
covariantly under a three-dimensional rigid rotation, and retains explicit
local-plus-ghost MPI semantics. The response also returns the exact direct
energy derivative with respect to its current surface tangents. Its
analytical matrix-free direction tangent matches residual
differences, satisfies Hessian symmetry, and transforms covariantly under a
three-dimensional rigid rotation in serial and two-rank tests. Its full
response linearization additionally differentiates the direction and
surface-tangent duals along an admissible moving-surface path.
Compatibility-force blocks, boundary moments and complete shell equilibrium
remain promotion gates.
For distributed use, `assembly.assemble_cell_residual(...)` maps scalar or
blocked-vector DG0 cell contributions to their explicit cell dofs, reverse-
scatters ghost contributions to owners, and then refreshes ghost entries. A
two-rank test now closes global bending-energy directional change against the
assembled PETSc residual work; the operator is no longer limited to a
rank-local diagnostic array.
`operators.cell_average_gradient(...)` establishes the complementary FEM
transfer needed by a displacement formulation. A reusable sparse UFL
coupling maps a scalar or vector continuous field to its physical DG0
cell-average gradient; its mass-inverse-weighted transpose maps arbitrary
local-plus-ghost cell-gradient duals back to a source-space PETSc residual.
Affine scalar and three-component fields are exact, and the global
forward/adjoint work identity closes under two MPI ranks even when each rank
contributes different ghost-cell duals. This is the first tested bridge from
the neighbouring-element chain back to displacement degrees of freedom.
`operators.convected_cell_fiber(...)` then consumes this transfer with
reference surface tangents and one fibre's reference tangent coordinates.
The public primary field remains the three-component surface displacement;
the operator derives current tangents, fibre stretch and unit direction on
local and ghost cells. Its exact directional derivative matches finite
differences, rigid-body rotation is reproduced, and its transpose maps
arbitrary tangent/stretch/direction duals back to the displacement PETSc
space with global work closure in serial and two-rank tests. Combining that
adjoint with the bending law's direction and direct surface-tangent duals now
assembles the complete displacement-level first variation of this bending
energy. Centered finite differences close the energy--virtual-work identity
in serial and over a two-rank partition. Matching exact linearizations of the
kinematic adjoint and bending response now compose into a matrix-free
displacement-level consistent tangent, which matches residual differences in
serial and over two ranks. The next gates are therefore boundary moments,
complete shell assembly, locking control and shell patch tests; this evidence
still does not constitute a shell Step.
`operators.displacement_fiber_bending(...)` is the public composition of this
chain. It owns no new mechanics: it gives one convected fibre family a compact
`energy(...)`, `residual(...)`, and `tangent_action(...)` interface while the
kinematic transfer, neighbour reconstruction, and bending response retain
their separate ownership and inspectable metadata. The composed energy is
objective, its residual and tangent action transform covariantly under a
superposed three-dimensional rigid rotation, and its displacement Hessian
action is symmetric.
The object satisfies the generic
`operators.NonlinearOperatorContribution` protocol. A future hybrid Newton
Procedure can therefore add it to local UFL membrane contributions without
teaching the solver about fibre-specific classes or lowering the neighbour
stencil into a fictitious local material law.
The corresponding internal promotion runtime now demonstrates that
composition. PETSc receives the assembled local Jacobian plus the exact
matrix-free bending action as its true operator, while the assembled local
matrix remains the independent preconditioner. Essential-boundary increments
and rows are handled once at this boundary. A loaded, constrained
displacement/bending problem converges through Newton and reproduces its
serial displacement integral, bending energy, maximum displacement, and
iteration count with two ranks. The runtime remains internal until standard
load incrementation, progress, checkpoint, result evidence, boundary moments,
locking control, and shell patch tests all use the ordinary Procedure
lifecycle.
Naive mixed P1/DG0 and P2/DG1 compatibility pairs were rejected after losing
rank under refinement; full-rank P2/CG1 and P2/DG0 candidates still showed a
decaying normalized inf-sup value in the tested H1/L2 norms. Those negative
results are retained in the architecture decision rather than hidden behind a
penalty constant.

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
- serial triangle/quadrilateral and two-rank meshes preserve exact interior-
  facet neighbourhoods without mistaking a partition interface for a boundary.
- neighbour-reconstructed in-plane fibre curvature is objective and shows
  near-second-order error reduction on a smooth rotating-direction field.
- geometry-cached reconstruction weights reproduce the one-shot result and
  can be reused across fields without storing a dense global matrix;
- the cached gradient and transpose satisfy scalar/vector inner-product
  identities in serial and across a two-rank computation stencil, with MPI
  reverse-scatter ownership kept explicit;
- the first matrix-free cell-gradient energy has an exact residual and
  symmetric positive-semidefinite tangent action, matches finite-difference
  energy derivatives, and annihilates constant fields;
- physical cell weights recover triangle/quadrilateral domain area and count
  the unit-square measure once under two MPI ranks;
- the neighbour-reconstructed independent-direction bending energy has an
  exact first variation and consistent matrix-free tangent, pointwise scale
  invariance, Hessian symmetry and three-dimensional rotation objectivity in
  serial and two-rank tests;
- DG0 scalar/vector residual assembly reverse-scatters ghost-cell
  contributions and closes the global two-rank energy--virtual-work identity;
- continuous scalar/vector fields transfer exactly to DG0 cell-average
  gradients, whose transpose closes global work back to source FEM dofs under
  two MPI ranks;
- displacement-derived current surface tangents, fibre stretch and direction
  reproduce a rigid rotation and close their exact derivative/adjoint work
  identity in serial and under two MPI ranks;
- the displacement-derived bending energy closes its complete first-variation
  identity, including both direction and surface-tangent paths, in serial and
  under two MPI ranks;
- its matrix-free displacement tangent matches residual differences in serial
  and under two MPI ranks.
- the assembled-local plus matrix-free PETSc tangent uses the local matrix as
  an independent preconditioner, preserves strong constraints, and solves a
  nonzero displacement/bending problem;
- the same solve has matching displacement, bending energy, maximum response,
  and Newton count in serial and with two MPI ranks;
- MPI energy operators reject a wider stencil when the available cell halo
  cannot make it partition complete.

No shell element patch test, contact benchmark, or drape experiment has yet
promoted this membrane foundation to a forming-capable fibrous shell.

## Promotion roadmap

1. **Fibrous shell kernel:** boundary moments and complete neighbouring-
   element rotation-free interpolation;
   membrane, transverse-shear,
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
