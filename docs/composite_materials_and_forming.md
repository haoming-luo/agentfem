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
energy. This is **material-point verified**, not FEM-integrated. Finite-rotation
shell kinematics, locking control, tool contact, friction, inter-ply slip,
explicit quasi-static controls, and forming observables remain separate
promotion gates.

That boundary reflects the literature. Boisse and co-workers identify yarn
tension, in-plane shear, and bending as separate contributors to deformation
and wrinkling, with bending mechanics not safely inferred from membrane
response. The fibrous-shell work of Liang, Colmars, Boisse and later Bai and
co-workers adds material-director/slip kinematics and in-plane bending rather
than treating the reinforcement as a conventional shell.

## Evidence and limits

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
- tension, trellising shear, and bending remain independent energy channels.

No forming benchmark, shell patch test, contact benchmark, or drape experiment
has yet promoted the fabric surface law to FEM-integrated maturity.

## References

- P. Boisse et al., “The bias-extension test for the analysis of in-plane
  shear properties of textile composite reinforcements and prepregs: a
  review,” *International Journal of Material Forming* (2022),
  <https://doi.org/10.1007/s12289-022-01682-8>.
- B. Liang, F. Colmars, and P. Boisse, “A shell formulation for fibrous
  reinforcement forming simulations,” *Composites Part A* 100 (2017),
  <https://doi.org/10.1016/j.compositesa.2017.04.024>.
- “A Shell Formulation for Textile Composite Forming Simulations,”
  *Procedia Manufacturing* 47 (2020),
  <https://doi.org/10.1016/j.promfg.2020.04.125>.
- R. Bai et al., “High-fidelity fibrous shell model based on a new
  kinematic assumption for multi-layer woven fabric forming simulation,”
  journal article and author summary,
  <https://fhclxb.buaa.edu.cn/en/article/id/236271a2-01d7-4910-b99c-79ccc4cb21f6>.
- Dassault Systèmes, “Defining composite plies,” Abaqus 2025 documentation,
  <https://docs.software.vt.edu/abaqusv2025/English/SIMACAECAERefMap/simacae-t-prpcompositesshellcontinuumplies.htm>.
- Dassault Systèmes, “Fabric material,” Abaqus 2025 documentation,
  <https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-fabric.htm>.
