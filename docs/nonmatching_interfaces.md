# Nonmatching cohesive interfaces: bounded development contract

The geometry operator is process-local. An experimental ordinary `model.step`
provider now consumes its coplanar common-refinement route for serial P1
tetrahedra or Q1 hexahedra and a purely elastic interface. Existing matched cohesive laws and
their transactions remain the constitutive owner; global damage evolution is
not enabled by this elastic provider.

The serial P1 lowering now adds the same residual and tangent to the existing
PETSc cohesive assembly. A separate two-block test verifies reaction,
interface opening and interface energy against the exact series compliance
`2/E + 1/Kn` for unit-area, unit-length blocks (Poisson ratio zero), using
matching and nonmatching tetrahedral meshes. This is global assembly evidence,
not an MPI, finite-strain or forming capability.

## Ordinary workflow (experimental)

Use `interfaces.elastic_cohesive(normal_stiffness=..., tangential_stiffness=...)`,
`interfaces.pair_nonmatching_triangles(negative, positive, tolerance=...)`, then
`fracture.nonmatching_cohesive_force(pairing, displacement, law,
negative_dofs=..., positive_dofs=...)`. Dof maps explicitly identify independent
P1 traces; matching coordinates alone cannot distinguish the two bodies.
Pass the resulting force as `model.step(target=displacement,
cohesive_force=force)` for a linear-static solid and call `solve_result()`.

The first provider accepts registered linear elastic bulk materials, loads and
ordinary strong constraints. It rejects damage laws, projected-only quadrature,
MPI, DG traces, eigenstrains, additional boundary models and direct K/F
overrides. Result output reuses the standard lifecycle and supplies U/RF,
free residual, bulk/interface energy and linear proportional-path natural and
prescribed-motion work. These work quantities assume a stress-free origin and
are not arbitrary-history work integration. Empty `field_variables=()` exports
only U. Solver failure restores the previous nodal field and discards trial
interface state. Durable Step restart is not yet supported.

The ordinary-Step, existing global cohesive, local interface/search and
architecture regression selection passes 112 tests in the local `fenicsx-env`.
This is targeted candidate evidence, not the release ladder or MPI acceptance.

## Ownership and numerical contract

- Pairing owns reference surface identity, projection, interpolation and
  positive integration weights. It reuses the reviewed triangle BVH.
- Kinematics owns the jump and basis transformation; a reference projection
  snapshot does not declare either deformable body rigid.
- The law owns traction, tangent and state evolution. No case-specific
  stiffness, fracture energy, part name or damage criterion enters pairing.
- The Operator scatters `B^T W t` and applies `B^T W D B`; this is the internal
  residual, not the restoring force. The transpose shares exactly the same
  interpolation as the jump so discrete virtual work is preserved.
- Procedure/State will own accepted updates, rollback and restart; Result
  retains pairing diagnostics separately from constitutive energy.

The first map explicitly selects the negative integration side and requires
coincident linear triangular reference surfaces with opposing locally
parallel normals. It rejects failed/ambiguous projections. Matching at the
quadrature points is **not a proof of complete surface overlap**. Three-point
triangle quadrature is not exact across projected master-element boundaries.
Side-swap and quadrature-refinement checks are mandatory before promotion.
Neither automatic finer-side selection nor arbitrary surface tessellation
eliminates these issues.

An explicit `quadrature_refinement` splits integration triangles without
changing their parent interpolation. A point-count ceiling rejects excessive
allocation before trace arrays are built. In the 3:2 structured regression,
the base rule has approximately 1.735% relative positive-side nodal-force
error under constant traction despite exact total-force balance; one
subdivision brings that particular fixture to roundoff. This is not a general
proof of exactness or coverage. Overlap integration remains the robust next
gate for unaligned partitions. The elastic material candidate has genuinely
zero damage and dissipation; no invented failure thresholds are used.

`constant_traction_audit()` compares both assembled nodal measures against
independent triangle-area integrals. A coarse-to-fine 1:3 regression exposes
more than 70% nodal-force error despite balanced total forces. This necessary
patch check neither certifies complete overlap nor estimates arbitrary
traction error. Refinement must be chosen before creating material state;
changing quadrature changes the checkpoint identity.

The bounded alternative `planar_overlap_pairing` constructs common-refinement
integration triangles while retaining both original P1 bases. Existing BVH
AABB queries select candidates; convex clipping supplies the narrow phase.
Every facet must have complete area coverage to relative tolerance `1e-9`,
and within-side overlapping triangles are rejected. Geometry is restricted to
coincident coplanar triangles. Three-point quadrature integrates products of
the two P1 traces exactly on each overlap, not arbitrary nonlinear tractions.
Tests include 1:3 and 3:2 ratios, perturbed internal nodes, rotated planes,
side-swap energy/residual equivalence and reference force/moment balance.
The pairing fingerprint includes the integration method. No Lagrange
multiplier or dual mortar formulation is claimed.

The trace preserves common rigid displacement at coincident points, including
finite rotation of a closed interface. This does **not** establish objective
finite-rotation traction or tangent for an already open interface. Convected
bases, reference-gap treatment, and their derivatives require separate tests.
Force balance alone does not prove moment balance when projection has a finite
offset. Finite initial gaps are outside the first physical scope.

Hex8 faces retain their original four-node bilinear basis: triangulation is
used only for geometric overlap integration, never as a P1 replacement.
Use `interfaces.reference_trace(vertices, cells, topology="quadrilateral")`
with cyclic perimeter connectivity, then `interfaces.pair_reference_traces`.
The first Q1 route accepts planar parallelograms only; warped/nonaffine faces
and mixed P1/Q1 pairs are rejected. Basix supplies the original Q1 basis and
degree-four overlap quadrature. The bilinear `u=xy` patch integrates its square
to `1/9` on the unit square; nodal traction, force, moment and side-swap tests
complement that interpolation check. Global two-block compliance tests use
independent 1:3 and 2:3 hexahedral interface partitions.

For explicit dynamics, combine `cohesive_force` with
`element_policy=elements.uniform_strain_hex8(...)` in ordinary `model.step`.
Only the undamaged elastic interface is admitted. The stable-step estimate
sums bulk and interface squared-frequency bounds, rather than taking the
minimum of two isolated stable steps. Histories distinguish physical bulk,
interface and artificial hourglass energy. Serial checkpoint/restart and
failed-increment retry reuse the existing Procedure/State lifecycle. This
does not yet claim arbitrary external-work closure, MPI, finite rotation,
damage evolution or compatibility with every commercial Hex8R formulation.

## Promotion sequence

Initial local evidence (2026-10-09): `test_nonmatching_pairing.py`,
`test_rigid_surface_search.py`, and `test_interfaces.py` pass 61 tests in the
local `fenicsx-env`. The new tests cover fixed-history tangent finite
differences, energy gradients, existing bilinear-law damage and rejected
checkpoint atomicity in addition to trace invariants. This does not establish
the complete patch-test, MPI or global Step gates below.

1. Pure trace tests: affine consistency, force/moment, virtual work, invalid
   geometry, immutable identity and matrix-free tangent action.
2. Reuse existing material transactions; verify residual/tangent, stored
   energy, damage rollback and matching-limit response. Add a genuine elastic
   interface law for undamaged bonding; never invent large damage thresholds.
3. Surface coverage and integration accuracy, side-swap sensitivity and
   fixed-plane nonmatching patch tests with independent expected nodal forces.
4. DOLFINx boundary-dof lowering and ordinary force/Step integration; compare
   a two-block specimen to an independent series-compliance reference.
5. Stable quadrature identity, portable restart and MPI owner exchange.
6. Convected finite-rotation kinematics and general quadrilateral traces before any
   claim about a large-deformation Hex8 industrial reproduction.

No migration capability is promoted by the initial local tests.

## Related Hex8R work

Uniform strain means volume-averaged gradients, not simply evaluating the
gradient at the geometric centre of a distorted element. A1 is small-strain
elasticity with stiffness stabilization, positive lumped mass, separate
physical/hourglass energy and a composed stability bound. Verify six rigid,
six constant-strain and twelve hourglass modes, distorted affine patches,
bending and waves. Large finite rotations belong to A2, not the small-strain
acceptance claim. Relaxation hourglass control needs its own history and
restart, not another unimplemented method string. Keep Basix/DOLFINx as the
mesh/DOF/assembly owner; add only the missing formulation contribution.

## References

- Paggi and Wriggers (2016), *Node-to-segment and node-to-surface interface
  finite elements for fracture mechanics*, CMAME 300, 540–560.
  https://doi.org/10.1016/j.cma.2015.11.023;
  author manuscript https://arxiv.org/abs/1604.05236.
  Their node-to-surface formulation informs the work-conjugacy review; the
  initial projected quadrature here is not claimed as an exact reproduction.
- Flanagan and Belytschko (1981), *A uniform strain hexahedron and quadrilateral
  with orthogonal hourglass control*, IJNME 17, 679–706.
  https://doi.org/10.1002/nme.1620170504.
- Belytschko and Bindeman (1993), *Assumed strain stabilization of the eight
  node hexahedral element*. https://doi.org/10.1016/0045-7825(93)90124-G.
- *3D Common-Refinement Method for Non-Matching Meshes in Partitioned
  Variational Fluid-Structure Analysis*, https://arxiv.org/abs/1711.01773.
  Common-refinement integration motivation; not a claim to reproduce that
  fluid-structure solver.
