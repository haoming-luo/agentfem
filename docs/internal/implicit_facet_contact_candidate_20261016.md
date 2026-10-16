# Fixed-facet implicit contact candidate — 2026-10-16

Status: experimental serial composition, not general die contact or a release
claim. This follows the analytical-plane finite-J2 contact candidate without
replacing its tested UFL route.

## Reusable ownership

- Model: an existing `RigidContactPair` binds a named slave boundary, rigid
  body, and frictionless penalty law. The pair does not contain search or
  Newton code.
- Operator: the DOLFINx boundary adapter supplies positive reference-area
  quadrature and interpolation. The existing reviewed triangle BVH projects
  the current slave points for every Newton trial. A separate solver-neutral
  facet linearization integrates force, potential, and the scalar-DOF COO
  tangent. For a fixed planar branch,
  `R_a = w N_a k g n` and `K_ab = w N_a N_b k (n ⊗ n)` when `g < 0`.
- Procedure: the ordinary finite-J2 Newton path adds the contact residual and
  tangent to the same material solve. PETSc insertion switches from UFL
  insertion to additive contact entries only after an assembly flush.
- State: J2 owns material trial/commit/rollback and serial checkpoint. The
  fixed rigid tool has no state to commit. Contact is reprojected, never
  recovered from a stale accepted gap.
- Result: contact dual force, distribution, potential, and a named generalized
  moment are inspectable. A completed solve still needs scientific
  verification; discrete path-work residual is not a general first-law proof.

## Deliberate admission boundary

Only one fixed, frictionless, triangulated rigid tool; 3D blocked displacement,
reviewed DOLFINx boundary trace, reference-area quadrature, and one MPI rank.
All active projections must lie on a planar facet branch or a coplanar
interior seam. Open tool boundaries, non-coplanar edges/vertices, invalid or
ambiguous projection, current-area quadrature, friction, prescribed tool
motion, and contact/strong-BC dof overlap are rejected. The latter has no
contact-lifting implementation yet. Periodic MPC is not admitted. A smooth
coplanar seam is not evidence for a curved triangle mesh or a finite-sliding
die with creases.

## Local evidence and next gate

The first solver-neutral and adapter regressions pass directional finite
differences of residual and potential. The ordinary public `model.step()`
J2 patch matches its analytical-plane counterpart for penalties 250, 500,
1000 on one cell per axis and penalty 500 on two cells per axis; displacement,
committed material state, and contact potential agree within the test
tolerances. A tilted coplanar tool also passes on one- and two-cell meshes,
giving nonuniform contact rather than only uniform compression. A separate
serial interrupted checkpoint matches uninterrupted state and rejects a
changed penalty; deliberate post-solve energy failure rolls back the joint
accepted state. One two-rank check confirms that both analytic and faceted
contact are rejected before distributed assembly. The public example writes
one reference-geometry XDMF *Uniform* grid with nodal and cell fields.

At source commit `623b13b0`, 48 focused tests passed (two one-rank skips),
the canonical example-language gate passed, Ruff and strict documentation
build passed, and the native macOS wheel plus sdist passed the installed-wheel
release smoke. The wheel SHA-256 was
`4ef1d2df80ba11cb984a517aed1ca3026883dcb2bab36b378113e554563ef386`.
The fixed-facet example also completed from an isolated installation of this
wheel, with contact potential `0.03425550070914266` in the dimensionless
example. The normal release smoke did not request an MPI run; the separate
two-rank *rejection* test is not distributed-contact execution. These checks
are not independent forming validation.

Before promoting beyond this slice, add an independent 3D structure reference
with nonuniform slave deformation and facet crossing; sample mesh, increment,
penalty, and tool tessellation sensitivity. Then implement reviewed
edge/vertex or smooth-surface transition linearization and moving-tool
force--moment--work with atomic pose rollback. Only after those gates should
the full contact residual and tangent enter periodic MPC reduction or an MPI
owner/ghost assembly path. No one of these can be inferred from the local
fixed-tool tests.
