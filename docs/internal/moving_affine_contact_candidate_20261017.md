# Moving rigid contact under affine-periodic J2 — source candidate

Status: experimental serial source candidate, not a release or general die-forming claim.

## Ownership and numerical route

The Model still owns one `RigidContactPair` and one prescribed rigid motion.
The affine Procedure owns the load coordinate and sets the current tool pose
before every increment trial. The existing rigid-contact Operator recomputes
projection at every Newton and line-search trial; the existing affine solver
assembles material plus contact in full space before applying `TᵀR` and
`TᵀKT`. No contact-specific Newton loop or alternative J2 material path was
introduced.

At each accepted station, the provider publishes its work-conjugate tool
force/moment and translation/rotation coordinate. A restartable accepted-path
history integrates their trapezoidal work. Trial or failed-increment samples
never enter this ledger. The Procedure restores pose, displacement, material
state, affine and tool dual histories together after failure. Checkpoints
bind the Operator identity and require the complete accepted tool path;
loading a damaged or incomplete path is rejected atomically.

The work integral is a path-discretization record, not a claim of exact first-
law closure for arbitrary plasticity and contact. The existing material-only
Hill–Mandel result remains unavailable when contact is present.

## Verified scope

- One serial, frictionless sphere/cylinder or reviewed triangular rigid tool
  with normalized `0 → 1` prescribed motion; the triangle active branch must
  remain on smooth planar facets or coplanar seams.
- The public `model.step(...)` affine-periodic J2 workflow completes with an
  active moving sphere, records tool work, and publishes a work-available
  constraint contract.
- The endpoint generalized force agrees with an independent finite difference
  of the contact potential with respect to tool motion.
- Partial-run checkpoint/restart matches uninterrupted displacement and tool
  force history. Injected projection failure restores displacement, committed
  material state, accepted factor, tool pose and both dual ledgers; retry
  succeeds.
- Focused local FEniCSx/packaging regression: 125 passed, with the existing
  published-version identity test excluded; the separate two-rank rejection
  check passed 6 tests per rank on 2026-10-17. This is source-checkout
  evidence, not installed-wheel or remote-CI evidence for this follow-on.

## Deliberate boundaries and next gate

Non-normalized physical-time tool schedules, multiple pairs, friction,
mixed pressure, active free or noncoplanar triangle edges, and MPI implicit
contact remain rejected. A general die surface needs either a reviewed smooth
geometry with a differentiable signed gap or a feature-aware nonsmooth
contact law; averaging adjacent facet normals while keeping the old gap is
not a variationally consistent substitute. Before promotion, compare a
curved structural problem against an independent reference and study mesh,
penalty and increment sensitivity, then run installed-wheel and remote gates.

## Smooth geometry extension: source candidate

The follow-on source change separates geometric differentiability from the
penalty law. `SmoothRigidSurface` now supplies the current-pose projection and
`SurfaceGapDifferential`; the latter binds the signed-gap Hessian to that exact
projection, checks finiteness, symmetry, closest-point consistency and the
signed-distance normal nullspace. A public, opt-in central-difference checker
compares the gap gradient and Hessian at selected smooth points without adding
finite differences to every Newton iteration. The plane, sphere and cylinder
share this contract. A separate `SmoothRigidSurface` implementation can enter
the same serial J2 `model.step` route without a new Step provider or contact
law. The affine-periodic composition uses the same admission and Operator.

This is *not* a claim of arbitrary CAD contact. The geometry provider must
guarantee a unique closest point on its declared valid branch and supply the
Hessian of its own signed gap; one spot check cannot certify its entire domain.
For a genuinely smooth CAD/parametric patch, projection and differential still
need independent patch-transition validation. For a physically sharp closed
feature, define its admissible side and feature-specific nonsmooth response
explicitly, with branch changes handled by a tested semismooth or cutback
procedure. Open trims and ambiguous projections remain rejected until their
physical side and edge behavior are declared. Do not silently replace supplied
geometry with normal averaging or a hidden fillet.

The focused follow-on check on 2026-10-17 passed 116 serial FEniCSx tests
covering smooth and facet contact, standard and affine J2, geometry contracts,
and rollback/restart. Six rank-sensitive tests skipped in that serial command
then passed separately under two MPI ranks (6 per rank), confirming the
unsupported distributed implicit-contact route still fails closed. An
externally defined smooth plane was accepted by the public J2 Step; this tests
the extension seam, not an independently validated CAD surface.

The first acceptance matrix should include smooth patch interiors and seams,
noncoplanar edge and vertex crossings, opening/closing, independent energy
directional differences, Newton residual/tangent differences, resultant force
and moment, rejected-trial rollback, and mesh/penalty/increment sensitivity.
Only then should a general imported curved die be admitted. The rationale is
consistent with the [closest-point solvability analysis](https://www.sciencedirect.com/science/article/pii/S0045782508000649)
and the [smooth-geometry contact formulation](https://www.sciencedirect.com/science/article/pii/S004578251000277X);
these references motivate the contract, not certify this implementation.
