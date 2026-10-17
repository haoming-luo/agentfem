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
