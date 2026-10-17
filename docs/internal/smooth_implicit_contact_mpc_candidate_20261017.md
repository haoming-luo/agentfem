# Smooth rigid contact and fixed-tool periodic MPC candidate — 2026-10-17

Status: source-level, experimental serial candidate. This note is not a
release claim or a validation of general die forming.

## Numerical and product contract

The Model continues to own one `RigidContactPair`. A solver-neutral contact
Operator composes the existing boundary trace, current-pose projection and
penalty point law. For a negative signed gap `g`, its reference-area penalty
potential is `k g²/2`. The current-trial structural residual is the first
variation; on a fixed smooth active branch, its Newton contribution is
`k(n ⊗ n + g ∇²g)`. Analytical spheres and cylinders supply the signed-gap
Hessian, including the cylinder's current rotated axis. Triangular facets
retain the previously reviewed planar-branch derivative and reject active
free/noncoplanar edges. The surface is reprojected for every Newton and
line-search trial; no old gap or force is reused.

The ordinary finite-strain J2 Step admits one such serial, frictionless tool
with strong boundary conditions. The separate affine-periodic J2 Step admits
one **fixed** tool. The Procedure assembles the complete material-plus-contact
residual and tangent before `TᵀR` and `TᵀKT`; the same complete residual is used
for the initial norm, every Newton update, line-search trial and periodic
reaction. Material State retains trial/commit/rollback and accepted-increment
checkpoint ownership. The fixed rigid tool owns no mutable pose state, so its
coordinate work is exactly zero. Its boundary traction nevertheless makes a
material-only Hill–Mandel report invalid; that output is rejected rather
than silently reported.

## Executable evidence

- Sphere and cylinder, interior and exterior admissible sides: pointwise
  energy variation and full active-branch tangent agree with independent
  central differences. A rotated cylindrical axis is also covered.
- On actual tetrahedral and hexahedral DOLFINx boundaries with CG1 and CG2
  traces, sphere and cylinder penalty energy, assembled nodal residual and
  tangent action agree with a separate UFL potential differentiated by
  DOLFINx. This checks quadrature and DOF assembly, not only the local formula.
- Ordinary public `model.step(...)` finite-J2 tetrahedral and hexahedral cases
  with a sphere or cylinder complete with active contact. A moving sphere also
  has a generalized tool force conjugate to the derivative of its penalty
  potential, and preserves accepted displacement, material state and pose
  after an injected energy failure.
- The affine-periodic public Step has active fixed-sphere and fixed-triangle
  contact cases. The converged *total* residual is reduced below the tested
  tolerance. The projected contact tangent matches a finite difference of
  the projected contact residual. A partial fixed-tool solve saves and resumes
  to the uninterrupted displacement, material state and constraint-dual
  history. A changed tool is rejected by checkpoint identity. An injected
  projection failure after a partial solve restores the last accepted
  displacement, material and dual ledger; a subsequent retry continues.
  Moving periodic contact is rejected in capability discovery and Step
  construction.
- A pure-Python candidate wheel includes all three new contact modules. An
  isolated installation of that wheel ran the public sphere example from
  outside the source checkout to a completed result with active contact.
  Both sphere and cylinder source-checkout examples also completed.

These are bounded local and small-global tests. No independent curved-tool
structure benchmark, mesh/penalty/increment sensitivity study, full native
installed-wheel release gate or remote CI is claimed here. One bounded
two-rank admission group confirmed that standard and affine-periodic
distributed implicit contact are rejected; this is not distributed numerical
validation. The same geometry assumptions and tolerances must be checked for
a real tool before promotion.

## Next gates

1. Validate a curved structural case against an independent reference and
   report force, moment, penetration, energy/work, mesh, penalty and increment
   sensitivity; do not use one visually plausible stress plot as the gate.
2. Develop a reviewed transition rule for noncoplanar triangle seams, free
   edges and vertices, or use a genuinely smooth surface representation.
   Neither a zero curvature tensor at a crease nor silent normal averaging is
   an acceptable consistent tangent.
3. Admit moving rigid tools with periodic MPC only after adding the accepted
   pose, tool work and rollback/restart evidence to the same affine lifecycle.
4. Add owner/ghost residual and tangent assembly, distributed reactions and
   rank-portable restart before MPI implicit-contact admission. Friction and
   mixed displacement-pressure remain separate numerical gates.
