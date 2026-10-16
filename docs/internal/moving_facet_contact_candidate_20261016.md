# Moving-facet implicit contact candidate — 2026-10-16

Status: stacked experimental serial slice after the fixed-facet J2 candidate.
It is not a general die-forming or periodic-contact release claim.

## Design and ownership

- Model retains the existing `RigidContactPair` and `RigidBody` with a
  `PrescribedRigidMotionSchedule`. The ordinary static Step admits only its
  normalized 0-to-1 form without an independent physical-time amplitude; the
  Step amplitude remains the sole load-path owner.
- Search retains one reference-geometry triangle BVH. Each Newton trial maps
  current slave points into the rigid reference frame, queries the BVH, then
  maps closest points and normals back to the current pose. It does not reuse
  a previous gap, normal, or force.
- Operator evaluates the penalty potential and the fixed-current-pose
  displacement residual/tangent on a smooth planar facet branch. The
  generalized tool dual contains the contact resultant and moment about the
  current rigid reference point, conjugate to prescribed translation and
  axis-angle coordinate along the proportional motion path.
- Procedure records trapezoidal force--coordinate work only on accepted
  increments. State snapshots and restores the tool factor with displacement,
  material trial/commit, and the accepted energy ledger. A failed increment
  cannot advance pose or work. Checkpoint identity includes the motion
  schedule through the contact pair.
- Result reports force, moment, potential, tool work, and their provenance;
  energy residual is a discrete accounting check, not a general first-law
  proof for arbitrary contact and plasticity.

## Local evidence and admission boundary

The moving coplanar triangulated tool agrees with the existing analytical
moving-plane J2 route for translation and small simultaneous z rotation:
displacement, committed material state, generalized force, penalty potential,
and accepted tool work agree within the regression tolerances. The tool-work
dual also agrees with a fixed-state finite difference of potential with
respect to the prescribed motion factor. A serial checkpoint resumes to the
uninterrupted state, while an intentionally failed energy update restores
the prior accepted pose, displacement, and plastic history. The public example
completes with `--tool-travel -0.006 --tool-rotation 0.01` and reports nonzero
tool work. An active free tool edge fails closed during the global solve and
preserves the prior accepted displacement, material state and pose. These are
local plane-limit checks, not independent curved-tool
validation or physical-time simulation.

One frictionless prescribed rigid triangle tool, serial 3D blocked
displacement, reference-area quadrature, and active projection on planar
interior/coplanar seams only. No free/noncoplanar edge transition, friction,
multiple pairs, contact/strong-BC overlap, periodic MPC, or distributed
implicit assembly. A non-normalized independent motion schedule is rejected
rather than silently remapped to the static Step; both capability discovery
and `model.step(...)` report the same stable diagnostic. Full installed-wheel
and remote CI gates must still pass for this candidate.

## Next review gates

1. Compare a non-planar smooth tool against an independent structural
   reference, including force and moment, mesh/penalty/increment sensitivity,
   and facet-switch stability. A planar-limit equivalence is insufficient.
2. Implement reviewed edge/vertex or smooth-surface transition derivatives
   before claiming general finite sliding on a tessellated die.
3. Reduce both material-plus-contact residual and tangent through periodic
   MPC, with identical line-search and reaction semantics.
4. Establish distributed owner/ghost assembly and portable restart evidence.

The accepted-path trapezoidal work choice is consistent with the contact
energy accounting discussed in the
[Abaqus contact-energy example](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEEXARefMap/simaexa-c-contactenergy.htm).
That reference also warns that contact energy balance depends on increment
size and contact transitions; its published numbers are not AgentFEM evidence.
