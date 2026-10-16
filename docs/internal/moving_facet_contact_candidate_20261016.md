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
moving-plane J2 route for translation and simultaneous z rotation through
0.2 rad on the bounded planar fixture:
displacement, committed material state, generalized force, penalty potential,
and accepted tool work agree within the regression tolerances. The tool-work
dual also agrees with a fixed-state finite difference of potential with
respect to the prescribed motion factor at 0.01 and 0.2 rad. A serial
checkpoint at both rotation magnitudes resumes to the
uninterrupted state, while an intentionally failed energy update restores
the prior accepted pose, displacement, and plastic history. The public example
completes with `--tool-travel -0.006 --tool-rotation 0.01` and reports nonzero
tool work. An active free tool edge fails closed during the global solve and
preserves the prior accepted displacement, material state and pose. These are
local plane-limit checks, not independent curved-tool
validation or physical-time simulation.

A bounded 0.2-rad moving-plane fixture was also scanned without changing its
penalty or motion path. With two cells per axis, 3/6/12 accepted increments
gave tool work `1.486774 / 1.508690 / 1.514850` and contact potential
`0.636862 / 0.636358 / 0.636349`. With six increments, 1/2/3/4 cells per
axis gave tool work `1.766163 / 1.508690 / 1.409196 / 1.363561` and
contact potential `0.923614 / 0.636358 / 0.543069 / 0.502516`. Differences
between successive levels shrink for these *integral* observables, but the
spatial series is not yet grid-independent. The discrete energy residual is
not monotone under the 3/6/12-step scan, so this is not an energy-convergence
claim. No pointwise maximum is used as a universal accuracy certificate.
At the fixed two-cell/six-increment resolution, penalties 250/500/1000/2000
gave tool work `1.138341 / 1.508690 / 1.841833 / 2.108082` and contact
potential `0.688932 / 0.636358 / 0.517584 / 0.375544`. This sensitivity
is expected for a penalty approximation and must not be mistaken for an
algorithmic performance comparison or a penalty-independent solution.

One frictionless prescribed rigid triangle tool, serial 3D blocked
displacement, reference-area quadrature, and active projection on planar
interior/coplanar seams only. No free/noncoplanar edge transition, friction,
multiple pairs, contact/strong-BC overlap, periodic MPC, or distributed
implicit assembly. A non-normalized independent motion schedule is rejected
rather than silently remapped to the static Step; both capability discovery
and `model.step(...)` report the same stable diagnostic. Full installed-wheel
and remote CI gates must still pass for this candidate.

At source commit `515d3468`, the macOS arm64 CPython 3.11 wheel
(`sha256:d8b6b06e7555f06153eebff11df7fa85bf3f0d9abbc713c0db82bc94c75b4f69`)
and sdist (`sha256:92866bdfc5aba1a9e61c8b1b75b5ec86f43fcfd846b44de3c316fe7d93c526b5`)
were built from clean source. The isolated installed-wheel `release_gate.py
--dist ... --smoke --require-platform macos` completed successfully. This is
local platform evidence, not a release tag or proof on Windows/Linux. The
remote PR gate remains a separate promotion decision.

## Next review gates

1. Compare a non-planar smooth tool against an independent structural
   reference, including force and moment, mesh/penalty/increment sensitivity,
   and facet-switch stability. A planar-limit equivalence is insufficient.
2. Implement reviewed edge/vertex or smooth-surface transition derivatives
   before claiming general finite sliding on a tessellated die.
3. Reduce both material-plus-contact residual and tangent through periodic
   MPC, with identical line-search and reaction semantics. The present affine
   J2 route assembles a material-only full-space form before `T^T R` and
   `T^T K T`, whereas the strong-boundary J2 contact route adds the external
   contact operator before Newton. A reviewed additive full-space operator
   seam, trial-pose refresh and joint acceptance transaction are required;
   merely admitting a contact parameter to the affine Step would silently
   omit contact from line search and reactions. The provider capability probe
   and Step lowerer now both reject any registered boundary model on this
   affine J2 path with `AFM-J2-CONTACT-MPC`; the material-only periodic
   regression still passes.
4. Establish distributed owner/ghost assembly and portable restart evidence.

The accepted-path trapezoidal work choice is consistent with the contact
energy accounting discussed in the
[Abaqus contact-energy example](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEEXARefMap/simaexa-c-contactenergy.htm).
That reference also warns that contact energy balance depends on increment
size and contact transitions; its published numbers are not AgentFEM evidence.
