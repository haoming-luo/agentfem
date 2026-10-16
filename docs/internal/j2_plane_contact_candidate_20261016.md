# Finite-J2 analytical-plane contact candidate — 2026-10-16

Status: bounded experimental composition, not a released industrial forming
capability. This note is an evidence snapshot for the 0.4.3 decision gate.

## Scientific and architecture boundary

- Ordinary three-dimensional strong-boundary finite-strain logarithmic J2.
- One analytical rigid plane with frictionless one-sided penalty contact; fixed
  or following prescribed translation/rotation. The existing contact potential
  contributes residual and semismooth tangent to the J2 Newton form.
- The J2 quadrature transaction remains the State owner. Accepted energy
  records material stored energy, plastic dissipation, contact penalty
  potential, natural/strong-boundary work, and generalized tool work separately.
- Failed finalization restores displacement, material history, energy ledger
  and tool pose. Serial checkpoints include the contact identity and accepted
  work history; a changed penalty is rejected on restore.
- No triangular tool geometry, edge/vertex transition, friction, periodic MPC,
  mixed pressure, cross-rank restart or MPI composition claim.

## Evidence collected on the local FEniCSx 0.11 environment

The source checkout passed 91 related tests across finite-J2, existing contact,
public workflows and validation after the rotating-tool additions. Seven
new/contact tests also passed from an isolated installed pure-Python wheel,
with `agentfem.__file__` resolved under its install target. The relevant
platform CI and independent structure-level reference remain open gates.

The one-cell uniaxial elastic limit is independently solved from
`E log(1+u)/(1+u) + k u = F` for a fixed plane and from
`E log(1+u)/(1+u) + k(u+d) = 0` for a moving plane. Numerical fields match
these scalar roots within the declared test tolerance. A separate active
plastic-contact patch has positive PEEQ, balances strong reaction + contact
resultant + natural force to `1e-6` absolute, and retains the same accepted
state after interrupted serial restart.
The elastic spring pair is checked at penalties 250, 500 and 1000, not merely
at one tuned penalty. This validates the bounded implementation's penalty
response against that scalar reference; it is not a mesh-independent contact
limit for general geometry.

| Active plastic-contact path | 3 increments | 6 increments | 12 increments |
| --- | ---: | ---: | ---: |
| Right displacement | 0.012399086 | 0.012403449 | 0.012405173 |
| Maximum PEEQ | 0.013502257 | 0.013518648 | 0.013528577 |
| Relative accepted-path ledger residual | 0.0063534 | 0.0026861 | 0.0009795 |

For six increments, prescribed translation produced tool work `0.016685405`
with relative ledger residual `1.54e-5`; translation plus a `0.02` rad small
rotation produced tool work `0.024131152` with residual `3.94e-5` and a
nonzero contact moment. These are internal small-model checks, not an external
forming validation or a claim of exact energy conservation at finite step.

After PR #110 entered main, the combined source branch passed 58 focused
FEniCSx tests spanning J2/contact, rigid contact and geometry-worker
regressions. The native extension was not built in that source environment,
so 99 native-only tests skipped; their prior PR #110 CI result must not be
misrepresented as a combined-branch native run. The public example was run
once continuously and once as three accepted increments plus checkpoint and
restart. Final displacement, plastic deformation gradient, PEEQ, plastic
dissipation and energy history matched exactly. Timing/event metadata did
not, as expected. A final nodal-displacement XDMF and a result manifest were
written without pretending that XDMF contains the unsmoothed quadrature state.
For a tilted plane, fixed-displacement differentiation of the contact
potential matches the accepted generalized force (translation plus moment)
on a two-cell-per-axis patch. A separate one/two/three-cell-per-axis scan
gave tool work `0.02413115 / 0.02332779 / 0.02293842`; the latter spatial
difference is smaller. This is internal spatial consistency only, not an
independent reference or penalty convergence claim.
An isolated macOS arm64 native wheel built from the combined branch resolves
both `_body_pipeline` and `_hex_native` under its installed target. A focused
installed-package run passed 159 J2/contact/native/geometry-worker tests.
The public example completed from this wheel with the same displacement,
plastic gradient, PEEQ, dissipation and energy history arrays as the source
run. This still does not substitute for full-platform CI or Windows hardware.
The composed MPI limit is exercised in one bounded two-rank test: capability
inspection reports unsupported with `AFM-J2-CONTACT-MPI`, and Step creation
raises that same diagnostic before assembly. Standalone distributed J2 remains
independent and supported; the limit applies only to this contact composition.

## Next gates before release promotion

1. Complete final platform CI and review old-provider dispatch on the merged
   candidate without changing the established explicit production case.
2. Extend the nonuniform contact patch to step/penalty sensitivity and include
   a public independent structure-level reference. Homogeneous spring and
   internal spatial-consistency checks are insufficient for broad contact
   validation.
3. For discrete rigid tools, implement reviewed current-geometry projection,
   facet/edge/vertex transitions and consistent residual/tangent in a separate
   candidate; compare a triangulated flat tool to the analytical plane.
4. Only then compose periodic MPC, ensuring both contact residual and tangent
   participate in the same reduction, trial/rollback and work ledger.

The discrete-tool candidate should be a boundary-quadrature contact
**Operator**, not a special J2 constitutive model. At each Newton trial it
must project current slave quadrature points onto current tool geometry,
assemble both the penalty residual and its branch-consistent linearization,
and report ambiguous facet/edge/vertex transitions rather than freezing a
stale normal. A smooth triangulated flat tool must first reproduce the
analytical-plane patch and cross-facet path; curved and nonsmooth tools then
need independent force/work and mesh/penalty checks. The
[FEBio surface-to-surface formulation](https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/)
motivates integrated contact constraints and explicit linearization rather
than assuming a node-to-facet force is mesh-neutral. The
[DOLFINx-MPC implementation](https://github.com/jorgensd/dolfinx_mpc)
provides kinematic reduction infrastructure, not a substitute for unilateral
sliding contact: assemble the complete material-plus-contact operator before
reducing it with the periodic transform. Neither source is evidence that our
unimplemented triangular composition works.

Performance is a separate evidence lane: this composition has no measured
full-run acceleration claim. Optimize body/contact/data flow against a frozen
installed candidate and accepted physical stations, per
`skills/agentfem-performance`; do not change penalty, inertia or time grid to
manufacture a timing win.
