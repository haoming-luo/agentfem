# Corotational history materials: implementation and acceptance plan

This is a development contract, not admission of a new global solver. Private
source materials stay outside the distribution. Model retains engineering
assets; constitutive providers own local updates; Operator owns stress/force
conversion; Procedure and State own acceptance and recovery.

## Ordered gates

| Gate | Deliverable and acceptance | Status |
| --- | --- | --- |
| 1 | Original compiled material oracle; frozen parameters, tensor ordering, full state, work and source hashes | Local reference established; no commercial-solver validation |
| 2 | Independent response-only batch implementation; proportional, reversal and random-path comparison | Local private implementation checked; not a packaged material |
| 3 | General corotational kinematics; proper frames, invalid-gradient rejection, objective stress transport, noncoaxial independent integration | Private core candidate, tested separately from any material name |
| 4 | Explicit material response contract with independently declared stability and energy capabilities | Private work-only response and serial quadrature lowering tested; independent explicit stability admission pending |
| 5 | Ordinary finite-Hex Step consumes the provider; single-element traction, multi-element patch, failed-trial recovery and checkpoint | Real Hex8 internal-force assembly and downstream trial rollback tested; ordinary time-integrating Step and full checkpoint pending |
| 6 | Multiple analytical rigid tools in that same Step; geometry validity, curvature stability, resultant/moment/work, joint rollback | Pending; reuse existing rigid surfaces, contact law and accepted work ledger |
| 7 | Imported geometry/section/interface audit and controlled incremental whole-model runs | Local geometry audit identifies nonaffine, globally noncoplanar traces and small facet warpage; general pairing and whole-model runs pending |
| 8 | Bulk viscosity and variable mass scaling when required for source equivalence, with separate work/inertia evidence | Serial Hex8 bulk-viscosity force/power and response-trial composition tested; accepted time integration, damping stability and variable mass scaling pending |
| 9 | Independent reference curves, mesh/time/penalty/hourglass sensitivity, performance and installed-use acceptance | Pending; commercial output is required before cross-validation claims |

Deletion and interface damage are not prerequisites for reproducing a source
that defines neither. General-purpose implementations of those features remain
separate roadmap items. A complete model may need gates 4--8 even though its
material-point behavior is already reproducible.

## Kinematics chosen for the candidate

For the right polar decomposition `F = R U`, the polar-frame stretching rate
is `R^T D R = sym(Udot U^-1)`. Given only endpoint gradients, the candidate
explicitly chooses a linear right-stretch path between `U0` and `U1`.
It computes its rate integral using the symmetric matrix
`B = U0^(-1/2) U1 U0^(-1/2)`:

`increment = sym(U0^(1/2) log(B) U0^(-1/2))`.

Reference material axes then rotate this increment into material coordinates.
Endpoint stress returns to spatial coordinates using `R_new Q_reference`.
This is generally NOT the difference of endpoint logarithmic strain tensors.
It is not a claim to reproduce Abaqus' discrete strain increment algorithm.
An actual intervening nonproportional stretch path can give another integral;
time refinement is mandatory. No automatic spin-sensitive provider admission
follows: relative spin and its time integration remain undeclared here.

`constitutive/_corotational.py` is vectorized across points and uses small
symmetric eigensystems rather than nested point/quadrature loops. It retains
no constitutive history. Read-only returned arrays, zero-length partitions and
proper-frame checks are covered by focused tests. It is deliberately private
until the response, stability and global lifecycle gates close.

## Energy and stability: do not falsify the existing contract

The private `ExplicitMaterialBatchOutput` now carries Cauchy stress, state and
reference-volume internal-work increment without a fabricated tangent. Storage
and dissipation remain independently optional, not zero-filled. Negative work
is legal during unloading; reported dissipation is evidence to assess, not
silently clipped into a passing thermodynamic claim. A requested time reduction
rejects the fixed-increment quadrature trial before state mutation.

Serial `update_explicit_material_points` reuses `MaterialQuadratureState` and
bounded point chunks. Every response is validated before writing trial fields.
The `ExplicitQuadratureResponse` owns real P/S/work fields, not placeholder
DPDF/SENER fields. Its existing Hex8 residual now exposes a separate private
trial evaluation which reuses geometry, mass, Piola-force scatter and hourglass
contributions. Missing stored energy stays `None` in the element response.
Downstream rejection restores all trial state and response fields. No public
Step admission or MPI claim follows from this force-assembly slice.
Explicit commit/rollback also includes accepted P/S/work, using the existing
quadrature transaction machinery. A late failure during response commit restores
both previously accepted material history and response fields. The focused
post-fix selection passes 45 tests, including fault injection; the private
120-increment real-Hex material exercise also passes again.

Current finite-Hex lowering requires reference-volume stored energy and
dissipation plus a reviewed tangent or envelope. A legacy material's cumulative
specific internal work is not stored energy. Reporting zero nonelastic energy
does not establish zero physical dissipation.

The next design must preserve separately: accepted internal work, optional
recoverable storage, optional dissipation, physical volume/mass convention,
and unavailability of a thermodynamic decomposition. A work-only material must
not receive the existing stored-energy verification claim. Do not construct a
fake symmetric tangent: damage updates may have nonsymmetric derivatives.
Response-only execution needs a separately justified explicit stability screen
before skipping numerical tangent generation. An initial elastic modulus alone
is not a global nonlinear stability proof.

Fixed-old-state numerical differentiation of the private source response at
three perturbation sizes confirms a nonzero antisymmetric part of dP/dF, both
at an elastic endpoint and at a damaging endpoint. This excludes reuse of the
existing symmetric-tangent envelope unchanged. These sampled derivatives are
diagnostic evidence, not a complete-path stability envelope. Next implementation
must separate effective wave-speed/time-resolution policy from implicit
consistent-tangent requirements, and record its scope and refinement evidence.

## Acceptance and efficiency policy

The private bulk-viscosity contribution reuses the reference-average Hex8
gradient and current F. With tension-positive volumetric rate `theta`, it uses
`q = rho*b1*c*L*theta - rho*(b2*L*min(theta,0))^2`. The contribution to Cauchy
stress is `q I`; artificial power is `current_volume*q*theta >= 0`.
Current density, wave speed and characteristic length are explicit inputs,
not inferred from a damaged material or a silently scaled mass.

`FiniteUniformHexResidual.assemble_bulk_viscosity` scatters this force through
the same existing DOF layout. The response-only trial accepts an explicit
velocity/coefficients pair and combines forces while returning artificial
power separately from material work/storage/dissipation. Existing Steps do
not enable it automatically. A per-cell frozen-configuration velocity-Jacobian
bound is returned; it is not a complete nonlinear global stability claim.
The owning Procedure must still integrate accepted artificial work and include
damping in its increment policy and restart before global admission.
Five focused checks cover analytical compression/expansion, rigid motion,
force and moment balance, all 24 assembled velocity derivative columns, invalid
coefficients, and downstream material-trial rollback. The combined existing
material/element/architecture selection passes 56 tests.

The response-only slice passes 87 selected tests including real serial
quadrature, Hex8 force assembly, rollback, kinematics, geometry and architecture.
The existing interface selection additionally passes 61 tests after reuse of
the new inverse-map adapter. These are development selections, not a release
or MPI qualification. A private original-material exercise passes 120 imposed
finite-deformation increments on two real Hex8 cells, including damage and
large superposed rotation. Prescribed deformation does not test a dynamic
time integrator, contact, stability or the full source model.

The source geometry audit rules out reusing coplanar parallelogram pairing
unchanged. The new private `_q1_surface_geometry` adapter uses Basix Q1 basis
values and derivatives, normalized batched Newton inversion, positive convex
corner checks, physical reconstruction and reference containment checks.
It handles nonaffine planar facets; the separate `q1_surface_pull_back` also
inverts on-surface queries on bilinear warped facets with a convex projection.
This retains real warpage and rejects off-surface queries, folded projections
and failed inversion. It does not perform nearest-point projection or admit
general nonplanar interface integration. Existing pairing keeps its affine
restriction. Geometry inversion alone is not a coverage or quadrature proof.
The private source patch's 4,874 facets pass 24,370 round-trip queries with
maximum local-coordinate error 1.21e-14 (about 0.042 s on this machine).
The subsequent warped-map check covers both original surfaces: 13,643 facets,
68,215 queries, maximum local-coordinate error 1.25e-14. Mother-surface and
patch query batches take approximately 0.070 s and 0.038 s respectively on
this machine. The related mapping and existing Q1 interface tests pass 33
checks. No cross-surface coverage or traction integration claim follows.

`q1_surface_quadrature` now integrates the actual bilinear surface Jacobian
and oriented normal without replacing a warped facet by two planes. It uses
Basix derivatives and positive tensor-product Gauss weights. Analytical warped
graph area, integrated oriented area, orientation reversal, planar trapezoid
area and empty batches are tested; related Q1 mapping/interface selection:
35 passed. Private source surfaces converge in area at orders 2/3/5 to
781.947900530708 and 781.754905210090 in the model's area units. Their roughly
0.025% mismatch is geometry evidence, not an integration error or an overlap
certificate. Before admitting pairing, measure reference gaps and coverage;
do not silently increase coincidence tolerance or flatten either surface.

A subsequent private bidirectional BVH diagnostic samples 54,572 true Q1
Gauss points. The maximum distances to triangular diagnostic surrogates are
0.003446 and 0.002817 in source length units; about 9.17% and 9.61% of sampled
area lies farther than 1e-4. These are sampled area estimates, not complete
coverage certificates. The new `q1_triangulation_distance_bound` bounds the
surrogate's geometric Hausdorff discrepancy using the corner-height range over
the admitted convex projected polygon. The target bounds are approximately
roundoff and 1.30e-7, respectively, so the observed mismatch cannot be explained
by triangulation alone. Mapping/quadrature/interface selection: 39 tests pass.

The source uses surface-based cohesive contact with no explicit eligibility
override. Abaqus's default limits bonds to original contacts; initialization
can change that active set. Thus a full-surface permanent elastic tie cannot
be assumed equivalent merely because all three stiffnesses match. Next work
must separate geometry projection, reference-gap/initial-contact policy and
accepted bond identity. Inspect secondary-node contact eligibility and initial
adjustment before deciding the original bonded set. No claim about the actual
Abaqus-initialized active set follows from Gauss-point distance samples alone.
Finite-offset subtraction alone also does not establish finite-rotation
objectivity or moment balance. Keep these gates explicit while developing the
general interface, rather than loosening the current coincidence contract.

Local candidate checks: 15 kinematic tests plus 26 architecture tests pass.
The independent noncoaxial oracle integrates the rate with 64-point Gaussian
quadrature. Random 3D frame tests check objective stress and virtual-power
invariance; invalid/complex/singular inputs fail before producing an increment.
In a 512-point, three-repeat single-thread microbenchmark, the same kernel's
median time is 0.03115 s through scalar Python calls and 0.001605 s in one batch
(19.4x). Compared strain arrays agree to the asserted absolute tolerance 1e-15.
This is a local development measurement, not an installed-release performance
certificate or an industrial-model speed claim.

Use serial focused checks first; one bounded MPI group at a time when the new
global path exists. No full CI or installer build is needed for this private
kinematic slice. Measure real batch and full-step costs separately. Retain
bounded scratch allocation for material plane searches; no array proportional
to `all_points * all_search_angles * full_tensor` is required. Do not claim
whole-model speedup from a local kernel benchmark.

## Primary design references

- [Abaqus surface-based cohesive behavior](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-cohesivebehavior.htm):
  original-contact eligibility and the influence of initialization on bonding.
  The source interaction must not silently become a permanently tied surface.

- [Abaqus VUMAT](https://docs.software.vt.edu/abaqusv2025/English/SIMACAESUBRefMap/simasub-c-vumat.htm):
  tensor shear, material corotational axes, Green--Naghdi rate, specific internal
  versus nonelastic energy, and independently declared effective moduli. Its
  current ABI differs from some older source signatures.
- [MOOSE ComputeFiniteStrain](https://mooseframework.inl.gov/source/materials/ComputeFiniteStrain.html):
  incremental corotational approaches and explicit decomposition choices.
  This motivates exposing the numerical path, not claiming implementation
  equivalence with MOOSE's incremental-rotation algorithms.
- [DOLFINx coordinate maps](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.html):
  nonaffine pull-back is nonlinear and requires convergence control.
- [DefElement Q1 quadrilateral](https://defelement.org/elements/examples/quadrilateral-lagrange-gll-1.html):
  the four-node tensor-product basis must not be replaced by a triangular field.
- [Abaqus explicit dynamics](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-expdynamic.htm):
  linear and compression-only quadratic artificial bulk viscosity. Matching
  these formulas alone does not establish identical characteristic lengths,
  velocity sampling, stable increments or whole-solver equivalence.
