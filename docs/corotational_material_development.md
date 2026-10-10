# Corotational history materials: implementation and acceptance plan

This is a development contract, not admission of a new global solver. Private
source materials stay outside the distribution. Model retains engineering
assets; constitutive providers own local updates; Operator owns stress/force
conversion; Procedure and State own acceptance and recovery.

## Ordered gates

### Immediate reusable interface work

Keep the responsibilities separate: geometry returns projections and gaps;
initialization defines the original bonded set and reference treatment; the
interface Operator uses one interpolation and its virtual-work transpose;
State owns accepted bond identities; Procedure owns advancement and recovery.
No geometry routine may implicitly activate a bond or adjust model coordinates.

1. True bilinear local projection: private candidate implemented. The batched
   `q1_interior_normal_projection` solves distance stationarity with the exact
   2-by-2 Hessian including curvature, normalized coordinates and per-query
   backtracking. Basix owns basis evaluation. Positive Hessian and strict
   interior convergence are required. Edge/corner minima and global ambiguity
   remain the responsibility of the upcoming search composition, not silently
   clipped local coordinates. No claim of global nearest-point uniqueness.
2. Search composition: a private reference now reuses BVH candidate discovery,
   with surrogate-error padding, then refines on original Q1 facets. Interior
   stationarity is reduced to a polynomial after eliminating one coordinate;
   its admissible roots compete with straight-edge/corner minima, subject to
   one-sided optimality checks. Original Q1 weights are retained. Candidate
   count and physical search radius are bounded explicitly. Identical smooth
   seams use a deterministic facet; competing positions or normals reject.
   This is a scalar narrow-phase reference, not yet production/MPI qualified.
   Coverage and side-swap/integration sensitivity remain required.
3. Initial bond semantics: private `initial_nodal_q1_pairing` consumes an
   explicit original-node eligibility mask and already coincident true-Q1
   geometry. It never moves coordinates or bonds later arrivals. Positive
   nodal area weights integrate the secondary trace basis. The existing
   elastic law and transpose force scatter are reused. Original-source
   initialization/adjustment policy remains separate and unresolved.
4. Global composition: the private response-only adapter now composes the
   nodal bond and two moving lateral-cylinder contacts, with joint disk
   restart and post-commit rollback. Equal three-direction stiffness is
   required; tests check virtual work, finite rigid rotations, and current
   moment balance. This is not admission of a finite-offset anisotropic bond.

Local projection verification uses independent analytic warped graphs, an
independent constrained minimizer, rigid rotations and three length scales,
orientation reversal, scalar/batch equality and deliberate rejection cases.
Backtracking evaluates only basis values, not discarded Hessians; converged
points are frozen. No whole-model performance improvement is claimed.
This is a narrow geometry increment, not a new public solver or a release.

The expanded geometry/interface selection passes 57 tests, including an
independent global optimizer, exact edges/corners, near-edge stationarity,
cross-facet search and ambiguity/budget rejection. A private 256-point-per-side
source-centroid sample accepts 248/256 and 256/256 respectively. Eight queries
are rejected for competing projections/normals; do not silently average their
normals or bind them anyway. This reveals a remaining crease/ambiguity policy
gate before initializing source bonds. The diagnostic search takes about
0.33/0.42 seconds including tree construction on this host; this is neither
full-surface coverage nor a time-step/whole-solver performance certificate.

Follow-up diagnosis corrects the earlier interpretation: those eight centroid
failures were near-seam unique minima incorrectly classified as ties by an
overly broad distance tolerance, not conflicting normals. Local-roundoff
comparison fixes them. Full-surface sampling additionally exposed polynomial
degree-reduction conditioning: roundoff leading coefficients could lose a
finite stationary root. Relative coefficient trimming and polishing the
original two-variable stationarity equations fix that failure. All 54,572
bidirectional Gauss queries now resolve on exact Q1 geometry. All 8,898/4,942
original nodes also resolve. This does not prove original bond eligibility.
For isotropic nodal bonding only, shared-edge normals need not agree, but
coincident competing projections must give identical global-node interpolation;
coincident disconnected DOFs still reject. Unique normals remain the default.

Global composition exposed and fixed a separate adapter defect: first-order
contact vertices were matched against all mesh DOFs, rejecting coincident
independent bodies. Matching now uses the facet's owning-cell DOFs, preserving
topological identity and replacing a full-mesh scan by a four/eight-node scan.
The focused serial selection passes 107 tests; a single bounded two-rank run
passes 42 contact/trace tests on each rank. New response-only execution itself
remains serial. These are targeted regressions, not full release acceptance.

| Gate | Deliverable and acceptance | Status |
| --- | --- | --- |
| 1 | Original compiled material oracle; frozen parameters, tensor ordering, full state, work and source hashes | Local reference established; no commercial-solver validation |
| 2 | Independent response-only batch implementation; proportional, reversal and random-path comparison | Local private implementation checked; not a packaged material |
| 3 | General corotational kinematics; proper frames, invalid-gradient rejection, objective stress transport, noncoaxial independent integration | Private core candidate, tested separately from any material name |
| 4 | Explicit material response contract with independently declared stability and energy capabilities | Private work-only response and serial quadrature lowering tested; independent explicit stability admission pending |
| 5 | Ordinary finite-Hex Step consumes the provider; single-element traction, multi-element patch, failed-trial recovery and checkpoint | Private response-only adapter now advances existing central difference with serial disk restart and whole-step rollback; ordinary model admission and full structure still pending |
| 6 | Multiple analytical rigid tools in that same Step; geometry validity, curvature stability, resultant/moment/work, joint rollback | Private response-only route composes two moving planes or bounded lateral cylinders with existing contact residuals, tool-work ledger and disk restart; curvature stability and ordinary-Step admission pending |
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

`mechanics/_response_hex_explicit.py` now connects response-only material
trials to the existing `problems.explicit_dynamics` Procedure. It does not
implement another time integrator. It records cumulative reference-volume
internal work and separate hourglass energy, and supplies an optional work
monitor without a fabricated recoverable-energy/dissipation decomposition.
Construction requires an explicitly stress-free virgin state. Its caller
increment ceiling is a numerical resolution choice, NOT a certified spectral
bound; no public automatic-stability or ordinary-model admission follows.

Full accepted material history, stress/work fields, deformation gradient and
work totals enter the existing binary checkpoint envelope. Recovery is serial,
same-partition and atomic; inconsistent committed/trial records are rejected.
Identity hashes numeric arrays directly rather than expanding mesh data into
Python lists. Whole-step rollback covers provider cutback and injected failure
after material commit. The selected lifecycle/material regressions pass 50 tests.

Private two-cell source-material dynamics now runs 100/200/400 increments at
2e-9/1e-9/5e-10 source-time units. At the lower initial velocity, successive
displacement differences are 4.72e-7 and 1.18e-7; maximum relative drift in
kinetic + material internal work + hourglass energy is 2.03e-4, 5.08e-5 and
1.27e-5. A fivefold-velocity path initiates weak viscous damage (about 1.76e-4)
and shows comparable refinement. Both interrupted/resumed paths reproduce
continuous displacement exactly. These are free-response material integration
checks, not the original specimen, severe damage validation, or Abaqus solver
equivalence. Source increment/spin equivalence remains unestablished.

Accepted body forces are cached with accepted displacement and material history.
Reaction sampling must not call a history material with a zero time increment
or advance it twice. Numeric checkpoint v3 preserves that cache and the bonded
interface state/energy. Two moving
contact contributions wrap the same body Operator and share the existing
`DynamicEnergyLedger`, without a two-tool special solver. Serial regressions
cover planes and finite lateral cylinder patches, nonzero tool work, identical
continuous/restarted fields and histories, and failure after the outer contact
has committed. These are small integration tests, not the original four-tool
specimen or a penalty/time convergence certificate.

`rigid_cylinder(..., axial_interval=(a, b))` measures the interval along the
supplied axis from the supplied axis point. Rigid motion transports both.
Strict-interior lateral projection is exact; rim/end queries return
`unsupported_rim`, not an invented cap or ignored contact. Infinite-cylinder
behavior remains the default. Source-tool admission must check this interval
throughout the accepted path, not just at initialization.

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

## Accepted artificial damping work (development gate)

The private response-only lifecycle now composes the existing volumetric
viscosity force with material and original-node interface trials. It records
artificial work separately, using the positive element-power contraction after
checking its agreement with assembled force dotted into the displacement
increment. Opposite nodal forces can cancel to a tiny negative floating-point
dot product during rigid translation; the check uses a contraction-scale
roundoff bound, not an arbitrary absolute energy tolerance.

The explicit policy declares physical reference density, a fixed supplied wave
speed, and a volume-equivalent current length `L0*cuberoot(J)`. The rate is the
end-configuration displacement secant. These are inspectable numerical choices,
not claims about Abaqus's characteristic length or velocity sampling. Initial
acceleration remains a Procedure/case responsibility; an initially resting,
stress-free body has zero initial damping force. No finite-rotation exactness is
claimed for this discrete velocity estimate. The conditional damping-adjusted
ceiling does not turn a caller-supplied elastic ceiling into a certified global
material/contact bound. Variable mass scaling remains unimplemented.

The private restart schema is now v4 and includes accepted artificial work,
damping rate and policy identity. Old private payloads are rejected. Targeted
serial checks: 27 pass, including body/bond/two moving contacts/damping combined
restart, outer-commit failure rollback, analytic homogeneous dilation work,
rigid translation, and damped-ceiling rejection. No public capability promotion
or new MPI claim follows from these checks.

## Original-size cost measurement

A confidential source-mesh preflight executed three prescribed affine trials
on 255,041 Hex8 cells / 888,222 displacement DOFs. This is not the industrial
loading simulation. Peak resident memory was approximately 1.91 GB; the
unprofiled trial times were 1.815, 1.870 and 1.829 seconds. Extrapolating 100,000
body-only increments gives about 51 hours before contact, interface, output and
whole-Procedure transaction costs. This estimate is a planning warning, not a
completion-time guarantee. Direct optimized compilation of the supplied source
preserved archived history responses to scaled error 4.8e-16; its 255,041-point
material-only batch was 1.63x faster than the private NumPy translation. Do not
advertise that local material gain as a solver speedup.

## Original-size lifecycle follow-up

The complete imported topology (255,041 cells, 888,222 displacement DOFs) now
prepares the original-node bond hypothesis and four bounded cylindrical tools.
This is a **startup integration pilot**, not the full 0.1-second analysis.
The first large checkpoint exposed per-contact-point dictionary/JSON expansion:
the interrupted process reached approximately 9 GB resident memory. Serial,
explicitly nonportable nested numerical State now uses typed projection arrays
with an explicit validity mask and status catalog. The common checkpoint codec
owns array storage and hashes. Legacy/global portable projection records remain
unchanged; no cross-partition capability is added.

The same two-increment pilot now completes and saves in 38.60 seconds including
31.32 seconds preparation, with peak RSS 3.23 GB and 167 KB JSON metadata.
Its second increment takes about 3.05 seconds rather than the earlier observed
12.74 seconds with dictionary transactions. These are single-run diagnostic
measurements, not a statistically controlled whole-solver performance claim.
The numerical auxiliary archive is approximately 318 MB. The delivered private
case marks full simulation and Abaqus cross-validation as false.
Independent continuous-four-step and two-step/restart/two-step executions now
agree bitwise on all eleven compared nodal/material response arrays, accepted
history and body evidence. Restore takes 0.54 seconds; peak RSS is 3.53 GB for
the continuous run and 3.35 GB for the resumed run. The physical endpoint is
only 4e-10 seconds: this demonstrates lifecycle correctness, not developed
forming response. The focused serial regression passes 57 tests (six MPI-only
skips); one bounded two-rank regression passes 41 tests per rank.

The next gates remain material-owned stability, explicit variable-mass State
and inertia accounting, body/body contact outside the original bond set,
source initialization evidence, and complete loading/refinement comparisons.
Neither smaller checkpoint overhead nor a completed startup run closes them.

## Current-configuration deformable contact: bounded slice

`_deformable_q1_contact` separates the fixed original bond mask from current
contact candidates. It retains the original four-node basis, reuses the BVH
and frictionless penalty law, and transfers equal/opposite residuals by the
transpose interpolation. A separate residual adapter composes with the existing
central-difference Procedure, accepted-state transactions and energy ledger.
No new case-specific integrator is introduced.

Ten focused tests cover open/closed analytical response, cross-facet motion,
finite rigid rotation, resultant and moment balance, independent energy
variation, actual two-body integration, bitwise disk restart, atomic rejection
and decreasing energy error under time refinement. This slice is serial,
one-sided reference nodal integration with a caller-specified increment ceiling;
it does not certify nonlinear stability or reproduce exact hard contact.

Active off-normal edge/corner projections are rejected, because substituting a
single facet normal there would violate the distance-potential gradient and
moment balance. The private source's 1,000-node unbonded hypothesis fails this
geometric admission at an ambiguous normal. A closed-solid sign/normal-cone
contract is still required; an open interface patch cannot silently stand in
for the solid exterior. Hence this contribution is not inserted into the
original case merely to obtain a completed run.

## Primary design references

- [Abaqus mass scaling](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-massscaling.htm):
  distinguish physical density from altered numerical inertia, preserve the
  physical time scale for rate-dependent laws, and evaluate added mass and
  response sensitivity. A requested element increment is not automatically a
  certified whole-system increment after contact and damping contributions.

- [Abaqus smooth-step amplitude verification](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-amplitude.htm):
  the fifth-order `s^3*(10-15s+6s^2)` law is distinct from our established
  half-cosine amplitude. The new `quintic_step` retains exact analytic first and
  second derivatives and passes the documented .12-to-.9 example. Explicit
  amplitudes now enter rigid-motion schedule identity and restart tests.

- [MOOSE contact module](https://mooseframework.inl.gov/modules/contact/index.html)
  and [mortar generation](https://mooseframework.inl.gov/source/constraints/AutomaticMortarGeneration.html):
  separation of contact enforcement and geometric projection/segmentation.
  Our bounded local projector is not a mortar implementation.

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
