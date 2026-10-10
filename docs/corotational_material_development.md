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
| 8 | Bulk viscosity and variable mass scaling when required for source equivalence, with separate work/inertia evidence | Pending; do not silently replace these by damping or a larger fixed increment |
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

## Acceptance and efficiency policy

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
It handles nonaffine planar facets, but does not admit warped-facet projection
or general nonplanar interface integration. Existing pairing keeps its affine
restriction. Geometry inversion alone is not a coverage or quadrature proof.
The private source patch's 4,874 facets pass 24,370 round-trip queries with
maximum local-coordinate error 1.21e-14 (about 0.042 s on this machine).

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
