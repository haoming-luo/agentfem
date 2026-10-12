# Corotational history materials: implementation and acceptance plan

This is a development contract, not admission of a new global solver. Private
source materials stay outside the distribution. Model retains engineering
assets; constitutive providers own local updates; Operator owns stress/force
conversion; Procedure and State own acceptance and recovery.

## Explicit response data flow (2026-10-12)

Accepted prescribed-tool work history now has a columnar numeric checkpoint
encoding, lowered through the existing numeric-tree archive. It retains every
station, the same work quadrature and history fingerprint; legacy JSON input
remains readable. The new encoding is selected only for the existing serial,
same-partition numeric contact checkpoint path. It is not a new MPI portability
claim. All columns are checked before State publication, including shape,
dtype, finite values, identity and strictly increasing accepted times.

In-memory rollback tokens share immutable blocks of at most 64 stations rather
than copying the full accepted history on every increment. Token creation is
constant-size, while durable save and restore remain linear in history length.
Old tokens remain valid across further commits and alternate continuation
branches. This separates transaction cost from durable archive encoding without
introducing another work ledger or dropping accepted observations.

Response-only Hex8 assembly reuses the prepared immutable first Piola array
after assigning the corresponding quadrature field. This avoids a redundant
field readback without dropping the field, its checkpoint identity, or its
trial/commit/rollback lifecycle. Displacement and viscosity velocity extraction
share a private detached-gather seam; NumPy remains the default implementation.
The seam is not a public native-backend or distributed-execution guarantee.

Private, opt-in native gathering was compared on identical original-mesh
continuations. Three 1,200-increment early/late/end pairs reduced advance time by
2.6%, 4.5% and 6.35%, respectively, with bitwise-identical final saved field and
material arrays. The early pair also retained exact equality at all 120 saved
reporting stations. A shorter late pair gave 6.2%; these are stage-dependent
paired timings, not a measured fresh full-run speedup or a default-backend
performance claim. No increment, mass, contact tolerance, or material law was
changed. A separate immutable-native-output experiment was not selected:
ownership tests passed but measured material-batch time increased.

Performance admission must distinguish preparation, restart, advancement,
checkpoint, export and full-process wall time. Overlapping body and contact
timers are not additive; speeding the hidden branch need not shorten the
critical path. Repeated same-checkpoint comparisons and a complete-run timing
are required before publishing a general percentage. Fixed-operator bound
contribution percentages are not CPU-time percentages.

Directional-increment research must include current stretch/rotation and the
stress-dependent geometric derivative. Agreement of a sampled symmetric-part
upper comparison does not certify a nonsymmetric/softening material, global
contact dynamics, or the complete loading path. Audit archives now need the
actual evaluated input time and content identity: filenames and target step
counts do not establish which material increment was sampled, and repeated
copies of one audit must not be counted as independent time stations.

## Fixed-operator increment attribution (2026-10-12)

`time._nodal_inertia.diagnose_nodal_increment` explains supplied nodal
stiffness and damping majorants without changing any mass or state. It reuses
the inertia planner's input contract and solves the row condition
`h*h*k + 2*h*c <= 4*m`, where `h = requested_increment / safety_factor`.
Stiffness and damping sources are added before deriving the limit; taking
the minimum of separate source time limits is not an equivalent calculation.
The report includes controlling node/component indices, each source's demand
ratio, finite node-limit quantiles, and explicitly unbounded zero rows.

This internal diagnostic certifies neither operator coverage nor a nonlinear
path. Its node indices refer to the supplied local ordering. Initial elastic
rows cannot silently stand in for damaged material, deformed geometry or
active deformable contact. Passing a target increment after planning mass for
that same increment is expected by construction, not independent validation.
The report does not automatically select an increment or alter a Procedure.

## Bounded contact retries and filtered geometry (2026-10-12)

Reviewed triangle topology now has a bounded byte-keyed cache (at most four
connectivity payloads of 4 MiB each). Vertex count and exact connectivity gate
reuse; coordinates, duplicate vertices, facet areas, normals and geometry
fingerprints are still checked/recomputed. The Q1 broad phase keeps coordinate
welding but numbers welded vertices by first facet occurrence, avoiding
topological renumbering under ordinary motion. This is computation reuse, not
permission to retain old contact projections or skip deformed-geometry checks.
Targeted surface, import, search and deformable-contact regression: 120 passed.

The polynomial Q1 oracle also has a rare recovery when floating-point
elimination produces no admissible stationary candidate. Exact rational
Gershgorin bounds must first establish strong convexity on the entire
parameter square. Bounded Newton witnesses are rounded back to binary64;
an exact normal-cone residual divided by the positive convexity modulus must
certify a parameter error no greater than `2**-48`. Unproved/nonconvex cases
remain rejected. This recovers cancellation-limited interior minima near a
corner without flattening a warped facet or accepting a guessed boundary point.

The swept Q1 separation proof first tries outward-rounded interval dot
products on proposed separating axes. A strict common sign excludes zero
from the trilinear control hull; uncertain arithmetic still reaches the exact
rational subdivision. The filter changes computational cost, not physical
contact tolerance. Tests include dyadic impacts over widely separated scales
and comparisons with the rational proof.

An unresolved swept path now raises `ContactPathResolutionError` with stable
code `contact_path_unresolved` and query indices. It is still a `ValueError`;
no contact force or state is accepted after this rejection. An internal
explicit advancement primitive accepts a smaller increment while retaining
its ordinary field, residual, material, prescribed-input and inertia rollback.
Its enclosing Procedure must own the macro transaction and intermediate work
ledger. Variable inertia must be prepared once at the macro station and held
constant through subincrements. The private serial pilot exercises bounded
dyadic retries, terminal-failure rollback, energy accounting and checkpoint
continuation. This does not yet expose a general adaptive explicit public API,
claim MPI subcycling, or certify arbitrary curved temporal trajectories.

The private compiled contact experiment additionally batches independent Q1
ray classifications. Unresolved arithmetic falls back to the exact reference;
two independent oriented rays must still agree. Frozen-geometry timings are
not whole-simulation speedups, and neither successful retries nor timing gains
establish vendor equivalence, full loading completion or mass sensitivity.

## Cancellation-limited closest-point recovery (2026-10-11)

Nearly affine Q1 elimination can produce a tiny admissible quadratic root and
an enormous inadmissible root. The small root now uses the stable root-product
formula rather than companion eigenvalues; unchanged physical stationarity
and constrained-minimum checks still decide admission. The rare exact
shared-edge ordering proof now accepts feasible edge witnesses near corners,
not just strict interior witnesses. Binary64 inputs are compared as rationals;
equal or unproved ordering remains an error. Synthetic tests cover both facet
orders, strongly separated root magnitudes and near-edge warped projections.

This repaired a private original-grid checkpoint reconstruction; it is not
evidence of completed loading or nonlinear stability. Numerical background:
[Kahan, quadratic root cancellation](https://people.eecs.berkeley.edu/~wkahan/Qdrtcs.pdf).

## Additive force assembly (2026-10-11)

The Hex8 scatter accepts a second cell-force contribution on the same topology.
Body and artificial-viscosity responses remain separate for work accounting;
assembly adds their components before nodal accumulation, without constructing
a full combined `(cells, 8, 3)` intermediate. The serial and chunked paths match
the previous precombined force bitwise. Targeted real-runtime tests: 29 passed;
both additive paths also passed on two MPI ranks. This changes neither the
material law nor numerical inertia, increment admission, or energy meaning.
Private compiled scatter experiments remain outside the distributed runtime.

Explicit material response transport now recognizes constructor-validated,
immutable byte payloads. An unchanged response crosses the provider boundary
with a fresh object and independent array metadata, without rescanning its
values. Replaced fields, changed shape/dtype/strides/data pointer, and subclasses
still take full constructor validation. This is not a security boundary against
malicious Python. Contract, quadrature ownership and rollback tests: 55 passed;
the private original-grid three-increment replay retained bitwise identical
stress, state and deformation-gradient arrays.

## Same-trial reuse and ordered contact identity (2026-10-11)

Profiling the private original-mesh startup identified duplicate full Hex8
Jacobian admission in response and damping evaluation. The response-only trial
now reuses its admitted gradient for damping inside the same guarded scope.
Standalone damping still checks complete cell geometry; provider mutation and
downstream failure retain rollback. Already canonical contact point IDs avoid a
second projection reorder/validation, retaining owned IDs and immutable fields.
No material, load, time step, or acceptance threshold is changed by this reuse.

Targeted real-runtime regression: **75 passed, 3 two-rank-only tests skipped**.
This is not MPI validation or completed original-case validation. The original
mesh startup is still inactive contact; speed measured there must not be reported
as full forming performance. Later loading, damage, source mass-scaling identity,
and commercial reference curves remain separate gates.

## Ordered gates

### Per-point closed-solid separation reuse

The explicit contact lifecycle now retains conservative positive path-distance
bounds independently per candidate. A single approaching point no longer forces
all separated candidates through narrow-phase projection. Expired/nonpositive
bounds receive a fresh projection; no closest point or normal is fabricated for
certified-separated points. Current closed-surface geometry validation and the
outside-to-outside swept-path rejection remain in place. Open patches cannot
use this optimization. Accepted state is unchanged by evaluation, and restart
continues to reconstruct geometry rather than trusting a serialized cache.

The focused contact/search/path suite passes **74 tests**. A mixed eight-point
case projects four points instead of eight with unchanged force and potential;
when the remaining certificates expire all eight are searched again. This is
not a measured full-model speedup, continuous collision detection, self-contact,
or a nonlinear stability certificate.

The accepted contact residual also exposes a frozen-projection PSD row-bound
contribution using the existing penalty-stencil machinery. It uses the actual
numerical nodal inertia and active point weights, including partial-projection
indexing. Independent dense normal-constraint eigenvalues are bounded in the
focused test; separated states contribute zero. This is **not** a total time
step bound: moving-projection curvature, body operators and damping are absent.
The source-case ledger must report that distinction explicitly.

### One enclosing response transaction

Response-only material preparation now produces the complete validated batch
without assigning quadrature fields. The Hex8 trial's existing outer transaction
then owns state, stress and work assignment together, avoiding duplicate nested
snapshots of the same large arrays. Standalone material updates retain atomic
assignment. Fifty focused tests include provider mutation, late assignment of
each response channel, downstream failure, commit failure and damping lifecycle.
This establishes unchanged rollback behavior; no measured end-to-end speedup is
inferred from the reduced snapshot count alone.

### Exterior projection ambiguity versus separation

A closed-solid exterior query can have unresolved nearest-face choices while
remaining strictly separated. Such a query now uses the existing conservative
Q1 distance enclosure and independent ray classification to establish zero
contact force; it does not choose a face, average normals or invent interpolation
weights. Interior and zero-bound ambiguities still reject, as do open-surface
ambiguities. Fresh evaluation during restore uses the same rule. The original
private early-loading run stopped at increment 1684 on this distinction; its
checkpoint is not evidence of a completed forming trajectory.

### Smooth Q1 contact curvature

The private local distance-potential Hessian now eliminates the two closest-point
coordinates analytically. Its Schur complement retains moving-projection
geometry: `K = K_fixed - S`, with `S` positive semidefinite at a strict interior
projection minimum. Separate assembled diagonal upper and negative-magnitude
enclosures do not hide geometric softening. Independent differences of the
existing contact force check every matrix column on flat and warped surfaces;
translation, rotation, scaling, singular projection and activation rejection
are included (31 focused contact tests pass).

This is an instantaneous smooth-interior diagnostic, not a certified nonlinear
time step or an implicit Step. Seams, rims, ambiguous projection, activation,
negative material curvature and whole-increment changes remain separate gates.
The original-case separated-contact pilot is not silently promoted by this work.

### Orientation-only admission cost

Repeated orientation checks now use the sufficient near-identity condition
`||F-I||_infinity < 1/2`, with the original LU determinant outside that strict
ball. The path `I+t(F-I)` proves sign preservation inside the ball; this is
neither an approximation to J nor a replacement for whole-cell Jacobian checks.
Bulk viscosity computes its required determinant once instead of twice. The
focused orientation, material-array, finite-Hex and damping selection passes
65 tests, including inversion, singularity, overflow and fallback boundaries.
No physical configuration, numerical inertia or accepted time step changes.

The separate batched determinant arithmetic retains LU for cancellation,
overflow and extreme nonzero inputs. The latter guard is necessary because an
intermediate product can underflow before a large third factor restores its
mathematical scale; checking the final sum alone is insufficient. A constructed
regression and 37 determinant/damping/explicit-lifecycle tests pass. A corrected
255041-matrix microbenchmark measured approximately 0.0090 s versus 0.0193 s
for NumPy LU; this component timing is not whole-simulation acceleration.

If the reference perturbation screen is inconclusive, an affine-removed check
now writes the current cell as `F X + r + translation`. The reference singular
value lower bound, multiplied by `sigma_min(F)`, bounds the permissible
nonaffine Jacobian perturbation. Orientation-reversing F is never admitted;
inconclusive cases still use Bernstein subdivision. Large proper rotations,
shears, random nonaffine perturbations and the existing inversion cases pass
47 focused tests. This improves admission cost, not allowable distortion or
the material's finite-strain domain.

Dynamic finite-Hex force scatter now reduces the complete owned-cell array
with three indexed reductions instead of looping over geometry chunks. Ghost
reverse accumulation remains unchanged; initial mass retains its original
summation order for checkpoint identity. Serial focused regression passes 24
tests; a bounded two-rank test independently checks old/new scatter and global
force sums on shared nodes (one test per rank). This is not admission of the
private response-only material route to MPI.

### Conservative separated-contact reuse

Closed Q1 contact now distinguishes exact projection records from force-free
separation certificates. Convex boxes and directional slabs of the original
four control vertices provide distance lower bounds; bounded Q1 subdivision
tightens them near warped patches. Triangular BVH padding only controls search
pruning and never substitutes a P1 field for a Q1 face. A numerical closest
distance may request more refinement, but is never itself admission evidence.
Relative nodal travel reduces the accepted lower bound. Only a strictly positive
remaining bound bypasses search; otherwise the existing narrow phase runs.
This applies to the existing reviewed, non-self-intersecting closed-solid scope,
not general continuous collision detection. Roundoff margins are conservative
engineering guards, not formal interval-arithmetic certification.

Bound-only responses carry no invented normal or closest point. Checkpoints
retain accepted geometry and reconstruct projection evidence on restore;
separated-contact continuous/restarted trajectories and histories agree exactly
in the focused regression. The original private 1,000 unbonded queries all have
positive refined bounds (minimum approximately `1.25515e-4` in source units).
That is a geometry result, not ALL EXTERIOR coverage or a completed simulation.

Far nodes on a larger closed-solid candidate domain now retain positive
separation bounds and explicit indices for the subset with exact projections.
The original open-patch search-escape rejection remains unchanged. A private
original-geometry audit covers 64,170 unbonded parent exterior nodes against
the complete patch: 1,000 need near-field projections, all have positive initial
separation bounds. A 20-increment original-mesh pilot includes this domain as a
strict separation audit, rejecting first possible body/body contact. It does
not certify active deformable-contact stability, symmetric contact, self-contact,
or commercial ALL EXTERIOR equivalence.

### Long-clock increment comparison and unchanged reference cells

The original-mesh pilot subsequently restored the accepted increment-66
checkpoint and completed its original 100-increment horizon at `6a13c303`.
It saved a new checkpoint and displacement visualization; physical time reached
`1e-4` s, not the source's `0.1` s. The endpoint ledger reports relative balance
error about `0.0065022` under its work-only accounting. This is nonzero early
contact/recovery evidence, not thermodynamic closure or commercial validation.
The initial frozen inertia plan remains heavily scaled (added/physical mass
about 767.77), so this trajectory cannot establish quasi-static fidelity.

An attempted restore into 1,000 configured increments was correctly rejected
because the checkpoint retains a 100-increment horizon. No checkpoint metadata
was edited to bypass compatibility. A separate continuous 1,000-increment pilot
is used for further bounded investigation and overlapping-history comparison;
completion and equivalence must be checked from its actual output.

That pilot stopped normally at its 2,400-second wall budget after 814 accepted
increments (`0.000814` s); checkpoint and displacement output were saved. The
recorded physical quantities at increments 70/80/90/100 exactly match the
increment-66 restart run (wall-clock timing excluded). At the last sampled
increment, 810, ledger relative error is `3.93e-6`, but numerical kinetic energy
is about 46% of accounted energy. A small ledger residual therefore does NOT
establish a quasi-static trajectory. The approximately three seconds per
increment would imply roughly 83 hours for 100,000 increments if cost stayed
constant; this is a feasibility extrapolation, not a validated full-run ETA.
Before spending that time, prioritize inertia/response sensitivity and the
combined controlling operators. Do not enlarge dt or accelerate source time
silently to fit a requested runtime.

The original-mesh initial-row ablation at `65bbb035` localizes this issue:
at target `1e-6`, body/hourglass alone requires added/physical mass 741.38,
versus 767.77 for the combined plan. Removing the body term leaves 25.39;
removing damping leaves 764.38. These are diagnostic omissions only, never
alternate admissible dynamics. This points first to the anisotropic body
majorant and thin-cell geometry, not contact-search speed or damping removal.
The current scalar modulus envelope discards material directions; investigate
a directional, provider-declared bound against independent assembled spectra
before using a tighter plan. An initial elastic tangent must not be promoted to
a bound over evolving damage/rotation. Mass and time sensitivity must still be
separated at common physical times.

The private Hex8 operator now provides `directional_cell_row_majorant` for a
provider-supplied symmetric PSD first-Piola upper tensor. It maps that tensor
through the reference gradients, adds fixed hourglass stiffness, and reduces
chunk-local element matrices to absolute nodal rows. It neither derives a
material envelope nor silently changes the source pilot's inertia plan.
Independent tangent-action column assembly verifies every row and the diagonal
majorant inequality; chunk invariance, invalid tensors, empty batches and a
thin anisotropic cell are covered. The latter demonstrates why a largest-modulus
scalar can be unnecessarily conservative, not a measured original-case gain.
The focused directional and finite-Hex selection passes 25 tests. Original
material tensor conversion, frame identity and damaged-path admission remain
separate gates before deployment in that case.

An initial-state-only private source audit at `4397a270` checks all nine
gradient directions against the compiled virgin material update; the tensor
shear/engineering-stiffness conversion agrees to the reported floating-point
precision. Source orientations used here are identity. On the original mesh,
the directional initial tensor gives added/physical mass 228.79 at target
`1e-6`, versus 767.77 with the previous scalar envelope (about 70.2% lower).
At `1e-7` the corresponding diagnostic is 1.52 versus 6.99. These are initial
planning comparisons only: no modified dynamic run, damaged/rotated-state
upper-bound proof, or achieved runtime improvement is claimed. Even the tighter
initial plan remains strongly mass-scaled. Keep the original pilot unchanged
until response sensitivity and evolving-domain checks justify another policy.

Fixed-inertia time refinement now explicitly separates the planning increment
from the integration increment in the private pilot. At common time `1e-4` s,
100 steps of `1e-6` and 200 steps of `5e-7` retain the same mass plan. Endpoint
maximum displacement differs by 0.597%, tool work by 0.00229%, and internal work
by 0.0592% (relative to the finer result). However, the two fixed-tool normal
forces differ by about 18--19%; their magnitudes are small during startup, but
that does not justify calling force convergence complete. A third refinement
is needed. These comparisons neither test mass sensitivity nor the full load.

The third run completed 400 steps at `2.5e-7`, again reaching `1e-4` with the
same initial mass plan. Exported geometry and topology arrays match exactly.
Successive displacement-coefficient relative Euclidean differences are 0.7474%
and 0.3891%; these are coefficient norms, not volume-weighted continuum L2
errors. Absolute maximum component differences are `4.11e-9` and `1.19e-9`
in source length units. Fixed-tool endpoint force differences decrease to
about 6--7%, but peak displacement is nonmonotone and the coefficient difference
ratio is only 1.92. This is an early-load refinement trend, not a second-order
certificate or a passed engineering force tolerance. The finest ledger relative
residual is about 0.00104. All three runs saved checkpoints and displacement
fields; no full `0.1`-second trajectory or mass-sensitivity admission follows.

An original-mesh nodal-contact pilot accepted 66 increments, then rejected
increment 67 because subtracting accepted times reconstructed `1e-6` with a
floating-point tail above an overly narrow tolerance. Provider limits now account
for endpoint clock ULPs as well as increment roundoff. A clock too coarse to
resolve the limit fails explicitly; a genuine `1e-8` relative excess remains
rejected in tests through the 100,000th increment. This does not increase the
physical/provider limit or add automatic retries. The rejected trial left the
accepted checkpoint intact.

The subsequent geometry fast path uses a sufficient perturbation bound, not a
small-displacement tolerance. The reference Bernstein determinant lower bound
and Q1 edge-based Frobenius upper bound imply a uniform lower singular-value
bound. A guarded perturbation strictly smaller than one quarter of that bound
cannot cross a singular Jacobian along the reference-to-current homotopy.
Inconclusive cells still receive full Bernstein admission, including large
rotations and possible inversions. This is floating-point guarded admission,
not interval arithmetic. Random distorted/scaled cells are compared with the
full check and independent Basix derivatives.

On the same original-mesh checkpoint, a 30-increment loaded segment (815--844)
took 90.15 s before and 81.14 s with this geometry admission, under profiling;
sampled physical histories match exactly. This approximately 10% reduction is
segment-specific, not a full simulation claim. Increasing kernel chunk size
to 8192 was not measured: re-preparation changed numerical-inertia byte identity
and restart correctly rejected it. No identity check was bypassed.

Material stretch-domain checks additionally use sufficient Gershgorin bounds
on F.T F and retain SVD for inconclusive points. The declared domain and its
existing endpoint tolerance are unchanged. Independent SVD-oracle tests cover
random rotated states, boundary values, reflections and invalid input. This
does not derive or strengthen the provider's material envelope.

The next targeted optimization keeps material batch size independent of
geometry/scatter chunking: `material_batch_points` controls provider calls
without changing reference preparation or inertia bytes. Contact numeric
snapshots encode repeated status runs before catalog reduction, preserving the
same sorted catalog and inverse indices for arbitrary strings. Finite-Hex
force and bulk-viscosity contractions use algebraically equivalent batched
matrix products; material and State ownership are unchanged. The combined
targeted selection passes 73 tests, with one MPI-only test skipped (not passed).

Private performance experiments keep compiled source material and optional
small-matrix kinematics outside the package. Their full-size kernel gain is
not advertised as a software-wide speedup. The loaded 30-increment checkpoint
segment is the comparison unit; neither it nor the microbenchmarks demonstrate
the requested complete 0.1-second simulation or its commercial equivalence.

### Explicit contact integration choice

The DOLFINx boundary adapter now accepts `integration="nodal_lumped"` for
first-order triangle/quadrilateral geometry and CG1 displacement. Tributary
reference areas are the existing positive quadrature integral of each shape
function; contact is then evaluated at its node. Default Gauss integration is
unchanged. Facet-local point IDs occupy a disjoint namespace, including under
repartitioning, so old Gauss histories cannot silently become nodal histories.
CG2/curved higher-order lumping is rejected, not approximated by corner sampling.

Four serial tests cover area, force, moment, virtual work, cylindrical contact
onset, and rejected combinations. One two-rank test passes for identity and area.
This is a discretization option, not vendor hard-contact/edge-contact equivalence.
Both [Abaqus general contact](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-contactgeneral.htm)
and [MOOSE contact](https://mooseframework.inl.gov/moose/syntax/Contact/index.html)
distinguish node/surface enforcement from face integration. The private startup
audit exposed a meaningful onset difference for a curved rigid tool; it must be
recorded explicitly in any comparison rather than hidden by advancing time.

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
| 7 | Imported geometry/section/interface audit and controlled incremental whole-model runs | True-Q1 pairing, bounded closed-solid contact and full original-mesh startup/restart tested; full loading, complete contact coverage and independently confirmed bond eligibility pending |
| 8 | Bulk viscosity and variable mass scaling when required for source equivalence, with separate work/inertia evidence | Serial accepted viscosity and variable-inertia transactions/restart tested; evolving material/contact/damping increment policy, mass sensitivity and vendor equivalence pending |
| 9 | Independent reference curves, mesh/time/penalty/hourglass sensitivity, performance and installed-use acceptance | Isolated review-wheel and startup acceptance tested; complete loading convergence and commercial comparison pending |

Deletion and interface damage are not prerequisites for reproducing a source
that defines neither. General-purpose implementations of those features remain
separate roadmap items. A complete model may need gates 4--8 even though its
material-point behavior is already reproducible.

### Evidence checkpoint, 2026-10-11

The bounded implementation has a 321-test targeted selection; subsequent polar
reuse and callback-transaction fixes passed their 42- and 38-test selections.
An isolated installed review wheel (`0.4.1+review.7d5de8ea`) passed 89 selected
tests without importing repository source. These selections overlap and must
not be added as unique tests. The quadrature-layout MPI check passed 18 unique
tests on two ranks; it does not admit the combined private response route to MPI.

All 255,041 original cells additionally completed 120 prescribed affine
stretch/shear/rotation updates, including damage. The exact-gradient material
reference gives maximum scaled stress error about 2.1e-11 and force/moment
balance below 1.1e-14. This is a manufactured kinematic path, not solved tool
loading or global equilibrium. Reusing the polar spectrum reduced the measured
median body-trial time by about 4.6% in this exercise; the approximately 13%
standalone kinematics gain is not a whole-solver speedup.

The installed-wheel four-increment original-mesh startup and its interrupted
restart agree bitwise in 11 accepted array fields and exactly in recorded
history. The endpoint remains 4e-10 s with zero displacement: no completed
0.1 s source simulation, force--displacement curve, or commercial validation
is claimed. Keep this capacity/recovery evidence separate from nonzero
manufactured-path evidence and eventual full-loading evidence.

The review package is private and excludes original confidential source assets.
Do not substitute zero inelastic work reported by a source routine for a proven
energy decomposition. Likewise, an initial mass/time screen is insufficient
for an evolving damaged/contacting model. Abaqus' own
[explicit-analysis guidance](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-expdynamic.htm)
calls for checking inertia when accelerating quasi-static analysis; a small
kinetic-energy ratio alone is not a replacement for response sensitivity.
Its [cohesive-contact documentation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-cohesivebehavior.htm)
also distinguishes original-contact eligibility from later contact. Our selected
near-coincident mask is a recorded hypothesis, not proof of the source solver's
actual initial bond set.

### Next bounded engineering increment (not an unbounded proof project)

Current integration sprint prioritizes (1) combined stiffness/contact/damping
inertia planning, (2) an actual nonzero accepted contact trajectory with recovery
and refinement, and (3) original-mesh feasibility before committing to a long
loading run. A user-reported commercial runtime of 3--4 hours is useful context,
not a matched performance benchmark: hardware, core count, increment history,
mass changes and output cadence have not been supplied.

The private radial-tool envelope now retains both signs of the geometric
Hessian. For active sphere/cylinder radial penalty its eigenvalues include
`k` and `k*(1-R/r)` (plus an axial zero for a cylinder). An explicit positive
radial-distance floor bounds the negative curvature of exterior contact.
Interior cavities have nonnegative tangential curvature. These fixed-reference
trace envelopes do not cover friction, follower area or a deformable master;
invalid/rim projections and out-of-domain endpoints reject. Mass scaling does
not remove the negative curvature. Independent force differentiation, signed
trace assembly, changed mass and rigid translation are tested. This remains a
private screening component, not automatic curved-contact admission.

A private nodal inertia planner combines supplied diagonal majorants of
stiffness and lagged-velocity damping through `h^2 K + 2 h C <= 4 M`, with
`h=target_dt/safety_factor`. It retains named additive contributions and never
reduces accepted mass. Its fixed-coefficient stability interpretation requires
symmetric PSD operators; a damaged or geometrically indefinite response does
not inherit that assertion. The existing VariableScaledInertia still owns
budgets, accepted mass, artificial momentum/energy and checkpoint identity.
Real Step tests compare an independent damped recurrence, bitwise disk restart
and budget-failure rollback. Random noncommuting SPD matrix tests check both
the matrix inequality and discrete amplification eigenvalues.

The Hex8 operator supplies a compact unscaled nodal majorant from a separately
declared material modulus, including its existing hourglass contribution.
It constructs only chunked 8x8 upper operators rather than global or element
24x24 tangents. Independent distorted anisotropic dense operators verify the
majorant. Neither this conversion nor the mass planner establishes the original
history material's modulus over the eventual damage path.

The response-only viscosity screen can now explicitly bind the same reviewed
numerical-inertia owner as the Explicit integrator. The Procedure checks that
identity through residual wrappers before advancement. Its frozen-geometry
damping spectral bound decreases by the minimum nodal mass factor; physical
density, viscosity force and artificial work are unchanged. A changed inertia
identity rejects restart. This is a conditional correction to the existing
caller-ceiling screen, not a new material stability assertion. Optional bulk
viscosity row majorants retain the default force path's allocation size.

The next response-only step policy should accept provider-declared effective
moduli or a reviewed wave-speed bound with units and provenance, combine it
with current element geometry, and account separately for interface/contact
penalty and damping restrictions. Do not force generation of a full implicit
consistent tangent merely to advance explicit dynamics. This follows the
practical separation illustrated by the
[VUMAT EFFMOD interface](https://docs.software.vt.edu/abaqusv2025/English/SIMACAESUBRefMap/simasub-c-vumat.htm):
automatic effective moduli can be nonconservative for strongly nonlinear laws,
and user-specified effective moduli provide an alternative input.

Such an engineering estimate must not be mislabeled a universal nonlinear
stability theorem. Admission should instead require a bounded demonstrated
domain, accepted-state energy/penetration checks, an explicit added-mass budget,
and predeclared response tolerances under increment and mass refinement. For
anisotropy, the declared bound must cover the relevant directions; copying
isotropic bulk/shear numbers without that justification is not sufficient.
The core owns the input/diagnostic contract and policy lifetime, while private
providers own material-specific bounds. Begin with small nonzero force-driven
tests and a representative contact submodel; only then spend a long-run budget
on the original mesh. No additional material library, implicit solver, or full
installer is required for this next gate.

For throughput, profile a nonzero accepted Step before further micro-optimization.
The measured original-mesh body trial alone is about 1.85 s/increment on this
host: extrapolating 100,000 identical-cost increments gives about 51 hours,
before contact/output, not an actual full-run ETA. Do not launch that expense
just to replace missing admission evidence with a long process. The next
performance comparison should examine chunk/temporary-array cost and, only if
kinematics dominate, an explicitly separate midpoint-rate integration candidate
against the current finite endpoint-stretch integral. Any cheaper integration
route must pass noncoaxial/objectivity tests and time refinement of stress,
state and work, without silently changing an existing provider's convention.

## Kinematics chosen for the candidate

### Provider-owned material increment restriction

The private response protocol now accepts an optional stateless
`explicit_increment_limit(request)` query returning an immutable
`ExplicitMaterialIncrementLimit`, with one positive finite limit per point in
the request's time units. A nonempty `explicit_increment_policy_identity`
must identify the algorithm and all fixed policy parameters. Dynamic history
must come from `request.state_old`, not an uncheckpointed callback closure.
Limits may govern local material-update accuracy or admissibility; they are
not automatically a geometric wave-speed, contact, or damping bound.

The query is numerically enforced before the corresponding batch's response
update. Exceeding it raises `explicit_material_increment_exceeded` with the
requested/allowed increments, policy identity and process-local point index.
A late chunk rejection preserves all trial/committed fields. Previously
evaluated chunks have not committed State. No automatic retry or enlargement
of a step follows, and existing providers without this optional contract keep
their caller-ceiling behavior. The response Hex adapter can explicitly require
the query, includes its identity in restart compatibility, and rejects identity
changes after construction. Provider evaluation also rejects mid-call changes.

Focused contract/quadrature/response/inertia selection: 70 tests pass, including
nonzero dynamic response with bitwise disk restart, a later accepted-time limit
change followed by whole-Step rollback, and a rejected late batch with a
correct global-in-this-process point index. Equality allows only floating-point
time-reconstruction noise, not a physical relaxation of the limit. This is an
enforced material restriction, **not** the still-pending combined body/contact/
damping increment strategy. No private source material is auto-promoted.

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

An explicit private `integration_rule="midpoint"` candidate now evaluates
`sym((U1-U0) inv((U0+U1)/2))`. The existing exact integral remains the default;
the returned method tag distinguishes them, including empty/stationary batches.
This is a numerical approximation to the same linear-stretch path, not an
equivalent finite logarithmic increment or a vendor algorithm. A provider
choosing it must include that choice in its numerical/restart identity.
Independent noncoaxial quadrature demonstrates second-order subdivision;
rigid rotations, material axes, virtual power and invalid gradients are tested.
The combined kinematics selection passes 27 tests.

Five alternating 65,536-point moving-batch runs measured median exact/midpoint
cost ratio 1.257 with chunk size 8,192. Maximum strain difference was 1.35e-8
for that generated input, not a universal accuracy bound. A private source
material path at 120/240/480 increments shows roughly fourfold decrease of
the rule difference in stress, state and internal work on each refinement.
This does not change the original case or establish full dynamic accuracy.
`tools/benchmark_corotational_rules.py` preserves the method distinction and
source fingerprint in its output instead of advertising algorithm equivalence.

The follow-up original-mesh manufactured-path run at `bc07740f` used all
255,041 cells / 888,222 displacement DOFs for 120 imposed increments. Body
trial median was 1.7230 s versus the earlier exact-rule run's 1.8459 s (6.66%
less time); total diagnostic elapsed was 233.07 s versus 248.67 s. These are
separate single runs, not an alternating repeated whole-body benchmark.
The same-rule exact-gradient point comparison had maximum scaled stress/state
differences 2.07e-11 / 2.25e-10; relative force/moment residuals were below
1.12e-14 / 7.62e-15. Peak RSS was 2.50 GB. This tests large-mesh lowering, not
the error between integration rules, which is assessed separately above.
No four-tool loading, free dynamics, interface forces or coupled stability was
tested by this imposed path. Neither the production case default nor its
checkpoint identity changed. The previously delivered review wheel/package
remains based on `7d5de8ea` and does not contain these later candidates.
After that run, the combined increment-limit, response/quadrature, variable
inertia and kinematics selection passed 97 tests in the real FEniCSx runtime.
No new MPI qualification or full pressing acceptance is inferred.

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
for the solid exterior.

The follow-up closed-solid candidate uses two independently oriented rays and
true bilinear patch intersections for sign, while retaining exact Q1 closest
coordinates for force transfer. Edge closure, orientation and vertex
manifoldness are checked; nonsingular ray classifications must agree. Force
directions follow the signed-distance gradient, not an arbitrarily averaged
facet normal. Concave-edge moment and virtual-work checks pass, as do warped
surface, rigid-rotation and real two-body disk-restart tests. This still assumes
a reviewed non-self-intersecting solid: it does not certify arbitrary geometry
or provide continuous collision detection.

On the private complete patch exterior (8,944 quads), all 1,000 initially
unbonded candidates are classified successfully. Importantly, all are outside,
at distances 0.0001255--0.0042484 in source units. Earlier open-patch negative
normal gaps must **not** be interpreted as proven solid overclosure. The source
initialization still needs independent confirmation; neither these openings nor
the unknown commercial adjustment should be silently bonded. The private case
can opt into this candidate contact, but its initial interface candidates are
not yet general ALL EXTERIOR contact.

Affine and convex planar Q1 closest points now avoid the general polynomial
solve; nonaffine planar interpolation still uses Basix Q1 inversion. The inverse
is converged to geometric roundoff rather than the looser default interpolation
tolerance, which was insufficient near shared seams. Genuine warpage retains
the polynomial path. Strictly identical contact configurations reuse the
accepted response, but material updates still execute at every increment.
Exactly stationary polar-frame kinematics avoid redundant spectral operations;
invalid nonidentity gradients are still checked. These are bounded algorithmic
optimizations, not evidence of full-loading throughput.

## Fixed numerical inertia foundation (not variable mass scaling)

The private immutable `FixedScaledInertia` retains a separate physical nodal
mass, constructs positive numerical inertia from explicit nodal factors, and
records added mass without changing constitutive density. The existing central
difference integrator consumes its inverse diagonal. Its optional immutable
identity now participates in the common checkpoint: scaled/unscaled or changed
factor restarts reject atomically rather than silently changing the dynamics.
This first identity is serial and not portable across partitions.

Response-only energy monitoring can report numerical kinetic energy for the
actual discrete equation alongside physical-mass kinetic energy and their
difference. A two-body test verifies the exact uniform mass/time rescaling
relationship and identical interrupted/restarted history. This is not an
automatic stable-step certificate, and no fixed factor is presented as
equivalent to the source's per-increment variable mass scaling. That next gate
must own accepted mass updates, acceleration consistency, added-mass momentum
and kinetic contributions, rejected-step rollback and restart.

### Variable numerical inertia transaction

The private serial `VariableScaledInertia` now participates in the existing
Explicit Procedure's increment and checkpoint transactions. A deterministic,
stateless, identity-bearing external policy proposes nondecreasing nodal
factors at the accepted configuration. The configured added-mass budget is
enforced before mutation. At fixed velocity, mass changes contribute
`0.5 * v.T * delta(M) * v` and `sum(delta(M) * v)` to numerical energy and
momentum inputs; accepted acceleration is rescaled to preserve the old force
before displacement prediction. These are not physical external work.
The artificial angular-momentum input about the coordinate origin is also
retained; equal/opposite physical forces do not cancel this numerical input.

The common dynamic ledger includes that separate numerical energy input in its
balance and checks restored history against inertia State. Inertia, mechanical
fields and material/contact state roll back together, including an injected
failure after material commit or during disk restoration. Independent restart
reproduces fields, history and mass state exactly. Free translation verifies
the added momentum/energy analytically. Prescribed nodal motion, MPC, MPI and
partition-portable recovery remain rejected for this bounded variable route.

This is the transaction mechanism, **not** a validated automatic mass-scaling
policy. A source-case policy must still establish an evolving body/contact/
damping bound and sensitivity to added inertia; neither a supplied target dt
nor an initial elastic bound can certify the entire damage/loading path.

An operator-neutral element planner consumes supplied positive element bounds
and physical element lumped masses. It scales only controlling elements, assembles
their nodal numerical mass and never decreases accepted nodal factors. Its
Rayleigh argument is checked against independently assembled dense operators.
The Hex8 envelope can now expose per-element positive and negative bounds
without changing the existing global maxima. Negative material curvature is
not removed by taking an absolute value or adding mass. This is a reusable
policy building block, not an automatic source-material certificate.

The variable-inertia gate also includes an independent oscillator whose mass
changes at a prescribed accepted station. Against its piecewise analytic
solution, displacement/velocity and energy-balance errors converge at second
order. Restart needs no material residual State and reproduces the complete
history. Unchanged proposals reuse the immutable mass arrays rather than
copying/hashing large diagonals on every increment. Artificial angular
momentum is reported about the spatial coordinate origin.

Quadrature-field initialization now consumes the backend's complete cell dof
table in bulk, retaining the existing cached permutation and ghost semantics.
Scalar, vector and tensor fields at three quadrature degrees agree with
independent cell-wise indexing on tetrahedra/hexahedra in serial and two ranks.
`tools/benchmark_quadrature_layout.py --size 64 --repeats 5` measured 0.1131 s
versus 0.00415 s median cold-layout construction for 262,144 cells (27.3x for
this initialization component only); all indices were identical. This is not
a full-simulation speedup or a change to quadrature/assembly ownership.

The moving polar-frame kernel reuses the old gradient's SVD spectrum to form
the right-stretch square root/inverse root, avoiding a redundant eigenproblem.
Independent noncoaxial quadrature, rotating prestress and virtual-power checks
are unchanged. Five alternating 65,536-point random moving-batch comparisons
against `91cec180` measured 1.13x component throughput at chunk sizes 1,024 and
8,192, with maximum strain difference 2.82e-18. No whole-step gain is inferred.

Response-only trials now also freeze the nodal configuration across provider
and downstream callbacks. A provider must not return stress for the admitted
gradient while changing the live displacement consumed by later assembly or
commit. Such mutation rejects the scope and restores nodal and quadrature
fields together; injected provider/consumer mutations have dedicated tests.

## Primary design references

The response-only Hex8 trial now combines material/hourglass and bulk-viscosity
cell forces before a single nodal scatter. Constitutive work and artificial
viscosity power remain distinct records. A two-cell regression compares the
result against independently scattered contributions and checks downstream
rollback; this is an assembly optimization, not a changed damping law.

Directional inertia planning also has a mass-shaped diagonal-majorant option:
the small material Gram supplies an element frequency bound, which multiplies
the physical element lumped mass before assembly. Independent dense actions
check the resulting Loewner upper bound. Neither this option nor absolute rows
is uniformly tighter; do not take their entrywise minimum, which need not be a
majorant. The supplied tensor's material domain remains the caller's obligation.

Explicit material responses use immutable byte-backed arrays. Revalidation
still checks shapes, finiteness and stress symmetry, but may reuse immutable
payload bytes with a fresh metadata view. Provider-side shape changes and field
replacement cannot alter a prepared consumer response. This removes redundant
payload copies without weakening the trial/commit/rollback contract.

Closed-solid explicit contact now rejects unresolved outside-to-outside paths
before updating material State. For linear nodal trajectories, two positive
endpoint gaps can only exclude a crossing when their sum exceeds the bound on
relative point/boundary travel. The guard removes common translation, includes
primary deformation, and deliberately permits conservative false positives
requiring a smaller increment. A real two-body contact regression verifies
that a through-solid jump with zero endpoint contact force is rejected. This
is **not** continuous collision detection, does not handle arbitrary curved
temporal paths, and does not establish ALL EXTERIOR coverage or valid geometry
through an inverting/self-intersecting increment.

The closed-surface ray broad phase now traverses the existing BVH with segment
slab clipping rather than the ray's entire axis-aligned bounding box. Exact Q1
intersection and two-ray agreement are unchanged. Three alternating private
original-geometry comparisons (1,000 queries) returned identical signed gaps:
median query time 11.812 s versus 0.641 s, 18.4x for this search component only.
No full-loading speedup is inferred. Parallel rays, endpoints, zero-length
segments, randomized exhaustive boxes and actual contact restart are tested.

Near a shared Q1 edge, squared-distance differences can be second-order small
even when an interior closest point is distinguishable from the edge. Search
now admits a bounded topological dominance screen: identical shared node
coordinates, a strictly interior stationary point, and a positive global
Gershgorin lower bound on the squared-distance Hessian. The stationarity margin
includes coordinate-roundoff allowance. If these checks are inconclusive,
the existing ambiguity rejection remains. Disconnected coincident traces do
not qualify. Ambiguity errors retain competing facet IDs and distance evidence.
This changes geometric admission only; it does not add bonded compression
contact, establish ALL EXTERIOR coverage, or validate a mass-scaled trajectory.

For unchanged closed-Q1 topology, contact can refit the existing BVH rather
than recursively sorting and rebuilding it at every geometry evaluation.
Refitting creates a new immutable tree: current leaf boxes are recomputed and
parent boxes are reduced bottom-up. Exact Q1 projection and solid-side tests
remain unchanged. Closed-shell topology is reused only through a vertex-only
refit operation; geometric validation remains active. The cache owns no
material/contact history, and a rejected increment can refit back to accepted
coordinates. Large deformation may degrade traversal efficiency; refitting is
not a guarantee of optimal tree quality or nonlinear physical validity.

### Explicit performance priorities after contact refitting

Filtered exact geometry is a robustness path, not a contact tolerance change.
For a shared-edge distance tie rejected by the existing floating-point screen,
an exact binary64-rational witness may certify strict ordering: the neighboring
Q1 squared-distance Hessian must be positive on the complete parameter square,
an exact edge minimizer must satisfy box KKT conditions, and a feasible witness
on the chosen patch must be strictly closer. An equality or unsuccessful screen
remains unresolved. No arbitrary-precision numerical root is treated as a proof.

For closed-solid classification, ambiguous floating-point rays have a bounded
rational-interval fallback. Exact elimination coefficients and isolating square
root intervals certify open-facet parameters, positive ray distance and oriented
crossing signs. Genuine seams, origin hits, tangencies and degeneracy still
reject. Two independent accepted directions must still agree. Ordinary rays
keep their existing fast path; the fallback is not a self-intersection test.
This follows the filtered/exact design principle described by
[Shewchuk](https://www.cs.cmu.edu/~quake/robust.html), not his orientation/incircle
predicate implementation. Full-run cost and numerical inertia remain separate
acceptance questions.

Near-touching contact coverage exposes a separate geometric cost: enclosure
refinement toward zero. Closed-solid projection now requests exterior distance
certificates only for positive signed gaps; interior and touching points retain
their already-computed signed gaps. Recursive nonnegative child bounds stop
after an exact zero minimum. Neither optimization relaxes projection tolerances
or introduces fabricated positive separation. Broad-phase bounds, actual Q1
projection and accepted path-crossing guards retain their separate roles.

Bonded-interface normal closure must be assessed as an explicit force-law
composition, not silently enabled during restart. A private case may test an
elastic reference bond in parallel with an independent unilateral current-face
penalty using the existing contact operator. Coverage must include the declared
bonded secondary nodes and change the operator/checkpoint identity. This is not
an automatic translation of a vendor hard-contact stiffness or ALL EXTERIOR
self-contact; the added stiffness requires a new stability/inertia review.

The optimization order is measured step cost first, then justified increment
count. Keep geometry, constitutive parameters, physical loading duration,
inertia and acceptance criteria fixed for a same-checkpoint replay. Compare
nodal arrays, material history and work records, not elapsed time alone.

### Local geometry certificates, not enlarged tolerances

The bounded Q1 contact path now distinguishes three different questions:

- Nearest-face ordering: binary64-distance ties may be ordered using exact
  rational strong-convexity bounds on the squared Q1 distance. Projected Newton
  iterates supply feasible witnesses only; the Hessian lower bound and gradient
  inequality supply the certificate. Genuine ties/nonconvex unresolved patches
  remain rejected. Shared-node boundary equivalence retains its existing trace
  and normal policies.
- Numerical zero gap: an exact feasible witness may establish distance within
  the existing closed-surface binary64 resolution. An ambiguous point then has
  no invented projection or normal, zero penalty force, explicit zero-gap
  indices/tolerance, and a **negative** resolution-band lower bound. It cannot
  be cached as positive separation. Resolvable interior penetration still
  requires an admissible projection.
- Swept separation: when the global travel estimate is inconclusive, the
  relative linear-point/moving-Q1 map has eight trilinear control vectors.
  Exact separating planes and bounded dyadic subdivision can exclude zero.
  Failure to certify still rejects the increment; this does not find impact
  times, admit curved temporal trajectories, or certify surface self-intersection.

These are reusable geometry/Procedure safeguards, not new material laws or a
claim that the private full-model reproduction is validated. Native batch
acceleration remains a separately audited private runtime prototype. Benchmark
frozen geometry separately from accepted dynamic trajectories; do not infer a
whole-run speedup from one contact evaluation.

The exact convexity screen also uses a two-dimensional comparison matrix.
With diagonal lower bounds `a,c` and off-diagonal magnitude upper bound `b`,
positive `a*c-b*b` gives the conservative eigenvalue bound
`(a*c-b*b)/(a+c)`. This covers skew, anisotropic traces for which the simpler
Gershgorin bound is negative. The minimum-distance recovery still needs an
exact normal-cone residual certificate; a better lower bound is not permission
to admit a nonconvex or unresolved patch. Closed-surface sign classification
has one internal batch seam retaining the same two independent oriented rays.
Triangle edge-incidence validation is vectorized, preserving manifold,
orientation and first-input-edge diagnostics rather than omitting validation.

Related method: [Tight Inclusion continuous-collision-detection reference](https://github.com/Continuous-Collision-Detection/Tight-Inclusion).
The present Q1 certificate uses exact trilinear convex-hull exclusion, not that
package's triangle CCD implementation or a complete time-of-impact algorithm.

1. Remove redundant geometry work and array transfers. Triangle coordinates
   are already finite and unique, so duplicate unordered connectivity also
   detects geometrically duplicate facets. A second per-facet Python coordinate
   sort is unnecessary. All vertex, degeneracy and manifold checks remain.
2. Profile native material batches and state snapshots after each improvement;
   worker waiting time is not by itself proof of a locking defect. Separate
   preparation, advancement, checkpoint and visualization costs.
3. Audit the critical material, contact and damping contributions before
   increasing the increment. Initial elastic bounds are not a certificate for
   an evolving damaged/contacting system. A converged Rayleigh quotient alone
   is not a conservative upper bound on the largest eigenvalue.
4. Assess added-inertia distribution and response sensitivity before changing
   mass scaling. Preserve the physical loading time of history-dependent laws.
   Frequency-selective mass scaling is a separate algorithmic project, not
   another name for scaling a selected subset of diagonal nodal masses.

References and their bounded implications:

- [Sierra/SM explicit performance guide](https://www.sandia.gov/files/sierra/SM_Users_5_26/main/appendix/performance.html)
  and [nodal stability theory](https://www.sandia.gov/files/sierra/SM_Theory_5_20/main/dynamics.html):
  compare assembled nodal modal-stiffness bounds before adopting an occasional
  Lanczos estimate. Sierra explicitly notes the latter's overhead and possible
  instability. Our existing initial directional/mass-weighted cell majorants
  are a starting point; evolving material and contact bounds remain necessary.
  Do not copy performance suggestions that change this reproduction's physics
  (substituting elastic material, disabling required contact, or skipping damage
  decisions). Timers and uniform-gradient Hex8 kernels are compatible priorities.
- [Abaqus explicit dynamics](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-expdynamic.htm):
  global versus element increment estimation and the effects of penalty
  stiffness and damping motivate a combined stability audit, not a fixed
  multiplier applied to an unverified nonlinear step.
- [DYNAmore mass-scaling guidance](https://www.dynasupport.com/howtos/general/mass-scaling):
  added mass changes inertia; inspect its distribution and perform reduced-mass
  sensitivity runs. A balanced numerical work ledger cannot replace this test.
- [DYNAmore selective mass scaling](https://www.dynasupport.com/howtos/general/selective-mass-scaling-sms-2):
  targeting high-frequency modes can better preserve low-frequency behavior,
  but introduces a non-diagonal mass solve and additional cost. Historical
  example speedups are not a prediction for this runtime or material law.
- [Wald, Boulos and Shirley, dynamic BVHs](https://graphics.stanford.edu/~boulos/research.htm):
  fixed-topology refitting motivates geometry-cache reuse. The graphics
  acceleration technique does not establish finite-element contact validity.

Fixed elastic interface and penalty-contact spectral rows now reuse one
bounded-memory stencil kernel. Signed interpolation and shared-node assembly
are retained; tests compare independent dense spectra and prior scalar rows.
This refactor does not admit moving-projection geometric stiffness or damaged
material tangents into an unproven whole-path stability estimate.

- [PBRT, ray--bounds intersections](https://pbr-book.org/4ed/Shapes/Basic_Shape_Interface):
  slab interval clipping and conservative roundoff expansion motivate the
  segment broad-phase optimization. Exact Q1 geometry remains the narrow phase.
- [Ramsey, Potter and Hansen, Ray Bilinear Patch Intersections](https://www.sci.utah.edu/~kpotter/publications/ramsey-2004-RBPI.pdf):
  two-plane elimination, both quadratic roots and robust parameter recovery.
  The private classifier rejects seam/tangent rays and checks another direction.
- [Bærentzen and Aanæs, signed-distance computation](https://doi.org/10.1109/TVCG.2005.49):
  why arbitrary single-facet normals do not define inside/outside at edges and
  vertices. We do not apply the triangle pseudonormal theorem to an open Q1 patch.
- [Eberly, Distance Between Point and Parallelogram](https://www.geometrictools.com/Documentation/DistancePointParallelogram.pdf):
  convex interior/boundary distance structure; independent polynomial-oracle
  comparisons protect the specialized Q1 fast path.

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
- [Tight Inclusion CCD](https://continuous-collision-detection.github.io/tight_inclusion/):
  bounded interval subdivision distinguishes a proof of separation from an
  unresolved query. A narrow curved gap can exhaust spatial proof depth even
  for a stationary path; cutting the physical time step then cannot resolve it.
  The swept-Q1 default proof budget is now 30 subdivisions (previously 18),
  retaining the independent 4096-box cap and exact strict separation test.
  This changes neither contact tolerance nor dynamics time step. Manufactured
  narrow gaps at three scales and true impacts pass the 33-test targeted suite;
  unresolved queries still reject. Full-path reproduction remains unverified.
  Follow-up acceptance: 289 targeted geometry, constitutive, viscosity,
  subincrement and restart tests passed. A private native depth-30 interval
  filter agreed with rational separation on 306 captured difficult paths
  (295 separated, 11 unresolved/rejected); 90 manufactured impacts at three
  scales remained rejected. An identical ten-step full-mesh replay retained
  bitwise field/material equality after topology-cache and filter changes.
  Cache-tiled material execution and a larger Newton iteration budget were
  measured but not selected: neither improved the tested runtime. These are
  bounded algorithm/runtime checks, not a completed loading-path validation.
- Contact geometry evaluation now has an isolated private array-input seam.
  The default residual remains serial: it validates geometry before body
  assembly, and retains its original rollback and commit order. The seam does
  not access PETSc vectors, but its geometry cache still requires exclusive
  ownership. A private overlap experiment keeps all PETSc work on the caller
  thread and joins geometry work before rollback; it is not a public threaded
  execution promise. Serial/parallel field identity, restart and injected
  failure tests must pass before any full-mesh timing claim.
- The exact Q1 convexity recovery additionally encloses the correlated Hessian
  determinant in the degree-(2,2) Bernstein basis. This is tried only when the
  cheaper diagonal/off-diagonal comparison is inconclusive. Strictly positive
  rational coefficients certify the whole parameter square; an inconclusive
  result still falls back. No contact tolerance, quadrature, or time step is
  changed. A generic warped patch, scaling checks, and indefinite/degenerate
  rejection are included in the targeted exact-geometry tests.
- Quadrature point counting now uses layout metadata rather than constructing
  a detached history array merely to obtain its length. Counts still include
  visible ghost points. The 48-test targeted state/layout suite passed; the
  20 layout tests also passed under a bounded two-rank TCP-backed MPI launch.
  This does not admit the private response-only full simulation to MPI.
- Response-only Hex8 displacement and velocity gathering uses axis-specific
  NumPy extraction, retaining detached arrays and the same cell ordering.
  A 91-test targeted finite-element/material/viscosity suite passed. Paired
  moving-contact, bonded-interface trajectories with and without viscosity
  agree bitwise with the original advanced-indexing implementation. Component
  timings are not a completed full-model performance claim.
- The central-difference integrator reuses the existing array update kernels
  instead of allocating eager temporary finite-element fields for every sum.
  Function-space checks precede mutation, and each target still performs its
  ghost update. The 124-test serial targeted suite passed (8 distributed-only
  checks skipped there); five dedicated kinematic tests and five distributed
  contact tests passed on two ranks. Stale-ghost algebra comparisons and a
  full material/contact/viscosity trajectory agree bitwise with field algebra.
- [Abaqus explicit dynamics](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-expdynamic.htm):
  linear and compression-only quadratic artificial bulk viscosity. Matching
  these formulas alone does not establish identical characteristic lengths,
  velocity sampling, stable increments or whole-solver equivalence.
- [Abaqus mass-scaling verification](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-mscale.htm)
  explicitly bases bulk viscosity on scaled mass after the first increment.
  Our current physical-density viscosity policy deliberately does not do this.
  Treat this as a separate vendor-comparison gap, not as harmless inertia-only
  bookkeeping. With unchanged stiffness and length, a uniform density factor
  `f` scales linear viscosity by `sqrt(f)` and quadratic viscosity by `f`.
  A frozen-kinematics work comparison cannot bound the changed trajectory.
  Nonuniform nodal inertia does not uniquely define element density factors;
  any conversion needs an explicit policy and independent damping/increment
  verification. Do not silently change damping halfway through a checkpoint
  continuation or reinterpret this diagnostic as vendor equivalence.
- Explicit Cauchy symmetry validation checks the three unique off-diagonal
  pairs under the unchanged full-tensor scale/tolerance. This avoids a second
  nine-component temporary per point. The 95-test material/response suite
  passed, including extreme scales and adjacent floating-point values at the
  symmetry threshold. Component timing is not a full-model speedup claim.
- The affine-removed Hex8 validity screen first bounds singular values using a
  strict Gram-matrix norm ball, with a floating-point guard. Only inconclusive
  cells retain the original SVD; unresolved geometry still takes the Bernstein
  path. This avoids a large SVD batch when reference-displacement admission
  becomes inconclusive during otherwise admissible finite motion. It changes
  neither deformation gradients nor force assembly. The 81-test targeted
  geometry/material/contact suite passed, including SVD-path equivalence at
  three scales and selective fallback for nonaffine or large-stretch cells.
- Serial quadrature commit/rollback can transfer from borrowed identity-layout
  source views when all destinations are concrete, same-mesh fields and no
  destination aliases any source. This removes an intermediate copy, not the
  caller's failure transaction. Open providers, custom assignment, permuted
  layouts, cross-aliases and distributed fields retain detached transfers;
  checkpoint snapshots remain detached in every case. The 73-test targeted
  ownership/layout/material/response suite passed, including failures after
  material commit. A paired 180-increment private full-mesh replay had bitwise
  equal saved state arrays; mean advance time changed from 0.20986 to 0.20234
  seconds per increment (3.6% lower in that pair). This excludes setup/export
  and is neither a fresh full-run timing nor Abaqus cross-validation. Duplicate
  outer/inner rollback snapshots are deliberately retained: removing them
  needs an explicit transaction-ownership design, not global suppression.

### Prepared body execution: ownership before fusion

The next performance route targets the complete body branch, not contact-only
microbenchmarks. Follow the restriction/kernel/backend separation described by
[libCEED](https://libceed.org/en/latest/libCEEDapi/), without changing Model or
claiming that the already matrix-free Hex8 force evaluation is a new algorithm.

1. Freeze accepted inputs once and allow contiguous sub-batches to retain that
   immutable storage with independent array metadata. A NumPy read-only flag on
   mutable storage is not sufficient. Concrete serial identity-layout history
   may be frozen directly; custom providers and other layouts retain packing.
2. Prepare topology, material groups and worker resources once. Evaluate each
   batch through gather, kinematics, material response, Piola mapping and force
   before publishing fields. A material-only chunk benchmark does not test this
   complete execution route.
3. Establish one explicit acceptance owner before removing nested snapshots.
   Workers must finish before rejection; callbacks, partial commit failure,
   retained output aliases, standalone calls and restart all require tests.
   Existing rollback protections remain until that contract is implemented.
4. Compare the original mesh from identical accepted states, first as a bounded
   screen, then repeated A-B-B-A runs of at least 1,000 increments in multiple
   deformation stages. Preserve mass, time schedule, physics and output. Report
   setup, advancement, persistence and full process wall time separately.

The immutable transport work is a prerequisite, not completion of resident
State or end-to-end fusion. No speedup target becomes a capability claim until
the full branch and trajectory comparisons support it.

Initial transport validation: 148 targeted serial tests passed; the layout and
snapshot-ownership selection passed on two ranks (27 tests per rank). The
private original-mesh 600-increment screen retained bitwise-equal saved fields
and history arrays and equal selected quantities at all 60 reporting stations.
Advance times were 121.588 s and 118.791 s in one reference/candidate pair.
This approximately 2.3% difference is not a demonstrated throughput gain: the
screen is shorter than the performance admission protocol and has no repeated
run uncertainty estimate. It supports numerical compatibility only; it neither
completes the fused body executor nor admits a larger time increment.

### Detached body continuation and accepted field ownership (2026-10-12)

The response-only Hex lowering now separates detached material/Piola/element
preparation from trial-field publication. Its default arithmetic is unchanged.
An execution lowering may fuse that continuation, but cannot commit history,
omit the material increment policy, or move PETSc work to background workers.
Preparation or force failure must leave the trial fields unchanged.

The serial response owner may explicitly lend the Procedure's existing accepted
field snapshots to nested field transactions. This is a joined increment, not
a nested savepoint: any child rejection poisons acceptance, even if an inner
callback catches the exception. Cleanup transactions remain usable so the
original scientific error is not replaced by a secondary cleanup error. The
outer Procedure restores motion, material history, contact state and time inputs
on rejection, including a failure after material publication. Independently
called field transactions retain their ordinary savepoint behavior. Distributed
borrowing is refused; the existing collective rollback path remains unchanged.

A private original-mesh experiment fuses response, Piola conversion and element
force work in the same material worker partition. It preserves the source
increment/domain checks and independent material audit. It is not a new public
provider capability or a material-specific kernel in the AgentFEM core. The
exploratory 1,200-increment pair took 235.640 s with independent preparation/savepoints
and 207.558 s with the joined candidate (advancement only; setup and export are
excluded). All saved numeric arrays were bitwise equal and selected physical
quantities matched at all 120 common reporting stations. Repeated runs and a
second trajectory segment are required before a broader performance claim;
this does not establish Abaqus equivalence or whole-run wall time.
The exploratory ownership timing predates restoration of the live-field finite
check at transaction entry and is not an admitted final performance result.
The final candidate retains that check; a private exact IEEE-754 classifier
avoids its temporary boolean array. A 4,096-cell cache-tiling experiment increased
dispatch overhead and was safely paused rather than selected for delivery.

The source-slice transport regression also now retains the byte offset into an
immutable owner's storage instead of reshaping the entire parent buffer.

Final full-check repeat evidence: the original mesh was replayed from accepted
increment 98,000 through 99,200 twice per route. Reference advancement took
235.640 / 237.554 s; the final whole-worker continuation plus accepted-snapshot
reuse and exact finite classification took 220.068 / 225.482 s. Mean advancement
time decreased 5.84% (not the requested 10%). Including startup/restart/export,
process wall times were 284.26 / 283.45 s versus 267.77 / 273.31 s. All saved
numeric arrays were bitwise equal, and selected physical quantities matched at
all 120 reporting stations in each pair. These are two independent repeats,
not 120 independent timing samples; no confidence interval or full-trajectory
gain is claimed. The second deformation-stage comparison remains undone.

Validation: 108 targeted serial tests passed, the two-rank default-rollback and
serial-only-borrowing guard passed on both ranks, and five private executor tests
covered tiled/untiled arithmetic, worker failure cleanup and exact finite checks.
The private fused executor is not promoted to a public backend, nor does this
evidence validate the underlying scaled-inertia case against Abaqus.
