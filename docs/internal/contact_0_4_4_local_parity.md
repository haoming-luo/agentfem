# 0.4.4 bounded implicit-contact sprint

Status: target and working evidence ledger, **not** a released capability or a
claim of Abaqus/Standard parity. Baseline: merged PR #117, commit `683eb429`.
Scientific input, implementation revision, independent reference and
acceptance tolerances must be recorded before a comparison is judged.

## Claim boundary

Target the ordinary `model.step(...)` route for serial three-dimensional
solids, prescribed rigid analytical or reviewed triangulated tools, finite
sliding, quasistatic implicit increments, finite-strain J2, named multiple
tool pairs and optional supported periodic MPC. Admit frictionless normal
contact first; admit a Coulomb/geometry combination only after its own
tangential-state and feature-switch tests. Comparison is on physical
observables and accepted histories at equivalent stations, not bitwise
equality with another solver or one cloud image.

Excluded from this claim: general self-contact, deformable--deformable
implicit contact, arbitrary CAD/STEP surfaces, shell thickness/offset,
distributed implicit contact, arbitrary history materials, impact and
thermomechanical contact. They remain separate capability gates rather than
being silently approximated by the bounded route.

## Live gate status

| Gate | Current evidence | Still needed for promotion |
| --- | --- | --- |
| 1. Comparator | Internal and public-protocol tests | Independent same-input finite-J2 contact run and frozen observables |
| 2. Surface | Reviewed geometry; exact plane and conservative closed-solid initial-overlap bounds on admitted first-order traces | Open facets, unsupported curved tools, ambiguity and sliding coverage at target geometry |
| 3. Integration | Planar/convex partitions; sampled nonconvex risk flag; certified zero-contact faces and certified per-face smooth-branch active subset of a closed concave tool | General active nonconvex feature-switch integral/tangent; separate admission or rejection of lower-dimensional edge/point contact |
| 4. Normal law | Penalty Step; fixed-trace AL operator candidate | Outer global Procedure, spatial stability and checkpoint composition |
| 5. Friction | Locally planar Coulomb State; rigid-master spin transport tested | Admitted geometry-by-friction matrix; sharp features remain rejected |
| 6. Transaction | Total-system and restart tests for penalty; serial affine/MPC accepted-feature path with legacy-incomplete migration | Repeat for each new enforcement/integration law |
| 7. Result | Pair work/pressure samples; AL local work sign | Cross-law first-law and pressure interpretation |
| 8. Science | Local/oracle/installed tests; global elastic-limit spring-plane matrix | Mesh x increment x law convergence and external observables |
| 9. Release | Scoped solver screens | Installed cross-platform gate and production runtime evidence |

“Partial” here is not a release status. The test target, geometry and
enforcement must all match before a gate can be marked closed.
The reentrant-tool sample carries an engineering quality failure whenever a
sampled slave face crosses nearest features without a certified transition
integral. Release additionally requires a certified integral even if no
switch or active point happened to be sampled. Exploratory inspection remains
possible; an absence of sampled switches is not a whole-face certificate.
For a reviewed closed solid, a current first-order slave face can now instead
be certified **strictly separated** by an adaptive 1-Lipschitz signed-distance
bound. A rigid-pose-aware complete AABB separation is checked first, so
obviously distant faces require no nearest-feature/winding point query. That
proves its normal contact integral is exactly zero at that accepted
configuration. A manufactured counterexample with three apparently separated
ordinary Gauss points and a penetrating face corner is correctly *not*
certified. An unresolved budget, ambiguous projection,
or any near/active contact leaves the integral uncertified. This is not a
between-increment path-intersection proof and does not admit active nonconvex
contact, friction, or lower-dimensional edge/point laws. Its reference
geometric cost also is not a large-CAD performance guarantee.
There is now a separate, deliberately stricter **active** subset: for a
reviewed closed tool, each current affine slave triangle must project
inside a completely covered coplanar master patch; its nearest-plane distance must
beat every noncoplanar master facet by more than twice the slave-face radius
(plus numerical margin). The difference of two distance-to-set functions is
2-Lipschitz, so this inequality certifies the closest feature over the
*whole* face, not just at Gauss points. Distinct slave faces may select
distinct certified master planes, but one face cannot cross a sharp feature.
The patch check subtracts the union of coplanar master triangles from the
projected slave polygon with a bounded halfspace clipper; an overlapping
master-patch area sum is not accepted as coverage. It admits a flat
triangulation seam but rejects a positive-area gap or proof-budget exhaustion.
This is a bounded floating-point coverage certificate, not an exact-arithmetic
CAD topology proof. It follows the classic reentrant polygon-clipping idea
of [Sutherland--Hodgman][polygon-clipping], adapted to subtract a triangle
union rather than to integrate a second contact surface.
The current Newton trial may then reuse the already proved analytical-plane
active partition, with the closed solid's own signed-gap projection checked
against each plane. Geometry proof is
repeated on each trial and rigid pose. A real serial finite-J2 solve has
matched the corresponding analytical plane in displacement, PEEQ and tool
work, with a contact directional tangent, interrupted restart and explicit
reentrant-edge rejection; the small runnable example exports one XDMF grid
and unsmoothed contact samples. This does **not** solve a feature switch or
physical edge/vertex contact: those remain rejected. The current per-face
smooth-branch and 1024-face/256-master-facet bounds are intentionally narrow,
and this is not an external commercial-solver comparator.
The closed-solid projector now takes the reviewed facet normal directly on a
smooth face, including a coplanar seam. Dividing a nearly zero
point-to-foot vector by the gap previously amplified tangential roundoff and
made strict projection/plane comparisons fail near closure. The geometric
edge/vertex branch and the signed-gap classifier are unchanged. A near-gap
two-sided normal regression and a serial finite-J2 seam-crossing Step exercise
this numerical correction.
The ordinary serial Step now also accepts an exact-name mapping from rigid
contact pairs to their already admitted integration modes. A missing or
unknown pair name is an early capability rejection, never a fallback to
sampled Gauss integration. Each lowered pair retains its own mode in
checkpoint identity; a two-tool mixed-mode finite-J2 path and interrupted
restart exercise this composition. A separate two-active-tool regression now
checks independently integrated pair forces, positive per-tool work, total
work sum and joint restart while one tool uses the certified closed smooth
branch and the other uses an analytical planar partition. A small rigid
rotation of the closed tool also matches its equivalent analytical plane
in a real finite-J2 Step. The serial periodic-MPC J2 route now admits this
*one named scalar mode* on a wide reviewed smooth patch, with total-residual
reduction, analytical-plane limit, interrupted restart and explicit
reentrant-edge refusal tested. The affine route now also uses the same
exact-name per-pair mode contract: one closed smooth branch and one analytical
plane can enter a *single* total MPC system, survive interrupted restart and
refuse a changed mode identity. This remains serial and does not admit a
sharp-feature transition merely because the second tool is supported.
The runnable closed-tool example now has a separate one-axis-at-a-time
mesh/increment/penalty sensitivity driver. Its eleven serial synthetic runs
all completed with certified smooth branches, but their terminal tool work
remains mesh- and penalty-sensitive, while the pointwise maximum PEEQ is not
monotone under mesh refinement (details in
`evidence/contact/2026-10-19-closed-planar-branch-ladder.md`). This is
evidence **against** promoting the present tiny fixture as a converged
hard-contact or external-solver benchmark. The driver is a reproducible
diagnostic, not a release threshold or a performance promise.
Both ordinary J2 and affine/MPC acceptance now validate the same typed
polyhedral diagnostic contract: a missing mixed-feature count, a non-Boolean
integration claim or inconsistent active face/edge/vertex totals rejects the
accepted step and restores its previous State. A visually plausible but
incomplete observation cannot silently enter the historical evidence.

Ownership for the next implementation is deliberately narrow: surface/search
admission belongs under `boundary_models/`; feature integration and the
fixed-multiplier law are Operators there, not new solvers; the outer AL loop,
Newton/MPC ordering and atomic acceptance belong to the existing finite-J2
Procedure; accepted histories and checkpoint identity belong to State; force,
work, field-location and trust claims belong to Result/Verification. The
external comparator and convergence ladder are test/evidence assets, not
hidden runtime heuristics. This keeps the public Model focused on declaring
the contact pair and its physical/numerical policy.

## Nine ordered gates and reading map

Each bibliography entry motivates a design question; none certifies AgentFEM.
The gate stays open until its listed evidence exists at the proposed scope.

1. **Frozen contract and external comparator.** Freeze geometry, mesh family,
   units, initial clearance, rigid pose history, material parameters, solver
   controls and observables. Prefer a reproducible external Abaqus/Standard
   input/results pair supplied or reviewed independently; record contact
   formulation and enforcement differences. Predetermine observable-specific
   tolerances and a comparison station set. Current state: internal examples
   and local regression exist, but no same-input external benchmark for the
   nonconvex finite-J2 fixture. Reading: [Abaqus' published finite-sliding
   rigid-surface verification/input list][abaqus-verification],
   its published [C3D8 solid against analytical rigid cylinder input][abaqus-c3d8-input]
   and [finite-sliding C3D4/C3D8 surface-to-surface input list][abaqus-surface-verification],
   [its small-sliding stress/displacement verification cases, including
   augmented-Lagrangian examples][abaqus-al-verification],
   [COMSOL's analytical/NAFEMS verification collection][comsol-verification]
   and [NAFEMS advanced contact benchmark catalog][nafems]. A published
   verification case is a comparator protocol, not an external result for
   our own geometry. In particular, that published three-dimensional
   meshed-rigid case deforms a **shell** around a cylinder, so its numerical
   output cannot be relabeled as a solid finite-J2 same-input comparator.
   The C3D8 rigid-cylinder deck is an **elastic contact protocol**: it
   specifies an analytical cylinder, a solid contact face, normal loading
   and later sliding, but changes friction and has distinct linear and
   geometrically nonlinear steps. It supplies an input, not numerical output
   for our finite-J2 same-input case. The C3D4/C3D8 surface-to-surface blocks
   use two deformable bodies rather than our prescribed rigid tool. Thus
   neither source closes the external J2 gate; the elastic rigid-cylinder
   portion can serve as an earlier interoperability checkpoint with matched
   material, constraints and observables.
   A more ambitious later comparator is [NAFEMS Contact
   Benchmark 3][nafems-contact-3], which includes elastoplastic contact and
   springback; its detailed material, geometry and published observables must
   be obtained and frozen before claiming a comparison. Keep this separate
   from the smaller same-input AgentFEM/Abaqus-Standard J2 plane/convex-tool
   pair needed for the bounded 0.4.4 claim.

2. **Surface and search admission.** Validate closed/open topology, outward
   orientation, self-intersection, initial clearance/overclosure and allowed
   free features; search and project in the current configuration on every
   trial. A local candidate cache may accelerate search only with a
   conservative coverage/invalidation proof, never by reusing old forces or
   normals. Test large sliding, seam crossing, geometry-scale changes and an
   intentionally missed-candidate trap. Current state: smooth projectors,
   triangulated BVH and reviewed closed-solid classifiers exist. Implicit
   rigid pairs now have an optional `initial_penetration_limit` in length
   units. On a first-order analytical-plane trace, the extremal nodal signed
   gap is an exact face-wise bound; this catches a corner penetration missed
   by quadrature. On a first-order triangular trace against a reviewed closed
   convex or nonconvex rigid solid, a bounded adaptive subdivision uses the
   true signed distance's 1-Lipschitz inequality to prove a conservative
   face-wise upper bound. It fails closed on ambiguous projections or if the
   declared depth/sample budget cannot prove the requested limit. For the
   convex subset, a support plane that
   dominates at every affine slave-face vertex dominates on the whole face;
   its nodal minimum directly bounds the initial penetration. Mixed-feature
   faces still use the adaptive certificate. A near-touching convex ordinary
   Step now passes a 1e-9 declared limit without refinement. This is only an
   initial-overlap result, not a later sliding/integration certificate. No open
   triangle sheet or arbitrary curved projector inherits that proof. The
   ordinary Step capability query reports unsupported geometry/order before
   Step construction. The explicit pair still rejects this policy. No branch
   silently moves slave nodes or the tool. Arbitrary large-CAD and ambiguous
   sharp features are not admitted. Reading:
   [Abaqus finite-sliding/general-contact formulation][abaqus],
   [Abaqus meshed-rigid-surface verification][meshed-rigid] and
   [Abaqus initial-overclosure choices][initial]. The conservative subdivision
   follows the classical 1-Lipschitz signed-distance property, distinct from
   any particular neural approximation; see the distance-function discussion
   in [Cavalletti--Mondino][distance].

3. **Contact-boundary/feature integration.** Make the active integration
   partition a trial-geometry Operator input. A zero-gap boundary and a
   nearest-feature switch are different discontinuities: the latter can
   contribute a line term to the Newton tangent. Keep positive weights and
   one consistent residual/tangent partition; refresh a changed partition
   before accepting a Newton trial. Verify manufactured narrow-strip,
   two-plane and curved-transition oracles, then the nonconvex L-tool response
   matrix. Also distinguish a *surface quadrature crossing a nearest-feature
   switch* from actual edge--surface, edge--edge or point contact. A corrected
   surface integral alone does not establish the latter's nonpenetration
   constraints or pressure units. Either admit those lower-dimensional
   branches with independent force/moment and tangent evidence, or reject
   them explicitly at the current geometry; do not silently assign an
   arbitrary averaged surface normal. Abaqus/Standard itself distinguishes
   the bounded small-sliding surface formulation from finite-sliding general
   contact: its [general-contact controls][abaqus-general-controls] state
   that small-sliding assignments deactivate edge and vertex contact. This
   is evidence that a surface-only approximation needs an explicit scope,
   **not** evidence that our finite-sliding route may silently do the same.
   Farah--Wall--Popp use separate point,
   line and surface multiplier families for non-smooth finite-deformation
   contact [point-line-surface], whereas Farah--Popp--Wall compare segment,
   element and hybrid mortar integration schemes [mortar-integration]. These
   are architecture comparisons, not drop-in algorithms or evidence that
   our one-sided P1 operator already has their coverage. Haikal--Hjelmstad's
   oriented-volume formulation is a third non-smooth contact direction
   [oriented-volume], but is developed for quadrilateral/hexahedral body
   elements; it does not convert our tetrahedral slave and triangular rigid
   tool into a validated volume-contact operator by renaming the gap.
   Current state: analytical planar clipper, research two-plane oracle
   and opt-in <=4-plane convex partition pass; the result now flags *sampled*
   mixed-feature slave faces for reviewed closed solids in both standard and
   periodic J2 Results, without claiming a complete transition integral or
   changing the Newton law. Nonconvex feature
   transitions remain open. Reading: [Duong--De Lorenzis--Sauer refined-boundary
   quadrature][duong] and [COMSOL contact quadrature guidance][comsol-settings].
   The convex partition now admits only the actual default three-point P1
   reference rule and its facet-ID grouping. Three arbitrary points whose
   shape values merely sum to one are rejected before their measure can be
   silently reinterpreted as that rule; this is an input-contract safeguard,
   not a new geometry or friction capability.
   An independent four-feature rectangular-window integral now matches the
   exact quadratic penalty energy, including the reference-area map; its
   perturbed directional derivatives and a real FEniCSx assembled tangent
   agree with the moving pairwise-line tangent. A small finite-J2 Step also
   completes and resumes from an interrupted checkpoint with the same
   displacement and tool work. A five-feature prism still fails closed.
   This only widens the reviewed **convex** slice; it does not establish
   nonconvex nearest-feature switching or physical edge/vertex contact.
   The cited isogeometric extended-mortar formulation is a comparison design,
   not a drop-in P1 facet implementation: its enriched pressure interpolation
   and two-half-pass force construction differ from our one-sided trace.

   The next implementation must distinguish *sampling* from *consistent
   differentiation*. For a slave face partitioned into current nearest-feature
   regions `A_i(u)`, write the scalar contact functional as
   `Pi(u) = sum_i integral_{A_i(u)} phi_i(u,x) dA`. At a feature-switch line,
   the adjacent `phi_i` agree because the selected distance is continuous, so
   the first shape-boundary term cancels. Their displacement derivatives need
   not agree when the selected normals differ; differentiating the assembled
   residual then has a moving-line contribution. This is a derivation for our
   piecewise-smooth penalty law, not a formula imported from another element.
   The candidate must reproduce both the integral and its directional
   pass the total Newton/MPC tangent check. A fixed quadrature rule, a frozen
   feature partition or a smaller Newton residual alone is insufficient.
   `tools/verify_contact_feature_transition.py` is already an independent
   two-plane unit-triangle counterexample: its moving-line term exactly
   cancels a nonzero fixed-branch bulk tangent. It proves why the term is
   necessary in that manufactured setting, not that the closed nonconvex
   three-dimensional implementation has been supplied.
   A second independent oracle, `tools/verify_reentrant_contact_integral.py`,
   computes signed distances directly to the six segments of an extruded
   L boundary, without calling the AgentFEM projector. One affine slave
   triangle has **all three ordinary face points outside** the rigid solid,
   so its sampled penalty potential is zero, yet positive-area material near
   its corners penetrates: 128/256/512 positive-subtriangle refinements give
   potentials 0.000348765/0.000349677/0.000349913 at unit penalty. The last
   two differ by about 0.07%. This is a numerical refinement reference, not
   an exact-arithmetic certified integral, and supplies a concrete target
   for the nonconvex feature-partition work rather than an excuse to turn on
   a sampled full Step as though it were complete.
   [Hiermeier--Wall--Popp's variationally consistent mortar study][variational-mortar]
   is a useful independent standard for varying *all* active contributions,
   but its symmetric two-body mortar formulation is not our algorithm.

   A 2026-10-18 fixed-input screen used
   `examples/finite_j2_closed_concave_contact.py` with 12 increments,
   penalty 20, direct linear solve, backtracking, one MPI rank and one
   OpenBLAS/OpenMP thread. The table reports the accepted end state, not a
   timing comparison. `q2` is the default positive face rule; `q5/q7` are
   explicit quadrature-degree requests.

   | Cells/axis | Rule | Tool force x | Active reference area | Max PEEQ | Tool work |
   | ---: | --- | ---: | ---: | ---: | ---: |
   | 2 | q2 | -0.307580 | 0.416667 | 0.00671873 | 0.00806207 |
   | 2 | q5 | -0.334865 | 0.447799 | 0.00670389 | 0.00895618 |
   | 2 | q7 | -0.336957 | 0.441456 | 0.00726967 | 0.00876881 |
   | 3 | q5 | -0.328318 | 0.445520 | 0.00812605 | 0.00888055 |
   | 3 | q7 | -0.319265 | 0.446202 | 0.00820966 | 0.00869530 |
   | 4 | q5 | -0.337238 | 0.428587 | 0.00841117 | 0.00874221 |
   | 4 | q7 | -0.320473 | 0.452688 | 0.00852799 | 0.00867533 |

   A further same-input q9/q11 check on the 2-cell mesh found tool-force x
   -0.336930/-0.321905 and active area 0.428296/0.440369; on the 3-cell
   mesh it found force -0.317957/-0.324226 and area 0.453882/0.450798.
   The higher positive Basix rules are now available as explicit study
   controls, not defaults. The q5/q7/q9/q11 force and active-area differences
   remain nonmonotone across this small mesh ladder. Completion, a small
   energy residual and a higher rule
   therefore do **not** establish a converged nonconvex result. Do not
   promote default q2 or q11 as a silent accuracy fix. The discriminating next
   experiment is a genuinely changing trial feature partition with verified
   residual/tangent, followed by a new mesh/increment/penalty matrix. This
   is a Gate 3 failure record, not a scientific benchmark result.
   An opt-in `DolfinxImplicitRigidContact.quadrature_sensitivity()` diagnostic
   now evaluates positive rules at the *same current* displacement and rigid
   pose, without another nonlinear solve or State mutation. On the accepted
   2-cell q9 end configuration the q9/q11 structure-force x values were
   -0.336930/-0.320336 and active reference areas 0.428296/0.440369. This
   isolates a roughly 5% force difference due to the rule at one fixed
   configuration; q11 is not a reference truth. It is frictionless and
   scalar-penalty-only, and reports
   `certifies_feature_transition_integral=false`. The next algorithmic
   candidate is trial-geometry segmentation or a variationally consistent
   mortar/feature treatment, with line-tangent and coverage tests.
   [Pinto Carvalho et al.'s rigid/deformable dual-mortar study][dual-mortar]
   motivates a distinct longer-term formulation, but its efficiency claims
   cannot be imported into this one-sided penalty implementation.
   A second opt-in fixed-configuration diagnostic now uniformly subdivides
   each affine P1 reference face into 4^level positive three-point subfaces.
   It preserves reference area and interpolation and is explicitly **not** a
   moving transition partition or line-tangent implementation. On that same
   2-cell accepted q9 state, level 3 (1,536 samples) gave potential
   0.008144 versus q9's 0.008140, but structure-force components
   (-0.334995, 0.010878, 0.020429) versus q9's
   (-0.336930, 0.016867, 0.012983). Close potential alone did not imply a
   reliable force vector. The level-0/1/2/3 active-area sequence was
   0.416667/0.437500/0.416667/0.430990, also not a convergence certificate.
   Level 4 (6,144 samples) changed the area again to 0.436523 and the
   force to (-0.325105, 0.017034, 0.024163), while potential remained near
   0.008136. Uniformly spending more points therefore did not close the
   feature-transition force question either.
   The production design now branches deliberately: a user-declared physical
   fillet/smooth tool can use the existing external `SmoothRigidSurface`
   extension and must pass the gap/Hessian checker and independent geometry
   comparison; the exact sharp nonconvex tool must wait for its own
   trial-geometry segmentation/line-tangent evidence. An implicit hidden
   normal average or silent fillet would change the model. This distinction
   follows [Abaqus' explicit advice to smooth analytical rigid corners for
   finite-sliding Standard contact][abaqus-rigid-smoothing], not an assumption
   that AgentFEM's present mesh already has those fillets.
   [Wolff--Bucher's assumed-gradient distance field][distance-field] is a
   different promising route for complex geometry, but its smooth interpolated
   gradient is a declared gap/geometry approximation. It cannot be advertised
   as the exact sharp-feature answer without an error study.
   A new algorithm may enter the Step only after both integral and
   differentiation checks, not merely a finer screenshot or work balance.
   The accepted closed-solid point diagnostic already distinguishes physical
   face, edge and vertex projections (not merely triangulation seams).
   Engineering verification now fails if a sampled *active* point belongs to
   a physical edge or vertex, even if no mixed-feature slave face was sampled;
   a corrected surface integral would not by itself define that separate
   lower-dimensional law. The diagnostic also rejects counts inconsistent
   with the active-point total. Exploratory results retain the observation
   without a trust upgrade; release still requires a certified integral. A
   real finite-J2 reentrant-edge Step now supplies the negative engineering
   test, not just synthetic metadata. Its Result exposes named face, edge and
   vertex active-point counts at the final accepted configuration beside the
   underlying per-pair metadata, so a user can inspect the reason without
   decoding a private trace. The runnable reentrant-tool example also prints
   the failed engineering feature claim beside `status=completed`; nonlinear
   completion is not silently presented as scientific acceptance. These final
   counts alone do not summarize earlier
   accepted increments. The ordinary serial finite-J2 Step now keeps a bounded
   per-pair path summary: maximum sampled mixed-feature faces and active
   physical edge and vertex points, plus whether every observed station used
   a certified transition integral. The initial and every accepted station
   contribute; failed increments roll back the summary and serial checkpoints
   restore it with count and pair-identity checks. Older checkpoints lacking
   it remain readable but mark the path incomplete. The summary is independent
   of output frames and the progress-event ring buffer, both of which may
   omit earlier states. It is still only *sampled* path evidence, not a
   whole-face coverage proof. The serial periodic/MPC path now records the
   same sampled maxima through its existing accepted-observer transaction:
   the initial and each accepted station are counted, late failures restore
   the prior record, and the identified recorder is checkpointed with the
   material and contact Operators. A damaged record is atomically refused.
   Older affine penalty checkpoints lacking the recorder load only through
   an explicit identity-compatible migration, marking historical coverage
   **incomplete**; subsequent observations cannot invent the missing past.
   An entirely observed flat convex path can now pass the narrow engineering
   claim of *no observed* uncertified feature transition, while its release
   integral-certificate claim still fails under plain Gauss quadrature.
   Thus a complete observation ledger is not a whole-face integration proof.
   For the separate *convex* route, accepted point diagnostics now say whether
   the reviewed changing-trial support-plane partition supplied its bounded
   feature integral. The default Gauss rule explicitly reports that no such
   certificate exists, even when no mixed-feature face happened to be sampled.
   Release-level verification now requires this certificate for every
   polyhedral pair; zero *observed* mixed-feature faces or active points no
   longer suffices. An empty or unattached accepted-path record cannot pass
   that claim vacuously; it must agree with current per-pair diagnostics. A
   future whole-face separation certificate may justify
   a separate no-contact release path, but point samples cannot.
   The narrow-strip tests in `test_contact_active_partition.py` and the
   missed-interior-overlap test in `test_initial_gap_bound.py` are concrete
   counterexamples to equating zero active quadrature points with separation.
   Engineering-level verification continues to report the narrower sampled
   diagnostic. This release guard does not itself supply a scientific reference
   or make the nonconvex operator accurate.

4. **Normal enforcement as an explicit law.** Retain penalty as one declared
   approximation. Prototype a separate opt-in augmented-Lagrangian law first
   on a fixed-quadrature analytical plane. With positive-outside gap `g`,
   fixed nonnegative multiplier `lambda`, and `k>0`, use
   `p=max(0,lambda-k*g)` during each Newton solve. Update the multiplier only
   after a converged solve; require separately specified global equilibrium,
   penetration and complementarity checks before the whole increment commits.
   Store stable point IDs, units, law and policy in State/checkpoint identity.
   Reject trial-changing contact quadrature until a conservative multiplier
   transfer is designed and tested. Current state: the ordinary Step remains
   penalty-only. A solver-neutral fixed-reference-trace AL candidate now has
   point-law, residual/tangent, spring-plane, independent scalar
   elastoplastic-bar indent/release and rollback tests. The scalar bar freezes
   old plastic history through each equilibrium solve, updates the normal
   multiplier only between converged solves, and commits both histories only
   at an accepted tool station; it is **not** an ordinary finite-element Step.
   A real serial DOLFINx tetrahedral trace also matches an independently
   UFL-differentiated
   plane/sphere functional, residual and tangent. There is still no global
   Step, MPI, or installed-package acceptance. A high-penalty L-tool trial
   failed Newton, so neither hard-contact convergence nor an AL benefit may
   be claimed. Reading: [Abaqus contact enforcement controls][acl],
   [COMSOL penalty/augmentation trade-off][comsol-method], and
   [Burman--Hansbo--Larson's augmented-Lagrangian FE analysis][burman]. The
   latter warns that a discontinuous multiplier approximation can need
   stabilization; [Hansbo--Rashid--Salomonsson's stabilized obstacle
   formulation][hansbo-stabilized] likewise explains that too many discrete
   multiplier constraints can make a system singular. Our independent point
   multipliers inherit neither proof and require a spatial-pressure/mesh
   check. A bounded opt-in fixed-trace rank diagnostic now builds the active
   weighted normal-constraint Jacobian and exposes duplicate point constraints:
   two active multipliers with the same nodal support and normal have row
   nullity one, whereas the three independent P1 triangle Gauss rows have
   full local row rank. This is a *negative screen*, not an inf-sup proof: it
   caps dense problem size and never certifies spatial stability. It can now
   also inspect `J T` when supplied the *actual* dense displacement reduction
   `u = T q + u_bar`; a three-point triangle that has full unreduced row rank
   drops to rank two when one normal DOF is fixed or tied by an MPC. This
   catches a different false positive than the duplicate-point screen, but
   the production Step still needs to pass its real reduction and a mesh
   ladder before using the result as an admission criterion.
   The global AL Step still needs a reduced-constraint rank/mesh-pressure
   study. An [exact-penalty alternative][exact-penalty] exists, but would not
   repair the unresolved nonconvex integration and is not a silent replacement
   for the declared AL policy. Nitsche is a serious *primal* alternative:
   [Chouly--Hild][nitsche-small] analyze small-strain unilateral contact
   without a separate multiplier or discrete inf-sup condition, and
   [Mlika--Renard--Chouly][nitsche-large] extend an unbiased formulation to
   large-deformation elastic frictional contact. Neither result is a proof for
   our finite-J2 material, one-sided rigid tool or sharp nonconvex integration.
   A fair method comparison would use the same analytical-plane J2 path,
   penetration, reaction, Newton work and mesh ladder, while counting the
   additional boundary traction and its consistent tangent that Nitsche
   requires. Do not add a second public enforcement choice on the basis of a
   1D penalty comparison alone.

   A private *global* feasibility probe now substitutes the fixed-trace AL
   Operator into the existing serial J2 Newton assembly, leaving its
   material update provisional while manually iterating the multiplier. A
   moving planar tool compressed a plastic cube; with declared penalty 3000
   and penetration/complementarity tolerances 1e-6 it required eight total
   Newton equilibria before one joint **test-owned** commit, with final PEEQ
   above 0.005. At penalty 300, another case still missed the tolerance after
   30 equilibria. The repeatable regression locks only this bounded
   feasibility observation; it is not a public Step, a robust penalty policy,
   atomic checkpoint, or a general convergence proof. The slow outer
   convergence is consistent with the linear-rate cost identified in
   [exact-penalty] and motivates comparing a simultaneous mixed semismooth
   route such as [Alart--Curnier][alart-curnier], rather than automatically
   nesting dozens of solves in the product. PETSc's bound-constrained
   [variational-inequality Newton options][petsc-vi] are another algorithmic
   reference, but a bound on displacement DOFs does not directly represent
   a nonmatching rigid-surface gap or cure multiplier-space instability.
   The next *research* comparison should hold the same fixed trace and J2
   path while solving the Alart--Curnier-style semismooth mixed residual
   `R_u(u,lambda)=R_bulk(u)-B(u)^T[lambda-k g(u)]_+` and
   `R_lambda(u,lambda)=lambda-[lambda-k g(u)]_+`. On an active point the
   latter has displacement derivative `k B` and zero multiplier diagonal;
   on an inactive point it has unit multiplier diagonal. This exposes why a
   rank-deficient *reduced* `B T` can make a simultaneous solve singular,
   even when a scalar example converges in one Newton step. It is a design
   candidate, **not** the implemented global law; a mesh-stable multiplier
   space or reviewed stabilization, consistent geometry derivatives, and
   joint accepted State remain mandatory before a public option exists.

   On the same 2-cell/axis, 12-increment, q7 L-prism diagnostic above,
   increasing the *penalty* from 20 to 200, 1,000, 10,000 and 100,000
   completed the nonlinear solves but changed the final active reference
   area from 0.4415 to 0.4415, 0.4415, 0.1914 and 0.0967, respectively;
   maximum sampled penetration fell from 0.0470 to 0.0319, 0.0170, 0.00690
   and 0.00174. Tool-force x changed from -0.337 to -1.927, -3.487, -5.099
   and -5.705. This is **not** a hard-contact or penalty-independent limit:
   both the contact footprint and force change substantially. The older
   high-penalty Newton failure was a different trial and remains valid at its
   own inputs; neither failure nor completion generalizes to every setting.
   Adding an outer multiplier loop before fixing or bounding the nonconvex
   integral would hide, not resolve, that spatial uncertainty.

5. **Friction and moving-feature State.** Verify normal-pressure coupling,
   stick/slip return mapping, tangent, finite sliding, force/moment and
   dissipated work independently for each supported geometry. Reproject the
   tangential basis and carry accepted history across a facet seam without
   inventing a sharp-edge traction law. If an active noncoplanar edge or vertex
   is unsupported, reject and roll back. Current state: analytical and locally
   planar faceted Coulomb routes exist. Accepted tangential elastic slip now
   follows the rigid master's incremental rotation before the minimal
   old-normal-to-new-normal transport; exact co-rotation of a contacting plane
   has zero relative slip, unchanged recoverable energy and no spurious
   dissipation in both implicit and explicit FEniCSx checks. This fixes spin
   about an unchanged normal, which normal-only transport cannot detect.
   Sharp-edge friction is not admitted.
   Reading: [Abaqus Coulomb law and finite-sliding term][friction],
   [Abaqus' local tangential slip/traction convention][vfric],
   [COMSOL frictional-contact solver advice][comsol-solver], and
   [Duong--Sauer's surface-potential finite-sliding formulation][duong-friction].
   The last is a thermodynamic and tangent-design reference, not evidence
   for an unverified sharp-edge friction law.

6. **One nonlinear Procedure and State transaction.** Assemble the total
   material, contact and load residual/Jacobian before strong-boundary or
   periodic-MPC reduction. Newton, line search, cutback and convergence checks
   use that same total system. A rejected trial restores displacement, J2,
   normal multiplier, tangential State, projection, rigid pose and work;
   an accepted increment commits them atomically. Check interrupted versus
   continuous paths and corrupted-checkpoint rejection. Current state: these
   contracts pass for existing bounded penalty pairs; new enforcement and
   changing integration identities each need their own tests. Reading:
   [PETSc nonlinear line-search/trust-region methods][petsc-snes] and
   [COMSOL contact solver sequencing][comsol-solver].

7. **Results and mechanical evidence.** Publish per-pair raw gap, pressure,
   normal/tangential force, moment, active area and slip with field location,
   interpolation and units. Keep rigid prescribed-motion work, frictional
   dissipation, material work/storage and numerical penalty or augmented
   functional distinct. An algorithmic Lagrangian is not physical stored
   energy. Current state: named force/moment/work and raw contact-point output
   exist on selected routes; the AL research candidate separately exposes
   physical tool reaction and generalized tool force/moment. Prescribed-plane
   translation/rotation finite differences confirm their work-conjugate signs. A
   general first-law closure and cross-method pressure convention are not
   established. Reading: [Abaqus contact output
   semantics][abaqus-output] and [COMSOL important contact variables][comsol-vars].

8. **Independent science gates.** Run a small ladder: spring/plane exact
   solution; open/closed and unloading; smooth Hertz/contact footprint;
   planar sliding friction; convex corner; reviewed nonconvex transition; J2
   loading/unloading; MPC and moving multi-tool composition. For each relevant
   rung check finite-difference total tangent, action/reaction and moment,
   mesh x increment x enforcement sensitivity, regional plasticity, failure
   rollback and restart. Distinguish integration-point fields from smoothed
   display fields and compare equivalent observables with the frozen external
   case. The first independent global scalar oracle now checks a nearly
   elastic finite-J2 unit bar and rigid-plane penalty against the series
   compliance limit `F=delta/(1/E+1/k)` at two mesh levels and two penalties;
   all four satisfy 5e-4 relative tolerance for delta=1e-4. This small-strain
   limit checks the total force path, not plastic flow, curved geometry or
   Abaqus equivalence. A second independent scalar oracle now constrains the
   *plastic* global path: a laterally fixed unit bar has
   `F=diag(lambda,1,1)`, `q_trial=-E log(lambda)` in compression,
   `dp=max(0,(q_trial-Y)/(3G+H))`, and
   `Pxx=E(log(lambda)+dp)/lambda` for `nu=0`. Solving
   `Pxx+k(lambda-x_tool)=0` predicts right displacement, contact force and
   PEEQ at two tetrahedral mesh levels; the tests compare all three without
   calling the AgentFEM material update as their reference. This checks a
   restricted homogeneous J2/contact state, not general forming. The
   same independent scalar return map also follows an eight-increment
   indent--release--reindent path. It checks displacement, contact force and
   PEEQ at loaded, detached and reloaded stations; the accepted mid-path
   checkpoint resumes to the same displacement, plastic history and tool
   work as the uninterrupted run. Under lateral constraint, release can
   reverse-yield before axial force vanishes, so a first-load PEEQ cannot be
   substituted for residual displacement. This is a bounded transaction and
   material-path reference, not nonconvex tool validation. The audit also
   found that automatic J2 incrementation could skip an interior tabular
   reversal, thereby changing the prescribed path while reporting Newton
   convergence. Interior table knots are now required accepted stations:
   fixed controls missing one fail before solving; automatic controls land
   on each knot without forcing an extra output frame. Tabular inputs are
   copied and their metadata identity frozen at creation, closing the
   caller-array mutation gap between runtime path and restart fingerprint. The
   denominator follows the [MOOSE radial-return documentation][moose-return].
   Requiring accepted stations at every interior tabular knot is AgentFEM's
   conservative stateful-path policy, not a claim about Abaqus' default
   increment scheduling. Abaqus documents explicit time points for evaluating
   a cyclic response at known load maxima/minima, and treats exact output
   times as a separate incrementation choice; these distinguish physical
   path resolution from output-frame frequency [Abaqus direct-cyclic time
   points][abaqus-time-points], [Abaqus output timing][abaqus-output-times].
   A separate **diagnostic, not acceptance** now runs
   `examples/finite_j2_hertz_contact.py` on a quarter finite block with a
   spherical tool. Holding `--quarter --grading 2 --indentation 0.002
   --radius 20 --increments 4` and the example's other defaults fixed gives:

   | cells/side | penalty | full tool force | active reference area | max penetration |
   | ---: | ---: | ---: | ---: | ---: |
   | 7 | 100,000 | 0.712466 | 0.142459 | 0.00007267 |
   | 9 | 100,000 | 0.696875 | 0.143011 | 0.00007231 |
   | 11 | 100,000 | 0.692319 | 0.138502 | 0.00007205 |
   | 7 | 300,000 | 0.726972 | 0.134057 | 0.00002812 |
   | 9 | 300,000 | 0.710146 | 0.135631 | 0.00002568 |
   | 11 | 300,000 | 0.706214 | 0.126903 | 0.00002499 |

   Reproduce one row with `python examples/finite_j2_hertz_contact.py
   --quarter --cells 11 --grading 2 --penalty 100000 --indentation 0.002
   --radius 20 --increments 4` in the FEniCSx environment. At 11 cells,
   tripling penalty still changes force by about 2.0% and active area by about
   8.4%; the area also changes nonmonotonically under refinement. Two versus
   four increments were nearly identical at the coarse elastic 2/3-cell
   levels, but that does **not** establish plastic-path or general time-step
   independence. On the 11-cell, 300,000-penalty row, increasing contact
   quadrature degree from 3 (4 points/face) to 5 (9) and 7 (16) gives tool
   forces 0.706214, 0.705531 and 0.705975 respectively, while active areas
   are 0.126903, 0.129983 and 0.127824. This makes force look much less
   sensitive than the active footprint; no quadrature degree is promoted on
   that observation alone. The half-space Hertz force printed by the example
   is only a scale reference: the solved problem is a finite, bottom-clamped block with
   finite penalty, finite-strain kinematics and sampled contact. Neither its
   Hertz mismatch nor completion alone certifies or refutes this formulation.
   The next gate needs a denser, independently checked finite-block reference
   and a declared pressure/area discretization tolerance, not a selectively
   chosen force match.
   An additional higher-order screen used the same quarter block,
   `--displacement-degree 2 --bulk-quadrature-degree 4
   --contact-quadrature-degree 7 --linear-policy gmres_gamg` and four
   increments. At penalty 100,000, 5/7 cells per side gave full tool forces
   0.685484/0.683987, active areas 0.149148/0.144965 and maximum
   penetrations 7.20347e-5/7.18637e-5. At 7 cells and penalty 300,000,
   these were 0.697151, 0.136021 and 2.4576e-5. The higher-order force
   varies less with this mesh change, but penalty and footprint sensitivity
   persist. This is a separate discretization screen, not a converged
   benchmark or a same-DOF performance comparison with P1.
   The nonconvex penalty sequence is not converged.
   Reading: [NAFEMS benchmark
   catalog][nafems], [Abaqus contact verification cases][abaqus-verification]
   and [COMSOL contact meshing advice][comsol-mesh].

9. **Performance and installed release.** Measure complete accepted-increment
   time, nonlinear iterations, search, assembly, linear solve, output and
   memory on frozen inputs. Keep robust direct solve as default; promote an
   optional iterative solve or cache only with same-path results and repeated
   platform-scoped evidence. Run ordinary installed `model.step` cases, early
   unsupported-combination checks, serial checkpoint/export, Linux/macOS CI
   and explicit platform availability report. Current state: opt-in GMRES/GAMG
   shows a same-host small-screen gain, not a universal speedup. A second,
   larger serial screen fixed the quarter finite-block sphere case at 15
   Hex cells per side (3,375 cells), `--grading 2 --penalty 100000
   --indentation 0.002 --radius 20 --increments 4`, with one OpenMP/BLAS
   thread. Only `--linear-policy direct` versus `gmres_gamg` changed.
   Reversed-order direct/GAMG/GAMG/direct runs took 28.071/11.675/11.326/
   28.084 seconds end to end (within-process case wall time), a 2.44x median
   ratio **on this host and input only**. All runs used 20 linear solves and
   returned tool force 0.68951518028 within 2e-12, active area 0.1453987193
   and maximum penetration 7.18696e-5; GMRES counted 540 Krylov iterations.
   The direct/iterative linear phases were about 18.37/1.61 seconds, so the
   observed gain has an identified source. This case remained elastic,
   finite-block Hertz is not an exact scientific reference, and the screen
   does not establish the same speed or robustness for plastic loading,
   friction, sharp features, another mesh or another machine. A separate
   4-cell/axis, 12-increment plastic J2 L-tool solve returned the same tool
   force, PEEQ, active area and **failed** engineering feature claim under
   direct and opt-in GAMG (differences below 1e-12 in reported scalar
   observables); this checks solver-policy consistency, not validation or a
   timed speedup on nonconvex contact. Keep the
   iterative method explicit opt-in pending those checks; distribution and
   independent scientific acceptance remain separate. Reading:
   [PETSc GAMG near-nullspace requirement][gamg], [PETSc profiling][profiling]
   and [Hoefler--Belli scientific benchmarking][benchmarking].

   A separate **plastic-contact runtime** screen on the same host explains
   why changing KSP alone is not the dominant lever here. With an 8-cell/axis
   tetrahedral reentrant L-tool, 12 increments, penalty 20, degree-5 contact
   quadrature, direct solve and one OpenMP/BLAS thread, the pre-change accepted
   increment ledger attributed about 9.65 s to backtracking, 6.38 s to
   material updates, 4.90 s to residual assembly, 0.56 s to tangent assembly
   and only 0.80 s to 60 linear solves. Opt-in GMRES/GAMG reduced linear time
   to 0.42 s but the complete case remained about 28 s in both policies.
   Backtracking had already evaluated the accepted next-state material and
   total residual; the following Newton iteration was evaluating both again.
   The standard J2 Procedure now carries that *accepted* trial response and
   PETSc residual into the next iteration. Rejected trials still destroy their
   residual; basic/no-line-search keeps its former path. This is the same
   function-value handoff described in [PETSc's nonlinear-solver iteration
   model][petsc-snes], adapted to AgentFEM's material State transaction.

   On the exact same source and machine, a temporary A/B harness restored the
   old discard-and-recompute behavior without changing tolerances or input.
   Bracketed direct runs took 28.59/28.51 s with discard versus 19.02/19.01 s
   with accepted-trial reuse (roughly 1.50x, or 33% less complete-case wall
   time on this input). The selected scientific JSON fields (quantities,
   histories, contact duals, mechanical ledger, feature path and verification)
   had identical SHA-256 in the old-behavior and new-behavior runs; both
   `fields.h5` and `fields.integration-points.h5` were byte-identical too.
   This includes the **same failed** engineering feature claim: the runtime gain
   does not validate the nonconvex contact law. It is not a cross-platform
   speed guarantee, and the finite-J2 affine/MPC Procedure needs its own
   ownership review before sharing this execution optimization.

## Promotion rule

Gates 1--4 and 6--8 are mandatory for any claim involving nonconvex rigid
features. Gate 5 determines the friction/geometry combinations allowed in the
claim; a locally planar Coulomb test never licenses sharp-edge friction. Gate
9 is mandatory for a release. If the nonconvex gate remains open, publish only
the actually verified smaller subset, still marked experimental where the
independent physical evidence is absent. Full Abaqus general-contact parity
is not a 0.4.4 claim.

## First implementation evidence (bounded)

The opt-in research module `boundary_models.normal_augmentation` separates
the signed algorithmic AL functional from physical contact energy. It binds
accepted multipliers to a contact-pair identity, fixed reference trace (point
IDs, node support, shape values, integration weights and reference query
geometry), rigid-surface fingerprint, units and explicit
acceptance/outer-iteration policy. It requires
a converged *total* residual
at each frozen-multiplier outer iterate, and atomically refuses mismatched
restart State. Its spring/plane exact solution and smooth plane/sphere
finite-difference tangent tests and the independent DOLFINx/UFL plane/sphere
operator oracle are local mathematical evidence only. They do not establish
the J2+contact+MPC product workflow. The next admission is a
single supported analytical plane in the full nonlinear Procedure, with the
same total-system and accepted-increment transaction as penalty contact; an
adaptive integration partition must continue to reject AL until multiplier
transfer is separately verified.

### Next code boundary: whole-increment augmentation, not a second solver

For that analytical-plane admission, keep `FiniteStrainJ2StandardProblem` as
the only Newton/line-search/linear-solve owner. The contact Operator reads a
fixed outer-iterate multiplier and contributes its total residual and tangent
before the existing MPC reduction. The Procedure snapshots displacement,
material State, tool pose/work and accepted multipliers *once* at the start of
the increment, then repeats: solve total Newton with a frozen multiplier;
test total equilibrium; test penetration and complementarity; if needed,
update only the trial multiplier and solve the total system again. Only a
converged final outer iterate permits one joint commit and one checkpoint.
An exhausted outer loop or Newton/cutback failure restores the complete
pre-increment snapshot. In particular, a converged inner Newton iterate is
not a valid checkpoint, and its algorithmic AL functional is not physical
stored contact energy. Result must reuse the ordinary constraint dual and
accepted-path work evidence, but mark physical contact-energy closure
unavailable until the finite-penetration enforcement residual is separately
bounded; it must not insert the AL functional into `contact_potential` merely
to make an energy equation appear closed. The current penalty
`ContactTraceAssembly` explicitly requires a nonnegative physical potential,
so reusing it as an AL wrapper would violate its contract. The new Operator
needs a contribution type that exposes the same residual/tangent and
force/moment protocol while keeping algorithmic and physical energy distinct.
Admission requires the existing penalty Step to remain unchanged by default
and the AL path to pass a fixed-trace restart identity,
one plastic-history path and one MPC-reduced total-tangent comparison. A
changing feature partition cannot use this State until a conservative,
identified multiplier-transfer rule has its own tests.

[abaqus]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-genlcontnumcontrolsstd.htm
[nafems]: https://www.nafems.org/publications/pubguide/list/
[nafems-contact-3]: https://www.nafems.org/downloads/FENet_Meetings/Glasgow_UK_Oct_2004/FENET_Glasgow_Oct2004_DLE_Becker.pdf
[abaqus-al-verification]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-contsmslcont.htm
[abaqus-general-controls]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-genlcontnumcontrolsstd.htm
[initial]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-adjustsurfaces.htm
[distance]: https://arxiv.org/abs/1803.09687
[meshed-rigid]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-finslcontdefmeshrig.htm
[comsol-verification]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/StructuralMechanicsVerificationExamples.pdf
[moose-return]: https://mooseframework.inl.gov/source/materials/RadialReturnStressUpdate.html
[abaqus-time-points]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-directcyclic.htm
[abaqus-output-times]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAECAERefMap/simacae-t-simodbeditorfield.htm
[duong]: https://arxiv.org/abs/1712.01179
[dual-mortar]: https://arxiv.org/abs/2201.01165
[variational-mortar]: https://doi.org/10.1016/j.cma.2018.07.020
[point-line-surface]: https://doi.org/10.1002/nme.5743
[mortar-integration]: https://doi.org/10.1007/s00466-014-1093-2
[polygon-clipping]: https://doi.org/10.1145/360767.360802
[oriented-volume]: https://doi.org/10.1016/j.cma.2007.06.002
[distance-field]: https://doi.org/10.1016/j.cma.2013.02.015
[abaqus-rigid-smoothing]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMODRefMap/simamod-c-rigidsurf.htm
[burman]: https://arxiv.org/abs/1609.03326
[hansbo-stabilized]: https://doi.org/10.1016/j.finel.2016.03.005
[exact-penalty]: https://www.sciencedirect.com/science/article/pii/S0045782520302929
[nitsche-small]: https://doi.org/10.1137/12088344X
[nitsche-large]: https://doi.org/10.1016/j.cma.2017.07.015
[alart-curnier]: https://doi.org/10.1016/0045-7825(91)90022-X
[petsc-vi]: https://petsc.org/main/manual/snes/#variational-inequalities
[duong-friction]: https://arxiv.org/abs/1808.10420
[comsol-settings]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.146.html
[acl]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-contactcontrolsstd.htm
[comsol-method]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.143.html
[friction]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-coulombfric.htm
[vfric]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAESUBRefMap/simasub-c-vfric.htm
[comsol-solver]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.152.html
[petsc-snes]: https://petsc.org/main/manual/snes/
[abaqus-output]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-contactgeneral.htm
[comsol-vars]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.155.html
[abaqus-verification]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-finslcontdefrigid.htm
[abaqus-c3d8-input]: https://docs.software.vt.edu/abaqusv2025/English/SIMAINPRefResources/ei34srsc_surf.inp
[abaqus-surface-verification]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-finslsurftosurf.htm
[comsol-mesh]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/StructuralMechanicsModuleUsersGuide.pdf
[gamg]: https://petsc.org/main/manualpages/PC/PCGAMG/
[profiling]: https://petsc.org/release/manual/profiling/
[benchmarking]: https://htor.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf
