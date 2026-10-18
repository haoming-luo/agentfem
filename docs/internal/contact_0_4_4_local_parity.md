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
| 3. Integration | Planar/convex partitions; sampled nonconvex risk flag and same-state integration screens | Nonconvex feature-switch integral and tangent |
| 4. Normal law | Penalty Step; fixed-trace AL operator candidate | Outer global Procedure, spatial stability and checkpoint composition |
| 5. Friction | Locally planar Coulomb State; rigid-master spin transport tested | Admitted geometry-by-friction matrix; sharp features remain rejected |
| 6. Transaction | Total-system and restart tests for penalty | Repeat for each new enforcement/integration law |
| 7. Result | Pair work/pressure samples; AL local work sign | Cross-law first-law and pressure interpretation |
| 8. Science | Local/oracle/installed tests; global elastic-limit spring-plane matrix | Mesh x increment x law convergence and external observables |
| 9. Release | Scoped solver screens | Installed cross-platform gate and production runtime evidence |

“Partial” here is not a release status. The test target, geometry and
enforcement must all match before a gate can be marked closed.

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
   [COMSOL's analytical/NAFEMS verification collection][comsol-verification]
   and [NAFEMS advanced contact benchmark catalog][nafems]. A published
   verification case is a comparator protocol, not an external result for
   our own geometry.

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
   declared depth/sample budget cannot prove the requested limit. No open
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
   matrix. Current state: analytical planar clipper, research two-plane oracle
   and opt-in <=3-plane convex partition pass; the result now flags *sampled*
   mixed-feature slave faces for reviewed closed solids in both standard and
   periodic J2 Results, without claiming a complete transition integral or
   changing the Newton law. Nonconvex feature
   transitions remain open. Reading: [Duong--De Lorenzis--Sauer refined-boundary
   quadrature][duong] and [COMSOL contact quadrature guidance][comsol-settings].
   The cited isogeometric extended-mortar formulation is a comparison design,
   not a drop-in P1 facet implementation: its enriched pressure interpolation
   and two-half-pass force construction differ from our one-sided trace.

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
   A new algorithm may enter the Step only after both integral and
   differentiation checks, not merely a finer screenshot or work balance.
   For the separate *convex* route, accepted point diagnostics now say whether
   the reviewed changing-trial support-plane partition supplied its bounded
   feature integral. The default Gauss rule explicitly reports that no such
   certificate exists, even when no mixed-feature face happened to be sampled.

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
   point-law, residual/tangent, spring-plane and rollback tests. A real serial
   DOLFINx tetrahedral trace also matches an independently UFL-differentiated
   plane/sphere functional, residual and tangent. There is still no global
   Step, MPI, or installed-package acceptance. A high-penalty L-tool trial
   failed Newton, so neither hard-contact convergence nor an AL benefit may
   be claimed. Reading: [Abaqus contact enforcement controls][acl],
   [COMSOL penalty/augmentation trade-off][comsol-method], and
   [Burman--Hansbo--Larson's augmented-Lagrangian FE analysis][burman]. The
   latter warns that a discontinuous multiplier approximation can need
   stabilization; independent point multipliers do not inherit its proof
   and require a spatial-pressure/mesh check.

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
   Abaqus equivalence. The nonconvex penalty sequence is not converged.
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
   shows a same-host small-screen gain, not a universal speedup; distribution
   and independent scientific acceptance remain separate. Reading:
   [PETSc GAMG near-nullspace requirement][gamg], [PETSc profiling][profiling]
   and [Hoefler--Belli scientific benchmarking][benchmarking].

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

[abaqus]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-genlcontnumcontrolsstd.htm
[nafems]: https://www.nafems.org/publications/pubguide/list/
[initial]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-adjustsurfaces.htm
[distance]: https://arxiv.org/abs/1803.09687
[meshed-rigid]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-finslcontdefmeshrig.htm
[comsol-verification]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/StructuralMechanicsVerificationExamples.pdf
[duong]: https://arxiv.org/abs/1712.01179
[dual-mortar]: https://arxiv.org/abs/2201.01165
[abaqus-rigid-smoothing]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMODRefMap/simamod-c-rigidsurf.htm
[burman]: https://arxiv.org/abs/1609.03326
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
[comsol-mesh]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/StructuralMechanicsModuleUsersGuide.pdf
[gamg]: https://petsc.org/main/manualpages/PC/PCGAMG/
[profiling]: https://petsc.org/release/manual/profiling/
[benchmarking]: https://htor.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf
