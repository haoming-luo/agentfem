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
   nonconvex finite-J2 fixture. Reading: [Abaqus general-contact scope][abaqus]
   and [NAFEMS advanced contact benchmark catalog][nafems].

2. **Surface and search admission.** Validate closed/open topology, outward
   orientation, self-intersection, initial clearance/overclosure and allowed
   free features; search and project in the current configuration on every
   trial. A local candidate cache may accelerate search only with a
   conservative coverage/invalidation proof, never by reusing old forces or
   normals. Test large sliding, seam crossing, geometry-scale changes and an
   intentionally missed-candidate trap. Current state: smooth projectors,
   triangulated BVH and reviewed closed-solid classifiers exist; arbitrary
   large-CAD and ambiguous sharp features are not admitted. Reading:
   [Abaqus finite-sliding/general-contact formulation][abaqus] and
   [Abaqus initial-overclosure choices][initial].

3. **Contact-boundary/feature integration.** Make the active integration
   partition a trial-geometry Operator input. A zero-gap boundary and a
   nearest-feature switch are different discontinuities: the latter can
   contribute a line term to the Newton tangent. Keep positive weights and
   one consistent residual/tangent partition; refresh a changed partition
   before accepting a Newton trial. Verify manufactured narrow-strip,
   two-plane and curved-transition oracles, then the nonconvex L-tool response
   matrix. Current state: analytical planar clipper, research two-plane oracle
   and opt-in <=3-plane convex partition pass; nonconvex feature transitions
   remain open. Reading: [Duong--De Lorenzis--Sauer refined-boundary
   quadrature][duong] and [COMSOL contact quadrature guidance][comsol-settings].

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
   point-law, residual/tangent, spring-plane and rollback tests, but no global
   Step, MPI, or installed-package acceptance. A high-penalty L-tool trial
   failed Newton, so neither hard-contact convergence nor an AL benefit may
   be claimed. Reading: [Abaqus contact enforcement controls][acl]
   and [COMSOL penalty/augmentation trade-off][comsol-method].

5. **Friction and moving-feature State.** Verify normal-pressure coupling,
   stick/slip return mapping, tangent, finite sliding, force/moment and
   dissipated work independently for each supported geometry. Reproject the
   tangential basis and carry accepted history across a facet seam without
   inventing a sharp-edge traction law. If an active noncoplanar edge or vertex
   is unsupported, reject and roll back. Current state: analytical and locally
   planar faceted Coulomb routes exist; sharp-edge friction is not admitted.
   Reading: [Abaqus Coulomb law and finite-sliding term][friction] and
   [COMSOL frictional-contact solver advice][comsol-solver].

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
   exist on selected routes; a general first-law closure and cross-method
   pressure convention are not established. Reading: [Abaqus contact output
   semantics][abaqus-output] and [COMSOL important contact variables][comsol-vars].

8. **Independent science gates.** Run a small ladder: spring/plane exact
   solution; open/closed and unloading; smooth Hertz/contact footprint;
   planar sliding friction; convex corner; reviewed nonconvex transition; J2
   loading/unloading; MPC and moving multi-tool composition. For each relevant
   rung check finite-difference total tangent, action/reaction and moment,
   mesh x increment x enforcement sensitivity, regional plasticity, failure
   rollback and restart. Distinguish integration-point fields from smoothed
   display fields and compare equivalent observables with the frozen external
   case. Current state: local/unit and installed software gates pass, but the
   nonconvex penalty sequence is not converged. Reading: [NAFEMS benchmark
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
accepted multipliers to a fixed reference trace (point IDs, node support,
shape values and integration weights), requires a converged *total* residual
at each frozen-multiplier outer iterate, and atomically refuses mismatched
restart State. Its spring/plane exact solution and smooth plane/sphere
finite-difference tangent tests are local mathematical evidence only. They do
not establish the J2+contact+MPC product workflow. The next admission is a
single supported analytical plane in the full nonlinear Procedure, with the
same total-system and accepted-increment transaction as penalty contact; an
adaptive integration partition must continue to reject AL until multiplier
transfer is separately verified.

[abaqus]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-genlcontnumcontrolsstd.htm
[nafems]: https://www.nafems.org/publications/pubguide/list/
[initial]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-adjustsurfaces.htm
[duong]: https://arxiv.org/abs/1712.01179
[comsol-settings]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.146.html
[acl]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-contactcontrolsstd.htm
[comsol-method]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.143.html
[friction]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-coulombfric.htm
[comsol-solver]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.152.html
[petsc-snes]: https://petsc.org/main/manual/snes/
[abaqus-output]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-contactgeneral.htm
[comsol-vars]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_modeling.05.155.html
[abaqus-verification]: https://docs.software.vt.edu/abaqusv2025/English/SIMACAEVERRefMap/simaver-c-finslcontdefrigid.htm
[comsol-mesh]: https://doc.comsol.com/6.3/doc/com.comsol.help.sme/StructuralMechanicsModuleUsersGuide.pdf
[gamg]: https://petsc.org/main/manualpages/PC/PCGAMG/
[profiling]: https://petsc.org/release/manual/profiling/
[benchmarking]: https://htor.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf
