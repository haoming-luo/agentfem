# AgentFEM Product Roadmap

## Direction

AgentFEM is building a readable, dependable, and extensible finite-element
platform for people and AI agents. It does not replace the finite-element
kernel already provided by FEniCSx, Basix, PETSc, and MPI. AgentFEM owns the
engineering language, numerical procedure, state lifetime, result contract,
verification evidence, and extension boundary around that kernel.

The public workflow remains:

```text
Study -> Model -> scientific assets -> model.step(...) -> SimulationResult
```

Capability claims are promoted by executable evidence. A formula, an example,
or a successful run is not by itself a validated engineering capability.

## Current product phase: the 0.4 foundation

### User-driven mesh/interface slice (2026-10-09)

The urgent next implementation is fixed-reference nonmatching cohesive trace
integration, followed by uniform-strain Hex8 stabilization. These are missing
general formulation capabilities, not an automatic translation of a specific
Abaqus forming model. Reuse triangle search and existing cohesive laws; keep
pairing, material state, force assembly and Procedure separate. The bounded
design and scientific promotion gates are in
[nonmatching interfaces](nonmatching_interfaces.md). Initial local trace tests
do not enable migration, finite rotations, quadrilateral traces or MPI.
Do not introduce artificial damage thresholds for an elastic-only interface,
or describe unstabilized one-point quadrature as Hex8R support.

The fixed coplanar P1 route now has common-refinement integration, per-facet
coverage checks, independent nodal-force audits and serial two-block
series-compliance evidence. Its experimental elastic `model.step` provider
reuses PETSc solving and Result output. The affine planar Q1 extension retains
the original four-node basis and degree-four common-refinement integration;
global 1:3 and 2:3 compliance tests exercise independent Hex8 partitions.
Global damage, MPI and finite-rotation kinematics remain separate gates.
The experimental [uniform-gradient Hex8 policy](hex8_uniform_strain.md) now
has compact batch execution, serial global affine/bending/wave evidence and
ordinary explicit `model.step()` integration. Isotropic/rotated anisotropic
elasticity, disjoint material regions, physical/artificial energy separation
and serial restart are exercised. Failed explicit increments and auxiliary
checkpoint rejection restore the accepted field state. Finite deformation,
damage, external-work closure and MPI remain separate gates. The elastic Q1
interface now composes with this explicit Step, including an additive spectral
stability bound, interface energy, serial restart and failed-commit retry.
Warped/nonaffine quadrilateral traces remain rejected. These are bounded
small-strain capabilities, not an industrial finite-deformation reproduction.

The 0.4 line is an architectural consolidation, not a feature-count release.
Its stable middle layer is:

```text
Model -> Operator -> Procedure -> State -> Result / Verification
                         |
                      Backend
```

Each owner answers one question:

- **Model:** what engineering problem is being solved?
- **Operator:** what mathematical contribution is assembled or applied?
- **Procedure:** how is the problem advanced and solved?
- **State:** what accepted and trial quantities must survive?
- **Backend:** which numerical runtime executes the formulation?
- **Result / Verification:** what was computed, and why is it usable?

For 0.4.0 and subsequent releases, AgentFEM must keep these boundaries acyclic, preserve one
provider-owned lowering route, retain atomic state and result lifecycles, and
pass a candidate-bound release ladder. The executable audit is:

```bash
python promotion_gate.py --target 0.4-foundation
```

A complete 0.4 candidate additionally requires complete serial tests,
representative two-rank state/nonlinear/output/checkpoint tests, clean installed
wheel acceptance, unchanged public examples and compatibility imports, Linux
and macOS acceptance, and benchmark evidence for every maturity change.
Windows runtime acceptance remains a separate product gate.

### 0.4.0 release and the next bounded cycle

0.4.0 consolidates the foundation rather than promoting experimental physics.
The release pipeline must gate publication on the same immutable wheel's
Linux/macOS foundation evidence. See [release scope](release_0.4.0.md).

After release, prioritize in this order:

1. installation/upgrade, examples, result readability, and actionable user
   failures; protect projects and results independently of runtime replacement;
2. supported mesh--element--formulation interoperability and early quality
   diagnostics, with small representative engineering regressions;
3. measured assembly/solver/state/output efficiency under fixed scientific
   inputs; no speed claim obtained by relaxing tolerances or changing physics;
4. one bounded external-validation track, beginning with the existing
   finite-strain RVE gaps, without reclassifying incomplete evidence as passed.

Each item needs a reusable owner, a finite acceptance test, and a stop/review
condition. New materials, forming, additional backends, or benchmark-specific
public APIs are not prerequisites for this consolidation. Keep routine CI
targeted; reserve the full installed-wheel ladder for milestones and releases.

#### Agile maintenance slice before coupling work

Do not turn the installation/mesh/performance priorities into another broad
foundation programme. The current bounded slice shares transient operator
accounting, fixes evidence loss after prepared-system release/continuation,
and reduces allocation in exact boundary-identity checks. It changes neither
the public modelling API nor scientific maturity. Targeted serial/restart and
two-rank tests are its exit gate; see the reproducible measurement instructions
in [development testing](development_testing.md#transient-lifecycle-maintenance).

The next bounded implementation is **sequential thermal--mechanical
coupling on a shared mesh**, using existing thermal fields, eigenstrain,
EngineeringStep and ordinary providers, not a new coupling god object.
Audit the following before implementing a public convenience layer:

1. Field transfer now guards same-mesh/space/component identity and explicitly
   declared equal time. Copy is exact-space only; interpolation is explicit.
   Equal array length is not evidence for compatible field transfer.
2. A transferred temperature may change only eigenstrain/RHS, or also material
   stiffness. Preserve this distinction in the existing time-input contract.
3. Define ownership of accepted source and target states, failed-target
   recovery, and restart provenance without confusing one-way sequencing with
   converged two-way coupling.
4. Begin with uniform free thermal expansion and restrained thermal stress;
   then one spatial-temperature, multi-region regression and two-rank restart.

The same-mesh slice is implemented with nodal field transactions and uniform/
linear-temperature analytical response tests, including two-rank rollback and
transfer after thermal restart. Temperature/material dependence retains the
ordinary operator lowering; no new global coupling solver is introduced.
The multi-region handoff regression now combines two conductivities, regional
E(T) and thermal expansion with exact cell-average stress checks. The existing
hot-wall example restores accepted heat and recomputes only mechanics; a
rejected structural trial rolls back field and transfer evidence together.
This is durable upstream reuse, not a combined multiphysics checkpoint.
The next two-way slice now has a bounded 3D verification prototype with a
reversible heat-feedback Operator, unrelaxed residuals, in-memory joint rollback
and an independently written monolithic oracle. See [coupling design](coupling_design.md).
The private experiment now verifies inward flux, prescribed-temperature heat
reactions, affine prescribed-motion path work, and joint durable recovery.
Registered strong boundaries and a bounded ordinary Step provider now reuse
the physical Procedure and joint restart, with explicit SI/participant-load
contracts and final-snapshot output. Accepted-window progress/checkpoint cadence
now reuses the existing reporter and policy, with explicit retention. Field-series
output remains explicitly final-snapshot-only. Local installed-wheel serial/MPI
acceptance passes for the candidate recorded under `evidence/coupling/`;
public-provider cross-partition recovery now also passes `1 -> 2 -> 1` with
prescribed motion and accepted force/work history. PR #90 is merged at
`1ecf41e0` after final-head FEniCSx/macOS/docs checks passed. The 0.4.1
release candidate retains the immutable-wheel publication gate. A separate smooth
Fourier-mode oracle now shows approximately second-order spatial and first-order
temporal convergence of both fields through the ordinary Step; its 1D mode
embedded in 3D is not general multidimensional validation. See the coupling
design for parameters, errors and fixed acceptance thresholds;
no general coupled capability is declared stable.

Execution order: finish this bounded thermal--mechanical product route first,
then audit contact against existing moving-tool/work/search implementations
before selecting any missing capability. A post-release two-rank audit passes
34 tests per rank for the explicit contact provider, distributed search and
accepted work. The serial search microbenchmark confirms existing BVH pruning
against exhaustive projection without modifying solver physics. Next measure
representative solve-stage costs before optimizing; implicit friction, wider
topology and extreme-scale routing remain separate gates. Do not reimplement already tested
contact features based on an outdated roadmap. Mesh compatibility and measured
performance changes follow demonstrated application needs, not a parallel
architecture rewrite. Small releases do not rebuild complete installers.

Stop after this vertical route has explicit field/response and state evidence.
Nonmatching-mesh transfer, broader nonlinear coupling, monolithic Jacobians,
and additional physics remain separate subsequent decisions. Installation or
mesh defects reported by users still interrupt this ordering when they block
an otherwise supported workflow; no speculative installer rewrite is planned.

### 0.3.8: the 0.4 foundation candidate

0.3.8 is the deliberate consolidation release before 0.4.0. It is promoted by
stable ownership and executable contracts, not by adding another material or
solver family. Its six product gates are:

1. one mesh--element--function-space compatibility matrix shared by inspection,
   validation, lowering, result identity, and checkpoint identity;
2. mesh-quality evidence and early, addressable rejection of unsupported
   topology or formulation combinations;
3. typed time inputs that distinguish right-hand-side changes, operator
   changes, state changes, and output-only observations before a Procedure
   chooses reuse or rebuild;
4. provider-owned reaction, force, work, and energy closure for exact MPC,
   weak constraints, and the first bounded contact route;
5. one MPI result, state-identity, and checkpoint lifecycle across supported
   procedures, including deterministic cross-rank-count restoration where the
   capability declares portability;
6. one independently installed extension that adds a material or Procedure
   through the public provider boundary without modifying AgentFEM core.

Every gate must preserve the stable Model, Operator, Procedure, State, Backend,
and Result/Verification ownership inventory. Passing only unit tests or adding
new public names is not sufficient evidence for promotion.

Gate 6 now has an executable installed-wheel acceptance path. The reference
extension is built as a separate distribution, discovered lazily through the
standard entry-point group, and activated only by an ordinary project's
declared dependency. `extension_gate.py` installs the candidate core and
extension wheels into an isolated environment, executes the same Model →
Procedure → Result/Verification lifecycle used by users, retains the provider
identity in the result, verifies the sealed manifest, and proves by before/after
hash that the installed AgentFEM package was not patched. The reference material
is intentionally only an acceptance fixture; new scientific behavior remains
owned by companion or third-party packages. Release-tier CI now runs
`foundation_gate.py` only after the complete serial suite, representative MPI
state/nonlinear/output/checkpoint checks, installed-wheel examples,
compatibility imports, and external provider gate have passed. The resulting
foundation certificate binds those stages to the exact clean commit and wheel
digest. The same release workflow then exercises that exact Linux-tested wheel
on a hosted macOS runner and performs one final evidence fan-in; promotion can
pass only when both immutable platform records match the candidate. This
cross-platform job is release-only, so ordinary targeted and core changes do
not consume macOS capacity. `promotion_gate.py --target 0.4-foundation` no
longer waits for a record that no workflow can produce or audits one platform
before the other exists.

Fresh-agent acceptance remains a separate product gate rather than a seventh
foundation abstraction. Release-tier CI now emits an immutable trial bundle
bound to the exact wheel, source commit, task and review instructions. A
genuinely fresh AI agent must still create, run, verify and explain the model
without repair prompts. The acceptance recorder then binds the project source,
structured result, transcript and reviewed explanation to that candidate.
Deterministic CI may prepare and audit this evidence, but cannot impersonate
the unfamiliar agent whose behavior the gate is intended to test.

For 0.3.8, that engineering contract and candidate-bound evidence route are
accepted as the release gate; continued zero-intervention runs by unfamiliar
users and agents remain field evidence rather than a reason to hold back the
foundation release.

Gate 3 now has one typed contract across the supported transient and ordinary
incremental nonlinear Procedures. Built-in and custom time inputs retain
separate RHS/operator/state/output effects. Linear implicit dynamics and
linear implicit-Euler heat transfer use those effects to select safe operator
reuse or refresh; nonlinear first-order and load-path Procedures declare
per-step or per-attempt residual/tangent assembly and restore the accepted
coordinate on rejection; matrix-free Explicit declares per-increment residual
evaluation and reports whether a fixed preflight stability bound must remain
conservative over an operator-changing path. Transient checkpoint schema v5
and ordinary nonlinear checkpoints bind the complete time-input identity
before authorizing restart. The runtime architecture audit now exposes both
owned-module counts and deliberately unowned utility roots in addition to
cycles and forbidden cross-layer dependencies.

Gate 4 is closed for its declared bounded routes: exact homogeneous rectangular
MPC, exact affine-periodic paths, scalar/normal/matrix elastic foundations, and
frictionless contact with one analytical rigid plane, fixed or following a
prescribed translation/rotation path. Force and work no longer share
one overloaded endpoint record. Providers publish `ConstraintDualEvidence` for
force and physical resultants; nonlinear and non-proportional Procedures publish
typed `ConstraintWorkEvidence` only after integrating accepted stations. The
homogeneous MPC contract states exact zero constraint work instead of inventing
a unit endpoint coordinate, while affine-periodic work retains its full path and
integration rule. Engineering verification checks that serialized duals and
path work agree with the balance contract. The contact provider owns its
potential, residual, tangent, nodal dual distribution, resultant, penetration
diagnostics, and contact energy in serial and under two MPI ranks. Its atomic
portable restart envelope covers the solution, increment ledger, events,
provider-dual history, natural-load work, bulk strain energy, contact potential,
and prescribed rigid geometry, with 1-to-2 and 2-to-1 MPI rank-count acceptance.
Moving tools publish force--translation and moment--rotation path work. General
surface search now has reviewed serial and distributed BVH infrastructure,
stable accepted/trial projection State, collective Procedure acceptance, and a
geometry-neutral pointwise frictionless penalty law. The local law preserves
the same conservative potential and explicitly separates structural residual
traction from conjugate prescribed-surface traction. It does not claim
a geometry-consistent Newton linearization or assembled path-work evidence. A
backend-neutral `ContactTrace` contract now carries stable point identity,
interpolation and explicit reference/current quadrature measure into a checked
reference assembly of nodal residual, conservative potential, resultant and
tool moment. The assembly accepts only matching projection records and query
coordinates. The DOLFINx trace adapter is executable for tagged exterior
triangles of tetrahedral meshes and quadrilateral faces of hexahedral meshes
with continuous blocked vector CG1 or CG2 displacement. It restricts the cell
basis to positive facet quadrature and evaluates the surface Jacobian from
complete Lagrange first- or second-order coordinate geometry. Both routes use
partition-independent facet/point identity, synchronized ghost values, and
collective rejection of unsupported spaces. Standard Q2 hexahedra also pass
the positive row-sum mass gate and the complete Explicit contact Procedure.
Standard P2 tetrahedra do not: their row-sum lumped mass has non-positive
vertex entries, so Explicit fails closed until a reviewed positive mass-lumped
high-order simplex element is available. Prism/pyramid mixed-facet adaptation,
implicit linearization, deformable master surfaces, free rigid-body dynamics,
and undeclared weak formulations remain outside this gate rather than
inheriting its evidence.

Reviewed triangle files can enter the same route as scientific geometry
assets with canonical identity and source hashes. The explicit contact Backend
automatically selects the serial BVH or the partitioned routed MPI BVH; the
Model does not own or expose search acceleration. This closes the import-to-
search ownership boundary, not CAD healing or general deformable-to-deformable
contact.

The first explicit Procedure hand-off is now executable for that narrow
DOLFINx slave trace. A composable residual evaluates the current predicted
displacement, repeats exact projection, applies the local frictionless penalty
law, integrates the trace, and adds only the owner-reduced contact vector to
the already assembled bulk residual. Its projection and work trials follow the
same increment commit/rollback decision as central difference. MPI-global
potential, force/reaction, active-point count and tool moment remain
inspectable. One explicit physical-time schedule can drive proportional rigid
translation and rotation; its initial and accepted generalized stations produce
force--translation plus moment--rotation work and interval power. Checkpoints
bind the schedule identity and restore the accepted path while recomputing the
memoryless projection. A caller-supplied whole-system stability ceiling
remains available. For fixed-normal or piecewise-planar projectors, the same
route can instead assemble a conservative contact spectral contribution from
the trace, penalties, Coulomb pressure coupling, lumped mass, and MPI
ownership, then add it to a supplied unsafed non-contact spectral bound before
selecting the time increment. Curved-normal geometric stiffness still
requires a reviewed caller ceiling. The same
residual now consumes the reviewed serial and routed distributed triangle
BVHs. Stable slave-point IDs cross changing rigid facets under finite sliding
in serial and at two MPI ranks, while accepted work remains rank-canonical and
pure tangential tool motion adds no false normal work. This closes one bounded
moving triangulated-tool route, not a shared multi-tool broad phase, arbitrary
higher-order topology, deformable contact, or implicit Newton linearization.
Multiple bounded pairs already compose through the residual chain,
including time propagation, atomic trial decisions, summed energy/work, and
nested checkpoint State; the reviewed mixed pair combines one analytical and
one triangulated tool without a case-specific solver.

Explicit time-step composition is now a Procedure-owned contract rather than
a contact or fracture special case. Body/material, cohesive-interface and
contact screens publish named equivalent spectral upper bounds. The Procedure
adds all simultaneously active stiffness contributions, converts the sum to
one central-difference limit and applies one safety factor. The legacy
finite-strain cohesive screen now delegates to this contract and retains its
body/interface compatibility fields only as a result view. This closes the
known unsafe `min(body_dt, interface_dt)` composition; formulation-specific
element eigenvalue bounds and state-dependent re-estimation remain later
accuracy promotions.

Gate 5 now has an executable checkpoint capability contract. Procedures must
separately declare the durable payload scope, accepted save boundary, atomic
publication, scientific identity, and MPI rank-count portability; a portable
scalar sweep ledger can therefore no longer masquerade as a full-field
restart. Transient, ordinary and affine nonlinear, harmonic, J2, creep,
viscoelastic, user-material, finite-strain J2, and cyclic-fatigue owners expose
that contract in their summaries. Cyclic bulk fields have moved from
partition-local shards to coordinate-keyed schema v2 and pass both 1-to-2 and
2-to-1 rank-count restoration. The composed global cyclic lifecycle now passes
those same migrations while preserving bulk fields, physical-facet-keyed
cohesive history, and the cycle-jump ledger before continuing to the
uninterrupted reference. The consumer boundary is now implemented:
`model.step()` resolves the exact
procedure contract before numerical execution, rejects malformed or
unsupported policies with stable `AFM-CHECKPOINT-*` diagnostics, and rolls
back any provider-registered Step objects when lowering fails. The same typed
contract is a top-level `SimulationResult` record, and engineering/release
verification checks every emitted checkpoint schema and portability claim
against it. Gate 5 is therefore closed for the declared procedures and tested
rank-count migrations. Collective high-throughput storage remains a scale
optimization, not a missing lifecycle contract.

## What is usable today

The installed capability catalog is authoritative. It covers the supported and
experimental boundaries for:

- linear solid mechanics, heat transfer, modal and structural dynamics;
- viscoelasticity, J2 plasticity, Chaboche hardening, creep, cohesive response,
  finite-strain hyperelasticity, and experimental finite-strain J2 workflows;
- multi-material regions, eigenstrain/thermal-strain semantics, material-aware
  result projection, periodic constraints, and selected Abaqus migration paths;
- mesh import, quality inspection, results, histories, progress,
  checkpoint/restart, provenance, convergence, and ParaView-oriented output;
- campaigns, scientific datasets, surrogate workflows, external providers,
  and learned-constitutive contracts.

Inspect the exact installed truth with:

```bash
agentfem capabilities
agentfem capabilities --json
```

## Near-term priorities

### 1. Trusted mechanics

The next scientific promotions focus on depth rather than catalog size:

1. complete external DCB, ENF, and MMB cohesive validation, including unstable
   propagation control and closed force--work--energy evidence;
2. advance finite-strain J2 through the remaining external RVE comparison.
   Ordinary strong-boundary J2 now records accepted-path dead-load
   work, prescribed-motion reaction work, stored energy, cumulative plastic
   dissipation, and their mechanical residual in the same rollback and
   checkpoint lifecycle. A four-to-forty increment refinement reduces the
   reference patch residual from about `9.4%` to `9.3e-6`, so the ledger
   exposes path-integration error instead of manufacturing closure. Current-
   configuration follower pressure now enters the same residual through its
   UFL-derived external tangent, while its accepted work integrates the
   configuration-dependent load vectors at both increment ends. Three-
   dimensional mixed P2/DG0 state now also passes one-to-two and
   two-to-one MPI checkpoint/continue with exact split primary fields and
   quadrature state. The external-beam gate is accepted from one
   executable package identity without relaxing a tolerance: the external
   curve, mesh, serial/MPI, checkpoint/restart, and `45 -> 90 -> 180`
   increment contracts all pass. The 90-to-180 RMS and maximum differences
   are `0.04460%` and `0.21498%`, within the fixed `0.2%` and `0.5%`
   contracts. For the separate Zhang Q2/DPC1 RVE, content-bound candidate and
   multi-axis audit records now replace caller-declared convergence flags;
   the scientific runtime, benchmark fixture/driver, executed high-order mesh,
   region/boundary tags, and periodic equations are bound explicitly. Path and
   quadrature slices must retain the identical discretization rather than
   relying on a nominal mesh-size label. A separate Appendix-B oracle
   evolving \(C_p^{-1}\) matches the provider's stress, plastic strain,
   inverse plastic metric and elastic energy on a non-coaxial path, so the
   remaining external gap is no longer attributed to a J2 normalization or
   state-variable mismatch.
   A 2026-10-07 source review found a phase-placement discrepancy between
   prose and Figure 10(a) in the inspected author manuscript. Explicit
   `geometry_source` now prevents mixing those models. Figure-based geometry
   at 2859 cells gives stress/energy/tangent errors of `0.099% / 0.014% /
   0.507%`, within the unchanged comparison contracts, but the last mesh pair
   changes energy by `5.3442%`; complete promotion remains blocked by energy
   convergence and remaining figure-specific evidence. The fine candidate
   has now been reproduced from clean commit `597e8795`, with identical stress
   and energy. A same-mesh 4/6/8-degree quadrature slice passes. Energy-channel
   attribution assigns 99.8554% of the last mesh change to the pressure
   constraint defect, not to the condensed channel; this is diagnostic
   attribution, not a license to switch the acceptance observable.
   Offline raw-point reconstruction localizes 97.19% of the defect in the
   worst 5% of cells, almost entirely in the matrix. An explicit interface
   refinement diagnostic uses 2448 rather than 2859 cells and reduces the
   defect by 17.30%, retaining the three external comparison checks. This
   remains a development experiment, not a mesh-convergence certificate or
   general adaptive-solver capability.
   A clean same-policy 1755/2448/3543-cell interface-size sequence now keeps
   all three external comparisons within their contracts. Its final stress
   and tangent changes pass at `0.06847% / 0.25812%`, but primal-energy change
   remains `0.85369%`, above `0.5%`; local refinement does not promote global
   mesh convergence. All raw-point energy reconstructions agree. Result-envelope
   auditing additionally exposed a self-hashed manifest; the writer now excludes
   only its own publication path while retaining external artifact verification.
   Original diagnostic archives retain the old envelope failure explicitly.
   The 2026-10-08 follow-up separates local from globally scaled refinement:
   local primal-energy change reaches `0.46575%`, but global change remains
   `0.72236%` (above `0.5%`). Local stress and global tangent also fail the
   decreasing-change requirement. Five clean runs verify their output seals;
   neither complete spatial axis is promoted. See the
   [spatial follow-up](https://github.com/haoming-luo/agentfem/blob/main/evidence/zhang_2021/2026-10-08-spatial-followup.md).
   Figure-specific 285-cell lifecycle verification now passes direct serial/MPI
   response equivalence, both cross-rank restart directions, and a local
   macro-tangent perturbation audit. A larger `1e-3` perturbation failed and
   remains recorded; MPI tangent condensation is still unsupported. These
   coarse lifecycle checks do not promote the fine spatial/formulation gates.
   See the [bounded verification record](https://github.com/haoming-luo/agentfem/blob/main/evidence/zhang_2021/2026-10-08-lifecycle-verification.md).
   The earlier diagnostics
   below use the prose geometry and must not be interpreted as unresolved
   constitutive errors or transferred to the figure geometry automatically.
   Initial 20/40/80-increment and degree-4/6/8 quadrature diagnostics show that
   path refinement and over-integration do not explain the remaining Table 5
   stress, energy and tangent gap. A 315/459/804/1590/2859-cell spatial
   sequence remains unconverged: at 2859 cells, close to the paper's stated
   2823, the published-energy error falls from about `20.5%` to `4.21%`, while
   the first-Piola error remains about `5.59%`. Element count alone therefore
   does not reproduce unpublished connectivity or geometry approximation. The
   fixed-old-state final-gradient differences now agree with the condensed
   macro tangent to about `1.43e-8`, even though that coarse tangent remains
   about `21.68%` from Table 5. This rules out an internally inconsistent
   Jacobian as the leading explanation without turning a self-consistency check
   into external validation. A controlled independently remeshed 1x1/1x2/2x1/2x2
   curved-geometry family now passes its 1% stress and 5% energy diagnostic
   contracts (observed maxima 0.7404% and 4.1493%). An independent Q9/DPC1
   hand-polynomial and curved-patch oracle passes. Serial/two-rank response
   equivalence and bidirectional 1-to-2/2-to-1 checkpoint continuation also
   pass. Exact discrete topology replication and full mesh/formulation
   convergence remain distinct open questions. The clean-source,
   content-bound `1e-5/1e-6/1e-7` tangent study now passes: its coarse-to-
   middle observed order is `1.998`, and the two finest finite-difference
   tangents differ by only `1.36e-8`. Blind refinement, parameter
   tuning, and tolerance relaxation are explicitly excluded;
3. extend the closed bounded MPC/weak/contact evidence contract only through
   independently verified provider routes; general contact remains a separate
   scientific promotion rather than a foundation blocker;
4. scale the now executable partition-independent raw integration-point HDF5
   result and portable quadrature-checkpoint identity from the compact
   root-gathered route to independently partitioned extreme-scale output;
5. retain every material and fracture capability at its proven maturity until
   its independent benchmark and failure tests pass.

The general-contact expansion after 0.3.8 is deliberately ordered by
ownership rather than by one forming example. Model assets will describe
surfaces, rigid-body motion, contact pairs and laws; Operators will own gap,
projection, residual, tangent, force, moment and energy; Procedures will own
search/update cadence and acceptance; State will own active projections and
history; Result/Verification will own penetration, generalized work and energy
closure. The immutable `RigidBody` asset now binds scientific surface,
fixed/prescribed kinematics, reference point, and rank-independent identity
without owning search or enforcement. `RigidContactPair` now binds one slave
boundary, rigid body, and scalar frictionless law without taking ownership of
search or Procedure stability. Prescribed analytical-plane motion and its accepted
force--moment--work contract are now the first executable slice. Next come
the solver-neutral projection contract (now executable for analytical planes,
circles/spheres and infinite cylinders), reviewed tessellated surfaces (the
serial triangle reference projector and fail-closed geometry audit are now
executable, together with a deterministic process-local AABB/BVH accelerator
and per-query work evidence plus stable barycentric facet coordinates),
distributed ownership/search (the stable-ID facet partition and all-gather
correctness reference now reproduce the serial oracle under MPI; scalable
neighborhood routing now has an oracle-checked two-stage rank-AABB path with
aligned integer/float `MPI_Alltoallv` payloads; owned exterior triangles from
first-order DOLFINx tetrahedral meshes now enter the same stable-ID contract,
including empty local shards; imported MeshTags and named BoundaryRegion
subsets now preserve their physical selection, while higher-order tessellation
and measured extreme-scale routing remain),
accepted/trial projection State (stable 64-bit contact-point identity,
facet-changing finite-slide trials, rollback, and accepted-boundary snapshots
are now executable without importing solver or contact-law ownership), then a
Procedure-owned search/update cadence (now executable as an every-evaluation
lifecycle with collective MPI rejection and explicit increment
commit/rollback) and an Operator that consumes the trial projection. Candidate
warm starts may optimize broad-phase work later, but exact gaps and normals
remain evaluation-local. Accepted discrete-surface projection evidence is now
serialized by stable global point ID and restored onto the current MPI
partition, including master facet and barycentric coordinates. It remains
audit State, not a search cache: projection is deliberately recomputed after
restart from portable accepted displacement and schedule time. The explicit residual hand-off
now supports fixed or proportional prescribed motion, a mandatory declared
stability ceiling, rank-canonical force--moment--work evidence, accepted-path
restart, and finite sliding over one serial or routed distributed triangulated
tool. Bounded rigid pairs compose with independent identity and summed
energy evidence. The first friction foundation is now solver-neutral: a pair
can declare penalty Coulomb friction; stable point-keyed tangential State
supports objective normal-rotation transport, stick/slip return mapping,
trial/commit/rollback and accepted-boundary snapshots; and its response keeps
recoverable penalty energy, irreversible sliding dissipation, and separation
release distinct. The bounded DOLFINx explicit residual now consumes that
contract. It follows the same master material point through prescribed
translation/rotation, forms the accepted slave-minus-master increment,
assembles normal and tangential residuals, and publishes separate recoverable,
dissipated, and separation-release energy channels. Point history is
checkpointed by global ID and restored onto the current MPI partition,
including empty local shards. The mandatory caller-supplied stability ceiling
now has a conservative automatic alternative for fixed-normal and
piecewise-planar geometry. It includes normal/tangential penalty stiffness,
Coulomb pressure coupling, cross-component contact coupling, real lumped mass,
and ghost-to-owner accumulation. It publishes contact and combined spectral
evidence, and deliberately rejects the unsafe shortcut of taking the smaller
of separately derived body and contact time limits. The external
finite-sliding gate is fixed to the public Abaqus/Explicit
deformable-body/rigid-surface family rather than a forming-specific solver.
Its parameters, two-stage protocol, and force/friction assessor are
machine-readable, while the registry explicitly retains
`protocol_solid_bridge_refinement_accepted_exact_b31_gate_pending`. A real deformable
tetrahedral-solid bridge now applies the fixed 500-unit resultant, accepts the
frictionless preload, atomically transfers it into a new stage, activates
`mu=0.3`, and completes the 0.1-unit rigid slide. Serial and two-rank runs
recover 500 normal force, the 150 Coulomb force, exact action--reaction,
positive dissipation, valid projections, actual stable-facet crossings over a
bounded tessellated tool, and the declared energy tolerance.
This is explicitly protocol-level evidence, not a pointwise B31 reproduction.
The bridge now also owns separated refinement evidence: one run halves both
stage increments at fixed physical duration, then another increases in-plane
mesh resolution while retaining the refined increments and duration. The
accepted certificate checks dissipation stability, energy non-growth, and
facet migration without claiming an observed convergence order from only two
levels.
Registered rigid pairs enter the
ordinary finite-strain Explicit Procedure without user-authored residual
plumbing. The Procedure builds reviewed traces, composes body, cohesive, and
every contact spectral contribution once, and preserves per-pair force,
moment, tool work, potential, friction dissipation, State, and checkpoint
evidence. Imported triangle tools and curved CG2 slave geometry now enter this
bounded route; Q2 hexahedra complete the Explicit Procedure while standard P2
tetrahedra fail the positive-mass gate. The next independent gate is an
element-equivalent external comparison if a reviewed beam-contact formulation
becomes available. Distributed multi-tool candidate routing, measured
extreme-scale behavior, additional topology, and implicit friction
linearization follow that bounded route. STEP
remains an optional geometry-adapter input that is repaired and tessellated
before the contact core consumes it. Shell/solid-shell forming and self-contact
remain later scientific promotions, not hidden extensions of the bounded
fixed-plane provider.

The first imported-tool boundary is now executable for triangle-only formats
supported by the optional `meshio` adapter, including STL. Import requires an
explicit coordinate scale because STL does not encode a dependable length
unit, binds the file SHA-256 as provenance, canonicalizes exact geometry to
stable facet identity, and exposes normal reversal only as an explicit user
decision. Mixed topology, degenerate/duplicate facets, non-manifold edges and
inconsistent winding fail before search. Approximate repair and STEP-to-mesh
tessellation remain external preparation capabilities; they do not silently
alter the contact Model asset.

### 2. Mesh and element foundation

Mesh breadth advances by preserved semantics and executable evidence, not by
recognizing more connectivity names.

- Maintain verified P1/P2 simplex and tensor-product import, patch, tag,
  quality, output, and MPI routes.
- Promote prism and pyramid routes only when the active DOLFINx I/O and solve
  stack preserves the external topology end to end.
- Keep beam, shell, cohesive, reduced-integration, hybrid, and stabilized
  elements behind dedicated formulations rather than topology aliases.
- Admit mixed-topology solve domains only when regions, assembly, results,
  checkpointing, and MPI preserve every block.
- Diagnose quality without silently moving nodes or remeshing a scientific
  input.

### 3. Composites and manufacturing

Composite development proceeds through reusable mechanics:

1. stable orientation, ply, thickness, orthotropic material, laminate, and
   failure-quantity semantics;
2. a public fibrous-shell Step with patch, locking, boundary-moment, state, and
   restart evidence;
3. contact, friction, inter-ply slip, forming controls, and per-layer output;
4. verified mapping of forming orientation, thickness, and defect state into a
   structural model.

Paper-specific geometries and conclusions remain outside the core.

### 4. Engineering workflow

- Keep one Step and one result lifecycle across procedures.
- Improve run comparison, artifact opening, imported regions/surfaces/sets,
  quality diagnostics, and selected Abaqus migration.
- Preserve project data independently of replaceable runtimes.
- Make long solves observable through bounded progress, stability and energy
  diagnostics, and restartable checkpoints.
- Keep concise human output and complete structured output as separate views of
  the same scientific record.

### 5. AI, data, and extensions

- Keep the core free of PyTorch and model-specific neural terminology.
- Let external providers own runtimes, devices, weights, and framework details
  while AgentFEM owns scientific conventions, state transactions, evidence,
  and failure semantics.
- Make campaigns resumable, deterministic, auditable, and independent of the
  physics definition of one case.
- Treat visualizations as views; retain a stable machine-readable scientific
  result as the data source.
- Keep MCP and other agent entrypoints thin: they operate AgentFEM rather than
  becoming another solver.

## Promotion discipline

Every capability moves through explicit maturity levels:

```text
contract -> experimental -> verified -> validated
```

- **contract:** syntax, ownership, and failure behavior are defined;
- **experimental:** a bounded implementation and regression evidence exist;
- **verified:** analytical, manufactured, or independent numerical benchmarks
  demonstrate correctness within a stated applicability domain;
- **validated:** suitable experimental or field evidence supports the declared
  physical use.

Performance evidence, a successful example, or a comparison with AgentFEM's
own earlier output cannot raise maturity by itself. Unsupported geometry,
missing evidence, incompatible constraints, and stale checkpoints fail closed.

## Later, after the foundation

The following remain valuable but must not destabilize the current spine:

- broader contact, fracture growth, phase-field, and multiphysics families;
- adaptive refinement and explicit, auditable mesh repair;
- richer CAD and assembly ingestion;
- a second finite-element backend as an architectural pressure test;
- larger graphical workflows built on the same Model/Step/Result contracts.

AgentFEM will not rewrite element tabulation, quadrature, distributed assembly,
or linear algebra merely to appear independent of FEniCSx. Its moat is the
stable scientific boundary that tells humans and agents what should be
computed, how it is advanced, what state is owned, and why the result can be
trusted.
