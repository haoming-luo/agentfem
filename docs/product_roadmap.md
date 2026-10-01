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

Before 0.4.0, AgentFEM must keep these boundaries acyclic, preserve one
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
owned by companion or third-party packages.

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

The first two executable slices of gate 3 are now present: built-in and custom
time inputs retain typed RHS/operator/state/output effects, linear implicit
dynamics uses those effects to select safe operator reuse or refresh, and
transient checkpoint schema v5 binds the complete time-input identity before
authorizing restart. Promotion still requires the same invalidation contract
across the remaining transient and nonlinear Procedures.

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
2. close the remaining finite-strain J2 external-beam increment-convergence
   gate without relaxing its fixed tolerance: complete the full refined load
   path through the sharp yield transition, regenerate the content-bound
   promotion report, and only then advance RVE mixed-MPI, follower-load, and
   prescribed-work promotion. The spectral analytical tangent, independent
   numerical oracle, collective inelastic-increment gate, rollback/cutback,
   and per-increment evidence are already implemented and tested;
3. extend the closed bounded MPC/weak/contact evidence contract only through
   independently verified provider routes; general contact remains a separate
   scientific promotion rather than a foundation blocker;
4. finish portable integration-point output and checkpoint identity across MPI
   partitions;
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
