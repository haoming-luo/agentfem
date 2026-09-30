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
Moving tools publish force--translation and moment--rotation path work; general
surface search, finite sliding, friction, multiple pairs, time-varying natural
loads, and undeclared weak formulations remain outside this gate and fail closed
rather than inheriting its evidence.

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
2. close the remaining finite-strain J2 increment-maximum gate with the new
   analytical tangent and per-increment evidence, then profile the remaining
   PETSc share before advancing RVE mixed-MPI, follower-load, and
   prescribed-work promotion;
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
closure. Prescribed analytical-plane motion and its accepted
force--moment--work contract are now the first executable slice. Next come
the solver-neutral projection contract (now executable for analytical planes,
circles/spheres and infinite cylinders), reviewed tessellated surfaces (the
serial triangle reference projector and fail-closed geometry audit are now
executable, together with a deterministic process-local AABB/BVH accelerator
and per-query work evidence plus stable barycentric facet coordinates),
distributed ownership/search (the stable-ID facet partition and all-gather
correctness reference now reproduce the serial oracle under MPI; scalable
neighborhood routing has started with an oracle-checked two-stage rank-AABB
path, while packed numeric exchange and ghost-aware imported geometry remain),
finite sliding, multiple pairs, explicit dynamics and finally friction. STEP
remains an optional geometry-adapter input that is repaired and tessellated
before the contact core consumes it. Shell/solid-shell forming and self-contact
remain later scientific promotions, not hidden extensions of the bounded
fixed-plane provider.

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
