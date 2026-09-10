# Changelog

AgentFEM records user-visible changes here. Scientific maturity remains
capability-specific: a package release does not silently promote an
experimental formulation to a validated one.

## [Unreleased]

## [0.3.3] - 2026-09-10

### Added

- Add a provider-neutral direct harmonic system over inspectable `K`, `M`, `C`,
  `K_loss`, and `F` operators. One prepared real-block problem now supports
  canonical frequency sweeps, named complex responses, forward/reverse
  execution and common `SimulationResult` histories without retaining every
  frequency field.
- Add atomic scalar-ledger checkpoint/restart for direct harmonic sweeps. The
  ledger binds the model assets, operator system, solver, response definitions,
  frequency axis and portable field identity; it can resume across execution
  order and MPI partitions while explicitly recording that full fields are not
  stored. Bounded progress events, status files and checkpoint retention use
  the shared execution lifecycle.
- Add reusable L2 projection preparation for repeated recovery of a live field.
  The projection space, mass matrix and KSP are assembled once while only the
  right-hand side changes, reducing Test 5H serial runtime and making long
  transient or frequency-domain recovery a first-class result operation.
- Add an automated external single-mesh comparison with NAFEMS R0016 Test 5H.
  The public 50-point forced-vibration sweep checks peak frequency, midspan
  displacement and explicitly recovered longitudinal stress, while independent
  residual, energy and loaded-area evidence and a machine-readable benchmark
  card retain the source, discretization and extraction boundary. The 1%, 2%
  and 3% acceptance limits are identified as AgentFEM gates rather than NAFEMS
  tolerances.
- Add a separate three-level Test 5H spatial-refinement stability certificate.
  The nonuniform complete-Q2 sequence requires decreasing successive changes
  and less than 1% change in the final pair for peak frequency, displacement
  and declared recovered stress. It deliberately reports neither observed
  order nor a continuum-error estimate, and does not relabel the nonmonotonic
  external-reference errors as convergence toward the published values.
- Add installed-wheel acceptance for the complete 50-point Test 5H sweep in
  serial and on two MPI ranks, plus scalar-ledger checkpoint transfers from
  one to two ranks and from two to one rank. Checkpoints bind executable
  operator, constraint, response, solver and software identity; they preserve
  accepted scalar evidence and explicitly do not restore finite-element fields.

### Changed

- Direct harmonic systems may select either nested field-split iteration or a
  monolithic direct/iterative policy. Evidence names the layout and PETSc matrix
  type; SPD-only Cholesky, ICC and CG policies are rejected because the dynamic
  real-block operator is not generally positive definite or symmetric.
- Prepared harmonic problems now freeze their actual lowering contract. Changes
  to operators, live coefficients, constraints, load phase, solution identity,
  material configuration or solver policy fail closed instead of mixing a
  cached matrix with new result metadata. NaN/Inf in the solution, residual or
  energy ledger is likewise rejected before evidence is published.
- The sweep maximum-displacement statistic now evaluates the exact maximum
  vector norm over one physical cycle. It no longer overestimates circular or
  otherwise non-proportional component motion by using the complex coefficient
  norm.
- Reusable harmonic and exact-MPC problems now expose one deterministic,
  idempotent resource lifecycle. Distributed PETSc allocations are released
  collectively by their owner instead of later through rank-local Python
  cyclic garbage collection; frozen results, summaries, public fields and
  provider-owned MPC graphs remain readable after close.

### Boundaries

- The provider-neutral small-strain structural direct-harmonic workflow is
  engineering maturity for the declared linear `K/M/C/K_loss/F` formulation,
  supported strong constraints and bounded frequency sweeps.
- Generalized-Maxwell harmonic response remains experimental. Test 5H exercises
  an elastic Rayleigh-damped structural system; it is not external validation of
  the generalized-Maxwell constitutive spectrum.
- Test 5H is an external numerical comparison, not experimental validation.
  Its acceptance limits are AgentFEM release gates, not NAFEMS tolerances, and
  its three-level evidence is observable stability rather than a GCI or
  continuum-error certificate.
- Complex modes, prestressed small-on-large response and nonlinear harmonic
  balance remain outside this release.

## [0.3.2] - 2026-09-09

### Added

- Add a public structural modal Study and SLEPc-backed generalized Hermitian
  Step. Strongly constrained degrees of freedom are removed before the solve;
  natural frequencies, residual evidence, and mass-normalized mode fields use
  the common `SimulationResult` lifecycle in serial and MPI.
- Add solver-independent FFT, FRF, free-decay damping, and modal-superposition
  tools that consume arrays or structured histories without embedding a
  beam-specific workflow.
- Add standard-linear-solid and generalized-Maxwell/Prony material-point
  spectra with exact state updates and dissipation, WLF/Arrhenius shifts, and
  a transparent fixed-spectrum fit. A restartable material-history procedure
  now returns accepted stress/internal-state histories and an independently
  integrated work--stored-energy--dissipation ledger through `SimulationResult`.
- Add a three-dimensional quasi-static generalized-Maxwell finite-element Step
  with committed quadrature state, regional materials, temperature shift,
  standard fields and work--storage--dissipation evidence. Prescribed uniform
  and nonuniform clocks are joined by error-controlled physical-time
  incrementation: one full step is checked against two half steps, rejected
  attempts roll back nodal and quadrature state atomically, and serial restart
  preserves the accepted path and next proposal. The adaptive decision is
  collective under MPI, and portable checkpoints are verified across one- and
  two-rank partitions; arbitrary production-scale repartition remains outside
  the release claim.
- Add a direct three-dimensional generalized-Maxwell harmonic Step. It lowers
  complex bulk/shear spectra and optional inertia to an exact real PETSc block
  system, publishes real/imaginary/amplitude/phase fields, and records stored,
  kinetic, dissipated and externally supplied cycle-energy evidence. A
  two-rank regression now verifies rank-identical phasors and energy evidence
  plus collective XDMF/ParaView output carrying all four fields. Successful
  solves also verify the unpreconditioned assembled global and per-block
  `A*x-b` residual instead of trusting only the KSP-reported norm.
- Add `ScientificFieldDataset`, a portable multi-field collection for neural
  operators and other field-learning workflows. It preserves channel-first
  shapes, units, geometry policies, masks, parameters, case evidence,
  deterministic train/validation/test splits, lossless NPZ storage and a
  content fingerprint without adding a machine-learning dependency to core.
- Tighten `FieldEncoding` and `NeuralOperatorSpec` validation, including
  normalized FNO/TFNO names, unique field and parameter identities, explicit
  boundary encoding and structured-grid compatibility.
- Add a versioned 0.3.2 release contract and fail packaging tests when commits
  beyond a published tag continue to report that tag's stable version.
- Add a machine-readable multi-source runtime distribution contract and an
  optional S3-compatible mainland China publication path. GitHub remains the
  canonical release identity; every regional download must preserve the exact
  accepted filename, byte count and SHA-256.
- Add a distinct assembled DCB finite-element benchmark provider using Q1 bulk
  elasticity, a paired zero-thickness interface and mesh-aligned precracks.
  It produces structural compliance/G evidence without relabelling the
  existing analytical beam oracle; ENF/MMB execution and convergence remain
  explicit promotion gates.
- Add a scoped DCB compliance convergence study and certificate. It requires
  three or more assembled mesh levels, decreasing reference and inter-level
  errors, bounded equilibrium residuals and stable evidence identities, while
  explicitly excluding cohesive evolution and crack-propagation validation.
- Add a distinct irreversible DCB propagation route with converged-state
  commit/rollback, automatic displacement-increment cutback, reaction and
  crack-front histories, and a bulk/cohesive/work energy ledger. A three-level
  nondimensional assembled regression now exercises the propagation
  certificate; source-identified DCB/ENF/MMB curves remain separate external
  promotion gates.
- Add a separate assembled ENF provider with three-point-bending supports,
  displacement control, a full vector cohesive tie, aligned precracks,
  reaction-derived compliance and pure Mode-II energy-release output. Its
  three-level certificate reaches the declared beam oracle without implying
  Mode-II propagation or external material-curve validation.

- Add committed finite-strain J2 `PDENER` state, quadrature output, volume
  quantity and homogenized history. The rate-independent linear-hardening law
  now separates recoverable `ELENER + HARDENER` from irrecoverable initial-
  yield work across trial, commit, rollback and restart.
- Add a solver-independent three-or-more-level DCB/ENF/MMB convergence
  certificate that checks reference error, last-refinement change, mode
  partition, cohesive-zone resolution and artificial dissipation. Malformed
  or internally inconsistent energy-release curves now fail before scoring.
- Add provider-owned MPC/weak/contact dual evidence with actual generalized
  force, accepted coordinate and physical resultant values. Static force and
  work ledgers now consume these values and remain unavailable when any channel
  is partial, duplicate or unmatched.
- Make exact affine periodic constraints the first native dual provider. The
  macro reaction comes from full-residual virtual work, accepted path work is
  checkpoint-safe, and periodic J2 tests independently match both
  \(V\mathbf{P}:\mathrm d\mathbf{F}/\mathrm d\lambda\) and Hill--Mandel work.
  Non-proportional path kinks retain distinct incoming and outgoing duals.
- Add MPI-aware construction diagnostics to exact rectangular periodic MPCs.
  Reports distinguish owned from ghost slaves and reject missing, multiply
  matched, or non-unit periodic relations before solve, while continuing to
  mark reaction distributions and macroscopic work unavailable until a real
  provider dual and work coordinate exist.
- Add a reusable prepared exact-MPC linear solve lifecycle with ordinary
  `LinearSolverOptions`, convergence evidence, safe augmented-to-public field
  transfer, and serial/MPI behavior. The periodic PDE benchmark now consumes
  this shared solver boundary instead of maintaining private lowering code.
  Evidence distinguishes reused compilation, allocation, and KSP state from
  matrix values that the current upstream backend still reassembles per solve.
  A distributed preflight also rejects Dirichlet conditions that were added
  after MPC slave selection and overlap owned slave DOFs.
- Lower one public `RectangularPeriodicMPC` directly through ordinary linear
  static, steady-heat, and constant-property transient-heat `model.step(...)`
  workflows. Strong Dirichlet data
  and exact elimination now follow separate assembly contracts; multiple or
  unknown providers fail before assembly, and reaction/work evidence remains
  unavailable until the active MPC provider supplies a physical dual. Linear
  transient runs prepare the operator lifecycle once per run instead of
  rebuilding it at every accepted time increment.
- Move the shared generalized work and cyclic energy ledger out of the fatigue
  domain module into the Result/Verification ownership layer while preserving
  the existing public compatibility imports.
- Add one runtime-independent `agentfem workspace` custody contract for the
  Complete WSL2 Runtime and ordinary Mamba installations, including verified
  migration to durable Windows storage and machine-readable safety status.
- Add a fail-closed Windows runtime remover that protects the project
  workspace and exports a complete recovery snapshot before unregistering the
  AgentFEM distribution.
- Add the NAFEMS LE10 curved three-dimensional elasticity benchmark with
  external stress evidence, mesh convergence and serial/two-rank equivalence.
- Add density-aware gravity and reference body-force lowering to ordinary and
  migrated Abaqus workflows, including the corresponding reaction and energy
  accounting paths.

### Changed

- Harden structural modal analysis with exact integer validation, deterministic
  serial/MPI mode orientation, operator-symmetry preflight, mass-orthogonality
  and stiffness-diagonalization evidence, and physically neutral mode-shape
  output semantics. Repeated eigenvalues now carry explicit cluster membership
  and invariant-subspace comparison, while incomplete truncations are marked
  rather than assigned false mode identity. A three-mesh slender-cantilever
  sequence adds analytical and discretization-convergence evidence. Complete
  user-supplied `K/M` systems remain a valid material-free expert path.
- Reject non-finite generic and generalized-Maxwell state at snapshot,
  restore, trial and commit boundaries; use cancellation-safe exact branch
  updates for very small increments and fail closed on singular spectral
  requests.
- Pin the documented conda installation route to the release-tested Python
  3.11 and DOLFINx 0.11 stack, publish a no-global-configuration TUNA mirror
  route for mainland China, and expose both commands to AI agents through the
  machine-readable documentation manifest.
- Keep Windows projects, results, and checkpoints in
  `Documents\AgentFEMProjects` by default while preserving the familiar
  `~/AgentFEMProjects` path inside WSL.
- Attach resolved project and output storage custody to project summaries and
  execution evidence, warning when a path remains owned by a replaceable WSL
  distribution.
- Make run directories and CLI summaries readable to people while retaining
  stable machine identities, latest-run lookup and structured JSON for agents.
- Add transactional replacement upgrades for the Complete Windows Runtime and
  independent global/China reliability routing under the same privacy schema.

### Boundaries

- Generalized-Maxwell harmonic analysis remains experimental until an
  independent structural frequency-response benchmark and robust
  near-resonance solver evidence are complete.
- DCB/ENF evidence verifies the declared assembled compliance and propagation
  workflows; it does not claim arbitrary delamination paths, general contact,
  or universal cohesive parameters.
- Windows remains supported through the WSL2 Complete Runtime Preview rather
  than a native PETSc solver stack.

## [0.3.1] - 2026-08-31

### Added

- Add a default-on, strictly whitelisted anonymous reliability channel with a
  bounded offline queue, coarse success sampling, message-free failure
  fingerprints, and a permanent `agentfem telemetry off` control.
- Add local diagnosis, private Codex/AI support tasks, sanitized feedback
  archives, repeated-failure escalation, and explicit authenticated GitHub
  issue submission without placing scientific models or results on the basic
  reporting path.
- Add an aggregate-only reference collector that rejects undeclared fields and
  retains no raw event, user identity, source IP, exception text, or scientific
  data.
- Add automated Complete Runtime Preview assembly for Windows through WSL2 and
  Apple Silicon macOS, including Gmsh redistribution evidence and installed
  runtime acceptance.
- Deploy the project-owned aggregate-only reliability collector and publish
  its reviewed HTTPS endpoint with the package.

### Changed

- Publish one compact Windows offline ZIP rather than duplicating its large
  `.wsl` image as two GitHub assets.
- Keep macOS runtime artifacts explicitly named `unsigned-preview`; 0.3.1 does
  not require an Apple Developer account or represent the package as Apple
  signed or notarized.

### Boundaries

- Online reliability delivery uses the project-owned Cloudflare endpoint. It
  remains fail-open, bounded, schema-whitelisted and permanently user-
  controllable with `agentfem telemetry off`.
- The Windows image is a public Preview until it passes acceptance on a real
  WSL2 host; container acceptance validates the embedded Linux runtime but
  does not impersonate Windows integration evidence.

## [0.3.0] - 2026-08-30

### Platform milestone

- Promote the public scientific language, provider-owned lowering, State and
  Result lifecycle, capability evidence, installed Linux/macOS routes,
  companion-provider seam, and fresh-agent workflow after all seven G1--G7
  gates passed against immutable package evidence.
- Establish `Study -> Model -> scientific assets -> model.step(...) ->
  SimulationResult` as the stable 0.3 public grammar while retaining reviewed
  0.2 compatibility delegates.
- Demonstrate that a genuinely fresh Codex session can install the candidate
  wheel, inspect its environment, construct and refine a plane-strain model,
  verify the structured result, and explain its scientific limits without
  project-specific history or human correction.

### Boundaries

- The 0.3 milestone stabilizes the scientific middle layer; it does not claim
  universal finite-element coverage, native Windows support, general contact,
  or universally validated nonlinear material and fracture formulations.

## [0.2.6 candidate] - 2026-08-30

### Added

- Add a machine-readable seven-boundary ownership contract covering Model,
  Constitutive, State, Operator, Procedure, Backend, and Result/Verification,
  together with dependency regression tests and an explicit restart/rollback
  State protocol.
- Add solver-neutral material-point schemas and validated user-material
  update contracts, preserving a path for reviewed external constitutive
  assets without moving executable foreign code into the engineering model.
- Add public finite-strain logarithmic J2 workflows for strong kinematics and
  affine-periodic RVE loading, including quadrature transactions, consistent
  tangents, cutback, MPI-portable restart, regional materials, void and
  multi-void realizations, non-proportional paths, and frozen Golden evidence.
- Add reviewed Abaqus migration and native-lowering gates that preserve the
  recursive source graph, Part/Instance scope, element identity, material and
  section assignments, and fail closed on unsupported semantics.
- Add framework-neutral neural-field contracts and the extension seam used by
  the separately installed AgentFEM-Learning provider.
- Add mixed-mode LEFM interaction-integral and DOLFINx field-extraction
  evidence with explicit path and equilibrium diagnostics.
- Add executable G1--G7 platform-promotion gates bound to exact source commits,
  wheel hashes, installed platforms, companion providers, and fresh-agent
  behavioral evidence.

### Changed

- Keep `Model` as the engineering facade while providers, builders, operators,
  State, the FEniCSx backend, and Result/Verification own their corresponding
  numerical and evidence responsibilities.
- Detect the MPI launcher family that matches the active `mpi4py` runtime
  instead of assuming the first system `mpirun` is compatible.
- Treat Linux and macOS installed-wheel evidence as the 0.3 platform gate;
  WSL2 remains the recommended Windows route and a separately tracked
  acceptance target.
- Publish the AgentFEM technical report as the project statement and durable
  software reference.

### Boundaries

- Finite-strain J2 remains experimental until an independent external
  structural benchmark passes; the current RVE evidence verifies the declared
  workflows rather than universal finite-strain plasticity.
- A fresh-agent trial and a companion-provider run remain independent evidence
  records; deterministic CI cannot impersonate either one.

## [0.2.5] - 2026-08-24

### Added

- Add one constraint-capability contract and preflight compatibility checks so
  unsupported analysis, procedure, or MPI combinations fail before assembly
  with stable diagnostics.
- Add a public rectangular `dolfinx_mpc` construction boundary shared by
  ordinary workflows and PDEAgent-Bench, plus periodic-pair diagnostics.
- Add installed-wheel platform-evidence aggregation and a separate fresh-agent
  trial recorder; deterministic CLI smoke evidence can no longer impersonate
  an unfamiliar AI agent.

### Changed

- Store two-dimensional displacement and other physical vectors as
  three-component visualization arrays on XYZ output geometry, with a zero
  out-of-plane component, explicit `U`/`Displacement` aliases, and warp
  metadata. The finite-element unknown remains two-dimensional.
- Preserve the one-grid XDMF/HDF5 result layout while making new 2D results
  directly usable by ParaView and PyVista Warp By Vector.
- Make `agentfem doctor` distinguish imported runtime version, installed
  distribution version, source shadowing, and MPI launcher mismatch.
- Record model-preflight failures, stable validation codes, stage and traceback
  in the common execution evidence instead of exposing backend attribute
  errors.

## [0.2.4] - 2026-08-24

### Added

- Add an experimental global Chaboche combined-hardening material with
  exponential isotropic saturation, multiple Armstrong--Frederick
  backstresses, backward-Euler quadrature integration, a fully discrete
  tangent, standard `ALPHA` output, cyclic amplitudes, rollback, cutback and
  restart through the ordinary `model.step(...)` workflow.
- Add source Study/procedure and accepted-time transfer metadata to
  `FieldHistory`, preserved through compact and MPI-portable archives.
- Add a layered sequential thermal/mechanical energy ledger that keeps heat
  balance and mechanical work residuals separate and explicitly refuses a
  monolithic conservation claim.
- Add engineering creep--fatigue V1 extraction from named scalar stress and
  temperature histories with declared dwell intervals, explicit reducers,
  user-owned rupture relations, retained sources, and addressable failure for
  missing or out-of-range evidence.
- Add formula-bearing scientific cards, official public model-definition
  references, focused material/global/restart tests, and a runnable 3D cyclic
  Chaboche example.

### Changed

- Preserve the released isotropic-J2 v4 serial checkpoint schema while using
  the extensible v6 schema for multi-backstress Chaboche state.
- Report Chaboche energy only as a state partition until dynamic-recovery
  dissipation and a structure-level cyclic energy benchmark close the full
  ledger.

## [0.2.3] - 2026-08-24

### Added

- Add an endpoint creep-rate time-integration error control for the global
  implicit creep Step. It remains independent of Newton convergence and the
  maximum accepted CEEQ increment, and rejected attempts restore displacement,
  quadrature state, stress, tangent, loading, and temperature atomically.
- Add a native small-strain axisymmetric solid formulation with shared
  `(r,z)` kinematics, full `(r,theta,z)` tensors, `2*pi*r` operator/load/result
  lowering, public total-force semantics, standard fields, linear elasticity,
  J2 plasticity, and implicit power-law creep. The NAFEMS R0027 Test 7 route
  now uses four Q2 radial cells and keeps radial, hoop, and axial stress errors
  below `0.03%`; the declared `0.5%` gate is an AgentFEM contract rather than
  an official NAFEMS tolerance.
- Add machine-readable lifecycle and replacement metadata for the Model
  vocabulary, expose it through Python and capability JSON, and report
  material/procedure-specific compatibility calls through `agentfem upgrade`
  without silently rewriting scientific Python.

- Add one immutable, inspectable Step execution-policy snapshot for declared
  solver, output, history, progress, and checkpoint controls; transient history
  requests may now be declared once on `model.step(...)` and are consumed by
  the common result lifecycle.
- Add one dependency-free product-language contract shared by package
  discovery, the Model facade, CLI capabilities, generated documentation,
  Agent Skill guidance, IDEs, and future GUI clients, with drift tests across
  every consumer.
- Freeze and index the current 558/645 PDEAgent-Bench snapshot with explicit
  micro/macro/minimum-family statistics, dimensional stratification, execution
  scope, upstream discussion, stronger manifest/report consistency checks, and
  repository-first evidence tools immune to an older installed wheel.

- Add a commit-pinned PDEAgent-Bench integration with a strict public-case
  schema, safe expression lowering, fourteen geometry specifications,
  Poisson/heat/linear-elasticity/Helmholtz/convection--diffusion/
  reaction--diffusion/wave solvers, strict grid sampling, and a failure-aware
  official-summary report.
- Add public scalar advection, SUPG, intrinsic-time-scale, and named reaction
  operators with formula-bearing knowledge assets and independent regression
  tests.
- Add accepted-physical-time `FieldHistory` capture for transient scalar
  fields, with explicit interpolation/range policy, compact persistence,
  scientific content identity, and coordinate-keyed portability across MPI
  partitions and rank counts.
- Let global Arrhenius creep consume a temperature history at attempted
  increment endpoints; cutback and restart restore the accepted temperature
  together with displacement and quadrature state.
- Add inspectable temperature-property tables and a temperature-dependent
  isotropic thermoelastic property record for sequential stiffness and thermal
  expansion.
- Add a three-dimensional heat-to-creep component contract in which accepted
  transient temperature states directly drive global Arrhenius creep on the
  same physical clock.
- Add conservative nonlinear transient heat transfer for tabulated
  conductivity and specific heat. The public `model.step(...)` route lowers
  automatically to a PETSc SNES solve using a sensible-enthalpy increment,
  with shared progress, heat ledger, rollback, checkpoint/restart, and MPI
  behavior.

### Changed

- Let global creep histories inherit the model's declared consistent time
  unit; an undeclared unit system no longer labels arbitrary model time as
  seconds.
- Print the scientific rejection reason when an automatic nonlinear increment
  is cut back.
- Move finite-strain and mixed hyperelasticity, explicit dynamics,
  finite-strain explicit dynamics, and implicit structural dynamics behind
  provider-owned scientific builders; their historical Model methods remain
  thin 0.2.x compatibility delegates.
- Separate the public direction roadmap from private release gates, technical
  debt, risks, benchmark tactics, and execution sequencing.

- Move linear-static/steady-conduction, J2, and implicit-creep scientific
  construction behind internal provider-owned builders; historical `Model`
  methods remain thin 0.2.x compatibility delegates.
- Make top-level workflow modules and heavy result-output modules lazy while
  preserving the public import surface, substantially reducing startup cost
  for CLI, agent, campaign-worker, and short scientific jobs.
- Lower case-varying known formulas to finite-element fields before assembly
  so repeated scientific inputs reuse compiled weak forms instead of invoking
  formula-specific C++ compilation.
- Record captured thermal histories in transient Step summaries and publish
  the heat-to-creep transfer contract through the manual, Agent Skill,
  roadmap, and scientific knowledge catalog.
- Reject silent history extrapolation and unsafe MPI NPZ history writes; field
  histories use the same physical-DOF identity as portable transient state.
- Keep linear and state-dependent heat steps on the same convection-ledger
  convention and reject ambiguous combinations of automatic tabulated
  properties with user-supplied `C` or `K` operators.

## [0.2.2] - 2026-08-19

### Added

- Add a framework-neutral neural-field executor boundary: user-owned PyTorch,
  JAX, DeepXDE, or laboratory solvers can consume
  `NeuralFieldExecutionRequest` through `model.step(..., executor=...)` and
  return the common `SimulationResult` without an official learning package.
- Add `agentfem.learning` as the public scientific-learning entry point while
  preserving `agentfem.surrogates` throughout the 0.2.x compatibility line.
- Add declarative neural-field contracts for residual, variational-energy,
  data, and constraint objectives; boundary/initial/interface conditions;
  sampling; neural representations; and inverse parameters. These contracts
  cover future PINN, VPINN, Deep-Ritz/DEM, XDEM, and related providers without
  claiming that those external trainers are bundled in the core package.

### Changed

- Clarify the open extension boundary: third-party learning engines belong in
  optional provider distributions, while confidential customer data,
  calibration assets, and domain products remain in independent private
  packages rather than private branches of the open core.
- Add a README-first agent route that lets a coding agent discover the
  environment, health check, project templates, execution lifecycle, and
  verification contract before a new user has to reproduce the manual setup.
- Establish AgentFEM-Learning as the optional official companion for maintained
  scientific-learning providers while keeping user-owned models directly
  executable through AgentFEM core.

## [0.2.1] - 2026-08-17

### Added

- Add progressive `core`, `advanced`, and `expert` public API discovery to
  Python, CLI capability JSON, and machine-readable documentation.
- Add model-owned execution context so every built-in provider can retain its
  target, material, model, and declared output product.
- Add inspectable Step option contracts with accepted/required keyword names
  to every built-in provider and expose them through capability JSON.
- Add progressive `core`, `advanced`, and `compatibility` discovery for the
  model-owned method vocabulary.
- Ship the AgentFEM coding-agent guidance as a standards-compatible,
  progressively disclosed Skill with versioned interface metadata.

### Changed

- Unify static, nonlinear, heat-transfer, Standard dynamics, Explicit
  dynamics, J2, and creep completion around `solve_result()` and
  `SimulationResult`.
- Allow output to be declared once through
  `model.step(..., output="results.xdmf").solve_result()` while preserving
  `solve_result(output=...)` and expert low-level routes.
- Let field constructors and models consume imported mesh facades directly;
  vector constraints also accept engineering axis names.
- Migrate installed templates and release-facing examples to the converged
  model-owned Step and result workflow.
- Reject misspelled or procedure-inappropriate `model.step(...)` options before
  form assembly, with stable issue codes and repair suggestions.
- Make bundled cases consistently use physical Study factories such as
  `static_solid`, `transient_heat_transfer`, and `dynamic_solid`.

### Fixed

- Make declarative output finalization idempotent during 0.2.x migration.
- Make the generated API reference consume progressive module declarations
  without importing the FEniCSx runtime.
- Make MathJax rendering deterministic across direct loads and MkDocs instant
  navigation without requiring a browser refresh; the pinned runtime and fonts
  are served with the manual rather than fetched from a third-party CDN.
- Preserve an explicitly selected Explicit/central-difference procedure during
  internal capability checks even when the Study prefers Newmark.

## [0.2.0] - 2026-08-13

AgentFEM 0.2.0 is the first non-prerelease distribution. Scientific maturity
remains capability-specific: experimental formulations are not promoted merely
because the package version is stable.

### Added

- Add a solver-integrated cyclic cohesive fatigue lifecycle with exact cycle
  coordinates, adaptive cycle blocks, rollback, restart and named interfaces.
- Add an explicit proportional mixed-mode cyclic law with complete local jump
  extrema, cohesive GI/GII energy ranges, BK/power interaction, material-aware
  fields and physical-facet restart across MPI rank counts.
- Add explicitly ordered closed non-proportional jump paths, segment-resolved
  mixed-mode fatigue driving, path evidence, global dispatch and atomic
  rollback/restart.
- Re-equilibrate and compare every degraded ordered-path station before a
  non-proportional cycle block can be committed.
- Add a transactional generalized work--energy ledger for natural loads,
  reference-point force/moment, prescribed motion, MPC/weak/contact channels
  and cycle-jump blocks.
- Add DCB/ENF analytical compliance and structural energy-release oracles plus
  an MMB contract with mandatory mode partition, process-zone resolution and
  numerical-dissipation guardrails.
- Track multiple cracks on one cohesive surface with persistent physical-facet
  identities and explicit birth, merge, split and death events.
- Fit Paris relations strictly as postprocessing evidence from accepted crack
  histories, without prescribing crack advance in the solver.
- Add MPI-portable quadrature state with stable physical identities for J2 and
  creep histories, including cross-rank restart tests and regional materials.
- Add a public thick-cylinder J2 benchmark based on an independently published
  elastoplastic structure, with analytical first-yield pressure and serial/MPI
  equivalence evidence.

### Changed

- Enable the public J2 global Newton route under MPI after the external
  thick-cylinder structure benchmark; global creep MPI remains experimental.
- Rework the README around the public value proposition, cross-platform
  installation, first runnable project, release examples, and extension path.
- Promote the package installation command from prerelease opt-in to ordinary
  `pip install agentfem` while retaining explicit maturity labels per workflow.

### Fixed

- Make quadrature ownership, ghost synchronization, transaction rollback, and
  checkpoint restore deterministic across MPI partition counts.
- Re-equilibrate every degraded station of an ordered mixed-mode fatigue path
  before accepting a cycle block.
- Align the package version, citation metadata, release contract, wheel payload
  checks, and documentation manifest for the immutable 0.2.0 artifacts.

## [0.2.0a2] - 2026-08-10

### Added

- Installed project templates for static solids, steady heat transfer, and
  structural dynamics, with one `check -> run -> inspect -> verify` lifecycle.
- Direct Abaqus C3D10H import and a P2 displacement/DG0 pressure mixed route
  for quasi-incompressible periodic hyperelastic cells.
- Engineering coordinates, mesh-set semantics, cohesive-interface lowering,
  portable cohesive state, and distributed interface assembly.
- Unified histories, field probes, path sampling, reactions, energy evidence,
  checkpoints, provenance seals, and result verification policies.
- Dynamic cohesive-fracture V0-V4 guardrails and Mooney-Rivlin finite-strain
  material support, both kept explicitly experimental where appropriate.
- Campaign-to-dataset-to-surrogate workflows with validation, applicability
  guards, PyTorch adapters, and FEM fallback.
- A scientific reference site, project upgrade preflight, capability reports,
  and machine-readable release scope.

### Changed

- Public examples consistently use `model.step(...)` and the shared
  `SimulationResult` lifecycle.
- Abaqus periodic-cell examples now preserve C3D10H formulation identity and
  express three-dimensional uniaxial-stress macro control directly.
- Release CI now builds distributions once, verifies the exact wheel to be
  uploaded, then publishes the same immutable artifacts. GitHub provenance
  attestation is added when repository visibility supports it.
- Installed-wheel smoke now compares the complete runtime payload and executes
  every bundled project template plus the release-facing workflow set.

### Fixed

- MPI result, checkpoint, and cohesive-state identity across partition counts.
- Documentation navigation, mathematical rendering, mobile layout, and
  scientific-reference organization.
- Stale same-version installations can no longer satisfy the release gate.

## [0.2.0a1] - 2026-08-03

- First public alpha preview of AgentFEM as an AI-native finite-element
  platform with readable study, model, step, result, campaign, and evidence
  contracts.

[Unreleased]: https://github.com/haoming-luo/agentfem/compare/v0.3.3...HEAD
[0.3.3]: https://github.com/haoming-luo/agentfem/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/haoming-luo/agentfem/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/haoming-luo/agentfem/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/haoming-luo/agentfem/compare/v0.2.6...v0.3.0
[0.2.6]: https://github.com/haoming-luo/agentfem/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/haoming-luo/agentfem/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/haoming-luo/agentfem/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/haoming-luo/agentfem/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/haoming-luo/agentfem/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/haoming-luo/agentfem/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/haoming-luo/agentfem/compare/v0.2.0a2...v0.2.0
[0.2.0a2]: https://github.com/haoming-luo/agentfem/compare/v0.2.0a1...v0.2.0a2
[0.2.0a1]: https://github.com/haoming-luo/agentfem/releases/tag/v0.2.0a1
