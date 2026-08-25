# Changelog

AgentFEM records user-visible changes here. Scientific maturity remains
capability-specific: a package release does not silently promote an
experimental formulation to a validated one.

## [Unreleased]

### Added

- Add `agentfem migrate-abaqus` and a fail-closed migration project containing
  the complete recursive source graph, scope-aware Part/Instance/ELSET
  relationships, element formulation identity, section assignments, material
  review candidates, preserved Step/load/output assets, and human plus machine
  migration reports.
- Add explicit migration findings for missing material properties,
  topology-only element declarations, incomplete include graphs, composite
  sections, and Abaqus user-material assets that still require source/ABI and
  constitutive verification.

### Changed

- Bind platform, companion-provider, and fresh-agent promotion evidence to the
  exact AgentFEM candidate instead of accepting stale records from an older
  version or commit; add an immutable fresh-agent trial bundle and transcript
  review contract.

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

[Unreleased]: https://github.com/haoming-luo/agentfem/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/haoming-luo/agentfem/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/haoming-luo/agentfem/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/haoming-luo/agentfem/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/haoming-luo/agentfem/compare/v0.2.0a2...v0.2.0
[0.2.0a2]: https://github.com/haoming-luo/agentfem/compare/v0.2.0a1...v0.2.0a2
[0.2.0a1]: https://github.com/haoming-luo/agentfem/releases/tag/v0.2.0a1
