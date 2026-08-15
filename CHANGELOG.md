# Changelog

AgentFEM records user-visible changes here. Scientific maturity remains
capability-specific: a package release does not silently promote an
experimental formulation to a validated one.

## [Unreleased]

No user-visible changes have been recorded after 0.2.1.

## [0.2.1] - 2026-08-15

### Added

- Add progressive `core`, `advanced`, and `expert` public API discovery to
  Python, CLI capability JSON, and machine-readable documentation.
- Add model-owned execution context so every built-in provider can retain its
  target, material, model, and declared output product.

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

### Fixed

- Make declarative output finalization idempotent during 0.2.x migration.
- Make the generated API reference consume progressive module declarations
  without importing the FEniCSx runtime.
- Make MathJax rendering deterministic across direct loads and MkDocs instant
  navigation without requiring a browser refresh.

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

[Unreleased]: https://github.com/haoming-luo/agentfem/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/haoming-luo/agentfem/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/haoming-luo/agentfem/compare/v0.2.0a2...v0.2.0
[0.2.0a2]: https://github.com/haoming-luo/agentfem/compare/v0.2.0a1...v0.2.0a2
[0.2.0a1]: https://github.com/haoming-luo/agentfem/releases/tag/v0.2.0a1
