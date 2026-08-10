# Changelog

AgentFEM records user-visible changes here. Scientific maturity remains
capability-specific: a package release does not silently promote an
experimental formulation to a validated one.

## [Unreleased]

- Continue external benchmark work for inelasticity and dynamic fracture.
- Extend three-dimensional cohesive interfaces and mixed finite-strain solids.

## [0.2.0a2] - Release candidate

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
  uploaded, then attests and publishes the same immutable artifacts.
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

[Unreleased]: https://github.com/haoming-luo/agentfem/compare/v0.2.0a1...HEAD
[0.2.0a2]: https://github.com/haoming-luo/agentfem/compare/v0.2.0a1...v0.2.0a2
[0.2.0a1]: https://github.com/haoming-luo/agentfem/releases/tag/v0.2.0a1
