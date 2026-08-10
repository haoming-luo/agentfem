# AgentFEM 0.2.0a2

**Channel:** public alpha  
**Released:** 10 August 2026

This release closes the loop between a readable finite-element model and a
repeatable installed-software workflow. The central release criterion is no
longer merely that the repository test suite passes: the exact wheel intended
for users must create projects, execute representative analyses, publish
structured results, and verify their evidence without importing the checkout.

## What becomes substantially stronger

- **One public analysis lifecycle.** Static, thermal, dynamic, nonlinear,
  inelastic, and campaign examples converge on `model.step(...)` and
  `SimulationResult` rather than accumulating case-specific top-level APIs.
- **Engineering mesh semantics.** Imported Abaqus sets, equations, C3D10H
  identity, mixed finite-strain dispatch, cohesive surfaces, and source
  provenance remain available above the converted numerical mesh.
- **Results that remain useful after the solve.** Fields, histories,
  reactions, energies, probes, checkpoints, verification decisions, and
  artifact-integrity seals share one result contract.
- **Simulation-to-learning without a second data pipeline.** Accepted campaign
  results can become a scientific dataset, feed built-in or user-provided
  models, be checked against an applicability domain, and fall back to FEM.
- **A credible experimental fracture route.** Finite-strain explicit
  dynamics, cohesive state, energy accounting, MPI interface assembly, and
  V0-V4 numerical guardrails are integrated without presenting arbitrary-path
  dynamic fracture as a finished general solver.

## Installation experience exercised by the release gate

The wheel must create and complete all bundled project templates:

```text
agentfem init -> agentfem upgrade -> agentfem check
              -> agentfem run -> agentfem inspect -> agentfem verify
```

The same immutable wheel then runs the flagship static, heat, wave, C3D10H,
J2, creep, and simulation-to-learning workflows. The gate compares every
runtime file in the installed distribution with the built wheel, so an older
same-version installation cannot substitute for the candidate.

## Capability maturity

| Capability | Maturity | Release evidence |
| --- | --- | --- |
| Linear static solid | Release | Golden result, serial and MPI smoke |
| Transient heat | Release | Golden result, lifecycle and checkpoint tests |
| Explicit waves | Engineering | workflow smoke, ABC and dynamics tests |
| C3D10H mixed periodic hyperelasticity | Engineering | import, formulation, equation, and homogenization tests |
| J2 plasticity and implicit creep | Engineering | path, cutback, rollback, history, and restart tests |
| Simulation to learning | Engineering | dataset acceptance, validation, guard, and FEM fallback |
| Dynamic cohesive fracture | Experimental | V0-V4 guardrails, energy and distributed-interface tests |

The machine-readable counterpart is
[`release/0.2.0a2.json`](https://github.com/haoming-luo/agentfem/blob/main/release/0.2.0a2.json).
It is checked by the same release gate as the distributions.

## Upgrade notes

- Existing scientific scripts continue to work where compatibility aliases
  remain, but new and migrated examples should use `model.step(...)`.
- Run `agentfem upgrade --project PATH --json` before editing an older project.
  It reports semantic changes without rewriting scientific Python silently.
- Gmsh remains optional. Abaqus/meshio interoperability is enabled through
  `agentfem[mesh-formats]`; visualization dependencies remain separate.
- Linux is the release CI platform. macOS is supported through a compatible
  conda-forge FEniCSx environment; WSL2 remains the recommended Windows route.
