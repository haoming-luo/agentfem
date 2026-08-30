# AgentFEM 0.3.0

AgentFEM 0.3.0 is the first release built on a stable scientific middle layer
between humans or AI agents and the FEniCSx/PETSc numerical runtime.

## One public workflow

The recommended language is:

```text
Study -> Model -> scientific assets -> model.step(...) -> SimulationResult
```

`Model` remains a readable engineering facade. Constitutive owns material
updates, State owns committed and trial history, Operator owns mathematical
contributions, Procedure owns numerical evolution, Backend owns finite-element
execution, and Result/Verification owns evidence and acceptance. Architecture
tests keep these responsibilities from silently collapsing back into one
numerical god object.

## Evidence-bound execution

The same execution lifecycle now carries solver options, output, history,
progress and checkpoint policy. Structured results retain convergence,
balances, failures, provenance, scientific quantities and artifact integrity.
Capability maturity remains bounded by executable Golden, failure, MPI or
external evidence rather than by the existence of an API name.

## Open extension seam

AgentFEM-Learning demonstrates that a separately installed package can
register neural-field and finite-domain XDEM providers without modifying the
core. The extension is discovered through the same provider boundary and
returns the same `SimulationResult` contract used by deterministic workflows.

## Human and agent acceptance

The release candidate passed clean installed-wheel acceptance on hosted Linux
and macOS. A genuinely fresh Codex session then consumed the immutable wheel,
inspected the runtime, constructed and refined a plane-strain cantilever,
verified force and work balance plus artifact integrity, and explained the
model assumptions without human correction. These are behavioral acceptance
conditions, not deterministic CI substitutes.

## Scientific foundation in this release

The 0.3 foundation includes reviewed Abaqus migration, solver-neutral user
materials, transient and nonlinear State transactions, finite-strain J2 RVE
workflows, mixed-mode LEFM extraction, cohesive and creep foundations,
parameter campaigns, scientific datasets and learning interfaces. Each keeps
its declared maturity and applicability boundary.

## Scope

AgentFEM 0.3.0 does not claim universal element, material, contact or fracture
coverage. Native Windows, general contact, arbitrary-path fracture, monolithic
thermo-mechanical coupling, and automatic PINN or neural-operator training for
arbitrary meshes remain outside the release claim. WSL2 remains the recommended
Windows route and a separately tracked platform target.
