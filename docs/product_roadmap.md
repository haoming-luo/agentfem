# AgentFEM Product Roadmap

## Direction

AgentFEM is building a readable, dependable, and extensible finite-element
platform for humans and AI agents. The near-term goal is not to reproduce every
feature of established commercial systems. It is to make selected engineering
workflows unusually clear, inspectable, reproducible, and easy to extend.

The public Python workflow remains the product:

```text
Study -> Model -> Mesh/Regions -> Fields -> Materials
      -> Loads/Constraints -> Step -> SimulationResult
```

FEniCSx, PETSc, and MPI provide the numerical foundation. AgentFEM adds the
engineering language, reusable scientific operators, workflow lifecycle,
verification evidence, automation, and extension boundary around that
foundation.

## What is usable today

AgentFEM currently provides executable foundations for:

- linear static solids and steady/transient heat transfer;
- central-difference, Newmark, and generalized-alpha structural dynamics;
- finite-strain Neo-Hookean and Mooney--Rivlin workflows;
- small-strain J2 plasticity, an experimental global Chaboche combined-
  hardening route, and implicit power-law creep, including regional MPI state,
  cross-rank-count portable restart, and bounded external Abaqus/NAFEMS
  benchmarks;
- accepted thermal-history transfer through shared E(T)/nu(T)/alpha(T)
  thermoelastic creep properties, plus source-preserving engineering
  creep--fatigue assessment, including named-history dwell extraction with
  project-owned rupture relations;
- fixed-path cohesive fracture, including experimental cyclic and mixed-mode
  routes;
- Abaqus/Gmsh mesh semantics, periodic equations, C3D10/C3D10H workflows, and
  MPI execution;
- common fields, histories, energy records, checkpoint/restart, provenance,
  convergence evidence, and ParaView-oriented output with one recommended
  visualization dataset per saved time in serial and MPI;
- resumable parameter campaigns, scientific datasets, surrogate models, and
  provider-neutral learning interfaces.

Each capability has an explicit maturity and applicability boundary. Query the
installed package with:

```bash
agentfem capabilities --json
```

The executable capability catalog and benchmark registry are authoritative;
the roadmap does not turn an experimental formula into a validated solver.

## Public development tracks

### 1. Trusted mechanics

Deepen the finite-element core before broadening the catalog:

- external and convergence evidence for nonlinear materials and cohesive
  fracture;
- richer thermal--mechanical and high-temperature histories;
- provider-owned reactions, work, and energy for MPC/weak/contact constraints;
  unsupported dual channels already fail closed rather than publishing a
  partial balance;
- selected cyclic plasticity, creep--fatigue, and fracture extensions.

### 2. Engineering workflow

Make real simulation projects easier to construct and maintain:

- one public Step and result lifecycle across supported procedures;
- richer imported-mesh regions, surfaces, sets, and quality diagnostics;
- multi-Step activation, inheritance, predefined fields, and engineering
  postprocessing;
- richer direct integration-point exchange and scalable collective checkpoint
  backends beyond the current portable laboratory-scale state contracts.

### 3. AI and data

Keep deterministic mechanics authoritative while making the complete workflow
naturally operable by agents and learning systems:

- stable machine-readable capabilities, validation issues, and results;
- simulation campaigns, observations, scientific datasets, and guarded model
  use;
- optional neural-field, neural-operator, surrogate, and user-model providers;
- future calibration and active-learning workflows with explicit provenance
  and applicability evidence.

PyTorch or any particular AI framework remains optional. Users may connect
their own models without inheriting an AgentFEM-specific neural-network base
class.

### 4. Open ecosystem

Keep the core useful on its own while supporting independent extensions:

- documented Python entry points and conflict-safe provider registration;
- companion packages for optional frameworks and specialized workflows;
- the same public contracts for scripts, agents, IDEs, future GUIs, and private
  domain products;
- contribution templates that require formulas, tests, examples, limits, and
  evidence appropriate to the claimed maturity.

## Capability maturity

A serious scientific capability advances through five evidence levels:

1. **Formula** — typed parameters and declared mathematical assumptions.
2. **Local** — analytical, invariant, or material-point verification.
3. **FEM integrated** — global assembly, state, convergence, output, and
   failure handling.
4. **Engineering** — representative benchmark and a bounded complete workflow.
5. **Release** — installed-artifact, platform, MPI, documentation, and
   regression gates protect the claim.

A public name does not imply the highest level. Experimental capabilities
remain useful, but their status must stay visible to users and agents.

## Toward 0.3

The 0.3 series is the point at which AgentFEM's platform contract becomes
coherent enough for wider extension and application development. Its focus is:

- one recommended engineering grammar centered on `model.step(...)`;
- provider-owned lowering instead of material-specific logic accumulating in
  the Model facade;
- one inspectable execution and evidence lifecycle;
- machine-readable compatibility guidance for existing projects;
- reproducible installed use on Linux, macOS, and Windows through WSL2;
- a proven extension path that does not require modifying the open core.

Compatibility methods remain executable during the transition. AgentFEM
reports preferred replacements but does not silently rewrite scientific Python
or change modeling intent.

The repository turns these conditions into an executable audit:

```bash
PYTHONPATH=src python promotion_gate.py --report promotion.json
```

Core architecture gates run directly. Installed-wheel platform, companion
extension, and unfamiliar-agent gates consume independent JSON acceptance
records; missing evidence remains `external_evidence_required` rather than
being inferred from development-machine success.

G1--G4 run directly from the repository. G5 is completed only by
installed-wheel records from Linux, macOS, and a real WSL2 kernel. G6 consumes
the independently built `agentfem-learning` extension record. G7 consumes a
fresh-context, zero-intervention AI-agent trial whose result and explanation
are retained and reviewed; the deterministic release smoke deliberately does
not impersonate that behavioral evidence. Acceptance artifacts can be
aggregated with `promotion_gate.py --evidence-directory ...`.

Every external record is bound to the candidate AgentFEM version and source
commit. Platform and fresh-agent records also retain the exact wheel digest;
passing evidence from an older release is deliberately ineligible.

## Beyond 0.3

Longer-term families include phase-field fracture, broader contact,
beam/shell formulations, deeper multi-physics coupling, scalable
tangent/adjoint responses, richer neural-field/operator providers, and more
external CAE interoperability. They will enter the trusted core only through
the same maturity and evidence process.

This roadmap communicates direction rather than a release promise. Current
truth remains in released code, capability records, benchmark evidence, and
release notes.
