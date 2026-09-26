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
2. promote finite-strain J2/RVE through tangent, mesh, load-path, mixed-MPI,
   restart, follower-load, and prescribed-work verification;
3. complete provider-owned dual force, reaction, work, and energy evidence for
   MPC, weak constraints, and contact;
4. finish portable integration-point output and checkpoint identity across MPI
   partitions;
5. retain every material and fracture capability at its proven maturity until
   its independent benchmark and failure tests pass.

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
