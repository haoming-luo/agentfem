# AgentFEM Workflow

AgentFEM uses a standard finite-element workflow. Application code should make
this sequence visible unless there is a strong reason to encapsulate it.

## Standard Sequence

1. Define the study context: analysis type, physics, dimension, and modeling
   assumptions.
2. Select a solution-procedure preference only when the same physical problem
   admits more than one route, for example Standard/Newmark,
   Standard/generalized-alpha, or Explicit/central difference. Keep this
   numerical choice separate from the study.
3. Define, import, convert, or read the mesh.
4. Create a lightweight model registry when auditability or agent inspection is
   useful.
5. Inspect the mesh summary and require expected material/boundary tags.
6. Define named mesh regions for boundaries, material zones, or point sets.
   Boundary regions carry `ds(tag)`; cell/material regions carry `dx(tag)`.
7. Create function spaces or application unknown fields.
8. Create state containers when needed.
9. Load or define material properties, then choose constitutive laws.
10. Define reusable amplitudes for prescribed time histories or scale factors.
11. Define essential constraints.
12. Define weak loads and boundary models.
13. Build operators or weak forms. Use model-first helpers such as
    `model.stiffness(...)` for standard registered assets, or operator-first
    constructors such as `operators.stiffness(...)` and
    `operators.combine(...)` when each contribution should be explicit.
14. Create and register an analysis step with `model.step(...)`. Pass an
    explicit `procedure=procedures...` only when overriding or making the
    Study's numerical preference visible. Legacy material/procedure-specific
    Step factories remain compatibility and expert implementation entrypoints,
    not parallel beginner workflows. The step should expose visible operators,
    such as `K U = F` or `(C / dt + K) T_next = C T_old / dt + Q`. For a
    nonlinear path, use `steps.automatic(...)` normally and
    `steps.fixed(...)` only for an intentionally fixed path.
15. Run `model.validate()` or `model.check()` before execution. When the case
    is an auditable artifact, write `model.write_ir(...)`.
16. Compile, assemble, and solve the step, or advance in time.
17. Solve to a `results.SimulationResult` when the result will feed more than
    one consumer. For transient heat or dynamics, prefer
    `step.solve_result(output="results.xdmf")`; it writes the time series and
    attaches the logical XDMF/HDF5 dataset to the same result.
    Declare compact transient probes, integrals, or other scalar histories
    through `history=(results.probe_history(...), results.history(...))` on
    this same call; they are sampled after every accepted increment.
    A paused/restarted step may use the same call; the result identifies the
    artifact as a continuation segment and records its physical start time.
    Stateful J2 and creep steps use the same `solve_result(output=...)` call:
    raw quadrature state stays inspectable in the result, while explicitly
    recovered `*_CELL` fields enter the visualization dataset.
18. Evaluate physical QoIs, diagnostics, and histories. Keep coefficient
    statistics distinct from assembled physical integrals.
19. Write visualization/output artifacts and attach them to the result. The
    standard transient lifecycle performs this automatically; low-level
    `run(output=...)` remains available for expert loops.
20. Apply `result.verify("exploratory" | "engineering" | "release")` and
    declare required quantities, histories, and artifacts. Add reference,
    invariant, discretization, or validation claims when the result will be
    described as verified or validated.

For an existing installed-use project, precede this sequence with
`agentfem doctor`, `agentfem check`, and `agentfem upgrade`. Only deterministic
project metadata may be migrated automatically; changes to scientific Python
require semantic review and re-verification.

For a collection of related cases, continue with:

21. Define a typed `campaigns.ParameterSpace` with bounds and units, directly
    or through a safe JSON campaign specification.
22. Create a deterministic design of experiments and fresh model variants.
23. Run or resume the campaign and extract declared `datasets.Quantity`
    outputs.
24. Require a named quality policy before admitting cases to a learning
    dataset; use a raw minimum trust level only for a project-specific policy.
25. Split the resulting `ScientificDataset` independently before training.
26. Validate a surrogate or reduced-order model, declare its applicability
    domain, and retain a high-fidelity FEM fallback where extrapolation would
    be unsafe.

## Module Map

- Mesh import, named regions, tags, summaries, checks, and measures: `mesh/`
- Local coordinate systems and engineering reference points: `coordinates.py`
- Consistent numerical unit contracts: `units.py`
- External CAE mesh inventory/conversion and named-set manifests:
  `mesh/formats.py`
- Study contexts: `studies.py`
- Numerical solution-procedure descriptions: `procedures.py`
- Model registry and checks: `models.py`
- Spaces and fields: `spaces.py`
- Time histories and scale factors: `amplitudes.py`
- Low-level dofs and vector access: `kernel/dofs.py`
- Constraints: `constraints/`
- Loads and natural boundary data: `loads.py`
- Constitutive laws and their queryable maturity catalog: `constitutive/`
- Material library: `materials/`
- Boundary models: `boundary_models/`
- Weak-form blocks: `forms.py`
- Assembly: `assembly.py`
- Operator families: `operators/`
- Time integration: `time/explicit.py`, `time/implicit.py`, and runtime cadence
- Shared transient checkpoint envelope and partition identity: `checkpointing.py`
- Model-owned analysis-step creation: `models.py`
- Stateful nonlinear solid mechanics: `mechanics/`
- Algebraic problems, reusable step containers, and state containers:
  `problems.py`
- Step incrementation and cutback: `steps.py`
- Solvers and convergence policy: `solvers.py`
- Scientific result/QoI/dataset bridge: `results/`
- Diagnostics: `diagnostics.py`
- Output: `io.py`
- Element/integration policies: `elements/`
- Verification benchmarks: `benchmarks/`
- Versioned scientific records: `ir/`
- Structured validation reports and issue codes: `validation.py`
- Scientific claims, trust levels, and mesh/time convergence evidence:
  `verification.py`
- Backend capabilities and compilation adapters: `backends/`
- Parameter spaces, sampling, case identity, and resumable execution:
  `campaigns/`
- Unit/shape-aware learning data and simulation provenance: `datasets/`
- Surrogate/ROM baselines, validation, applicability guards, and
  neural-operator/PINN contracts: `surrogates/`

For publication or sensor-aligned field evidence, use one
`surrogates.AffineCoordinateMap` with `datasets.fem_observation_sample`, then
convert to `datasets.RectilinearObservation`. Do not hide axis, scale,
reference/current configuration, or outside-domain masks in plotting code.
Dynamic-fracture research conditions can be exchanged through
`fracture.DynamicFractureEvidenceBundle`; its seal proves artifact integrity,
not scientific validation.

For cohesive research restarts, use
`interfaces.save_portable_cohesive_state(...)` and
`interfaces.load_portable_cohesive_state(...)`. The accepted state follows an
ordered physical facet key across facet order and MPI rank-count changes, and
the writer rejects inconsistent owner/ghost histories. For a split mesh, the
same `fracture.mode_i_cohesive_force(...)` call selects serial or MPI execution;
`step.save_checkpoint(..., portable=True)` retains coincident independent
nodal identities and cohesive history across rank-count changes. The MPI path
uses a one-time physical-node owner schedule and sparse `MPI_Alltoallv`
trace/force payloads. For conforming 2D cell regions,
`interfaces.split_conforming_cell_interface(...)` derives the checked path
without a hand-written facet list. This is not yet a 3D or extreme-scale
neighbor-collective scalability claim.

## Design Principle

The workflow should be easy for a human researcher to read and easy for an
agent to audit. A simulation file may call reusable helpers, but it should still
show the finite-element meaning of each step.

Prefer `model.tree()` and `model.manifest()` before a solve when an agent,
reviewer, or notebook user needs to inspect the current model state.

Beginner workflows should prefer the stable public path:
`studies`, `mesh`, `models`, `fields`, `materials`, `constitutive`,
`constraints`, `amplitudes`, `loads`, `operators`, `problems`, `solvers`,
`time`, `io`, `campaigns`, `datasets`, and `surrogates`.

First-level Python modules are reserved for the main FEM workflow. Subpackages
hold reusable asset families, such as constitutive laws, material records,
boundary models, element policies, operator families, and benchmarks.
Low-level implementation helpers such as `constraints/boundary.py` and
`kernel/dofs.py` should not be featured in beginner workflows.
