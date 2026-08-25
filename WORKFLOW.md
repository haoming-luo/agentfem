# AgentFEM Workflow

AgentFEM uses a standard finite-element workflow. Application code should make
this sequence visible unless there is a strong reason to encapsulate it.

## Standard Sequence

For an existing Abaqus project, first run `agentfem inspect-abaqus` and then
`agentfem migrate-abaqus`. Review the recursive source graph, Part/Instance
scope, regions, section/material assignments, formulation suffixes, and every
addressable issue before entering the standard sequence below. The generated
project is a source-preserving migration workspace, not an executable
whole-deck translation.
If the narrow native route is eligible, `agentfem lower-abaqus` emits a draft
only after `--reviewed-by` and `--unit-system` are explicit. Activation remains
a separate choice; never bypass a blocked finding by editing the decision
record.

1. Define the study context: analysis type, physics, dimension, and modeling
   assumptions.
   For a revolved small-strain solid use `dimension=2,
   assumption="axisymmetric"`; this declaration lowers kinematics, operators,
   loads, and standard results consistently.
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
17. Solve to a `results.SimulationResult` as the standard completion path.
    Output may be declared while constructing the step and consumed without
    repeating it: `model.step(target=u, output="results.xdmf").solve_result()`.
    Passing the same request directly to `solve_result(output=...)` remains
    supported. For transient heat or dynamics, this writes the time series and
    attaches the logical XDMF/HDF5 dataset to the same result.
    Declare compact transient probes, integrals, or other scalar histories
    through `history=(results.probe_history(...), results.history(...))` on
    this same call; they are sampled after every accepted increment.
    A paused/restarted step may use the same call; the result identifies the
    artifact as a continuation segment and records its physical start time.
    Stateful J2 and creep steps use the same `solve_result(output=...)` call:
    raw quadrature state stays inspectable in the result, while explicitly
    recovered `*_CELL` fields enter the visualization dataset. For implicit
    creep, keep Newton convergence, maximum accepted CEEQ increment, and the
    optional endpoint creep-rate time-error tolerance as separate acceptance
    decisions. Declare a model unit system when time histories require a
    physical unit label.
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

Public discovery is progressive. `agentfem.public_api("core")` is the daily
engineering language; `"advanced"` adds campaigns, fracture, mechanics, and
learning bridges; `"expert"` exposes backend and extension seams. Calling
`public_api()` without a level preserves the complete 0.2.0 inventory.
Within the model facade, `models.model_api("core")` returns the recommended
engineering verbs, while `"advanced"` discloses deliberate lower-layer seams
and `"compatibility"` identifies historical aliases that remain executable but
should not be generated in new cases. Built-in Step providers publish their
accepted and required keyword contracts through `models.step_providers()` and
`agentfem capabilities --json`.

For a collection of related cases, continue with:

21. Define a typed `campaigns.ParameterSpace` with bounds and units, directly
    or through a safe JSON campaign specification.
22. Create a deterministic design of experiments and fresh model variants.
23. Declare reusable scientific inputs such as source meshes, materials,
    loading bases, and observer plans so coverage and fingerprints are explicit.
24. Run serially, through `campaigns.local_processes(...)`, or as deterministic
    external shards; do not nest across-case processes inside within-case MPI.
25. Audit mesh/time/other refinement axes with explicit fixed coordinates and
    observable-specific convergence policies.
26. Require a named quality policy before admitting cases to a learning
    dataset; use a raw minimum trust level only for a project-specific policy.
27. Split the resulting `ScientificDataset` independently before training.
28. Validate a surrogate or reduced-order model, declare its applicability
    domain, and retain a high-fidelity FEM fallback where extrapolation would
    be unsafe.

## Module Map

Package paths below are relative to `src/agentfem/`.

- Mesh import, named regions, tags, summaries, checks, and measures: `mesh/`
- Local coordinate systems and engineering reference points: `coordinates.py`
- Consistent numerical unit contracts: `units.py`
- External CAE mesh inventory/conversion and named-set manifests:
  `mesh/formats.py`
- Scope-aware Abaqus project migration and fail-closed scaffolds:
  `mesh/abaqus_migration.py`
- Study contexts: `studies.py`
- Numerical solution-procedure descriptions: `procedures.py`
- Model registry and checks: `models.py`
- Spaces and fields: `spaces.py`
- Time histories and scale factors: `amplitudes.py`
- Low-level dofs and vector access: `kernel/dofs.py`
- Constraints: `constraints/`
- Loads and natural boundary data: `loads.py`
- Constitutive laws and their queryable maturity catalog: `constitutive/`
- Engineering damage and life-assessment consumers: `assessments.py`
- Material library: `materials/`
- Boundary models: `boundary_models/`
- Weak-form blocks: `forms.py`
- Assembly: `assembly.py`
- Operator families: `operators/`
- Time integration: `time/explicit.py`, `time/implicit.py`, and runtime cadence
- Cyclic cohesive damage, cycle jumps and 3D crack observations:
  `fatigue_fracture.py`
- Structure-level cyclic peak/valley control uses
  `fatigue_fracture.global_cyclic_fatigue_step(...)`. It owns accepted cycle
  coordinates, post-damage equilibrium checks, automatic cycle cutback and
  bulk/interface restart. Use
  `fracture.FiniteStrainCohesiveEquilibrium(...)` for the native
  hyperelastic bulk-plus-interface Newton route; a custom equilibrium callback
  remains a supported extension boundary for other global formulations.
- Several disjoint cohesive cracks should be split together through
  `interfaces.split_conforming_named_interfaces(...)`, then lowered to one
  named force collection with `fracture.named_mode_i_cohesive_forces(...)`.
- Shared transient checkpoint envelope and partition identity: `checkpointing.py`
- Stable Model facade and model-owned analysis-step entry: `models.py`
- Provider-owned scientific Step builders: `_step_builders.py`
- Analysis lowering and inspectable execution policies: `step_providers.py`
- Stateful nonlinear solid mechanics: `mechanics/`
- Algebraic problems, reusable step containers, and state containers:
  `problems.py`
- Step incrementation and cutback: `steps.py`
- Solvers and convergence policy: `solvers.py`
- Scientific result/QoI/dataset bridge: `results/`
- Diagnostics: `diagnostics.py`
- Threshold events, first-passage localization, and censoring: `events.py`
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
- Multi-axis, observable-aware convergence certificates: `convergence.py`
- Campaign-backed finite-difference response experiments: `responses.py`
- Unit/shape-aware learning data and simulation provenance: `datasets/`
- Unified scientific-learning entry, including surrogate compatibility,
  neural-operator contracts, provider-neutral neural-field objectives, and the
  user-owned executor boundary:
  `learning/`
- Exact surrogate/ROM baselines, validation, applicability guards, and the
  0.2.x PINN compatibility API: `surrogates/`

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

Beginner workflows should begin with `agentfem.public_api("core")`: studies,
mesh, models, fields, materials/constitutive behavior, constraints, loads,
steps, results, and verification. Campaigns, datasets, learning, fracture,
operator-first construction, and direct problem/time containers are deliberate
advanced disclosures, not additional beginner routes. Expert backend, form,
assembly, IR, and extension modules remain available without appearing in the
first modeling vocabulary.

The tier inventories, Model verbs, CLI commands, and machine workflow stages
have one dependency-free source in `_api_contract.py`. Do not duplicate these
lists in a frontend, Skill, generated manifest, or documentation script.

First-level Python modules are reserved for the main FEM workflow. Subpackages
hold reusable asset families, such as constitutive laws, material records,
boundary models, element policies, operator families, and benchmarks.
Low-level implementation helpers such as `constraints/boundary.py` and
`kernel/dofs.py` should not be featured in beginner workflows.
