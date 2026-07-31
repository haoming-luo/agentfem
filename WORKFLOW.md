# AgentFEM Workflow

AgentFEM uses a standard finite-element workflow. Application code should make
this sequence visible unless there is a strong reason to encapsulate it.

## Standard Sequence

1. Define the study context: analysis type, physics, dimension, and modeling
   assumptions.
2. Define, import, convert, or read the mesh.
3. Create a lightweight model registry when auditability or agent inspection is
   useful.
4. Inspect the mesh summary and require expected material/boundary tags.
5. Define named mesh regions for boundaries, material zones, or point sets.
   Boundary regions carry `ds(tag)`; cell/material regions carry `dx(tag)`.
6. Create function spaces or application unknown fields.
7. Create state containers when needed.
8. Load or define material properties, then choose constitutive laws.
9. Define reusable amplitudes for prescribed time histories or scale factors.
10. Define essential constraints.
11. Define weak loads and boundary models.
12. Build operators or weak forms. Use model-first helpers such as
    `model.stiffness(...)` for standard registered assets, or operator-first
    constructors such as `operators.stiffness(...)` and
    `operators.combine(...)` when each contribution should be explicit.
13. Create and register an analysis step with `model.step(...)` or
    `model.linear_static_step(...)`. The step should expose visible operators,
    such as `K U = F` or `(C / dt + K) T_next = C T_old / dt + Q`. For a
    nonlinear path, use `steps.automatic(...)` normally and
    `steps.fixed(...)` only for an intentionally fixed path.
14. Run `model.validate()` or `model.check()` before execution. When the case
    is an auditable artifact, write `model.write_ir(...)`.
15. Compile, assemble, and solve the step, or advance in time.
16. Solve to a `results.SimulationResult` when the result will feed more than
    one consumer.
17. Evaluate physical QoIs, diagnostics, and histories. Keep coefficient
    statistics distinct from assembled physical integrals.
18. Write visualization/output artifacts and attach them to the result.

For a collection of related cases, continue with:

19. Define a typed `campaigns.ParameterSpace` with bounds and units, directly
    or through a safe JSON campaign specification.
20. Create a deterministic design of experiments and fresh model variants.
21. Run or resume the campaign and extract declared `datasets.Quantity`
    outputs.
22. Split the resulting `ScientificDataset` independently before training.
23. Validate a surrogate or reduced-order model, declare its applicability
    domain, and retain a high-fidelity FEM fallback where extrapolation would
    be unsafe.

## Module Map

- Mesh import, named regions, tags, summaries, checks, and measures: `mesh/`
- External CAE mesh inventory/conversion and named-set manifests:
  `mesh/formats.py`
- Study contexts: `studies.py`
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
- Time integration: `time/explicit.py`, future `time/implicit.py`, and runtime cadence
- Model-owned analysis-step creation: `models.py`
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
- Backend capabilities and compilation adapters: `backends/`
- Parameter spaces, sampling, case identity, and resumable execution:
  `campaigns/`
- Unit/shape-aware learning data and simulation provenance: `datasets/`
- Surrogate/ROM baselines, validation, applicability guards, and
  neural-operator/PINN contracts: `surrogates/`

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
