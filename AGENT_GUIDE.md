# AgentFEM Agent Guide

Use this guide when an AI agent is asked to build, review, extend, or debug a
finite-element simulation with AgentFEM.

## First Steps

1. Read `WORKFLOW.md` to identify the standard finite-element sequence.
2. Read `CONCEPTS.md` to align terminology before changing code.
3. Identify or create the `studies.Study` before choosing constitutive laws or
   operators.
4. Inspect or create a mesh summary before reasoning about boundary tags,
   material regions, or output dimensions.
5. Inspect the target application only after mapping it to AgentFEM concepts.
6. Prefer existing `agentfem` APIs before adding new helpers.
7. Add reusable code only when it belongs to a standard FEM concept.

## Progressive Reading

- Mesh, domain, or region modeling: read the Mesh Region, Selector, and Region
  Set concepts in `CONCEPTS.md`. Prefer named regions and region sets over raw
  integer tags in user-facing workflows. Use low-level `MeshTags` helpers only
  when importing, validating, or teaching the tagging implementation.
- Mesh or boundary tagging: read `docs/module_map.md`, then use
  `mesh.summarize_mesh`, `mesh.require_cell_tags`, and
  `mesh.require_facet_tags`.
- Study setup: use `studies.linear_static`, `studies.first_order_transient`,
  `studies.implicit_dynamics`, or `studies.explicit_dynamics` before building
  operators. Use neutral `studies.second_order_dynamics` only when downstream
  code deliberately selects the numerical procedure.
- Solution procedure: inspect `procedures.py` when Standard/Explicit or a
  particular time integrator must be selected. Do not encode the algorithm by
  inventing a new physical analysis type.
- Model audit: use `model.tree()`, `model.manifest()`, `model.summary()`, and
  `model.validate()` when multiple fields, regions, loads, constraints, or
  steps must be inspected. Use `model.check()` to stop before execution on
  validation errors and `model.write_ir(...)` when a persistent scientific
  record is required.
- Model registration: use `model.field(...)`, `model.material(...)`,
  `model.fix(...)`, and `model.traction(...)` in application workflows when the
  assets should stay visible and auditable.
- Application unknowns: use `fields.py` before dropping to `spaces.py`.
- Function spaces: inspect `spaces.py`; only inspect `kernel/dofs.py` for
  implementation-level dof work.
- Essential boundary conditions: read `CONCEPTS.md`, then use `constraints/`.
- Natural loads: read `CONCEPTS.md`, then use `loads.body_load`,
  `loads.neumann`, or `loads.boundary_load`.
- Constitutive laws: read `docs/nonlinear_materials.md`, query
  `constitutive.capabilities()`, then use `constitutive/`. Never turn a
  material-point law into a claimed FEM step without state/tangent/solver
  evidence.
- Absorbing or Robin-like terms: use `boundary_models/`.
- Assembly or lumped operators: inspect `assembly.py`.
- Time stepping: inspect `time/` and `problems.py`.
- Solves: inspect `solvers.py`.
- Step incrementation: use `steps.automatic(...)` by default or
  `steps.fixed(...)` only when exact fixed subdivision is scientifically
  intended. `max_increments` limits accepted increments; Newton `max_it`
  limits iterations in one attempt.
- Analysis steps: prefer `model.step(...)` or `model.linear_static_step(...)`
  when a model owns fields, materials, loads, and constraints. Use
  `problems.linear_static` or `problems.first_order_transient` when a workflow
  intentionally starts from explicit K/C/F operators without model ownership.
- Problem summaries: use `problems.FEMProblem` when a workflow needs a
  broader structured audit record.
- Results: read `docs/results_and_campaigns.md`; use `solve_result()`,
  `results.SimulationResult`, assembled QoIs in `results`, then attach XDMF/CSV
  artifacts from `io`. Distinguish execution status from
  `result.trust_level`, and attach explicit claims before describing a result
  as verified.
- AF-IR and repair: inspect `ir/` and `validation.py`. Treat AF-IR 0.1 as an
  experimental record, not as proof of backend-neutral executability.
- Backend work: inspect `backends/` and
  `docs/air_architecture_roadmap.md`. Preserve the full FEniCSx path and do not
  advertise a backend until its capabilities are independently tested.
- Parameter campaigns and learned models: read
  `docs/ai_native_learning.md`, then inspect `campaigns/`, `datasets/`, and
  `surrogates/`. Build a fresh model per case; retain units, output shapes,
  case IDs, validation data, and applicability behavior. Do not present a
  training residual or successful prediction call as independent scientific
  validation. For release or learning data, use
  `require_dataset(minimum_trust_level="verified")` when appropriate.
- External meshes: inventory with `mesh.inspect_external_mesh(...)` before
  choosing cell/facet types. Preserve conversion manifests and never describe
  mesh conversion as full Abaqus/ANSYS deck import.
- Example workflows: inspect `examples/` after reading `WORKFLOW.md`.

## Agent Rules

- Do not mix constraints, loads, and boundary models.
- Do not hide the finite-element workflow inside overly broad abstractions.
- Do not make concrete geometry helpers, such as circle/disk/box predicates,
  the core modeling concept. Treat them as selectors used to build named
  regions. Regions are the stable interface for materials, loads, constraints,
  boundary models, output, and imported CAE mesh groups.
- Keep application-specific geometry, source definitions, and parameter choices
  outside the core platform.
- Validate imports, syntax, and a small runnable case after structural changes.
- Report structured validation codes and paths when they are available; do not
  replace them with an unaddressed generic error.
- When changing a public concept, update `CONCEPTS.md`, `WORKFLOW.md`, and the
  relevant skill references.
- Keep solve increments and result frames distinct. Use output
  `every="increment"` or `intervals=...`; reserve “frame” for saved/read result
  states.
- State whether a nonlinear capability is FEM-integrated, material-point
  verified, or a postprocessor. Use the benchmark registry as evidence.
- Never call one converged mesh a convergence study. Use coarse-to-fine
  `verification.ConvergenceStudy` evidence and record a reference theory's
  applicability domain.
- Do not silently extrapolate a learned model. Reject the request or invoke an
  explicit high-fidelity fallback and report which source produced the result.

## Current Platform Focus

AgentFEM currently focuses on a dependable DOLFINx/PETSc path: linear
elasticity, implicit and explicit linear structural dynamics, implicit heat
transfer, sequential thermoelasticity, a bounded Neo-Hookean nonlinear solve,
result/campaign/data flow, external mesh conversion, and deliberately staged
path-dependent material tools. The first global J2 provider is serial-only;
Arrhenius power-law creep is still a local law. Product depth and verification
take priority over adding material names or backend abstractions.
