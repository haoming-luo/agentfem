# AgentFEM Agent Guide

Use this guide when an AI agent is asked to build, review, extend, or debug a
finite-element simulation with AgentFEM.

## First Steps

1. For an installed case, run `agentfem doctor --json`, then
   `agentfem check --json`. For an existing case also run
   `agentfem upgrade --json`; treat `semantic_review=true` as a requirement to
   inspect the finite-element meaning, not as permission for blind text
   replacement. For repository work, also read this guide.
2. Read `WORKFLOW.md` to identify the standard finite-element sequence.
3. Read `CONCEPTS.md` to align terminology before changing code.
4. Identify or create the `studies.Study` before choosing constitutive laws or
   operators.
5. Inspect or create a mesh summary before reasoning about boundary tags,
   material regions, or output dimensions.
6. Inspect the target application only after mapping it to AgentFEM concepts.
7. Prefer existing `agentfem` APIs before adding new helpers.
8. Add reusable code only when it belongs to a standard FEM concept.

## Progressive Reading

- Mesh, domain, or region modeling: read the Mesh Region, Selector, and Region
  Set concepts in `CONCEPTS.md`. Prefer named regions and region sets over raw
  integer tags in user-facing workflows. Use low-level `MeshTags` helpers only
  when importing, validating, or teaching the tagging implementation.
- Mesh or boundary tagging: read `docs/module_map.md`, then use
  `mesh.summarize_mesh`, `mesh.require_cell_tags`, and
  `mesh.require_facet_tags`.
- Study setup: use `studies.static_solid`, `studies.steady_heat_transfer`,
  `studies.transient_heat_transfer`, or `studies.dynamic_solid` for common
  workflows. Use the general factories only when the common vocabulary does
  not fit. `dynamic_solid(method=...)` changes the procedure, not the physics.
- Solution procedure: inspect `procedures.py` when Standard/Explicit or a
  particular time integrator must be selected. Do not encode the algorithm by
  inventing a new physical analysis type.
- Model audit: use `model.tree()`, `model.manifest()`, `model.summary()`, and
  `model.validate()` when multiple fields, regions, loads, constraints, or
  steps must be inspected. Use `model.check()` to stop before execution on
  validation errors and `model.write_ir(...)` when a persistent scientific
  record is required.
- Capability dispatch: inspect `models.step_capability(model)` or run
  `model.check()` before solve. It queries the same provider predicates as
  `model.step(...)`, including target-field shape and the material protocol
  needed by the default lowering; never work around `AFM-STUDY-002` by
  silently changing the Study.
- Model registration: use `model.field(...)`, `model.material(...)`,
  `model.fix(...)`, `model.traction(...)`, `model.surface_force(...)`,
  `model.distributing_coupling(...)`, and `model.elastic_foundation(...)` in
  application workflows when the assets should stay visible and auditable.
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
- Time histories: define reusable assets in `amplitudes.py`, register important
  histories with `model.amplitude("name", history)`, and reference the name
  from loads or prescribed data. Static single-solve, nonlinear normalized
  step time, and transient physical time have distinct documented coordinates.
  Attach histories to model loads, prescribed values, or supported boundary
  models. Transient procedures update registered histories automatically; use
  callbacks only for genuinely application-specific state.
- Solves: inspect `solvers.py`.
- Step incrementation: use `steps.automatic(...)` by default or
  `steps.fixed(...)` only when exact fixed subdivision is scientifically
  intended. `max_increments` limits accepted increments; Newton `max_it`
  limits iterations in one attempt.
- Multi-Step activation: use `model.stage(...)` and pass its result as
  `configuration=` to `model.step(...)`; keep asset inheritance separate from
  incrementation and nonlinear solver controls.
- Hybrid solids: a known C3D10H source mesh requires
  `fields.displacement_pressure(...)` and
  `constitutive.mixed_neo_hookean(...)`; do not route it through a
  displacement-only provider.
- Analysis steps: prefer `model.step(...)` or `model.linear_static_step(...)`
  when a model owns fields, materials, loads, and constraints. Use
  `problems.linear_static` or `problems.first_order_transient` when a workflow
  intentionally starts from explicit K/C/F operators without model ownership.
- Problem summaries: use `problems.FEMProblem` when a workflow needs a
  broader structured audit record.
- Results: read `docs/results_and_campaigns.md`; use `solve_result()` or
  `solve_result(output="results.xdmf")` for the standard static/transient
  solve/output/result lifecycle. Use `results.SimulationResult`, MPI-safe
  point/path probes, region integrals/averages, boundary resultants and field
  extrema in `results`, then attach additional XDMF/CSV artifacts from `io`.
  Model-generated static elasticity produces projected `S/E/MISES`
  automatically; `SENER` is opt-in through `field_variables`. Use
  `results.small_strain_partition_fields` for reviewed
  regional projections and `results.reaction_resultant(..., on=..., component=...)`
  only for named strong-constraint reactions.
  For imported physical surfaces use `mesh.tagged_boundary_region(...)` for
  both strong and weak conditions, call `model.audit_boundaries(strict=True)`,
  and use `results.region_measure(on=...)` rather than application-owned UFL.
  Request `field_extrema(..., location=True)` when publishing a peak. Serial
  `solve_result(output=...)` writes one Uniform Grid with mixed point/cell
  attributes; low-level `io.XDMFTimeSeries` retains DOLFINx multi-Grid
  semantics.
  For field-learning data, create a fixed `surrogates.ObservationGrid` and call
  `datasets.fem_observation_sample`; preserve its units, layout, and mask.
  Distinguish
  execution status from
  `result.trust_level`, apply a named quality policy for routine checks, and
  attach explicit scientific claims before describing a result as verified.
- Result provenance: register every numerical/report artifact before
  publishing, then use `agentfem verify ... --json`. Treat `verified` here as
  byte-integrity evidence only; scientific trust remains in
  `SimulationResult.verification`.
- For long transient jobs, pass one `checkpointing.every(...)` policy through
  `model.step(checkpoint=...)`; do not implement cadence in a case loop. Set
  `keep_last` for bounded storage; do not delete checkpoint directories in
  application code.
- For transient probes or scalar histories, pass reusable
  `results.probe_history(...)` / `results.history(...)` requests to
  `run(history=...)` or `solve_result(history=...)`. The callback receives the
  public Step and physical time. Keep the same named requests across restart;
  AgentFEM rejects a changed history schema instead of producing ragged data.
- Restart: heat, Standard dynamics, and Explicit dynamics share
  `run(until_step=...)`, `save_checkpoint(...)`, `load_checkpoint(...)`, and
  resumed `solve_result(output=...)`. Treat `output_scope="continuation_segment"`
  as a partial field series even though the result retains full accepted-time
  and execution evidence. Do not change MPI size or mesh partition when loading
  the current partition-bound checkpoint.
- Installed projects and external frontends: read `docs/getting_started.md`
  and `docs/agent_gui_integration.md`. Keep `case.py` as modeling truth, use
  `project.current_run()` for artifacts, publish a `SimulationResult`, and
  consume versioned JSON instead of scraping terminal prose.
- Installed private extensions: run `agentfem extensions --json` before
  accepting a project's `[extensions].required` list. Treat every extension as
  trusted executable code; a missing package is not permission to install an
  unknown distribution. Keep private products in separate repositories and
  integrate them through the versioned extension API.
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
  validation. Use `require_dataset(quality="engineering")` for ordinary
  learning data and a release policy when every sample carries a scientific
  release contract.
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
path-dependent material tools. The first global J2 and Arrhenius power-law
creep providers are serial-only foundations. Product depth and verification
take priority over adding material names or backend abstractions.
