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
- Abaqus migration: run `agentfem inspect-abaqus source.inp --json` before
  conversion and read `docs/abaqus_migration.md`. Preserve element formulation
  suffixes and report topology-only support; never substitute a native element
  formulation silently.
- If `migration.json` reports an eligible narrow native route, use
  `agentfem lower-abaqus PROJECT --reviewed-by NAME --unit-system SYSTEM` to
  create `case.native.py` and `lowering.json` without activation. Read both,
  then use `--activate --force` only when accepted. Never invent units or
  delete a blocking finding. Multiple constant isotropic materials are lowered
  only through exact, non-overlapping ELSET/Section coverage. Dependent
  material tables, partial or ambiguous Section coverage, non-unit 2D
  thickness, and Step/BC inheritance require a dedicated lowering route rather
  than a first-row or final-value approximation.
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
- API discovery: begin with `agentfem.public_api("core")` or the
  `public_api.core` section of `agentfem capabilities --json`. Escalate to the
  advanced or expert layer only when the task requires it; the flat
  `public_modules` inventory remains a compatibility contract. These generated
  views share one product contract; never reconstruct a competing vocabulary
  from one example or an old compatibility method.
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
  `constitutive.chaboche(...)` is an experimental three-dimensional global
  combined-hardening route with quadrature rollback/restart and `ALPHA`
  output. Keep its external-definition evidence distinct from a calibrated
  structure-level stabilized-hysteresis validation.
- Named materials: use `materials.define(...)` when material identity, source,
  reuse, or separate mechanical/thermal roles matter. The Study selects the
  required physics role at `model.material(...)`; it does not choose or rewrite
  the constitutive equation. Packaged `load_definition(...)` cards are
  reference-only unless their source and applicability establish otherwise.
- Project materials: `materials.load("materials/active.py")` explicitly loads
  reviewed, trusted Python code and records its content hash. Do not use it to
  inspect downloaded code. TOML may select an asset but is not a constitutive
  language; installed or private collections belong in extension packages.
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
  For many related loading modes, use `amplitudes.basis(...)` and
  `basis.combine(...)`; preserve coefficient order, derivatives, endpoint
  audit, and the amplitude fingerprint. Do not represent a frozen campaign
  waveform only as an anonymous callable.
- Solves: inspect `solvers.py`.
- Step incrementation: use `steps.automatic(...)` by default or
  `steps.fixed(...)` only when exact fixed subdivision is scientifically
  intended. `max_increments` limits accepted increments; Newton `max_it`
  limits iterations in one attempt.
- Global implicit creep has an additional physical-time accuracy decision:
  `creep_strain_error_tolerance` bounds the endpoint creep-rate change times
  the attempted increment. Keep it distinct from Newton convergence and
  `maximum_inelastic_increment`; all three may independently trigger atomic
  rollback and cutback. Result time units come from the model `UnitSystem` and
  remain undeclared when no consistent-unit contract was supplied.
  For one-way thermal input, `capture_history(...)` records source Study,
  procedure, accepted-time role and content identity. Use
  `assessments.sequential_energy_ledger(...)` to keep heat and mechanical
  residuals separate. For engineering creep--fatigue from result histories,
  declare `DwellInterval` objects and pass a source-identified project rupture
  relation to `creep_fatigue_from_result(...)`; do not embed normative curves.
- Multi-Step activation: use `model.stage(...)` and pass its result as
  `configuration=` to `model.step(...)`; keep asset inheritance separate from
  incrementation and nonlinear solver controls.
- Hybrid solids: a known C3D10H source mesh requires
  `fields.displacement_pressure(...)` and
  `constitutive.mixed_neo_hookean(...)`; do not route it through a
  displacement-only provider.
- Analysis steps: prefer `model.step(...)` when a model owns fields,
  materials, loads, and constraints. Pass a `SolutionProcedure` object when
  the numerical route must be explicit or must override the Study preference;
  capability inspection and lowering consume that same object. Treat named
  legacy Step factories as expert/compatibility entrypoints. Use
  `problems.linear_static` or `problems.first_order_transient` when a workflow
  intentionally starts from explicit K/C/F operators without model ownership.
- Problem summaries: use `problems.FEMProblem` when a workflow needs a
  broader structured audit record.
- Scientific formulas: when a formula arrives from JSON, a benchmark,
  experiment metadata, or agent-authored configuration, use
  `expressions.expression(...)`, `expressions.as_ufl(...)`, or
  `expressions.interpolate(...)`. The language accepts coordinates, declared
  parameters, arithmetic, and reviewed mathematical functions; never replace
  it with `eval`, and do not silently reinterpret an invalid formula. Use
  `as_ufl` for symbolic weak-form physics and `interpolate` for known fields;
  the latter uses the same validated AST without a per-expression C++ JIT.
- Results: read `docs/results_and_campaigns.md`; use `solve_result()` or
  `solve_result(output="results.xdmf")` for the standard static/transient
  solve/output/result lifecycle. Use `results.SimulationResult`, MPI-safe
  point/path probes, region integrals/averages, boundary resultants and field
  extrema in `results`, then attach additional XDMF/CSV artifacts from `io`.
  A model-owned case may instead declare a path in
  `model.step(..., output="results.xdmf")` and call `solve_result()` once; do
  not send the same output request through custom case loops.
  Model-generated static elasticity produces projected `S/E/MISES`
  automatically; `SENER` is opt-in through `field_variables`. J2 and creep
  expose the same completed-result writer through `solve_result(output=...)`;
  raw quadrature fields remain result evidence and explicit `*_CELL` fields
  are written for visualization. Use `results.small_strain_partition_fields`
  for reviewed
  regional projections and `results.reaction_resultant(..., on=..., component=...)`
  only for named strong-constraint reactions. Inspect
  `SimulationResult.metadata["constraint_balance_contract"]` before claiming
  global equilibrium: MPC, weak, contact, projection, or multiplier channels
  remain unavailable until their active provider supplies actual dual forces.
  For imported physical surfaces use `mesh.tagged_boundary_region(...)` for
  both strong and weak conditions, call `model.audit_boundaries(strict=True)`,
  and use `results.region_measure(on=...)` rather than application-owned UFL.
  Request `field_extrema(..., location=True)` when publishing a peak. Serial
  `solve_result(output=...)` writes one Uniform Grid with mixed point/cell
  attributes; low-level `io.XDMFTimeSeries` retains DOLFINx multi-Grid
  semantics. Under MPI, open the `field_output` metadata's
  `recommended_visualization_artifact` (normally PVD), not the scientific
  multi-Grid XDMF; that route needs no Extract Block step.
  For field-learning data, create a fixed `surrogates.ObservationGrid` and call
  `datasets.fem_observation_sample`; preserve its units, layout, and mask.
  For a direct solver or external benchmark contract, use
  `results.sample_rectilinear_grid(...)`; its physical-axis `shape=(nx, ny,
  nz)` is returned in array order `(nz, ny, nx)` and its `inside` mask must be
  retained for non-rectangular domains.
  For the pinned PDEAgent-Bench contract, use
  `integrations.pdeagent_bench.solve_case(...)` and read
  `docs/pdeagent_bench.md`; inspect `SUPPORTED_FAMILIES` rather than guessing
  capability from a case name. Do not read oracle-only mesh, FEM, solver, or
  manufactured-solution fields, dispatch on case IDs, or infer success from
  execution alone. Preserve the benchmark commit and failure taxonomy. For
  mixed flow, periodic MPC, or split fourth-order cases, report the resolved
  field pair, pressure reference, periodic method, and auxiliary boundary
  closure from `solver_info` rather than presenting them as implicit defaults.
  When observations and the FEM mesh use different axes or origins, pass an
  explicit `surrogates.AffineCoordinateMap`; for publication maps convert the
  result to `datasets.RectilinearObservation` before comparison. Never infer a
  transpose, unit conversion, or reference/current transformation from a
  figure. A `fracture.DynamicFractureEvidenceBundle` is the portable handoff
  for one fracture condition, but its provenance seal is not a validation
  claim.
  Distinguish
  execution status from
  `result.trust_level`, apply a named quality policy for routine checks, and
  attach explicit scientific claims before describing a result as verified.
- Result provenance: register every numerical/report artifact before
  publishing, then use `agentfem verify ... --json`. Treat `verified` here as
  byte-integrity evidence only; scientific trust remains in
  `SimulationResult.verification`.
- Runtime freeze: use `provenance.freeze_runtime(...)` before a frozen or blind
  campaign and `provenance.require_runtime(...)` before continuing it. A
  runtime match establishes execution compatibility, not scientific validity.
- Scientific inputs: pass reusable mesh/source files as `Path` values and
  materials, loading bases, procedures, and observer plans through
  `Campaign(scientific_inputs=...)`. Review fingerprint coverage; an opaque
  input is an explicit incomplete identity, not permission to ignore it.
- For a single run, attach the same assets with
  `result.add_scientific_inputs(...)` before writing its manifest. Pass source
  files as `Path`, not filename strings, when byte identity matters.
- Events: use `events.first_passage(...)` for threshold timing. Preserve its
  bracket, localization method, and censoring state; for discontinuous damage
  or contact, refine the solve rather than presenting linear interpolation as
  an exact event time.
- For long transient jobs, pass one `checkpointing.every(...)` policy through
  `model.step(checkpoint=...)`; do not implement cadence in a case loop. Set
  `keep_last` for bounded storage; do not delete checkpoint directories in
  application code.
- For transient probes or scalar histories, pass reusable
  `results.probe_history(...)` / `results.history(...)` requests to
  `model.step(history=...)`, `run(history=...)`, or
  `solve_result(history=...)`. Construction-time requests are consumed
  automatically by the common result lifecycle and retained with solver,
  output, progress, and checkpoint declarations in execution metadata. The
  callback receives the public Step and physical time. Keep the same named
  requests across restart; AgentFEM rejects a changed history schema instead
  of producing ragged data.
- Restart: heat, Standard dynamics, and Explicit dynamics share
  `run(until_step=...)`, `save_checkpoint(...)`, `load_checkpoint(...)`, and
  resumed `solve_result(output=...)`. Treat `output_scope="continuation_segment"`
  as a partial field series even though the result retains full accepted-time
  and execution evidence. Do not change MPI size or mesh partition when loading
  the current partition-bound checkpoint.
- Cohesive state portability: fixed-path cohesive transactions may instead use
  `interfaces.save_portable_cohesive_state(...)` and
  `interfaces.load_portable_cohesive_state(...)`. Physical facet keys survive
  local order and MPI rank-count changes. `create_dolfinx_split_mesh(...,
  comm=...)` plus the unchanged `fracture.mode_i_cohesive_force(...)` factory
  provides the experimental sparse-payload MPI force path. When positive-side
  cell identities are known, prefer
  `interfaces.split_conforming_cell_interface(...)` to a hand-written facet
  list. Portable transient checkpoints also distinguish coincident independent
  nodes. Do not infer 3D imported surfaces or extreme-scale performance from
  this 2D owner-scheduled contract.
- Cyclic cohesive equilibrium: use
  `fracture.FiniteStrainCohesiveEquilibrium(...)` when a UFL finite-strain bulk
  residual and paired-facet interfaces should enter one native Newton solve.
  Supply the physical scalar load through `load_parameter` or `set_load`; keep
  interface history commit/rollback under the owning Step. Strong-constraint
  reactions are supported through the residual reaction field, but do not reuse that
  definition for MPC, weak, contact or multiplier reactions.
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
- API lifecycle: inspect `models.model_api_contract()` or
  `agentfem capabilities --json` before generating a Model call. Run
  `agentfem upgrade` before adapting an older project. Compatibility findings
  that require semantic review are guidance, not permission for blind textual
  replacement.
- Backend work: inspect `backends/` and
  `docs/air_architecture_roadmap.md`. Preserve the full FEniCSx path and do not
  advertise a backend until its capabilities are independently tested.
- Parameter campaigns and learned models: read
  `docs/ai_native_learning.md`, then inspect `campaigns/`, `datasets/`, and
  `learning/`. Build a fresh model per case; retain units, output shapes,
  case IDs, validation data, and applicability behavior. Do not present a
  training residual or successful prediction call as independent scientific
  validation. Use `require_dataset(quality="engineering")` for ordinary
  learning data and a release policy when every sample carries a scientific
  release contract. A user-owned neural-field model may enter through
  `model.step(target=spec, executor=...)` without an official adapter, but it
  must return `SimulationResult`; do not serialize the live callable into
  provenance or infer scientific validation from training loss.
- Predefined fracture fields: construct reusable straight 2D cracks through
  `fracture.segment(...)` and `fracture.crack_set(...)`. Preserve stable
  crack/tip identity, orientation, and fingerprints through every provider.
  Never infer fracture mode from an input label, average the two sides of a
  discontinuous field into one nodal value, or report one SIF without its local
  convention and ring-sensitivity evidence. The first geometry contract
  deliberately rejects intersecting, touching, and curved cracks.
- Neural energy integration: use `learning.IntegrationPlan` to distinguish the
  optimization rule from explicitly independent validation and refinement
  rules. Attach `integration_consistency_check(...)` evidence to the result.
  Loss reduction, held-out integration, refinement convergence, boundary
  checks, and physical balance are separate decisions.
- Across-case execution: use `campaigns.local_processes(workers=...)` only for
  independent cases. It deliberately uses `spawn`. Do not nest it inside a
  within-case MPI communicator; use deterministic plan shards for separate MPI
  jobs or schedulers. Build/evaluate callables must be importable and
  serializable, and each case must construct fresh mutable solver state.
- Convergence certificates: use `convergence.audit(...)` with one explicit
  refinement axis at a time, fixed values for every other varying parameter,
  and an observable-specific relative, absolute, or exact comparison. Missing,
  failed, ambiguous, or insufficient sequences remain inconclusive.
- Response experiments: use `responses.finite_difference(...)` when baseline
  and perturbation cases should share Campaign identity, cache, failures, and
  evidence. Keep the parameter perturbation mode and output quantities
  explicit. A failed perturbation makes the response report incomplete.
- External meshes: inventory with `mesh.inspect_external_mesh(...)` before
  choosing cell/facet types. Preserve conversion manifests and never describe
  mesh conversion as full Abaqus/ANSYS deck import.
- Abaqus projects: run `agentfem inspect-abaqus ... --json` before
  `agentfem migrate-abaqus ...`. The migration project must preserve the
  recursive source graph, Part/Instance scope, section/material assignments,
  source locations, and unresolved issue codes. A generated material candidate
  is reviewable data, not permission to execute it. Do not remove formulation
  suffixes, flatten same-named sets across scopes, or bypass the fail-closed
  guard before native operators and verification evidence are selected. Never
  invent a missing material property or silently omit a Step, load, boundary,
  interaction, or output listed under `pending_assets`.
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
creep providers use regional quadrature materials. J2 MPI global equilibrium
is public after partition-interface, cutback/rollback, cross-rank-count
restart, and external thick-cylinder structural tests; distributed creep
equilibrium retains its separate NAFEMS promotion gate. Their constitutive
transactions and quadrature archives are MPI-safe and portable across rank
counts. Product depth and external verification take
priority over adding material names or backend abstractions.
