---
name: agentfem
description: Build, review, run, validate, migrate, or extend AgentFEM finite-element projects. Use for AgentFEM studies, meshes, materials, constraints, loads, solution steps, results, campaigns, scientific datasets, surrogate/PINN/neural-operator integration, verification, public API extensions, and agent or GUI integration.
---

# AgentFEM

Use this skill when building, reviewing, or extending finite-element simulations
with AgentFEM.

## Reference Routing

When this file is installed as a Codex skill, the reference paths below are
relative to the skill directory `skills/agentfem/`. In the generated
documentation site, use the left navigation pages `Workflow`, `Concepts`, and
`Module Map` instead.

1. Read `references/workflow.md` for every model construction, execution, or
   review task.
2. Read `references/concepts.md` when choosing or explaining scientific
   objects and maturity levels.
3. Read `references/module_map.md` when locating implementation code or
   deciding ownership.
4. Read `references/validation.md` before changing executable scientific code
   or promoting a verification claim.
5. Read `references/extension_rules.md` before adding a public helper, provider,
   constitutive family, or extension boundary.

## Rules

- Keep the finite-element workflow visible.
- Identify the study context before selecting constitutive laws or operators.
- Inspect mesh summaries and required tags before building weak forms.
- Use lightweight models for registry, checks, summaries, and model-owned
  analysis steps; do not hide K/F systems or finite-element meaning.
- Prefer model registration helpers such as `model.field(...)`,
  `model.material(...)`, `model.fix(...)`, `model.symmetry(...)`,
  `model.traction(...)`, `model.surface_force(...)`, and `model.pressure(...)`
  for application examples. Use `surface_force` when a continuum-solid end
  resultant should be distributed over a named reference boundary.
- Use `FEMProblem.summary()` or equivalent structured summaries when auditing a
  workflow.
- Use `model.validate()` for addressable issue reports, `model.check()` before
  execution, and `model.write_ir(...)` when a persistent AF-IR record is part
  of the task.
- Treat `model.check()` and `models.step_capability(model)` as the executable
  Study/provider preflight. Do not advertise or lower a combination that no
  registered provider accepts.
- Use AgentFEM modules before writing ad hoc DOLFINx/PETSc boilerplate.
- Discover modules with `agentfem.public_api("core")` first. Disclose
  `"advanced"` and `"expert"` only when the requested workflow needs them.
  Within the model facade, generate methods from `models.model_api("core")`;
  do not choose names reported under `models.model_api("compatibility")` for
  new cases.
- Treat `agentfem capabilities --json` and `/agentfem.json` as generated views
  of one product contract. Do not reconstruct a competing API vocabulary from
  a single example or an old compatibility method.
- Treat `ir` and `validation` as public inspection/record interfaces. Treat
  `backends` as an advanced numerical boundary and `extensions` as the explicit
  installed-package boundary. FEniCSx is the only production backend in the
  current release.
- Prefer `step = model.step(target=u)` when a model has a Study and registered
  materials, constraints, and loads. New analysis/material families belong in
  a registered step provider; do not add one public model method per material.
  Every built-in provider must declare a `StepOptionContract`, and agents
  should read provider option summaries from `agentfem capabilities --json`
  rather than guessing keywords from one example.
  Use `model.stiffness(...)`,
  `model.external_force(...)`, and `operators.combine(...)` when an example
  must expose individual contributions.
- For nonlinear paths, use `steps.automatic(...)` as the normal step control.
  Treat `max_increments` as an accepted-increment ceiling and solver
  `maximum_iterations` as the Newton-iteration ceiling for one attempt. Prefer
  `solvers.newton(...)` over backend-specific solver classes. Use
  `results.output_plan(...)` to combine field, history, diagnostic, and
  presentation requests; do not use result frames as solve controls.
- Keep `Study` and `SolutionProcedure` distinct. A dynamics Study may lower to
  Newmark, generalized-alpha, or central difference; do not encode the
  algorithm by changing the physical problem name.
- Prefer `studies.static_solid`, `studies.steady_heat_transfer`,
  `studies.transient_heat_transfer`, and `studies.dynamic_solid` for common
  cases. Attach `amplitudes` to loads, prescribed values, and supported
  boundary models so procedures update them automatically.
- Use `amplitudes.basis(...)` for multiple named loading modes. Preserve
  coefficient order, value/velocity/acceleration behavior, endpoint audit, and
  the content fingerprint. Anonymous callables remain valid but are not a
  frozen scientific input record.
- Keep operator notation such as `K = operators.stiffness(...)`,
  `F = operators.load_vector(...)`, and
  `step = problems.linear_static(K, F, study=..., ...)` available for
  transparent research/debugging examples.
- Prefer `step = problems.first_order_transient(...)` for first-order transient
  workflows instead of hand-combining effective matrices in tutorial code.
- Put local response relations under `constitutive/`.
- Query `constitutive.capabilities()` before using nonlinear materials. State
  whether a law is FEM-integrated, material-point verified, or a
  postprocessor; never infer a global solver from a material-point update.
- Stateful materials must use quadrature-owned committed/trial state and prove
  rollback plus restart equivalence. The current global J2 route has declared
  serial/MPI structural evidence; global Arrhenius power-law creep remains a
  3D small-strain foundation with a stricter MPI maturity boundary. Other creep
  laws remain material-point or assessment consumers.
- Prefer sequential heat-transfer then thermal-stress analysis when coupling
  is one way. Do not claim fully coupled thermo-mechanics unless temperature
  and mechanics are solved in one consistent nonlinear system.
- For an evolving one-way thermal input, call
  `temperature_history = heat_step.capture_history(name="temperature", unit="K")`
  before solving and pass that object to the receiving Arrhenius creep Step.
  Treat its coordinate as physical time, not an output frame or increment
  number. Keep interpolation and out-of-range behavior explicit; cutback must
  restore the temperature to the accepted start time. Use `save(...)` and
  `FieldHistory.load(...)` when the handoff crosses runs: nodal archives use a
  physical-DOF identity and are portable across MPI partition counts. The
  compact archive is root-gathered, so do not present it as an extreme-scale
  parallel field database.
- Use `materials.temperature_property(...)` and
  `constitutive.temperature_dependent_thermoelastic(...)` for tabulated
  sequential properties. Do not hide interpolation in an anonymous callback
  or silently extrapolate outside laboratory data. A field-valued UFL
  coefficient requires an explicit bounded extrapolation policy.
- If conductivity or specific heat is tabulated, keep the ordinary
  `model.step(target=temperature, dt=..., steps=...)` call. AgentFEM lowers it
  to a conservative enthalpy-based nonlinear heat step. Do not also pass
  user-defined `C=` or `K=` operators, and retain its convergence, heat-ledger,
  rollback, and checkpoint evidence.
- Put weak boundary physics under `boundary_models/`.
- Put Dirichlet, periodic, and MPC relations under `constraints`.
- Put Neumann, traction, flux, and body sources under `loads`.
- Treat positive `loads.pressure(...)` as inward. State whether it is a
  reference dead load or current follower load.
- For imported physical surfaces, create
  `mesh.tagged_boundary_region(domain, facet_tags, tag=...)`. Use that same
  region for strong constraints and weak loads; do not reconstruct a physical
  group with a broad coordinate marker. If an independent marker is supplied,
  run `model.audit_boundaries(strict=True)` before solving.
- Validate small cases before reporting success.
- Use `solve_result()` and `solve_result(output="results.xdmf")` for static or
  transient outputs that feed
  visualization, reporting, campaigns, or datasets. Do not treat XDMF/CSV as
  the scientific result itself.
- After publishing, run `agentfem verify <result.json> --json` when results are
  copied, reused, or admitted to a campaign. This checks manifest/artifact
  integrity, not convergence or scientific validation.
- Before a frozen or blind campaign, use
  `provenance.freeze_runtime(...)`; require it on continuation with
  `provenance.require_runtime(...)`. Review an intentional mismatch rather
  than editing the stored lock.
- Declare source meshes as `Path` values and reusable materials, loading,
  procedures, and observers through `Campaign(scientific_inputs=...)`. Reject
  claims of reproducibility when fingerprint coverage is incomplete unless the
  opaque input has been reviewed and recorded.
- For an individual result, call `result.add_scientific_inputs(...)` before
  `write_manifest(...)`; pass file assets as `Path` so their bytes are hashed.
- Use `events.first_passage(...)` for threshold timing and retain its bracket,
  localization, and censoring. Refine discontinuous damage/contact events
  instead of presenting interpolation as exact.
- For heat, Standard dynamics, and Explicit dynamics, pause with
  `run(until_step=...)`, save with `save_checkpoint(...)`, rebuild the same
  step, then `load_checkpoint(...)`. Resume through
  `solve_result(output=...)`; inspect `metadata["transient"]["output_scope"]`
  before presenting the XDMF/HDF5 series as complete. Current transient
  checkpoints require the same MPI size and mesh partition.
- Use `results.region_integral`, `results.region_average`,
  `results.region_measure`,
  `results.boundary_resultant`, and `results.field_extrema` for common
  MPI-safe quantities instead of case-owned UFL for standard measures or
  rank-local array reductions. Use `field_extrema(..., location=True)` when a
  reported peak needs coordinates and DG0 cell identity.
- Use `results.probe`, `results.sample_points`, and `results.sample_path` for
  physical-coordinate field sampling. Every MPI rank must request identical
  coordinates. Put discontinuous-field probes inside the intended cell rather
  than relying on an interface-side convention.
- Use `expressions.expression`, `expressions.as_ufl`, and
  `expressions.interpolate` for formulas supplied through JSON, benchmark
  specifications, or agent-authored configuration. Never execute those
  formulas with Python `eval`; reject syntax outside the reviewed mathematical
  language. Lower symbolic physics with `as_ufl`; use `interpolate` for known
  loads, coefficients, initial values, and boundary data so cases reuse stable
  finite-element forms instead of recompiling each formula.
- Use `results.sample_rectilinear_grid` when an external protocol requires a
  direct mesh-independent array. Preserve its `inside` mask and declared axis
  order; do not infer a transpose from array dimensions.
- For PDEAgent-Bench, use the commit-pinned
  `integrations.pdeagent_bench` contract and fixed solver entry point. Consume
  only the public agent-view case, never case identifiers, oracle numerics, or
  manufactured solutions. Read `SUPPORTED_FAMILIES` before selecting a case.
  Preserve mixed-field, pressure-reference, periodic-MPC, and fourth-order
  closure evidence reported in `solver_info`; these are scientific decisions,
  not disposable backend details.
  Report fixed-adapter capability separately from an AI-agent A/B experiment
  and from any official library-track submission.
- Model-generated static elasticity returns projected `S/E/MISES` fields
  automatically; request `SENER` explicitly when needed. Use
  `results.small_strain_partition_fields` for an
  explicit regional material projection. Use
  `results.reaction_resultant(..., on=..., component=...)` only for named
  strong Dirichlet reactions;
  do not reuse that definition for affine MPC, weak, or contact constraints.
- For FNO-style data, use one reviewed `surrogates.ObservationGrid` across the
  campaign and `datasets.fem_observation_sample(..., outside="mask")`. Keep the
  exported axes, order, components, units, and mask with every sample.
- Use `checkpointing.every(...)` through the public Step for automatic
  transient checkpoints. Cadence follows accepted increments, not output or
  progress frames.
- In an installed project, run `agentfem doctor --json`,
  `agentfem check --json`, and `agentfem upgrade --json` before execution.
  If `[extensions].required` is declared, inspect it with
  `agentfem extensions --json`; do not install or activate an untrusted package
  merely to make a project check pass. Required extensions are executable code.
  Never apply a `semantic_review=true` migration without inspecting regions,
  loads, constraints, materials, forms, output meaning, and verification. Use
  `project.current_run()` and
  publish the result so terminal, GUI, and agent consumers receive the same
  run identity, artifacts, and manifest.
- Treat `agentfem run --json` as the machine boundary. Read the versioned
  execution and result records; do not infer success by matching console text.
- Use `learning` as the public umbrella but preserve exact roles. A surrogate,
  neural operator, neural-field solver, and learned constitutive model are not
  interchangeable. PINN/DEM/XDEM providers must consume explicit objectives,
  conditions, sampling, and evidence contracts; do not hide them behind a
  generic AI model label.
- For a user-owned PINN, DEM, XDEM, or other neural-field implementation, use
  `model.step(target=spec, executor=...)`. The executor receives one immutable
  `learning.NeuralFieldExecutionRequest` and must return `SimulationResult`.
  Do not require an official companion package, inherit an AgentFEM neural
  model base class, or put live executable objects in result provenance.
- Inspect external meshes before conversion, retain the conversion manifest,
  and do not call mesh conversion a full Abaqus/ANSYS model import.
- Read an Abaqus `C3D10H` source directly with `mesh.read_abaqus_mesh(...)`,
  inspect `cell.element_definitions`, and consume it only through the verified
  P2-displacement/DG0-pressure mixed route. Do not derive another mesh when the
  source already declares `C3D10H`, and do not treat neutral `tetra10` geometry
  as sufficient formulation evidence.
- When Abaqus constraints use node labels, preserve labels separately from
  backend dof ordering. Treat `*EQUATION` as an auditable constraint graph;
  reject duplicate slaves, cycles, missing node matches, and unsupported
  parallel ownership rather than silently weakening periodicity.
- For finite-deformation periodic cells, report the macro deformation
  gradient, every load increment's convergence, equation mismatch, sampled
  `det(F)` bounds, complete homogenized tensor histories, and deformed geometry
  at scale one. Read the actual macro gradient from solved reference-point
  motion when any macro component is free; do not replace it with the nominal
  loading predictor. A three-dimensional uniaxial-stress periodic cell leaves
  both transverse normal macro components free and suppresses macro shear;
  do not fix one transverse stretch to unity.
- Prefer `solve_result(output=...)` for serial multi-field static output and
  the unified XDMF/HDF5 backend for finite-strain time series. Verify
  shared topology, retained reference coordinates, deformed geometry, time
  values, and all point/cell attributes. The low-level `io.XDMFTimeSeries`
  follows DOLFINx and can expose one Grid per Function. Use PVD only for a
  consumer that specifically requires it. Treat
  `completed_with_output_errors` as a completed solve with failed optional
  output, not as a lost numerical result.
- Do not claim that experimental AF-IR records make arbitrary UFL workflows
  backend neutral.
- For parameter collections, build a fresh case per sample, retain case/run
  evidence, split validation data independently, and guard learned-model
  applicability. Do not silently extrapolate or imply that neural-operator and
  PINN contract records are executable trainers.
- Use `responses.finite_difference(...)` for campaign-backed baseline and
  perturbation cases. Declare absolute or relative steps and output Quantity
  contracts. A failed perturbation makes the response incomplete; do not fill
  a missing Jacobian column with zeros.
- A campaign evaluator may return a declared mapping, `CaseOutcome`, or
  `SimulationResult`. Prefer a safe JSON campaign specification for parameter
  and sampling policy when non-programmers or agents need to edit a sweep;
  keep trusted model construction in Python.
- Use `campaigns.local_processes(workers=...)` for independent local cases. It
  uses spawned processes; do not nest it inside within-case MPI or replace it
  with threads. Use plan shards for separate MPI jobs and schedulers.
- Use `convergence.audit(...)` only with explicit refinement axes, fixed
  coordinates for every other varying parameter, and declared relative,
  absolute, or exact observable policies. Preserve failed, missing, and
  ambiguous sequences as inconclusive evidence.

## When Extending AgentFEM

Before adding a public helper, read `references/extension_rules.md`. If the new
helper is application-specific, keep it in the application package.
Private material libraries and domain products belong in separate packages
using the versioned `agentfem.extensions` entry point, not in a long-lived
private branch of the open core.
