# AgentFEM Skill

Use this skill when building, reviewing, or extending finite-element simulations
with AgentFEM.

## Required Reading Order

When this file is installed as a Codex skill, the reference paths below are
relative to the skill directory `skills/agentfem/`. In the generated
documentation site, use the left navigation pages `Workflow`, `Concepts`, and
`Module Map` instead.

1. Read `references/workflow.md`.
2. Read `references/concepts.md`.
3. Read `references/module_map.md`.
4. Read focused references only as needed.

## Rules

- Keep the finite-element workflow visible.
- Identify the study context before selecting constitutive laws or operators.
- Inspect mesh summaries and required tags before building weak forms.
- Use lightweight models for registry, checks, summaries, and model-owned
  analysis steps; do not hide K/F systems or finite-element meaning.
- Prefer model registration helpers such as `model.field(...)`,
  `model.material(...)`, `model.fix(...)`, `model.symmetry(...)`,
  `model.traction(...)`, and `model.pressure(...)` for application examples.
- Use `FEMProblem.summary()` or equivalent structured summaries when auditing a
  workflow.
- Use `model.validate()` for addressable issue reports, `model.check()` before
  execution, and `model.write_ir(...)` when a persistent AF-IR record is part
  of the task.
- Treat `model.check()` and `models.step_capability(model)` as the executable
  Study/provider preflight. Do not advertise or lower a combination that no
  registered provider accepts.
- Use AgentFEM modules before writing ad hoc DOLFINx/PETSc boilerplate.
- Prefer the stable public workflow modules first: `studies`, `mesh`, `models`, `fields`,
  `materials`, `constitutive`, `constraints`, `loads`, `operators`,
  `problems`, `procedures`, `results`, `solvers`, `steps`, `time`, `io`, `campaigns`, `datasets`, and
  `surrogates`.
- Treat `ir` and `validation` as public inspection/record interfaces. Treat
  `backends` as an advanced extension boundary. FEniCSx is the only production
  backend in the current release.
- Prefer `step = model.step(target=u)` when a model has a Study and registered
  materials, constraints, and loads. New analysis/material families belong in
  a registered step provider; do not add one public model method per material.
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
  rollback plus restart equivalence. The current global J2 route is 3D small
  strain; Arrhenius creep remains material-point only.
- Prefer sequential heat-transfer then thermal-stress analysis when coupling
  is one way. Do not claim fully coupled thermo-mechanics unless temperature
  and mechanics are solved in one consistent nonlinear system.
- Put weak boundary physics under `boundary_models/`.
- Put Dirichlet, periodic, and MPC relations under `constraints`.
- Put Neumann, traction, flux, and body sources under `loads`.
- Treat positive `loads.pressure(...)` as inward. State whether it is a
  reference dead load or current follower load.
- Validate small cases before reporting success.
- Use `solve_result()` and `results.SimulationResult` when outputs feed
  visualization, reporting, campaigns, or datasets. Do not treat XDMF/CSV as
  the scientific result itself.
- For heat, Standard dynamics, and Explicit dynamics, pause with
  `run(until_step=...)`, save with `save_checkpoint(...)`, rebuild the same
  step, then `load_checkpoint(...)`. Resume through
  `solve_result(output=...)`; inspect `metadata["transient"]["output_scope"]`
  before presenting the XDMF/HDF5 series as complete. Current transient
  checkpoints require the same MPI size and mesh partition.
- Use `results.region_integral`, `results.region_average`,
  `results.boundary_resultant`, and `results.field_extrema` for common
  MPI-safe quantities instead of rank-local array reductions.
- Use `results.probe`, `results.sample_points`, and `results.sample_path` for
  physical-coordinate field sampling. Every MPI rank must request identical
  coordinates. Put discontinuous-field probes inside the intended cell rather
  than relying on an interface-side convention.
- Use `results.small_strain_cell_fields` for projected standard small-strain
  output. Use `results.reaction_resultant` only for strong Dirichlet reactions;
  do not reuse that definition for affine MPC, weak, or contact constraints.
- For FNO-style data, use one reviewed `surrogates.ObservationGrid` across the
  campaign and `datasets.fem_observation_sample(..., outside="mask")`. Keep the
  exported axes, order, components, units, and mask with every sample.
- Use `checkpointing.every(...)` through the public Step for automatic
  transient checkpoints. Cadence follows accepted increments, not output or
  progress frames.
- In an installed project, run `agentfem doctor --json` and
  `agentfem check --json` before execution. Use `project.current_run()` and
  publish the result so terminal, GUI, and agent consumers receive the same
  run identity, artifacts, and manifest.
- Treat `agentfem run --json` as the machine boundary. Read the versioned
  execution and result records; do not infer success by matching console text.
- Inspect external meshes before conversion, retain the conversion manifest,
  and do not call mesh conversion a full Abaqus/ANSYS model import.
- When Abaqus constraints use node labels, preserve labels separately from
  backend dof ordering. Treat `*EQUATION` as an auditable constraint graph;
  reject duplicate slaves, cycles, missing node matches, and unsupported
  parallel ownership rather than silently weakening periodicity.
- For finite-deformation periodic cells, report the macro deformation
  gradient, every load increment's convergence, equation mismatch, sampled
  `det(F)` bounds, complete homogenized tensor histories, and deformed geometry
  at scale one. State explicitly when a user material or user MPC is
  unavailable and has been substituted.
- Prefer the unified XDMF/HDF5 backend for finite-strain time series. Verify
  shared topology, retained reference coordinates, deformed geometry, time
  values, and all point/cell attributes. Use PVD only for a consumer that
  specifically requires it.
- Do not claim that experimental AF-IR records make arbitrary UFL workflows
  backend neutral.
- For parameter collections, build a fresh case per sample, retain case/run
  evidence, split validation data independently, and guard learned-model
  applicability. Do not silently extrapolate or imply that neural-operator and
  PINN contract records are executable trainers.
- A campaign evaluator may return a declared mapping, `CaseOutcome`, or
  `SimulationResult`. Prefer a safe JSON campaign specification for parameter
  and sampling policy when non-programmers or agents need to edit a sweep;
  keep trusted model construction in Python.

## When Extending AgentFEM

Before adding a public helper, read `references/extension_rules.md`. If the new
helper is application-specific, keep it in the application package.
