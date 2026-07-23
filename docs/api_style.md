# AgentFEM API Style

AgentFEM APIs should be readable by finite-element researchers and predictable
for AI agents.

## Naming

- Use nouns for assets: `Amplitude`, `LoadSet`, `ConstraintSet`,
  `TransientState`.
- Use verbs for actions: `assemble_vector`, `copy_function`, `solve_matrix_system`.
- Use explicit qualifiers when ambiguity matters: `DirichletConstraint`,
  `NeumannLoad`, `ViscousAbsorbingBoundary`.

## Inputs

- Prefer explicit function-space arguments named `V`.
- Use `measure` for UFL integration measures.
- Use `test_function` and `trial_function` rather than single-letter names in
  public APIs.
- Use `comm`, `model_rank`, and `gdim` explicitly for mesh import/read helpers.
- Keep optional file-format dependencies lazy. Import packages such as `meshio`
  inside conversion functions so core AgentFEM imports stay lightweight.

## Outputs

- Return DOLFINx/PETSc objects when the function is a low-level operation.
- Return dataclasses for reusable FEM assets.
- Give reusable dataclasses a `summary()` method when the object represents a
  finite-element concept that humans or agents may audit.
- Give material-like dataclasses an `as_dict()` method when constants and
  derived quantities are useful for logs or validation.
- Avoid hidden global state.

## Semantic Constructors

- Provide readable constructors for common concepts, such as
  `constraints.dirichlet(...)`, `loads.neumann(...)`, and
  `loads.body_load(...)`.
- Provide application-level constructors when they reduce boilerplate, such as
  `constraints.fixed(...)` for fixed-value Dirichlet conditions.
- Field arithmetic should be eager and explicit for same-space field algebra:
  `field_a + scalar * field_b` returns a numerical AgentFEM field, not a hidden
  symbolic solve.
- Use `amplitudes` for reusable time histories and scale factors that can
  drive loads, constraints, sources, or prescribed data.
- Hide backend scalar-type details inside concept constructors such as
  `loads.traction(...)`, `constraints.fixed(...)`, and material-property
  loaders.
- Application examples should not call `PETSc.ScalarType` directly. Use
  `kernel.constants.scalar_value(...)` in low-level helpers and
  `amplitudes.Amplitude` with
  `constraints.time_dependent_component_dirichlet(...)` for time-dependent
  boundary data.
- For vector fields, application-level constraint constructors should make the
  common case short: `constraints.fixed(displacement, on=left)` fixes all
  displacement components, while `components=0` or `components=(0, 1)` selects
  individual dofs.
- Prefer `on=...` for geometric targets in user-facing APIs. Keep `location=...`
  as a compatibility spelling for explicit region objects.
- Provide model-level registration helpers such as `model.field(...)`,
  `model.fix(...)`, and `model.traction(...)` when they keep the workflow
  readable without hiding finite-element meaning.
- Provide model-level operator helpers such as `model.stiffness(...)`,
  `model.internal_force(...)`, and `model.external_force(...)` for daily
  application scripts. These helpers should delegate to `operators` and
  preserve inspectable operator summaries.
- Prefer model-owned step helpers such as `model.linear_static_step(...)` for
  beginner workflows. The step should expose its K/F system through summaries
  so it remains auditable rather than becoming a black box.
- Provide operator-level constructors for engineering FEM notation, such as
  `operators.stiffness(...)`, `operators.capacity_operator(...)`,
  `operators.conduction_operator(...)`, and `operators.load_vector(...)`.
- Provide contribution-combination helpers such as `operators.combine(...)` for
  explicit operator algebra. The `+` operator may be convenient shorthand, but
  public examples should prefer `combine(...)` when names and summaries matter.
- Represent weak boundary-model residual terms with operator helpers such as
  `operators.boundary_model_vector(...)` instead of writing UFL forms in
  application examples.
- Keep explicit family constructors such as `operators.elastic_stiffness(...)`
  available when ambiguity would be harmful.
- Prefer problem-level constructors such as
  `problems.linear_static(K, F, study=..., unknown=..., constraints=...)` and
  `problems.first_order_transient(capacity=C, stiffness=K, history=..., dt=...)`
  when an example is intentionally teaching explicit operator notation.
- Keep lower-level dataclasses available for advanced users who need direct
  control.
- Use concept names in errors and summaries: constraint, load, boundary model,
  constitutive law, operator, state, and diagnostic.

## Error Messages

Raise errors that explain the modeling issue, not only the Python issue. For
example, say that a normal vector is required for normal/tangential absorbing
boundaries.
