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
- Keep optional file-format dependencies lazy. Import packages such as
  `meshio` and Gmsh only at the capability boundary, and raise an actionable
  `OptionalDependencyError` naming the relevant AgentFEM extra.

## Outputs

- Return DOLFINx/PETSc objects when the function is a low-level operation.
- Return dataclasses for reusable FEM assets.
- Give reusable dataclasses a `summary()` method when the object represents a
  finite-element concept that humans or agents may audit.
- Give material-like dataclasses an `as_dict()` method when constants and
  derived quantities are useful for logs or validation.
- Use `to_ir()` for a versioned scientific record. Do not use `repr()` as a
  persistent fallback for backend objects because it may contain memory
  addresses and no scientific semantics.
- Validation failures intended for repair should carry a stable code and an
  object path in a `ValidationIssue`.
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
  `loads.traction(...)`, `loads.pressure(...)`, `constraints.fixed(...)`,
  `constraints.symmetry(...)`, and material-property
  loaders.
- Application examples should not call `PETSc.ScalarType` directly. Use
  `kernel.constants.scalar_value(...)` in low-level helpers and
  `amplitudes.Amplitude` with
  `constraints.time_dependent_component_dirichlet(...)` for time-dependent
  boundary data.
- For vector fields, application-level constraint constructors should make the
  common case short: `constraints.fixed(displacement, on=left)` fixes all
  displacement components, while `components=0` or `components=(0, 1)` selects
  individual dofs. Engineering axis names (`"x"`, `"y"`, `"z"`) are accepted
  wherever a public component selector is unambiguous.
- Prefer `on=...` for geometric targets in user-facing APIs. Keep `location=...`
  as a compatibility spelling for explicit region objects.
- Provide model-level registration helpers such as `model.field(...)`,
  `model.fix(...)`, `model.symmetry(...)`, `model.pressure(...)`, and
  `model.traction(...)` when they keep the workflow
  readable without hiding finite-element meaning.
- Provide model-level operator helpers such as `model.stiffness(...)`,
  `model.internal_force(...)`, and `model.external_force(...)` for daily
  application scripts. These helpers should delegate to `operators` and
  preserve inspectable operator summaries.
- Prefer the model-owned `model.step(...)` procedure-dispatch entry point for
  beginner workflows. The step should expose its K/F system through summaries
  so it remains auditable rather than becoming a black box.
- Every built-in Step provider must publish a `StepOptionContract`. The
  contract is the single source for accepted and required keyword names,
  pre-assembly typo detection, CLI capability JSON, and future IDE/GUI forms.
  Third-party providers without a contract remain supported during 0.2.x, but
  a provider is not considered workflow-ready until it declares one.
- Prefer `solve_result()` as the common completion verb. An output product may
  be supplied to `solve_result(output=...)` or declared once on
  `model.step(..., output=...)`; provider constructors must not reinterpret it
  as a numerical solver option.
- Provide operator-level constructors for engineering FEM notation, such as
  `operators.stiffness(...)`, `operators.capacity_operator(...)`,
  `operators.conduction_operator(...)`, and `operators.load_vector(...)`.
- Provide contribution-combination helpers such as `operators.combine(...)` for
  explicit operator algebra. The `+` operator may be convenient shorthand, but
  public examples should prefer `combine(...)` when names and summaries matter.
- Give matrix, vector, residual, and scalar operators explicit roles and check
  those roles against UFL form arity before assembly. Preserve the visible
  relation `K_t = dR/du` when UFL automatic differentiation linearizes a
  nonlinear residual.
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
- Prefer a repair address such as `model.materials[1].region` and a specific
  hint over a generic execution error.
- Parameter-study APIs must carry bounds, units, and deterministic case IDs.
  Prefer a fresh `build(parameters)` case factory over mutating one live backend
  model across samples.
- Learning outputs must be declared with names, shapes, and units before a
  campaign runs. Do not infer a scientific schema only from the first returned
  NumPy array.
- A surrogate prediction API should report its source, uncertainty semantics,
  applicability decision, and fallback behavior. It must not silently
  extrapolate or describe residual scale as calibrated epistemic uncertainty.

## Error Messages

Raise errors that explain the modeling issue, not only the Python issue. For
example, say that a normal vector is required for normal/tangential absorbing
boundaries.

## Model vocabulary and compatibility

Bundled application cases should begin with the physical factories
`studies.static_solid(...)`, `studies.steady_heat_transfer(...)`,
`studies.transient_heat_transfer(...)`, `studies.dynamic_solid(...)`, or
`studies.creep_solid(...)`. Generic analysis-order factories remain available
for new physics and expert formulation work, but ordinary examples should not
repeat `physics="solid_mechanics"` or `physics="heat_transfer"` when the
physical factory already states it.

The canonical application language is returned by `models.model_api("core")`.
It includes registration and engineering verbs such as `field`, `material`,
`clamp`, `traction`, `step`, and `check`. Operator construction, remote
coupling, machine records, and other deliberate escape hatches are disclosed
through `models.model_api("advanced")`.

Historical `add_*` registration spellings and material/procedure-specific
`*_step` factories remain executable throughout 0.2.x, but
`models.model_api("compatibility")` identifies them so documentation, agents,
IDEs, and future GUIs do not present them as parallel beginner workflows. New
examples must use:

```python
model.material(material)
step = model.step(target=unknown, output="results.xdmf")
result = step.solve_result()
```

Do not add another alias merely to save one word. A new public name must either
express a distinct engineering concept or replace an older name through an
explicit migration path.
