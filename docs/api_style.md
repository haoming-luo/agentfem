# AgentFEM API Style

AgentFEM APIs should be readable by finite-element researchers and predictable
for AI agents.

## Naming

- Use nouns for assets: `LoadSet`, `ConstraintSet`, `TransientState`.
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
- Hide backend scalar-type details inside concept constructors such as
  `loads.traction(...)`, `constraints.fixed(...)`, and material-property
  loaders.
- For vector fields, application-level constraint constructors should make the
  common case short: `constraints.fixed(displacement, location=left)` fixes all
  displacement components, while `components=0` or `components=(0, 1)` selects
  individual dofs.
- Provide operator-level constructors for engineering FEM notation, such as
  `operators.stiffness_operator(...)`, `operators.mass_operator(...)`, and
  `operators.boundary_load_vector(...)`.
- Keep lower-level dataclasses available for advanced users who need direct
  control.
- Use concept names in errors and summaries: constraint, load, boundary model,
  constitutive law, operator, state, and diagnostic.

## Error Messages

Raise errors that explain the modeling issue, not only the Python issue. For
example, say that a normal vector is required for normal/tangential absorbing
boundaries.
