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
- Avoid hidden global state.

## Error Messages

Raise errors that explain the modeling issue, not only the Python issue. For
example, say that a normal vector is required for normal/tangential absorbing
boundaries.
