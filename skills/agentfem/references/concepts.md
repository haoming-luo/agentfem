# Concepts Reference

- Mesh summary: dimension, local/global entity counts, and cell/facet tag
  availability.
- Mesh region: named geometric location where constraints, loads, or material
  data are applied.
- Unknown field: application-level bundle containing space, solution, trial, and
  test objects.
- Constraint: essential or algebraic dof restriction.
- Fixed constraint: application-level fixed-value Dirichlet condition on a
  target unknown and geometric region. Vector unknowns default to all
  components; selected components are explicit.
- Load: weak right-hand-side source term.
- Boundary model: weak boundary physics such as Robin, impedance, or absorbing
  behavior.
- Constitutive law: local response relation mapping state to stress, flux, or
  tangent quantities.
- Material record: SI-unit constants plus model name and source note. It is data,
  not the equation itself.
- Material properties: typed parameter object used by constitutive relations.
- Problem: inspectable description of mesh, spaces, fields, materials,
  constraints, loads, boundary models, and forms.
- State: fields grouped for a solver or time integrator.
- Diagnostic: quantity used to inspect correctness, stability, or physics.
- Benchmark: verification case with expected quantities and tolerances.

Never treat Neumann data as a Dirichlet constraint.
