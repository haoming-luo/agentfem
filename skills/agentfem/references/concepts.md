# Concepts Reference

- Study: early analysis context containing analysis type, physics, dimension,
  and modeling assumptions.
- Model: lightweight registry for mesh, regions, fields, amplitudes, materials,
  constraints, loads, and checks. It is not a solver.
- Model helpers: `model.field`, `model.material`, `model.fix`, and
  `model.traction` register assets without hiding operator construction.
- Mesh summary: dimension, local/global entity counts, and cell/facet tag
  availability.
- Mesh region: named geometric location where constraints, loads, or material
  data are applied. Boundary regions provide `ds(tag)`; cell/material regions
  provide `dx(tag)`.
- Unknown field: application-level bundle containing space, solution, trial, and
  test objects.
- Field algebra: same-space AgentFEM fields support eager arithmetic such as
  `u + dt * v`, returning a numerical field rather than a symbolic weak form.
- Amplitude: named time history or scale factor used to drive prescribed data;
  it is not a spatial finite-element field.
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
- Analysis step: solve stage under a study, such as linear static or implicit
  Euler, built from visible operators.
- Problem: discrete algebraic or transient system, such as `K x = F`,
  `C xdot + K x = F`, or `M a + C v + K u = F`.
- Operator algebra: use `model.stiffness(...)` for registered material regions
  and `operators.combine(...)` when individual contributions must be explicit.
- State: fields grouped for a solver or time integrator.
- Explicit dynamics integrator: time integrator such as
  `time.explicit.central_difference(...)`; central difference is Newmark with
  beta=0 and gamma=1/2.
- Diagnostic: quantity used to inspect correctness, stability, or physics.
- Benchmark: verification case with expected quantities and tolerances.

Never treat Neumann data as a Dirichlet constraint.
