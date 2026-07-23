# Concepts Reference

- Constraint: essential or algebraic dof restriction.
- Load: weak right-hand-side source term.
- Boundary model: weak boundary physics such as Robin, impedance, or absorbing
  behavior.
- Constitutive law: material response mapping local state to stress, flux, or
  tangent quantities.
- State: fields grouped for a solver or time integrator.
- Diagnostic: quantity used to inspect correctness, stability, or physics.

Never treat Neumann data as a Dirichlet constraint.
