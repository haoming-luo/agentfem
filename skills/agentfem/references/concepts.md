# Concepts Reference

- Study: early analysis context containing analysis type, physics, dimension,
  and modeling assumptions.
- Axisymmetric solid: a two-dimensional `(r,z)` meridian whose displacement is
  `(u_r,u_z)`, whose stress/strain tensors retain `(r,theta,z)`, and whose
  physical integrals use the full-revolution `2*pi*r` measure.
- Solution procedure: Standard/Explicit family, equation order, integration
  algorithm, statefulness, and global-solve requirements. It describes how a
  Study is solved without redefining the physical problem.
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
- Amplitude basis: ordered serializable histories plus named coefficients,
  derivatives, endpoint audit, and content identity.
- Constraint: essential or algebraic dof restriction.
- Fixed constraint: application-level fixed-value Dirichlet condition on a
  target unknown and geometric region. Vector unknowns default to all
  components; selected components are explicit.
- Load: weak right-hand-side source term.
- Boundary model: weak boundary physics such as Robin, impedance, or absorbing
  behavior.
- Constitutive law: local response relation mapping state to stress, flux, or
  tangent quantities.
- Constitutive maturity: `fem_integrated`, `material_point_verified`, or
  `postprocessor`. Query it; do not infer it from the law name.
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
- Quadrature state: committed and trial history owned at integration points;
  rejected Newton iterates and cutbacks never mutate committed history.
- Implicit dynamics integrator: Newmark or generalized-alpha solution of a
  second-order system through an assembled effective operator.
- Explicit dynamics integrator: time integrator such as
  `time.explicit.central_difference(...)`; central difference is Newmark with
  beta=0 and gamma=1/2.
- Diagnostic: quantity used to inspect correctness, stability, or physics.
- Simulation result: named QoIs, fields, histories, metadata, scientific-input
  identity, and artifact links from one analysis; it bridges solves,
  campaigns, and datasets.
- Runtime lock: compatibility-relevant package, scalar, MPI, and
  source/distribution identity; paths remain evidence but not equality gates.
- Event observation: threshold crossing with bracket, localization rule, and
  observed/left-censored/right-censored status.
- Benchmark: verification case with expected quantities and tolerances.
- AF-IR document: experimental versioned scientific record; it is not yet a
  complete backend-neutral executable serialization.
- Validation issue: stable code, object path, severity, message, and repair
  hint for an addressable model finding.
- Backend adapter: explicit compilation boundary. FEniCSx is currently the only
  production backend.
- Parameter space: ordered typed simulation inputs with admissible values,
  bounds, units, and scale.
- Campaign: deterministic related cases with fresh construction, case IDs,
  output contracts, scientific-input fingerprints, serial/spawned/sharded
  execution evidence, failure records, and resume behavior.
- Scientific-input manifest: content identity for declared files, arrays,
  public scientific objects, and callable code/state; opaque coverage remains
  explicitly incomplete.
- Convergence certificate: explicit refinement-axis slices with fixed
  coordinates and relative, absolute, or exact observable policies.
- Response experiment: named output derivatives with respect to named
  parameters, first implemented as Campaign-backed finite differences.
- External mesh conversion: a topology/set conversion with a manifest, not a
  full commercial solver-deck import.
- Scientific dataset: numeric inputs/outputs plus units, shapes, field
  encodings, case identities, provenance, and artifacts.
- Surrogate: learned or reduced-order mapping whose scientific asset includes
  independent validation, applicability, and out-of-domain behavior.
- Neural operator: function-to-function model requiring explicit field,
  geometry, boundary, mesh, and projection encodings.
- Neural-field solver: per-problem field optimization through residual,
  variational/energy, data, and constraint objectives. PINN, DEM, and XDEM are
  methods in this family, not surrogate aliases. User-owned implementations
  can consume `NeuralFieldExecutionRequest` directly and return
  `SimulationResult`; an installed provider is optional.
- Physics-informed model: learning contract with explicit strong, weak, or
  discrete residuals and conditions; arbitrary UFL is not automatically a PINN
  residual.

Never treat Neumann data as a Dirichlet constraint.
