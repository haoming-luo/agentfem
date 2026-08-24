# Creep and inelasticity

Time-dependent and path-dependent materials require more than a rate formula.
AgentFEM treats committed/trial state, consistent linearization, cutback,
rollback, restart, output, and benchmark evidence as part of the material's
global finite-element contract.

## Current boundary

- 3D global power-law creep with regional materials, backward Euler, shared
  quadrature state, analytical tangent, physical-time cutback, creep fields,
  dissipation, and portable full-Step restart;
- MPI-safe regional quadrature updates and rank-count-portable J2/creep
  full-Step archives; J2 distributed global Newton has public thick-cylinder
  structural evidence, while the creep route retains a NAFEMS structural
  promotion gate;
- scalar, finite-element or accepted-history temperature input consumed at
  creep integration points by the normalized Arrhenius rate, shared
  E(T)/nu(T)/alpha(T), thermal strain and consistent tangent;
- natural-load work, prescribed-motion work, elastic energy, creep
  dissipation, internal energy and mechanical residual histories;
- Kachanov–Rabotnov, Sinh, and related relations at explicitly marked
  material-point/evaluation maturity unless a global consumer is present;
- modified-theta projection as a curve-assessment tool;
- source-identified creep time fractions and explicit creep--fatigue
  interaction diagrams as an engineering postprocessor;
- small-strain J2 plasticity through a stateful global route.

The maturity label is part of the interface: the presence of a formula does not
silently imply a validated global analysis procedure.

## Go deeper

- [Thermal stress and creep procedures](../solution_procedures_and_thermal_creep.md)
- [Nonlinear materials](../nonlinear_materials.md)
- [Scientific trust and verification](../scientific_verification.md)
- [Implicit-creep example](../examples/index.md#implicit-creep-relaxation)
