# Creep and inelasticity

Time-dependent and path-dependent materials require more than a rate formula.
AgentFEM treats committed/trial state, consistent linearization, cutback,
rollback, restart, output, and benchmark evidence as part of the material's
global finite-element contract.

## Current boundary

- 3D isothermal global power-law creep with backward Euler, shared quadrature
  state, analytical tangent, physical-time cutback, creep fields, dissipation,
  and serial restart;
- Kachanov–Rabotnov, Sinh, Arrhenius, and related relations at explicitly marked
  material-point/evaluation maturity unless a global consumer is present;
- modified-theta projection as a curve-assessment tool;
- small-strain J2 plasticity through a stateful global route.

The maturity label is part of the interface: the presence of a formula does not
silently imply a validated global analysis procedure.

## Go deeper

- [Thermal stress and creep procedures](../solution_procedures_and_thermal_creep.md)
- [Nonlinear materials](../nonlinear_materials.md)
- [Scientific trust and verification](../scientific_verification.md)
- [Example gallery](../examples/index.md#creep-and-time-dependent-solids)
