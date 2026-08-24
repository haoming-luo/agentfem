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
  structural evidence, while native axisymmetric creep now reproduces NAFEMS
  R0027 Test 7. An independent distributed creep component benchmark remains
  distinct from the existing MPI regional patch and restart tests;
- scalar, finite-element or accepted-history temperature input consumed at
  creep integration points by the normalized Arrhenius rate, shared
  E(T)/nu(T)/alpha(T), thermal strain and consistent tangent;
- natural-load work, prescribed-motion work, elastic energy, creep
  dissipation, internal energy and mechanical residual histories;
- Kachanov–Rabotnov, Sinh, and related relations at explicitly marked
  material-point/evaluation maturity unless a global consumer is present;
- modified-theta projection as a curve-assessment tool;
- source-identified creep time fractions and explicit creep--fatigue
  interaction diagrams as an engineering postprocessor; declared dwell
  intervals may be extracted from named stress and temperature histories;
- small-strain J2 plasticity through a stateful global route, including an
  experimental multi-backstress Chaboche combined-hardening material using
  the same Step, quadrature transaction, output and restart lifecycle.

The maturity label is part of the interface: the presence of a formula does not
silently imply a validated global analysis procedure.

For cyclic plasticity, `constitutive.chaboche(...)` accepts one or more
`(C_i, gamma_i)` pairs plus optional exponential isotropic saturation. It
publishes `S`, `PE`, `PEEQ`, total backstress `ALPHA`, and `MISES`. Its local
consistency, reversal response, global cycle and restart are automated; a
structure-level stabilized-hysteresis comparison and complete recovery-
dissipation energy closure remain promotion gates.

## Go deeper

- [Thermal stress and creep procedures](../solution_procedures_and_thermal_creep.md)
- [Nonlinear materials](../nonlinear_materials.md)
- [Scientific trust and verification](../scientific_verification.md)
- [Implicit-creep example](../examples/index.md#implicit-creep-relaxation)
