# 0019 — Preserve source meaning before native lowering

## Decision

Abaqus migration has three independent gates: source inventory, reviewed
native lowering, and executable constitutive integration. A declaration that
can be parsed is not automatically executable.

For one geometrically linear static Step, AgentFEM may lower a supported
relative tabular amplitude to its value at the declared Step end because the
native consumer solves only the final linear equilibrium state. The complete
table, time span, Step duration, reference magnitude, final multiplier, and
final magnitude remain in the decision record and generated source asset.

A UMAT or UHYPER Fortran file first enters a source-only inspection gate. The
report fingerprints the source and identifies entry points, includes, Abaqus
utilities, and project calls. It never claims compilation or execution.

## Consequences

- Intermediate Abaqus increments are not silently presented as reproduced.
- Nonlinear, history-dependent, multi-Step, absolute, or unsupported amplitude
  semantics continue to fail closed.
- Existing industrial material assets receive an addressable migration route
  without making the AgentFEM public model language depend on Abaqus.
- Compiled adapters must still pass material-path, tangent, one-element, and
  global FEM evidence gates before becoming executable providers.
