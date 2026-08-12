# Decision 0011: Model complete interface kinematics before case constraints

## Decision

A zero-thickness interface is a vector mechanical object. AgentFEM stores and
assembles the complete displacement jump, normal/tangential decomposition,
traction vector and algorithmic tangent. Normal-only slip, strict tangential
penalty, normal-damage-driven shear transfer, and mixed-mode damage are
explicit interface modes behind one paired-facet and MPI contract.

The recommended scalar Mode-I consumer uses a non-degrading tangential penalty
tie. A scalar normal law contains neither shear strength nor Mode-II fracture
energy, so silently deriving shear failure from normal damage would invent an
undeclared constitutive assumption. The legacy normal-only consumer remains
available as intentional frictionless slip, and normal-damage-driven shear
release remains an explicit approximation. Significant shear failure uses the
mixed-mode law.

Mixed-mode initiation uses a quadratic nominal-traction interaction. Energy
evolution selects either BK or power-law interaction. Compression contact is
separate from tensile/shear damage; optional friction is labeled as a smooth
regularization. Post-peak arc length is a solution procedure consuming the
same residual and transaction, not a material option.

## Why

Duplicating cohesive nodes disconnects the bulk graph. A normal-only interface
can therefore leave entire bodies free to translate or rotate tangentially.
Fixing one arbitrary point may suppress a singular solve but changes the
mechanics and does not restore interface shear transfer. The missing physics
belongs in the reusable interface contract and model preflight.

## Consequences

- Standard outputs are `JUMP_N`, `JUMP_T`, `TRACTION_N`, `TRACTION_T`,
  `DAMAGE`, and `MODE_MIXITY`.
- A split-topology rigid-mode audit measures exact translation/rotation rank
  from declared supports and intact interface modes before DOLFINx execution.
- A runtime Mode-I audit can reject excessive tangential jump.
- Scalar cyclic cohesive fatigue remains normal-opening fatigue; using a
  shear-carrying kinematic mode does not relabel its calibration mixed-mode.
- Mixed-mode state follows physical facet and quadrature identity through
  serial/MPI restart.
- General non-proportional mixed-mode fatigue, contact active sets, free crack
  paths, and external structural validation remain separate promotion gates.
