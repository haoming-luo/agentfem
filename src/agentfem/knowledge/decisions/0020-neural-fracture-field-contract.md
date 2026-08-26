# Decision 0020: method-neutral fracture assets before a general XDEM provider

## Status

Accepted foundation; general Mode I/II XDEM remains experimental future work.

## Decision

AgentFEM core owns predefined crack geometry, stable crack/tip identity,
provider-neutral field access, stress-intensity evidence, independent
integration identity, and result trust semantics.

AgentFEM-Learning owns PyTorch modules, discontinuous and Williams-enriched
representations, optimizers, device policy, checkpointing, and warm starts.
Project repositories own particular crack layouts, interaction coefficients,
material parameters, and publication figures.

The public core vocabulary uses `CrackSegment2D`, `CrackSet2D`,
`FractureField2D`, `WilliamsField2D`, `InteractionIntegralSamples2D`,
`StressIntensityReport`, and `IntegrationPlan`. It does not name a core solver
XDEM and does not accept a requested Mode I/II label as an answer. The mode is
computed from the declared geometry and loading.

## Consequences

- The Williams Mode III provider remains a packaging and scientific-evidence
  regression.
- It records crack geometry, paired one-sided trace samples, independent
  validation integration, and a refinement integral.
- The interaction-integral reducer and its mixed-mode analytical Williams
  Golden tests are independent of the neural solver. Ordinary FEM benchmark
  evidence remains the next admission gate for the FEM adapter.
- Multi-crack XDEM enters only after the single-crack vector-elastic provider,
  SIF extraction, and unsupported-geometry failures are verified.
