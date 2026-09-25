# Decision 0012: Separate learning semantics from execution providers

## Decision

AgentFEM exposes `learning` as the public umbrella for scientific-learning
workflows while retaining exact roles for surrogates, neural operators,
neural-field solvers, and learned constitutive models. The released
`surrogates` module remains public throughout 0.2.x.

The open core owns field meaning, objective and condition semantics, sampling,
physical parameter identity, result evidence, and extension compatibility. It
does not absorb every neural-network architecture or training framework.

PINN, VPINN, Deep Ritz, DEM, and XDEM are treated as neural-field methods:
they optimize a field for one physical problem. A neural-field implementation
enters the normal `model.step(...)` lifecycle through either the
framework-neutral user-executor boundary or an installed Step provider and
returns `SimulationResult`. It is not registered as an operator-assembly
backend and is not treated as a surrogate.

Learned constitutive models use the same ownership rule at a material-point
boundary. AgentFEM owns named material parameters, state identity, Cauchy-
stress/small-strain tangent conventions, scalar and batch requests, atomic
trial/commit/rollback, applicability, artifact identity, and result evidence.
An activated extension owns executable model loading, tensor frameworks,
devices, automatic differentiation, and architecture-specific state codecs.
The core never names one learned architecture or assumes a fixed state size.

## Why

The stable asset is the scientific problem and its evidence, not PyTorch,
DeepXDE, PhysicsNeMo, XDEM, FNO, or a future architecture. Separating the
contract from the provider lets open adapters and private domain products use
the same model and result boundary without copying framework-specific details
into AgentFEM core.

Residual-based PINNs alone are too narrow for energy methods. A general
contract must represent residual, energy, data, and constraint objectives and
must keep physical coefficients separate from numerical loss weights.
Neural representations are declared separately so one network may own several
fields while discontinuity, signed-distance, Fourier, RBF, or crack-tip
enrichments remain visible rather than hidden in provider code.

## Consequences

- `learning.NeuralFieldSpec` is declarative until an installed provider binds
  its implementation identifiers, or a user supplies an explicit executor.
- User-owned models require no AgentFEM base class or companion package. They
  receive an immutable `NeuralFieldExecutionRequest` and return the common
  scientific result; live executable code is not serialized into provenance.
- Existing `PINNSpec` records can be lifted without breaking 0.2.x projects.
- XDEM belongs in the optional `agentfem-learning` companion under a narrow
  `neural_fields.xdem` domain with examples and independent verification, not
  as paper-specific core code or as a standalone gatekeeper.
- FNO and other neural operators continue to consume scientific datasets and
  field encodings; they do not share a PINN trainer merely because both use
  PyTorch.
- Confidential materials, calibration assets, workflow policy, and customer
  services remain in separate packages and repositories using the public
  extension protocol.
- A learned material is not promoted to implicit global execution merely
  because its material-point inference succeeds. Its fixed-old-state tangent,
  rollback, checkpoint/restart, MPI behavior, and structure-level response must
  pass joint provider/core evidence first.
