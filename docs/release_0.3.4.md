# AgentFEM 0.3.4

AgentFEM 0.3.4 strengthens the scientific boundary between an engineering
model and its numerical execution. The release closes exact reaction evidence
for declared constraints and weak boundaries, consolidates nonlinear path
control, and establishes a reproducible finite-strain J2 RVE foundation while
preserving explicit maturity limits.

## Physical reactions owned by their providers

- Rectangular multi-point constraints recover their multiplier from the owned
  slave residual and publish the corresponding physical nodal reaction,
  global resultant, constraint gap and virtual work. Serial and two-rank paths
  share the same mechanical contract.
- Conservative elastic foundations accept isotropic, normal-only or symmetric
  positive-semidefinite matrix stiffness. Their provider owns the nodal
  reaction, resultant and stored energy, so force, work and energy audits do
  not silently fall back to strong-Dirichlet assumptions.
- Provider reaction fields enter the same compact XDMF/HDF5 result dataset as
  the solved fields. Two-dimensional vector output is padded to three
  components for direct ParaView use without changing the internal FE field.
- Unsupported weak constraints or contacts still fail closed when they cannot
  supply a physical dual; metadata alone cannot mark an incomplete balance as
  verified.

## Nonlinear procedure and fracture execution

One shared monotonic-target controller now owns bounded target advancement,
cutback, rollback and progress records. Constitutive and fracture providers
retain ownership of trial and accepted state. The assembled DCB and ENF paths
reuse this execution policy instead of maintaining benchmark-specific stepping
loops.

This is an execution-architecture improvement, not a new claim of general
cohesive propagation. DCB/ENF evidence remains limited to the declared fixed
interfaces, material laws, discretizations and verification contracts.

## Finite-strain J2 RVE foundation

The experimental mixed finite-strain route combines multiplicative isochoric
J2 state, exact affine-periodic kinematics, mixed displacement/mean-Kirchhoff-
stress fields, transactional quadrature state and a condensed homogenized
algorithmic tangent. Checkpoint state is portable without exposing backend
mixed-vector layout.

A clean-source, content-bound multi-void RVE certificate compares 2, 4 and 8
load increments on one fixed mesh. For the accepted 4-to-8 comparison it
records relative changes of approximately:

| Quantity | Relative change |
| --- | ---: |
| Homogenized first Piola stress | 0.00121% |
| Mean equivalent plastic strain | 0.000993% |
| 95th-percentile equivalent plastic strain | 0.0751% |

The certificate binds the exact source commit, scientific inputs and runtime
fingerprint. It is load-path stability evidence for this declared problem; it
is not a mesh-convergence result or external validation for arbitrary porous
microstructures. The long installed-wheel path is opt-in CI evidence rather
than a cost imposed on every ordinary change.

## Modal and harmonic ownership

Modal and direct-harmonic workflows now bind the operators, constraints,
request and distributed resources actually executed. Structured region
averages and point probes synchronize local evaluation before reduction,
restart identity is rank-neutral, and rank-local failures terminate together.
The provider-neutral direct-harmonic workflow retains the NAFEMS R0016 Test 5H
external comparison introduced in 0.3.3.

## Maturity boundary

- **Engineering:** declared exact rectangular MPC and conservative elastic-
  foundation duals; structural modal analysis; provider-neutral small-strain
  direct harmonic response; declared fixed-interface delamination benchmarks.
- **Experimental:** mixed finite-strain J2 RVE and generalized-Maxwell harmonic
  routes.
- **Not claimed:** general contact duals, arbitrary weak constraints,
  universally verified finite-strain plasticity, arbitrary porous-RVE
  homogenization, external validation of the condensed tangent, or general
  free-path fracture.

The release is published only after the exact wheel passes metadata and
payload checks, serial and MPI verification, installed-template acceptance,
strict documentation, the optional PyTorch bridge and the versioned release
contract.
