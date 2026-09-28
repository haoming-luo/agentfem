# Establish contact ownership with one bounded rigid-plane formulation

## Decision

AgentFEM's first contact route is frictionless, small-strain contact between a
declared boundary and one fixed rigid plane. Its normal points into the
admissible half-space and its signed gap is

\[
g(\mathbf{u}) = g_0 + \mathbf{u}\cdot\mathbf{n}.
\]

The contact Operator owns the one-sided penalty potential

\[
\Pi_c(\mathbf{u}) = \frac{1}{2}\int_{\Gamma_c}
k_c\langle-g(\mathbf{u})\rangle_+^2\,\mathrm{d}\Gamma,
\]

its exact first variation, and its UFL-derived tangent. Procedure owns
incremental Newton and load acceptance. Result/Verification asks the same
provider for the converged nodal reaction, physical-space resultant, contact
energy, penetration norm, and active-contact measure.

This provider does not perform surface search, friction, moving-obstacle
kinematics, or deformable-to-deformable coupling. It accepts only one contact
asset and ordinary strong displacement constraints. Other configurations fail
before assembly.

## Reason

Contact is a variational inequality, not another Dirichlet boundary. A general
contact claim would require geometric search, changing active sets, parallel
ownership, consistent linearization, frictional history, and external
verification at once. Starting with a bounded conservative potential closes
the important ownership chain without embedding a partial general-contact
algorithm in Model or Result.

The contact potential is system internal energy. Its endpoint value must not
be added again as prescribed-motion or external work. Conversely, a nonlinear
work integral cannot be reconstructed from one endpoint force. The ordinary
nonlinear Procedure therefore preserves provider duals only at accepted
stations. A fixed obstacle publishes a zero translation coordinate, proving
zero obstacle work while retaining nonzero contact energy as internal energy.

## Consequences

- Model declares the obstacle, normal, gap, penalty, and candidate boundary;
- Operator owns contact energy, residual, and consistent tangent;
- Procedure owns Newton increments and rollback, not contact physics;
- Result obtains dual evidence from the Operator that assembled the contact;
- a unit normal is mandatory so gap and penalty units remain unambiguous;
- penalty sensitivity and penetration are visible verification evidence;
- serial and MPI results use the same provider-owned reaction path;
- failed Newton attempts never enter the accepted constraint-work ledger;
- general contact remains unsupported until a dedicated backend earns its own
  capability and verification evidence;
- restart promotion remains blocked until ordinary nonlinear checkpoints carry
  the accepted dual history atomically with the solution state.

## Verification

- an active unit patch matches the analytical series stiffness of bulk and
  contact penalty;
- contact reaction plus natural load closes global force balance;
- an open gap has zero contact force, energy, penetration, and active measure;
- accepted fixed-obstacle path work is exactly zero and excludes cutback trials;
- invalid penalty and non-unit normals fail before lowering;
- nodal reaction, resultant, energy, and diagnostics agree in serial and under
  two MPI ranks.

## References

- E. Burman, P. Hansbo, and M. G. Larson, "The Penalty-Free Nitsche Method and
  Nonconforming Finite Elements for the Signorini Problem":
  <https://doi.org/10.1137/16M107846X>
- ASiMoV Contact, a DOLFINx contact library with Nitsche contact and parallel
  examples: <https://github.com/Wells-Group/asimov-contact>
