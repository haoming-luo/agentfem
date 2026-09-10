# Recover exact-MPC duals from owned slave residuals

## Decision

For the homogeneous rectangular relation `u_s = C u_m`, the constraint
provider recovers `lambda = r_s` from the owned slave entries of the full,
unconstrained residual. It reconstructs the physical nodal distribution as
`B.T @ lambda`, using backend-global scalar degree-of-freedom identities so
off-rank master contributions are accumulated by PETSc rather than gathered
as Python data.

The provider publishes a compact `ConstraintDualEvidence` record containing
the physical resultant, constraint virtual work, multiplier and gap norms,
and an optional live nodal distribution. The Result layer transports that
field and consumes the compact dual pair; it does not inspect the elimination
graph.

## Reason

Reduced-system convergence proves only `C.T @ r = 0`. It does not by itself
identify the reaction distribution or its work. For one-master homogeneous
relations, the owned slave residual is the constraint multiplier under the
declared sign convention, and `B.T @ lambda` supplies equal-and-opposite
physical forces without duplicating backend assembly in Model or Result.

Storing every multiplier in `result.json` would scale poorly. A live finite-
element field is the correct large-data carrier; the manifest keeps only
norms, counts, resultant, work, field identity, and provenance.

## Consequences

- construction remains immutable and carries no solved state;
- dual recovery is available only after a converged problem consumed the same
  provider instance;
- MPI communication follows PETSc vector ownership and does not all-gather the
  constraint graph;
- scalar temperature relations expose a flux-like algebraic dual, while solid
  Steps interpret vector distributions as nodal reactions;
- the current homogeneous relation contributes zero constraint work up to
  numerical tolerance; nonzero affine-path work stays with the affine provider;
- weak, contact, arbitrary nonunit, and nonlinear MPC providers must implement
  their own reviewed extraction rather than inheriting this special relation.

## Verification

- serial and two-rank constant-field solves retain exact periodic values;
- a spatially varying source creates nonzero recovered multipliers;
- reconstructed reactions have near-zero physical resultant;
- measured slave/master gaps and `lambda.T @ gap` remain near zero;
- the nodal distribution is attached to `SimulationResult` without embedding
  its full array in JSON metadata;
- missing provider duals continue to fail closed.
