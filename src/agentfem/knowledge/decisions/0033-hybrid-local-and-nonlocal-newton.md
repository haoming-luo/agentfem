# Keep local assembly and nonlocal tangent actions as separate operators

## Context

The first rotation-free fibrous-bending contribution is intentionally
neighbour reconstructed. Its energy depends on cell values across interior
facets, so its residual and consistent tangent are not cell-local UFL forms.
The existing woven-membrane provider, loads, and ordinary boundary conditions
are cell/facet-local UFL forms solved through the standard DOLFINx nonlinear
path.

Encoding the neighbour stencil as a nominal material expression would make a
constitutive object own mesh topology and MPI ghost exchange. Reassembling a
dense global bending matrix from repeated tangent actions would preserve the
equations but destroy scalability. Replacing the standard membrane provider
with shell-specific Newton code would duplicate load paths, constraints,
progress, checkpoint, output, and verification lifecycles.

## Decision

AgentFEM keeps three distinct owners:

1. local membrane, load, and boundary contributions remain ordinary UFL
   operators assembled by the FEniCSx backend;
2. neighbour-reconstructed bending remains an inspectable nonlinear operator
   exposing local owned energy, an assembled PETSc residual, and an exact
   matrix-free tangent action;
3. a future hybrid nonlinear Procedure composes both contributions, applies
   constraints once, and owns Newton, line search, convergence, progress,
   checkpoint, and failure evidence.

The hybrid Jacobian will therefore be an additive PETSc operator: the local
assembled Jacobian plus the nonlocal matrix-free action. The local assembled
matrix is also the first preconditioning operator; a provider may add a
documented nonlocal preconditioner later without changing the public model
language. Essential-boundary rows and increments are handled at the Procedure
boundary, not independently inside each scientific contribution.

`DisplacementFiberBendingOperator` is the first contribution satisfying this
contract. It composes displacement-derived kinematics, the cached neighbour
gradient, and the fibre-bending response behind `energy`, `residual`, and
`tangent_action`. This does not by itself create a shell Step.

The Procedure contract must remain general enough for other nonlocal terms,
including gradient damage, phase field, peridynamic-like interactions, and
learned energy corrections. It must not expose fibrous-shell names in the
generic nonlinear algebra layer.

## Promotion gates

The hybrid Procedure is not public until:

1. the additive residual equals the directional derivative of total energy;
2. the additive tangent matches residual differences and is symmetric for a
   conservative problem;
3. essential constraints, natural loads, reactions, external work, and
   stored energy are counted exactly once;
4. serial and MPI partitions give the same converged observables and preserve
   ghost ownership;
5. the assembled local preconditioner gives mesh-scalable convergence on a
   declared benchmark, or the limitation is explicit;
6. load incrementation, line search, checkpoint/restart, progress, output,
   and failure evidence use the standard Procedure and SimulationResult
   lifecycle;
7. shell-specific boundary moments and membrane/shear/bending patch tests
   pass before a fibrous-shell provider is promoted.

## Consequences

- Constitutive laws remain independent of topology, DOF ownership, and PETSc.
- The backend continues to own UFL compilation, local assembly, ghost
  communication, and linear algebra.
- The Procedure, not Model, owns the mixed assembled/matrix-free solve.
- A working bending operator cannot be mistaken for a complete shell or
  forming capability.
- The same algebra boundary can support future nonlocal physics without a
  second public workflow.

## Primary references

- Steer et al. (2021),
  <https://doi.org/10.1016/j.ijsolstr.2021.03.001>.
- Duong, Itskov, and Sauer (2022),
  <https://doi.org/10.1002/nme.6937>.
- PETSc matrix-free operator guidance,
  <https://petsc.org/release/manual/mat/>.
