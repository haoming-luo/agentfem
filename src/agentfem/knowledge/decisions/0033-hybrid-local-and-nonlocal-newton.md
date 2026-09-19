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

## Current implementation evidence

The first internal promotion runtime now implements the additive PETSc
operator. The true matrix action is the assembled local Jacobian plus every
exact matrix-free contribution; PETSc receives a separately owned assembled
preconditioning matrix. It defaults to the local Jacobian, but may be an
independent approximation when the local physics is singular in directions
stabilized by the nonlocal contribution. This approximation never replaces
the true operator. A FEniCSx adapter scatters each
increment before applying a nonlocal contribution. Essential increments and
residual rows are projected once at the composition boundary.

A constrained local-spring plus displacement-derived fibre-bending problem
converges through this runtime. The same mesh gives matching displacement
integral, bending energy, maximum displacement, constrained reaction, free-
residual norm and Newton iteration count in serial and with two ranks. The
total energy/residual and residual/tangent derivative identities are tested as
composed quantities rather than inferred only from component tests. Rejected
attempts restore the primary field to its pre-attempt state. This closes the
algebra, reaction and partition portions of the decision. An internal
attempt-solver seam also routes the hybrid solve through the existing ordinary
nonlinear load controller, proving fixed increments, automatic cutback,
progress events, snapshots and `SimulationResult` without duplicating a shell-
specific state machine. The runtime remains internal: checkpoint/restart,
shell boundary moments, locking evidence and patch tests are still required
before a public Procedure.

The first embedded fabric-membrane plus nonlocal warp-bending composition
builds with this split. Its attempted transverse strip solve is negative
evidence rather than a claimed patch test: a displacement-only edge condition
leaves slope free for a rotation-free curvature operator. Boundary
slope/rotation and conjugate moment ownership must be explicit before the
shell Procedure is promoted.

The exercise also exposed a partition condition hidden by affine tests. One
shared-facet ghost layer makes a one-ring cell stencil complete, but does not
make an arbitrary two-ring stencil complete. Conservative neighbor energies
therefore fail closed when their stencil exceeds the available halo rather
than accepting a partition-dependent result.

## Primary references

- Steer et al. (2021),
  <https://doi.org/10.1016/j.ijsolstr.2021.03.001>.
- Duong, Itskov, and Sauer (2022),
  <https://doi.org/10.1002/nme.6937>.
- PETSc matrix-free operator guidance,
  <https://petsc.org/release/manual/mat/>.
