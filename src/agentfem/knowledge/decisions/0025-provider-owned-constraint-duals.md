# Providers own non-Dirichlet constraint duals

## Decision

MPC, weak and contact providers own the dual quantities created by their
enforcement method. After convergence they may publish a named
`ConstraintDualEvidence` containing the generalized force, its accepted
work-conjugate coordinate, a physical-space resultant and provider identity.
The shared Result/Verification layer consumes those values; it does not infer
them from a strong-Dirichlet residual or from a success flag.

Force balance is complete only when every declared non-Dirichlet constraint
supplies a physical resultant. Work is complete only when every such
constraint supplies its generalized force and coordinate. Missing, partial,
duplicate or unmatched evidence fails closed.

## Reason

The meaning of an MPC multiplier, weak-boundary flux or contact traction is
defined by the provider that assembled it. Reconstructing those quantities in
the Model facade would duplicate numerical knowledge and eventually turn the
facade into a solver object. Treating an absent channel as zero would be more
dangerous: a numerically converged solve could then be labelled globally
balanced while omitting a real reaction or prescribed-motion contribution.

## Consequences

- Model continues to own engineering constraint declarations, not multipliers;
- Procedure/provider code extracts dual values after convergence;
- a constraint publishes them through `dual_evidence(problem)`; the Step
  collects records automatically after solve convergence;
- Result/Verification owns the common balance and work ledger;
- each new enforcement backend must define its dual extraction before claiming
  complete force or work evidence;
- the exact affine provider evaluates its macroscopic path reaction as
  ``R_full . du_affine/dlambda`` and its physical resultant from the same full
  displacement residual. It does not sum reference-node or eliminated-slave
  residuals in isolation;
- one nonlinear endpoint never closes path work. The affine nonlinear problem
  therefore records the provider dual at every accepted boundary and
  trapezoidally integrates the actual generalized-force/path-coordinate
  history; the endpoint evidence remains intentionally incomplete for generic
  static-work consumers;
- a non-proportional piecewise-linear path stores both incoming and outgoing
  generalized forces at every kink. Each increment uses the two forces
  projected on its own path tangent; one ambiguous knot value is never reused
  across two different deformation directions;
- existing strong-Dirichlet workflows retain their unconstrained-residual
  route and public result names.
- rectangular periodic MPC construction may prove graph integrity without
  claiming a dual: owned and ghost slaves, master cardinality and unit
  coefficients are diagnosed collectively, while eliminated multipliers,
  face reactions and macroscopic work remain unavailable.

## Verification

- complete named provider evidence closes both force and work channels;
- force evidence without a coordinate cannot close work;
- a generalized force without a physical resultant cannot close global force;
- duplicate or unmatched records are rejected;
- non-callable providers and records attributed to another constraint are
  rejected before any balance is published;
- old calls without provider evidence remain fail-closed.
- proportional and non-proportional periodic J2 tests compare the provider
  dual against ``V * P : dF/dlambda`` and require zero global resultant in the
  unloaded homogeneous cell.
- accepted affine-path work is checked against the independently assembled
  Hill--Mandel macroscopic work and survives portable checkpoint/restart.
- two-rank rectangular MPC tests verify owned/ghost accounting and require one
  unit-coefficient master relation for every globally owned slave DOF.
