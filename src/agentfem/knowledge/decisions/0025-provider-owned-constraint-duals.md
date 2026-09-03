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
- Result/Verification owns the common balance and work ledger;
- each new enforcement backend must define its dual extraction before claiming
  complete force or work evidence;
- existing strong-Dirichlet workflows retain their unconstrained-residual
  route and public result names.

## Verification

- complete named provider evidence closes both force and work channels;
- force evidence without a coordinate cannot close work;
- a generalized force without a physical resultant cannot close global force;
- duplicate or unmatched records are rejected;
- old calls without provider evidence remain fail-closed.
