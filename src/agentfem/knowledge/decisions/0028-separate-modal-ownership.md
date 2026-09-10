# Separate modal intent, backend execution, and result evidence

## Decision

Structural modal analysis uses three explicit owners. `mechanics.modal` owns
the procedure, including rigid-mode filtering, targeted or low-mode selection,
cluster completeness, and accepted modal state. `backends._modal` owns
distributed Dirichlet reduction, PETSc/SLEPc execution, operator Gram data,
and deterministic destruction of every backend resource. `results._modal`
owns fields, frequencies, metadata, and publication through the common result
lifecycle.

`problems.modal_analysis(...)` remains the compatible construction entry point
but delegates to the mechanics-owned Step. The public
`model.step(target=..., modes=...)` workflow is unchanged.

## Reason

An eigensolver is not the engineering meaning of a modal analysis. Mixing DOF
reduction, eigenpair selection, field construction, evidence assembly, and
resource destruction in one problem class obscured responsibility and made
collective solver lifetimes depend on a high-level container. It also allowed
procedure-specific diagnostics to incorrectly name another analysis family.

The separation keeps AgentFEM's stable middle layer intact: a procedure decides
which modes answer the Study, a backend computes them, and a Result explains
what was accepted. It does not introduce a second public grammar or reimplement
finite-element assembly.

## Consequences

- strong Dirichlet elimination and SLEPc allocation remain backend details;
- rigid-mode thresholds, target-frequency selection, and repeated-eigenspace
  completeness remain scientific procedure decisions;
- PETSc/SLEPc objects are destroyed on successful and exceptional paths before
  a modal solve returns;
- returned DOLFINx mode fields remain usable after backend teardown;
- nonzero, time-dependent, remote and otherwise unsupported modal constraints
  fail before eigensolver assembly and name modal analysis rather than implicit
  dynamics;
- accepted residuals decide eigensolver convergence, whereas cluster
  completeness separately decides whether individual-mode comparison is
  scientifically meaningful;
- every published Result binds a portable executable identity of the mesh,
  live stiffness/mass coefficients and exact constrained-degree set, and
  post-solve drift fails closed;
- rank-local identity, boundary-reduction, candidate-eigenvalue and reduced-
  vector failures are synchronized before the next collective, so all ranks
  fail at the same named stage;
- future modal backends can return the same raw candidate and Gram evidence
  without changing the public Step or Result language.

## Verification

- the existing analytical, mesh-refinement, target-frequency, symmetry, and
  result-output tests retain their values and evidence;
- the public compatibility entry returns the mechanics-owned Step;
- unsupported constraints fail with the modal-specific compatibility message;
- nonzero, history-driven and remote strong constraints are rejected before
  backend execution;
- coefficient changes alter the result fingerprint, and an incomplete
  executable identity prevents publication;
- a small three-dimensional solid cantilever reproduces the first
  Euler--Bernoulli bending-frequency limit within its declared tolerance;
- a two-rank modal solve followed by distributed result tests completes without
  relying on Python garbage-collection order and retains the same executable
  fingerprint on every rank;
- injected rank-local backend and identity failures reach every rank, after
  which a communicator barrier still completes.
