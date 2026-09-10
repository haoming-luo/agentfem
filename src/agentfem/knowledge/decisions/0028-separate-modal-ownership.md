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
- unsupported modal constraints fail before assembly and name modal analysis
  rather than implicit dynamics;
- future modal backends can return the same raw candidate and Gram evidence
  without changing the public Step or Result language.

## Verification

- the existing analytical, mesh-refinement, target-frequency, symmetry, and
  result-output tests retain their values and evidence;
- the public compatibility entry returns the mechanics-owned Step;
- unsupported constraints fail with the modal-specific compatibility message;
- a two-rank modal solve followed by distributed result tests completes without
  relying on Python garbage-collection order.
