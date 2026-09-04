# Share the MPC solver lifecycle, not benchmark glue

## Decision

Exact MPC construction belongs to `constraints`; compiled variational forms,
PETSc allocation, KSP policy, repeated solve execution, convergence evidence,
and transfer from the augmented MPC layout to the public field belong to
`solvers`. Scientific adapters consume that common lifecycle rather than
instantiating `dolfinx_mpc.LinearProblem` privately.

The provider that assembled a constraint continues to own physical dual
recovery. A generic solver does not infer multipliers, face reactions, or work
coordinates merely because the reduced linear system converged.

## Reason

Periodic PDE stepping, thermal cells, structural cells, and future engineering
providers need the same exact-MPC numerical lifecycle. Duplicating it in an
evaluation adapter makes solver choices and MPI state transfer drift, while
placing it in `Model` would turn the engineering facade into a numerical
owner. The solver layer is the narrow reusable boundary.

The current upstream implementation reuses compiled forms, allocated PETSc
objects, and KSP state but reassembles matrix values on every call. AgentFEM
records that distinction instead of advertising an unsupported constant-matrix
optimization.

## Consequences

- `prepare_mpc_linear_problem(...)` uses the ordinary `LinearSolverOptions`;
- augmented MPC ghost layouts never leak into the public solution field;
- repeated solves expose one stable `LinearSolveInfo` and solve count;
- Dirichlet conditions introduced after constraint construction are checked
  against owned MPC slaves on every rank and rejected when they overlap;
- adapters translate domain-specific missing-dependency errors only at their
  boundary;
- a later constant-operator optimization can replace the internal lifecycle
  without changing its callers;
- reaction/work completeness remains fail-closed until the constraint provider
  supplies real `ConstraintDualEvidence`.

## Verification

- one prepared object solves two independently known constant periodic fields;
- the right-hand side changes without reconstructing the public object;
- scalar and vector serial/two-rank solutions reproduce their analytical
  constants, including blocked displacement-style layouts;
- late Dirichlet/MPC conflicts fail before assembly in serial and MPI, while a
  condition declared during constraint construction remains admissible;
- the migrated periodic Burgers benchmark retains its matching-face result;
- summaries state both allocation reuse and matrix-value reassembly.
