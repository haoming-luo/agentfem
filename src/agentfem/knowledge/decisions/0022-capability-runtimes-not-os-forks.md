# Select numerical runtimes by capability, not operating system

## Decision

AgentFEM has one public scientific language and may have multiple numerical
runtimes. Step providers declare required capabilities such as linear solve,
PETSc nonlinear solve, distributed mesh, or exact MPC. Provider resolution
checks those requirements before importing and invoking a backend-specific
builder.

The initial profiles are:

- `fenicsx-petsc`, the full distributed PETSc route;
- `fenicsx-native-serial`, DOLFINx native assembly with SciPy and optional
  PyAMG for a useful serial core, including native Windows.

Scientific code must not select equations, materials, outputs, or public APIs
by checking the operating-system name.

## Reason

Maintaining a Windows edition and a Unix edition would duplicate workflows and
allow their scientific meaning to drift. The actual portability boundary is a
numerical capability: native DOLFINx can assemble forms on Windows while the
current conda-forge win-64 stack does not provide the complete PETSc,
petsc4py, and dolfinx_mpc route used by AgentFEM's nonlinear, exact-MPC, and
distributed procedures.

Capability dispatch preserves one user and agent experience while making that
boundary explicit and testable.

## Consequences

- New reusable physics, materials, operators, results, and project tooling are
  implemented once.
- A new Step provider records `requires`; it does not contain `if Windows`.
- Unsupported procedures fail before lowering with a stable capability error.
- Runtime reports and result provenance record the selected profile and
  installed numerical packages.
- Every advertised profile needs installed-wheel execution evidence for its
  claimed procedures.
- Future Windows PETSc, GPU, or external providers can expand capabilities
  without changing application scripts.
