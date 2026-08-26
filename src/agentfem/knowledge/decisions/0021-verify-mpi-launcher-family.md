# Verify the MPI launcher family before execution

## Decision

AgentFEM does not treat every executable named `mpiexec` or `mpirun` as
interchangeable. The runtime detects the MPI implementation used by `mpi4py`,
inspects available launchers, and selects a compatible launcher from the
active Python environment before considering `PATH`.

`agentfem run --mpi N` and `agentfem mpi-run -n N -- ...` share this decision.
`agentfem doctor --json` records the vendor, candidates, selected path,
compatibility result, and a stable diagnostic code.

## Reason

Conda FEniCSx environments commonly use MPICH while a macOS workstation also
has Homebrew Open MPI on `PATH`. Open MPI can start Python but cannot safely
initialize an MPICH-linked `mpi4py`/PETSc process. That failure occurs before
finite-element assembly and otherwise looks like an intermittent solver or
sandbox problem.

## Consequences

- A verified environment launcher is used automatically.
- A mismatched `PATH` is recoverable and visible as
  `AFM-MPI-LAUNCHER-RECOVERED`.
- If no compatible launcher exists, AgentFEM fails before spawning ranks with
  `AFM-MPI-LAUNCHER-MISMATCH` or `AFM-MPI-LAUNCHER-MISSING`.
- Documentation and agents use `mpi-run` for arbitrary MPI tests instead of
  constructing environment-specific absolute paths.
- Scheduler-specific launching remains an explicit future adapter rather than
  an inferred shell workaround.
