# Separate portable scientific projects from execution profiles

## Decision

An AgentFEM project contains one scientific model. Materials, meshes, loads,
constraints, procedures, outputs, and verification rules remain in `case.py`
and project-local assets. Named `agentfem.toml` execution profiles may declare
only a runtime, a default MPI rank count, and required runtime capabilities.

Projects may be transported as deterministic `.afm` bundles. Every bundled
file is registered by logical relative path, size, and SHA-256 digest. Outputs,
caches, credentials, version-control data, and symbolic links are excluded or
rejected. A bundle is verified before inspection, unpacking, or execution, and
its identity remains attached to the result evidence.

Numerical agreement across native serial, PETSc serial, and PETSc/MPI routes
is assessed through explicitly selected scientific quantities and declared
tolerances. Bitwise equality is not required, and incidental diagnostics are
not silently promoted into an acceptance contract.

## Reason

A Windows project and a cluster project must not become two models that drift
apart. Conversely, pretending that sparse solver, MPI partition, and runtime
identity never change would hide material evidence. Keeping execution intent
outside the model makes the unchanged project transferable while preserving
the numerical route in provenance.

An ordinary archive without path and digest validation is insufficient for
agent-operated workflows: missing meshes, stale tables, path traversal,
accidentally bundled secrets, or source-tree imports can otherwise produce an
apparently successful but irreproducible run.

## Consequences

- Project code contains no operating-system branches.
- Execution profiles cannot contain scientific or solver-input keys.
- Unsupported profiles fail during runtime preflight, before numerical
  lowering or MPI launch.
- Direct `.afm` execution writes results outside its temporary workspace and
  records the bundle SHA-256.
- Server scheduling remains an executor/plugin concern; it does not expand
  the core project schema with Slurm-, SSH-, or vendor-specific fields.
- Cross-runtime promotion evidence names the quantities and tolerances being
  compared.
- Partition-independent restart remains procedure-specific and fail-closed;
  a partition-bound checkpoint is never relabelled portable.
