# Fail-closed balance evidence and explicit result routing

## Decision

AgentFEM reports a complete force or work balance only when every declared
constraint has a reaction definition consumed by the active solver provider.
The unconstrained assembled residual is authoritative for strong Dirichlet
elimination. It is not silently reused for MPC, weak, contact, projection, or
multiplier constraints, whose conjugate forces require provider-owned dual
variables. Missing channels produce structured `unavailable` evidence and a
named `constraint_balance_contract`, not a partial number labeled as global
equilibrium.

Scientific storage and presentation routing are similarly explicit. Serial
results recommend the single-grid XDMF/HDF5 artifact. MPI results retain the
collective DOLFINx XDMF/HDF5 record and recommend the single-geometry PVD/PVTU
artifact for ParaView. `SimulationResult.metadata["field_output"]` records both
roles and whether Extract Block is required.

## Consequences

- Adding a constraint backend also requires declaring reaction and work
  evidence semantics.
- A solver provider may promote an unavailable channel only by returning the
  actual conjugate reaction and accepted-path coordinate history.
- Users and agents do not infer the preferred visualization file from an
  extension or from backend implementation details.
- Scientific checkpoint state, compact field storage, and presentation files
  remain separate products with a shared result manifest.

## Verification

- unresolved periodic/MPC assets suppress strong-only force and work balance;
- ordinary Dirichlet static cases retain their existing reaction and energy
  evidence;
- serial output recommends one XDMF Uniform Grid;
- MPI output recommends one PVD dataset per time and requires no Extract Block.
