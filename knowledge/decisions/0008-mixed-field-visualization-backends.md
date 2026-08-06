# Decision 0008: Preserve one result contract while separating storage backends

## Decision

AgentFEM keeps checkpoint/restart state, scientific field semantics, and
ParaView presentation as distinct responsibilities under one
`SimulationResult`/`OutputPlan` contract. XDMF/HDF5 remains the compact
interoperability and scientific-data route. An optional presentation backend
may use VTX/BP or VTKHDF only when its element and parallel-I/O constraints are
explicitly satisfied.

VTX is not made the silent default for mixed solid output. DOLFINx requires all
functions in one `VTXWriter` to share one mesh and one element type, whereas a
normal solid result intentionally combines `U` on continuous Lagrange space
with stress/strain/state fields recovered to discontinuous cell spaces. A
backend switch must not erase that distinction or duplicate geometry without
warning.

The next collective single-grid route will therefore be selected by evidence:

1. evaluate VTKHDF for one temporal mesh carrying point and cell attributes;
2. retain VTX for homogeneous field groups where its contract naturally fits;
3. otherwise use a collective HDF5/XDMF writer with stable global point/cell
   identity;
4. keep integration-point truth in scientific storage and declare every
   recovery, averaging, or smoothing operation used for presentation.

## Why

ParaView should open one result and expose one deformable geometry with all
requested fields. But solving this by coercing every field to one element type
would make output look cleaner while losing scientific meaning. Restart data
also needs committed integrator/material state that a visualization file does
not represent.

The official DOLFINx I/O contract documents VTX's same-mesh and same-element-
type restriction. Backend choice must follow this real constraint rather than
product aesthetics alone.

## Consequences

- The public API remains `OutputPlan`/`SimulationResult`, not writer-specific
  calls in every case file.
- Serial unified XDMF/HDF5 remains useful and tested.
- Collective MPI single-grid visualization stays an urgent, explicit gap.
- Presentation fields always retain source position and processing metadata.
- Checkpoints never masquerade as ParaView output, and ParaView output never
  claims to be sufficient restart state.

Reference: [DOLFINx I/O API](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.io.html)
