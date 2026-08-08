# Decision 0009: Route cohesive traces and forces by physical node ownership

## Decision

Distributed fixed-path cohesive assembly uses a one-time schedule keyed by
durable split-mesh input-node identity. Every physical facet remains owned by
one deterministic rank. At each Explicit increment, the facet owner receives
only the remote displacement traces required by its facets, evaluates the
local cohesive kernel, and returns only those nodal force contributions to the
MPI ranks that own the displacement entries.

The schedule uses one collective metadata exchange during construction and
numeric `MPI_Alltoallv` payloads during time stepping. Energy remains a scalar
global reduction. Irreversible state remains keyed by ordered physical facet
geometry and quadrature, so the communication backend does not become state
identity. Each rank's material-point assembler is compacted to the interface
nodes touched by its owned facets; it no longer allocates temporary force and
trace arrays proportional to all volume nodes.

Two-dimensional conforming interfaces may be recovered from a declared cell
partition with `interfaces.split_conforming_cell_interface(...)`. The caller
still declares the positive side; AgentFEM derives the shared manifold edges,
duplicates only that side, and rejects non-manifold or disconnected
partitions. Abaqus/Gmsh adapters should lower reviewed ELSET/physical-group
semantics to this contract instead of creating solver-specific cohesive loops.

## Why

The correctness reference reduced arrays proportional to all split-mesh nodes
on every increment. That proved force and restart semantics but coupled
communication volume to the bulk mesh rather than to the lower-dimensional
interface. It also required users and importers to enumerate an otherwise
derivable conforming edge path.

The physical interface is the stable scientific object. Rank-local DOFs,
partition adjacency, and the communication schedule are execution details.
Routing by source-node and facet identity preserves that separation while
allowing the MPI payload to scale with interface traces and contributions.

## Consequences

- The public `mode_i_cohesive_force(...)` call remains unchanged.
- Distributed force vectors contain contributions only on owning displacement
  entries; shared nodes are summed from all facet owners exactly once.
- Communication summaries expose remote trace/force value counts and peer
  counts so scaling claims are inspectable.
- Local cohesive integration storage scales with nodes touched by locally
  owned facets rather than the complete split volume mesh.
- Serial and distributed nonuniform traction states must agree node by node,
  including compression, elastic loading, softening, energy, and rollback.
- Cross-rank-count restart continues to use physical nodal/facet identity and
  is independent of the MPI schedule.
- The current implementation is a sparse-payload collective foundation, not a
  claim of extreme-scale neighbor-graph optimality. A neighborhood collective
  or custom kernel may replace it behind the same contract after profiling.
- Three-dimensional cohesive surfaces, mixed mode, contact, and direct
  Abaqus/Gmsh internal-surface adapters remain separate promotion gates.
