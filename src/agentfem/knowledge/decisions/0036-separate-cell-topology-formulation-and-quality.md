# ADR 0036: Separate cell topology, formulation, and quality

## Status

Accepted.

## Decision

AgentFEM describes mesh compatibility through three independent contracts:

1. **source connectivity** identifies node ordering, named sets, and geometry;
2. **solver formulation** identifies interpolation, integration, mixed fields,
   stabilization, and analysis-specific kinematics;
3. **quality evidence** evaluates the active coordinate map before solving.

Reading a source cell never proves formulation equivalence. A C3D8R block may
lower to neutral hexahedral geometry, but reduced integration and hourglass
control are not inherited from its connectivity. Likewise, line, shell, and
cohesive topology do not become beam, shell, or interface formulations without
an explicit provider.

The compatibility matrix is machine readable and uses `verified`,
`conditional`, and `blocked`. Verified means the neutral route has executable
import and solver evidence. Conditional identifies the precise missing order,
format, or formulation evidence. Unknown cells fail closed.

Simplex quality uses normalized mean ratio. Tensor-product, prism, and pyramid
quality uses the minimum sampled scaled Jacobian from the actual coordinate
element, including high-order curvature. Thresholds remain project policy;
invalid or folded geometry is never accepted by a threshold label.

## Consequences

- External-mesh inspection reports compatibility before conversion.
- `agentfem capabilities meshes` exposes the installed matrix to people and
  agents.
- Mixed topology remains split into explicit solver domains until the backend
  and AgentFEM lifecycle can preserve it without ambiguity.
- New cells advance only with import, ordering, quality, patch, output, and MPI
  evidence; topology availability alone is insufficient.
