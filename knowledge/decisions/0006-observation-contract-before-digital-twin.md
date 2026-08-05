# Decision 0006: Establish observation contracts before a digital-twin service

## Decision

AgentFEM owns the scientific observation, model-state, verification, and
fallback contracts needed by a future digital twin. It does not absorb sensor
transport, plant historians, dashboards, or asset-management infrastructure
into the finite-element core.

The first executable boundary is a deterministic map from FEM fields to named
physical points, paths, and structured grids with coordinates, units,
components, layout, masks, and provenance. Online assimilation will only be
added after observation identity, model/checkpoint identity, uncertainty, and
applicability decisions have explicit records.

## Why

A neural network or a live plot does not by itself constitute a scientific
digital twin. Measurements, inferred state, deterministic computation, and
learned prediction have different epistemic meanings. A stable observation
contract lets external agents, neural-operator libraries, GUIs, and industrial
data systems interact without erasing those distinctions.

## Consequences

- FEM field sampling is reusable independently of any ML framework.
- Neural models remain replaceable external consumers.
- A future online service can use the same result and checkpoint identities as
  offline campaigns.
- Sensor protocols and commercial front ends can evolve without coupling the
  scientific core to one vendor stack.
- Digital-twin claims require state updating, uncertainty, and lifecycle
  evidence beyond the currently implemented offline foundation.
