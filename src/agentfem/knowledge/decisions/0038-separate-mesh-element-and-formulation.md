# Separate mesh topology, element space, and numerical formulation

## Decision

AgentFEM treats the following as distinct, inspectable identities:

```text
source element declaration
        -> neutral mesh connectivity and coordinate map
        -> runtime topology
        -> finite-element function space
        -> Step-provider formulation
```

No stage may silently infer the last stage from an earlier one. In particular,
node count and topology do not reproduce reduced integration, hourglass
control, incompatible modes, mixed pressure, shell directors, beam sections,
cohesive kinematics, or a vendor's private element implementation.

The `agentfem.mesh` layer owns source-cell and runtime-topology compatibility
plus geometry-quality evidence. The `agentfem.elements` layer owns the actual
UFL/Basix element identity bound to each field and checks it against the Model
mesh and Study. Step providers own analysis-specific stability and formulation
acceptance. Basix and DOLFINx continue to own tabulation, dof maps, quadrature,
and assembly.

The coordinate finite element is a separate first-class identity. Its basis
family, variant, degree, map, node count, topological dimension, and geometric
dimension must not be inferred from the solution field. High-order coordinate
maps are assessed from their sampled Jacobian, including positive near-singular
degradation, rather than only from corner-cell shape or fold detection.

## Validation policy

Ordinary `Model.validate()` performs metadata-only topology and element checks.
It must not traverse every cell or assemble a form. Explicit
`elements.audit(..., check_quality=True)` performs the collective cell-level
quality calculation before a long run, benchmark, or release gate.

Invalid or folded cells fail. A positive quality threshold is project policy,
not a universal constant: cells below it warn by default and fail only when
the caller explicitly requests rejection. Every report retains the chosen
metric and threshold.

## Consequences

- Imported connectivity can be useful without pretending to reproduce a
  source solver formulation.
- Humans and agents can inspect the real element instead of guessing from a
  field name.
- Future shell, beam, cohesive, mixed, and user-defined elements have one
  compatibility boundary without growing `Model` into an element registry.
- Conditional prism, pyramid, and interval topology remains visible and
  inspectable, but cannot become release evidence without provider and
  benchmark promotion.
- Narrow topology capabilities carry their own evidence. A verified P1 prism
  patch does not imply a verified mixed, nonlinear, shell, or source-vendor
  formulation.
- High-order neutral import maturity is earned per source cell layout. The
  triangle6, quad9, tetra10, hexahedron20, and hexahedron27 routes pass a real
  meshio/XDMF/DOLFINx read, coordinate-element identity, sampled-quality, and
  P2 affine-patch contract. Quad8 remains conditional because the current
  DOLFINx XDMF reader rejects its eight-node coordinate layout; AgentFEM
  reports that boundary explicitly instead of treating conversion as solve
  readiness.
