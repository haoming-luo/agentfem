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
