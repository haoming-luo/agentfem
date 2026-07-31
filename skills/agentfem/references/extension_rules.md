# Extension Rules Reference

Add a helper to AgentFEM only when it is a reusable FEM concept or workflow
operation.

Keep problem-specific geometry, source histories, benchmark constants, and paper
parameters in application packages or examples.

Use `materials/` for reusable SI-unit material-property records. Use
`constitutive/` for equations that map state to stress, flux, tangents, or other
responses.

Use `elements/`, `operators/`, and `benchmarks/` only for reusable asset
families. Keep first-level modules focused on the standard FEM workflow.

Use operator-level language for beginner application tutorials. Use weak-form
language only when the tutorial is teaching formulation details.

When a new concept is added, update:

- `CONCEPTS.md`
- `WORKFLOW.md` if the workflow changes
- `docs/module_map.md`
- The relevant skill reference

For a new constitutive capability, also add:

- an entry in `constitutive/catalog.py`;
- a benchmark obligation in `benchmarks/registry.py`;
- material-point evidence before any global FEM convenience step;
- explicit state/tangent/increment-control evidence before marking it
  `fem_integrated`.
