# AgentFEM 0.3.7

AgentFEM 0.3.7 consolidates the scientific core and makes mesh compatibility a
first-class product contract.

## Highlights

- A provider-neutral small-strain learned-constitutive contract now shares the
  ordinary material state, batch update, tangent, checkpoint, and global Newton
  lifecycle without adding a PyTorch dependency to AgentFEM core.
- Cyclic plasticity gains reusable material histories, explicit Chaboche energy
  semantics, faster batch integration, and public path and structural evidence.
- Multi-material thermoelasticity now uses explicit regional eigenstrain,
  physical stress semantics, discontinuity-preserving result projection, and
  rigid-mode and material-partition preflight.
- Partitioned finite-strain explicit dynamics accepts compatible regional
  materials and loads instead of imposing an artificial one-material form.
- Composite foundations now include multilayer fabric semantics, ply failure
  assessments, and a bounded rotation-free fibrous-shell development path.
- Mesh inspection now reports a machine-readable cell compatibility matrix.
  Triangle, quadrilateral, tetrahedral, quadratic-tetrahedral, and hexahedral
  neutral routes are verified; high-order tensor, prism, and pyramid routes
  remain explicitly conditional.
- Mesh quality now covers simplex mean ratio and coordinate-map sampled scaled
  Jacobians for quadrilateral, hexahedral, prism, and pyramid cells, including
  high-order geometry where supported.

## Inspect a mesh before solving

```bash
agentfem inspect-mesh model.msh
agentfem capabilities meshes
```

These commands distinguish source connectivity, solver formulation, and
quality evidence. Importing a cell block never claims equivalence with source
reduced integration, hybrid variables, hourglass control, shell, beam, or
cohesive formulations.

See [Mesh interoperability](mesh_interoperability.md) for the compatibility
matrix, quality semantics, and promotion roadmap.

## Release boundary

0.3.7 does not claim universal element coverage, general contact, a production
composite shell, or automatic repair of poor meshes. Unknown cell types fail
closed, conditional routes remain labelled, and mesh-quality thresholds remain
project decisions rather than universal engineering limits.
