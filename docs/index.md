# AgentFEM Engineering Documentation

AgentFEM is a FEniCSx-first finite-element environment designed to keep
engineering models readable for researchers and operable by AI agents.

Start with:

- the [product roadmap](product_roadmap.md) for priorities and maturity gates;
- [nonlinear materials](nonlinear_materials.md) for current constitutive scope;
- [nonlinear solid architecture](nonlinear_solid_architecture.md) for the P1
  public API, state contract, verification ladder, and explicit non-goals;
- [results and campaigns](results_and_campaigns.md) for simulation-to-data
  workflows;
- [stable steps and compact output](step_and_output_architecture.md) for the
  provider and unified XDMF/HDF5 decisions;
- [mesh interoperability](mesh_interoperability.md) for external CAE meshes;
- the [module map](module_map.md) for code routing.

The repository root also contains the installation guide, full workflow,
concept vocabulary, examples, and agent guide. The dependency-free
`build_docs.py` command builds those sources into the complete local static
site.
