# AgentFEM Engineering Documentation

AgentFEM is a FEniCSx-first finite-element environment designed to keep
engineering models readable for researchers and operable by AI agents.

Start with:

- the [installed project workflow](getting_started.md) for project creation,
  CLI runs, MPI, and result locations;
- the [agent and GUI integration contract](agent_gui_integration.md) for
  conversational MVPs, machine interfaces, and future service boundaries;
- the [product roadmap](product_roadmap.md) for priorities and maturity gates;
- [nonlinear materials](nonlinear_materials.md) for current constitutive scope;
- [nonlinear solid architecture](nonlinear_solid_architecture.md) for the P1
  public API, state contract, verification ladder, and explicit non-goals;
- [results and campaigns](results_and_campaigns.md) for simulation-to-data
  workflows;
- the [digital-twin direction](digital_twin_direction.md) for observation,
  state updating, learned prediction, FEM fallback, and system boundaries;
- [scientific operator contracts](operator_contracts.md) for K/M/C/F,
  residual/tangent structure, composition, and validation;
- [stable steps and compact output](step_and_output_architecture.md) for the
  provider and unified XDMF/HDF5 decisions;
- [mesh interoperability](mesh_interoperability.md) for external CAE meshes;
- [platform and optional dependencies](platform_support.md) for Windows/WSL2,
  Gmsh, and runtime evidence;
- the [module map](module_map.md) for code routing.

The repository root also contains the installation guide, full workflow,
concept vocabulary, examples, and agent guide. The dependency-free
`build_docs.py` command builds those sources into the complete local static
site.
