# AgentFEM

AgentFEM is an agent-oriented finite-element workflow platform built on
DOLFINx/PETSc. The Python package is named `agentfem`, matching the platform
name used by the documentation and application code.

The current implementation is strongest in reusable FEM workflow steps,
linear-elastic constitutive laws, explicit dynamics, weak loads, constraints,
absorbing boundary models, diagnostics, and ParaView output.

## Agent Entry Points

- `AGENT_GUIDE.md`: first file for AI agents working with AgentFEM.
- `WORKFLOW.md`: standard FEM workflow and module order.
- `CONCEPTS.md`: shared vocabulary for humans and agents.
- `docs/`: detailed platform rules and extension guidance.
- `examples/`: runnable workflow examples for researchers and agents.
- `skills/agentfem/`: skill-ready progressive-disclosure package.
- `mkdocs.yml`: optional static documentation-site configuration.

## Standard Workflow

1. Build or read a mesh with application geometry plus `mesh` import helpers.
2. Create spaces and fields with `spaces`.
3. Locate dofs and construct strong constraints with `dofs`, `boundary`, and
   `constraints`.
4. Construct weak loads such as body forces, Neumann fluxes, and tractions with
   `loads`.
5. Build weak forms with application physics plus reusable `forms` blocks.
6. Compile and assemble with `assembly`.
7. Step in time with `runtime.TimeStepper` plus kernels from `time`, or solve
   algebraic systems with `solvers`.
8. Measure distributed diagnostics with `diagnostics`.
9. Manage output locations with `io`.

## Modules

- `mesh`: mesh import/read/write, boundary facet location, facet tags,
  boundary measures
- `mesh_formats`: optional external CAE mesh conversion through `meshio`
- `spaces`: scalar/vector spaces, trial/test functions, named fields
- `dofs`: dof location, owned/ghost arrays, field copies
- `constraints`: Dirichlet constraint containers and periodic constraint specs
- `constitutive`: material laws and constitutive-model helpers
- `boundary_models`: reusable weak boundary models, such as absorbing boundaries
- `boundary`: Dirichlet constants and BC application
- `loads`: time functions, time-dependent Dirichlet data, body loads,
  Neumann/boundary loads, and load sets
- `forms`: common UFL virtual-work building blocks
- `assembly`: form compilation, vector/matrix assembly, lumped operators/mass
- `time`: explicit central-difference/Newmark kernels
- `problems`: standard problem/state containers
- `runtime`: time-step metadata, progress printing, and elapsed-time formatting
- `solvers`: PETSc KSP setup and standard linear problem solve
- `diagnostics`: distributed norms, energy-like quantities, and named scalar
  diagnostic sets
- `io`: MPI-safe directories, CSV logs, and XDMF time-series output

## Core API Direction

- Use `assembly.assemble_lumped_operator(V, coefficient, measure)` for generic
  diagonal operators on a function space `V`.
- Use `mesh.import_gmsh_model(...)` or `mesh.read_gmsh_mesh(...)` to convert
  Gmsh meshes into DOLFINx meshes with physical tags.
- Use `mesh.read_xdmf_mesh(...)` and `mesh.write_xdmf_mesh(...)` for DOLFINx
  XDMF mesh exchange.
- Use `mesh.convert_external_mesh_to_xdmf(...)` or
  `mesh_formats.convert_to_xdmf(...)` for Abaqus `.inp`, NASTRAN `.bdf/.nas`,
  VTK, and other meshio-supported external CAE mesh formats.
- Use `assembly.assemble_lumped_mass(V, density, measure)` as the mass-specific
  wrapper.
- Use `problems.TransientState` for first-order transient unknowns.
- Use `problems.SecondOrderDynamicsState` for displacement/velocity/acceleration
  dynamics. `ExplicitDynamicsState` remains as a compatibility alias.
- Use `constitutive.elasticity.estimate_elastic_wave_speeds(material)` when a
  wave solver needs scalar wave-speed estimates from an elastic material.
- Use `constraints` for strong constraints. Use `loads.NeumannLoad` or
  `loads.BoundaryLoad` for natural boundary data that enters the weak form.

## Boundary Data Rule

- Dirichlet data is essential/strong data. It modifies dofs directly or enters
  matrix/vector assembly through `bcs`.
- Neumann data is natural/weak data. Force, flux, and traction terms belong in
  the right-hand-side weak form through `loads`.
- Robin or absorbing data is weak boundary physics. It should be represented as
  a boundary bilinear/linear form, not as a Dirichlet constraint.

## Absorbing Boundaries

Absorbing and impedance boundaries are reusable weak boundary models. They are
typically represented as boundary damping or impedance terms proportional to
velocity, flux, or another boundary response variable:

```python
abc = boundary_models.absorbing.lysmer_kuhlemeyer_boundary(
    ds(tag),
    density=rho,
    pressure_wave_speed=cp,
    shear_wave_speed=cs,
    normal=ufl.FacetNormal(domain),
)
form = abc.form(velocity, test_function)
```

This keeps boundary location and tags in the application layer while reusable
absorbing-boundary formulas live in `agentfem.boundary_models`.

## Naming Rules

- Use verbs for actions: `assemble_vector`, `apply_dirichlet_bcs`,
  `copy_function`.
- Use nouns for finite-element assets: `TimeFunction`,
  `TimeDependentDirichlet`.
- Keep physics-specific formulas outside `agentfem`; pass them in as callables,
  UFL expressions, or material/source objects from the application layer.

## What Belongs Here

- Generic FE operations: function spaces, dof maps, weak-form blocks, matrix/vector
  assembly, time-integration kernels, KSP solves, diagnostics, and output.
- Reusable physics primitives: standard strain/stress operators and material-law
  containers live under `constitutive`, not as first-level workflow modules.
- Application assets stay outside: geometry, case-specific sources, benchmark
  constants, and paper-specific parameters belong in applications or examples.

## Examples

Run examples from the parent directory of the source checkout:

```bash
python agentfem/examples/static_elasticity_2d.py
python agentfem/examples/wave_packet_plate_2d.py
```

The examples write XDMF output to `agentfem/examples_output/` for ParaView.
