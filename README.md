# AgentFEM

AgentFEM is an agent-oriented finite-element workflow platform built on
DOLFINx/PETSc. The Python package is named `agentfem`, matching the platform
name used by the documentation and application code.

The current implementation is strongest in reusable FEM workflow steps,
linear-elastic material properties and constitutive relations, explicit
dynamics, weak loads, constraints, absorbing boundary models, diagnostics, and
ParaView output.

## Agent Entry Points

- `AGENT_GUIDE.md`: first file for AI agents working with AgentFEM.
- `WORKFLOW.md`: standard FEM workflow and module order.
- `CONCEPTS.md`: shared vocabulary for humans and agents.
- `docs/`: detailed platform rules and extension guidance.
- `examples/`: runnable workflow examples for researchers and agents.
- `materials/`: data-backed material library.
- `skills/agentfem/`: skill-ready progressive-disclosure package.
- `mkdocs.yml`: optional static documentation-site configuration.

## Standard Workflow

1. Build or read a mesh with application geometry plus `mesh` import helpers.
2. Inspect mesh size and required tags with `mesh.summarize_mesh(...)`,
   `mesh.require_cell_tags(...)`, and `mesh.require_facet_tags(...)`.
3. Define named geometric regions with `mesh.boundary(...)` or mesh tags.
4. Create application unknowns with `fields`, or low-level spaces with `spaces`.
5. Define a lightweight `problems.FEMProblem` when a workflow needs a structured
   model summary for logs, reviews, or agents.
6. Construct strong constraints with `constraints`.
7. Construct weak loads such as body forces, Neumann fluxes, and tractions with
   `loads`.
8. Build weak forms with application physics plus reusable `forms` blocks.
9. Compile and assemble with `assembly`.
10. Step in time with `time.TimeStepper` plus kernels from `time`, or solve
   algebraic systems with `solvers`.
11. Measure distributed diagnostics with `diagnostics`.
12. Manage output locations with `io`.

## Modules

- `mesh`: mesh import/read/write, named boundary regions, boundary facet
  location, facet tags, boundary measures, structured mesh constructors, mesh
  summaries, and tag checks
- `mesh.formats`: optional external CAE mesh conversion through `meshio`
- `spaces`: scalar/vector spaces, Lagrange space constructors, trial/test
  functions, named fields
- `fields`: application-level unknown bundles such as displacement and
  temperature
- `kernel.dofs`: internal dof location, owned/ghost arrays, field copies
- `constraints`: Dirichlet constraint containers and periodic constraint specs
- `constitutive`: stress-strain, flux-gradient, and other local response
  relations
- `materials`: reusable material-property records, property containers, and
  loaders
- `boundary_models`: reusable weak boundary models, such as absorbing boundaries
- `loads`: time functions, time-dependent Dirichlet data, body forces,
  tractions, fluxes, heat sources, semantic constructors, and load sets
- `forms`: common UFL virtual-work building blocks
- `assembly`: form compilation, vector/matrix assembly, lumped operators/mass
- `operators`: engineering-level K/M/C/F operators and system containers
- `time`: time-step cadence, progress metadata, elapsed-time formatting, and
  explicit central-difference/Newmark kernels
- `problems`: standard problem descriptions and state containers
- `solvers`: PETSc KSP setup and standard linear problem solve
- `diagnostics`: distributed norms, energy-like quantities, and named scalar
  diagnostic sets
- `io`: MPI-safe directories, CSV logs, XDMF time-series output, and named
  result writers
- `elements`: reserved namespace for reusable element and integration policies
- `benchmarks`: reserved namespace for standard verification problems

## Core API Direction

- Use `assembly.assemble_lumped_operator(V, coefficient, measure)` for generic
  diagonal operators on a function space `V`.
- Use `mesh.import_gmsh_model(...)` or `mesh.read_gmsh_mesh(...)` to convert
  Gmsh meshes into DOLFINx meshes with physical tags.
- Use `mesh.rectangle(lower, upper, cells, cell_type="quadrilateral")` for a
  structured rectangular mesh in beginner examples. Choose the analysis element
  degree later with `fields` or `spaces`.
- Use `mesh.summarize_mesh(...)` or `FEMMesh.summary()` before assembly when
  an agent or reviewer needs to confirm dimensions, cell counts, and tags.
- Use `mesh.require_cell_tags(...)` and `mesh.require_facet_tags(...)` when an
  input file must contain specific material or boundary labels.
- Use `spaces.scalar_space(...)` and `spaces.vector_space(...)` as concise
  Lagrange defaults; use `lagrange_space(...)` or `vector_lagrange_space(...)`
  when the element family should be explicit in code.
- Use `fields.displacement(...)` and `fields.temperature(...)` in beginner
  application workflows to avoid exposing trial/test bookkeeping.
- Use `mesh.boundary(...)` to name geometric locations before applying loads
  or constraints.
- Use `constraints.fixed(target, location=..., value=...)` for application-level
  fixed-value boundary conditions. Vector fields default to all components; use
  `components=0` or `components=(0, 1)` for selected components.
- Use `mesh.read_xdmf_mesh(...)` and `mesh.write_xdmf_mesh(...)` for DOLFINx
  XDMF mesh exchange.
- Use `mesh.convert_external_mesh_to_xdmf(...)` or
  `mesh.formats.convert_to_xdmf(...)` for Abaqus `.inp`, NASTRAN `.bdf/.nas`,
  VTK, and other meshio-supported external CAE mesh formats.
- Use `assembly.assemble_lumped_mass(V, density, measure)` as the mass-specific
  wrapper.
- Use `problems.TransientState` for first-order transient unknowns.
- Use `problems.SecondOrderDynamicsState` for displacement/velocity/acceleration
  dynamics. `ExplicitDynamicsState` remains as a compatibility alias.
- Use `constitutive.elasticity.estimate_elastic_wave_speeds(material)` when a
  wave solver needs scalar wave-speed estimates from an elastic material.
- Use `constitutive.elasticity.stress(displacement, properties)` to apply an
  elastic constitutive relation to a displacement field.
- Use `materials.load_material(name, model=...)` for reusable material records
  stored in material-centered JSON files under `materials/data`; keep material
  parameters in SI units.
- Use `constraints` for strong constraints. Use `loads.NeumannLoad` or
  `loads.BoundaryLoad` for natural boundary data that enters the weak form.
- Use `loads.traction(...)`, `loads.body_force(...)`, `loads.heat_flux(...)`,
  and `loads.heat_source(...)` as readable natural-load constructors in
  application code.
- Use `problems.FEMProblem(...).summary()` to expose the mesh, spaces,
  materials, constraints, loads, boundary models, and forms in a single
  inspectable record.
- Use `io.ResultWriter(path, domain, fields)` when a workflow writes the same
  fields repeatedly.
- Use `io.interpolate_for_xdmf(...)` when a higher-order analysis field must be
  written to XDMF on a linear mesh.
- Use `operators.stiffness_operator(...)`, `operators.mass_operator(...)`,
  `operators.force_vector(...)`, and `operators.LinearSystem(...)` when a
  workflow should be written in engineering matrix notation such as `K x = F`.
- Use `problems.LinearSystemProblem(system=..., solution=...)` to solve an
  operator-level `K x = F` system without manually handling variational forms.

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
- Material-property records live under `materials`; constitutive equations live
  under `constitutive`.
- Future element, operator-family, and benchmark assets belong under
  `elements`, `operators`, and `benchmarks` rather than crowding the first-level
  workflow modules.
- Application assets stay outside: geometry, case-specific sources, benchmark
  constants, and paper-specific parameters belong in applications or examples.

## Materials And Constitutive Laws

Material properties and constitutive laws are deliberately separate:

```python
from agentfem.constitutive import elasticity

properties = elasticity.isotropic_elastic(
    name="my_material",
    young=250e9,
    poisson=0.23,
    density=3100.0,
)
sigma = elasticity.stress(displacement, properties)
```

Here `properties` stores constants, while `elasticity.stress(...)` is the
stress-strain relation. This keeps material data reusable across different
analysis workflows.

## Operator Layer

The operator layer is for users who think in finite-element matrices and
vectors:

```python
from agentfem import constraints, fields, loads, mesh, operators

displacement = fields.displacement(domain, degree=1)
left_boundary = mesh.boundary(domain, left_marker, name="left")
right_boundary = mesh.boundary(domain, right_marker, name="right")
fixed_left = constraints.fixed(
    displacement,
    location=left_boundary,
    value=0.0,
)
right_traction = loads.traction(
    value=(0.0, -1.0e6),
    location=right_boundary,
)
K = operators.stiffness_operator(displacement, properties)
F = operators.force_vector(
    target=displacement,
    loads=[right_traction],
)
system = operators.LinearSystem(stiffness=K, force=F)
problem = problems.LinearSystemProblem(
    system=system,
    unknown=displacement,
    bcs=fixed_left.bcs,
)
problem.solve()
```

The system still exposes forms internally for DOLFINx assembly/solve, but
application code can be read as `K x = F`. Dynamics workflows can use
`SecondOrderSystem` to record `M`, `C`, `K`, and `F`.

For beginner application code, prefer unknown bundles:

```python
from agentfem import fields, loads, mesh, operators

displacement = fields.displacement(domain, degree=1)
K = operators.stiffness_operator(displacement, properties)
right_boundary = mesh.boundary(domain, right_marker, name="right")
right_traction = loads.traction(
    value=(0.0, -1.0e6),
    location=right_boundary,
)
F = operators.force_vector(
    target=displacement,
    loads=[right_traction],
)
```

The bundle stores the function space, solution field, trial function, and test
function internally. Advanced workflows can still use `spaces` directly.

## Material Library

The material library is material-centered and data-backed. Edit one JSON file
per material under `materials/data/` when adding persistent material records:

```python
from agentfem.materials import list_material_models, list_materials, load_material

print(list_materials(model="isotropic_linear_elastic"))
print(list_material_models("silicon_nitride_generic"))
material = load_material("silicon_nitride_generic", model="isotropic_linear_elastic")
```

Each record must declare `id`, `display_name`, `family`, `unit_system: "SI"`,
`source`, and a `models` mapping. The loader validates records and returns
material-property objects suitable for `agentfem.constitutive` relations.

## Examples

Run examples from the parent directory of the source checkout:

```bash
python agentfem/examples/static_elasticity_2d.py
python agentfem/examples/transient_heat_2d.py
python agentfem/examples/wave_packet_plate_2d.py
```

The examples write XDMF output to `agentfem/examples_output/` for ParaView.
