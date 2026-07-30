# Module Map Reference

- `mesh/`: mesh import/read/write, boundary/cell regions, summaries, tag
  checks, tags, and measures
- `mesh/formats.py`: optional external CAE mesh conversion
- `studies.py`: analysis context, physics, dimensions, and modeling assumptions
- `models.py`: lightweight model registry, amplitudes, material assignments,
  checks, summaries, and model-first operators
- `spaces.py`: function spaces and named functions
- `fields.py`: application-level unknown bundles
- `amplitudes.py`: time histories and scale factors for prescribed data
- `kernel/dofs.py`: low-level dof lookup and field copying
- `constraints/`: constraint containers and semantic Dirichlet constructors
- `loads.py`: loads, natural boundary data, and semantic constructors
- `constitutive/`: local response relations
- `materials/`: material-property records and loaders
- `boundary_models/`: weak boundary models
- `forms.py`: UFL weak-form blocks
- `assembly.py`: matrix/vector/lumped assembly
- `operators/`: engineering-level K/M/C/F operators and system containers
- `time/`: explicit/implicit time integration routes, progress, and cadence
- `problems.py`: analysis steps, system problems, and state containers
- `solvers.py`: PETSc solver wrappers
- `diagnostics.py`: norms and scalar diagnostics
- `io.py`: output writers and scalar logs
- `elements/`: element and integration-policy namespace
- `benchmarks/`: verification benchmark namespace
- `ir/`: experimental versioned AF-IR scientific records
- `validation.py`: structured issue codes, paths, hints, and reports
- `backends/`: backend descriptors and advanced lowering adapters
- `campaigns/`: typed parameters, deterministic sampling, case plans,
  resumable execution, and MPI-aware persistence
- `datasets/`: unit/shape-aware learning data and case provenance
- `surrogates/`: baseline learned/ROM models, validation, applicability guards,
  and neural-operator/PINN contracts
