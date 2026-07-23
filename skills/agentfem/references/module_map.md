# Module Map Reference

- `mesh/`: mesh import/read/write, named regions, summaries, tag checks,
  tags, and measures
- `mesh/formats.py`: optional external CAE mesh conversion
- `spaces.py`: function spaces and named functions
- `fields.py`: application-level unknown bundles
- `kernel/dofs.py`: low-level dof lookup and field copying
- `constraints/`: constraint containers and semantic Dirichlet constructors
- `loads.py`: loads, natural boundary data, semantic constructors, and time functions
- `constitutive/`: local response relations
- `materials/`: material-property records and loaders
- `boundary_models/`: weak boundary models
- `forms.py`: UFL weak-form blocks
- `assembly.py`: matrix/vector/lumped assembly
- `operators/`: engineering-level K/M/C/F operators and system containers
- `time/`: time integration kernels, progress, and time-step cadence
- `problems.py`: problem summaries, system problems, and state containers
- `solvers.py`: PETSc solver wrappers
- `diagnostics.py`: norms and scalar diagnostics
- `io.py`: output writers and scalar logs
- `elements/`: element and integration-policy namespace
- `benchmarks/`: verification benchmark namespace
