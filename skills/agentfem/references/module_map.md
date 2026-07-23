# Module Map Reference

- `mesh.py`: mesh import/read/write, mesh tags, and measures
- `mesh_formats.py`: optional external CAE mesh conversion
- `spaces.py`: function spaces and named functions
- `dofs.py`: dof lookup and field copying
- `boundary.py`: low-level Dirichlet helpers
- `constraints.py`: constraint containers
- `loads.py`: loads and time functions
- `constitutive/`: material laws
- `boundary_models/`: weak boundary models
- `forms.py`: UFL weak-form blocks
- `assembly.py`: matrix/vector/lumped assembly
- `time.py`: time integration kernels
- `problems.py`: state and problem containers
- `runtime.py`: progress and time-step cadence
- `solvers.py`: PETSc solver wrappers
- `diagnostics.py`: norms and scalar diagnostics
- `io.py`: output helpers
