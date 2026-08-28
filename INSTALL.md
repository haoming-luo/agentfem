# Install AgentFEM

AgentFEM supports Linux, macOS, and a native Windows serial core. It depends on
the compiled FEniCSx/DOLFINx stack, so use conda-forge unless you already
maintain a compatible build.

## Recommended Environment

```bash
mamba create -n agentfem-env -c conda-forge \
  python=3.11 fenics-dolfinx=0.11 mpich mpi4py petsc4py h5py
mamba activate agentfem-env
```

Install AgentFEM from PyPI:

```bash
python -m pip install agentfem
```

For source development, install from the repository root:

```bash
python -m pip install -e .
```

The AgentFEM conda-forge recipe is currently in review. Until it is published,
conda supplies the compiled FEniCSx/PETSc/MPI runtime and PyPI or the source
tree supplies AgentFEM. Once the recipe is available, both layers can be
installed in one conda environment command.

## Windows

For native Windows, use Miniforge Prompt or PowerShell with conda initialized:

```powershell
mamba create -n agentfem-env -c conda-forge `
  python=3.11 fenics-dolfinx=0.11 mpi4py h5py scipy pyamg
mamba activate agentfem-env
python -m pip install agentfem
agentfem doctor
```

This route keeps the same AgentFEM public workflow and uses DOLFINx native
matrix/vector assembly with SciPy/PyAMG. It supports serial linear statics,
steady and transient heat transfer, linear implicit dynamics, explicit
dynamics, standard results, campaigns, and project/agent tooling.

PETSc, petsc4py, and dolfinx_mpc are not currently available as a complete
conda-forge win-64 stack. Native Windows therefore rejects PETSc nonlinear,
exact MPC, and distributed procedures before lowering with
`AFM-BACKEND-CAPABILITY-001`. Use WSL2, Linux, macOS, or a supported cluster for
those capabilities. This is one package and one scientific API, not a reduced
Windows fork.

## Smoke Test

From the `agentfem` repository directory:

```bash
python examples/static_elasticity_2d.py
```

From the parent development directory:

```bash
python agentfem/examples/static_elasticity_2d.py
```

Expected output:

```text
Model: cantilever_model
...
Static elasticity result: .../examples_output/static_elasticity_2d.xdmf
```

For a two-rank smoke test:

```bash
mpiexec -n 2 python examples/static_elasticity_2d.py
```

The C3D10H mixed macro-control reference currently runs in serial:

```bash
python examples/abaqus_c3d10h_periodic_cell/case.py \
  --displacement 0.20
```

Distributed, fully prescribed displacement-periodic formulations use the
separate `dolfinx_mpc` backend.

The launcher and `mpi4py` must come from the same MPI implementation. On macOS,
Homebrew Open MPI can appear before a conda MPICH launcher on `PATH`; check
`which mpiexec` and use the environment's `bin/mpiexec` when they differ.

## Optional Tools

- `meshio`: `python -m pip install 'agentfem[mesh-formats]'` for external CAE
  mesh conversion, including Abaqus `.inp` and NASTRAN `.bdf/.nas` formats.
- `gmsh`: `python -m pip install 'agentfem[gmsh]'` only for direct Gmsh model
  or `.msh` import. Gmsh is separately licensed under GPL-2.0-or-later with
  its published exception; it is not bundled in the Apache-2.0 AgentFEM core.
- `dolfinx_mpc`: `python -m pip install 'agentfem[parallel-mpc]'` for
  distributed multi-point constraints. Its minor version must match DOLFINx.
- `mkdocs`, `mkdocs-material`, `pymdown-extensions`: documentation site.
- `jupyterlab`, `ipykernel`: notebook workflows.

`requirements.txt` records the tested MVP stack, but direct pip installation of
DOLFINx/PETSc/MPI packages may be fragile across platforms.

Inspect the actual runtime before reporting an installation issue. The report
includes the exact Python executable and imported AgentFEM directory, which
reveals when a source checkout shadows another installed release:

```python
from agentfem import platforms
print(platforms.runtime_report().format())
```
