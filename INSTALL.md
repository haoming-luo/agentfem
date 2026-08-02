# Install AgentFEM

AgentFEM depends on the FEniCSx/DOLFINx stack. For the MVP release, use
conda-forge unless you already maintain a compatible MPI/PETSc/DOLFINx build.

## Recommended Environment

```bash
mamba create -n agentfem-env -c conda-forge \
  python=3.11 fenics-dolfinx=0.11 mpich mpi4py petsc4py h5py
mamba activate agentfem-env
```

Install the released AgentFEM wheel from PyPI:

```bash
python -m pip install agentfem
```

For source development, install from the repository root:

```bash
python -m pip install -e .
```

AgentFEM is not currently packaged on conda-forge. Conda supplies the compiled
FEniCSx/PETSc/MPI runtime; PyPI or the source tree supplies AgentFEM.

## Windows

The recommended first-release route is WSL2 with Ubuntu, Miniforge/Mambaforge,
and the Linux environment above. This uses the same package family exercised
by AgentFEM's Linux CI.

Native Windows is experimental, not a release-supported route. FEniCSx 0.11
has `win-64` packages, but AgentFEM currently uses PETSc-based solver APIs and
the distributed Abaqus-equation example uses `dolfinx_mpc`; the corresponding
complete conda-forge Windows stack is not available and AgentFEM has no native
Windows CI gate. A hand-built expert environment may work for a subset, but it
is not an installation promise.

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

The Abaqus `*EQUATION` periodic-cell example also supports within-case MPI:

```bash
mpiexec -n 2 python \
  examples/abaqus_c3d10_periodic_cell/agentfem_periodic_hyperelastic.py \
  --stretch 1.05
```

This path requires `dolfinx_mpc` with the same minor version as DOLFINx.

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

Inspect the actual runtime before reporting an installation issue:

```python
from agentfem import platforms
print(platforms.runtime_report().format())
```
