# Install AgentFEM

AgentFEM supports Linux, macOS, and Windows through WSL2. It depends on the
compiled FEniCSx/DOLFINx stack, so use conda-forge unless you already maintain
a compatible MPI/PETSc/DOLFINx build.

For new users who do not want to manage a scientific Python environment,
AgentFEM's release workflow also produces self-contained offline runtime
installers for Apple Silicon macOS and Windows through WSL2. See
[Offline Runtime Installers](docs/runtime_installers.md). These packages embed
the compatible finite-element stack and do not contact conda, PyPI, or GitHub
during installation.

## Recommended Environment

```bash
mamba create -n agentfem-env --override-channels -c conda-forge \
  python=3.11 fenics-dolfinx=0.11 agentfem
mamba activate agentfem-env
agentfem doctor
```

This installs AgentFEM and the compatible FEniCSx/PETSc/MPI foundation in one
environment.

The explicit Python and DOLFINx versions are the current release-tested
runtime contract. They also prevent an older solver stack from being selected
from a stale package index.

## Mainland China Mirrors / 中国大陆镜像

For a new environment in mainland China, use one conda-forge mirror for the
entire compiled numerical stack. This one-shot command uses TUNA and leaves
the user's global conda configuration unchanged:

```bash
mamba create -n agentfem-env --no-rc --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  python=3.11 fenics-dolfinx=0.11 agentfem
mamba activate agentfem-env
agentfem doctor
```

If `mamba` is unavailable, use conda's equivalent command (conda does not
accept mamba's `--no-rc` option and dependency resolution will usually be
slower):

```bash
conda create -n agentfem-env --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  python=3.11 fenics-dolfinx=0.11 agentfem
```

On Windows, run either command inside Ubuntu on WSL2.

Before installing, the synchronized package can be checked without creating
an environment:

```bash
conda search --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  agentfem
```

For a reproducible run, append the required AgentFEM version, for example
`agentfem=0.3.1`, and retain the resulting environment specification with the
simulation evidence. When a newly published version is not yet visible, run
`conda clean -i` and query again after the mirror has synchronized. Do not mix
several conda mirrors or combine `defaults` and `conda-forge` packages in this
environment; PETSc, MPI, HDF5, and DOLFINx should be resolved as one stack.

The TUNA PyPI mirror is appropriate only when a compatible FEniCSx/PETSc
environment already exists:

```bash
python -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  agentfem
agentfem doctor
```

PyPI does not provision DOLFINx or PETSc for AgentFEM. A bare pip install is
therefore not the recommended first installation on any platform.

### Persistent project storage on WSL

When this environment is inside WSL, establish a durable project workspace:

```bash
agentfem workspace --protect
```

This keeps the familiar `~/AgentFEMProjects` path while storing its project
inputs, results, and checkpoints in Windows `Documents\AgentFEMProjects`.
Mamba upgrades then replace packages without owning project data, and removing
the WSL distribution cannot silently remove the protected workspace.

## PyPI and Source Development

PyPI remains available when AgentFEM needs to be installed into an existing
compatible FEniCSx environment:

```bash
mamba create -n agentfem-env --override-channels -c conda-forge \
  python=3.11 fenics-dolfinx=0.11 mpich mpi4py petsc4py h5py
mamba activate agentfem-env
```

```bash
python -m pip install agentfem
```

For source development, install from the repository root:

```bash
python -m pip install -e .
```

AgentFEM is available from conda-forge as a `noarch` package. Its numerical
dependencies remain platform-specific compiled packages. Conda resolves these
together; users should not mix a system MPI launcher with the environment's
MPI libraries.

## Windows

The recommended first-release route is WSL2 with Ubuntu, Miniforge/Mambaforge,
and the Linux environment above. This uses the same package family exercised
by AgentFEM's Linux CI.

Do not run `wsl --unregister` until `agentfem workspace --json` reports
`protected_from_distribution_removal: true`. For the Complete Runtime, use its
bundled `Remove-AgentFEM.ps1`; an ordinary Mamba environment may be removed
normally because its protected projects are independent of that environment.

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
