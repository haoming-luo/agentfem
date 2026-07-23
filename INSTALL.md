# Install AgentFEM

AgentFEM depends on the FEniCSx/DOLFINx stack. For the MVP release, use
conda-forge unless you already maintain a compatible MPI/PETSc/DOLFINx build.

## Recommended Environment

```bash
mamba create -n agentfem-env -c conda-forge \
  python=3.11 fenics-dolfinx=0.11 gmsh mpi4py petsc4py \
  meshio matplotlib jupyterlab ipykernel
mamba activate agentfem-env
```

Then install AgentFEM from the repository root:

```bash
python -m pip install -e .
```

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

## Optional Tools

- `meshio`: external CAE mesh conversion, including Abaqus `.inp` and NASTRAN
  `.bdf/.nas` formats where supported by meshio.
- `mkdocs`, `mkdocs-material`, `pymdown-extensions`: documentation site.
- `jupyterlab`, `ipykernel`: notebook workflows.

`requirements.txt` records the tested MVP stack, but direct pip installation of
DOLFINx/PETSc/MPI packages may be fragile across platforms.
