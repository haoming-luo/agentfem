# Platform and Optional-Dependency Support

## First-release support policy

AgentFEM reports platforms by executable evidence, not by whether Python can
theoretically import one package.

| Route | 26 August 2026 level | Evidence and boundary |
| --- | --- | --- |
| Native Linux | CI verified | Full tests, two-rank MPI, wheel and release Demo gates run on Ubuntu. |
| Native macOS | Developer verified | Maintainer development and release verification run on Apple Silicon; the MPI launcher must match `mpi4py`. |
| Windows through WSL2 | Recommended Windows route | Uses the Linux FEniCSx/PETSc/MPI stack exercised by CI. Host GUI and filesystem integration remain local setup concerns. |
| Native Windows | Experimental | FEniCSx 0.11 has `win-64` builds, but AgentFEM's PETSc route and `dolfinx_mpc` 0.11 do not form a complete conda-forge native-Windows stack, and AgentFEM has no native-Windows CI gate. |

The current machine can produce a compact issue-report record:

```python
from agentfem import platforms

print(platforms.runtime_report().format())
```

This reports the operating-system route, Python and core-package versions,
and the availability of meshio, Gmsh, PyVista, PyTorch, and `dolfinx_mpc`.

## Windows recommendation

Install WSL2 with Ubuntu, install Miniforge inside WSL, and follow
`INSTALL.md`. Keep the project inside the WSL Linux filesystem for compilation
and I/O performance. ParaView may run either inside WSLg or on Windows while
opening output copied or exposed from the WSL filesystem.

Native Windows should become supported only after all of the following exist:

1. a supported linear/nonlinear algebra provider for AgentFEM's public steps;
2. serial solve, XDMF/HDF5, JIT-form, and result tests on `windows-latest`;
3. a documented MPI story;
4. an explicit limitation or replacement for workflows requiring
   `dolfinx_mpc`;
5. an installed-wheel Demo gate.

## Gmsh is an optional adapter

AgentFEM has three independent mesh routes:

1. DOLFINx structured meshes such as `mesh.rectangle(...)` require no Gmsh.
2. XDMF meshes and meshio-based Abaqus/NASTRAN conversion require no Gmsh.
3. `mesh.read_gmsh_mesh(...)` and in-memory Gmsh models require the optional
   Gmsh Python API.

Install only the route required by the model:

```bash
python -m pip install 'agentfem[mesh-formats]'
python -m pip install 'agentfem[gmsh]'
```

The Gmsh project distributes its software under GPL-2.0-or-later with a
published exception. AgentFEM does not vendor or bundle Gmsh, does not list it
as a core dependency, and loads it only at the direct-Gmsh capability
boundary. Distributors assembling a combined product remain responsible for
reviewing the licenses of every component they distribute.

Primary references:

- [FEniCSx download and installation](https://fenicsproject.org/download/)
- [DOLFINx installation dependencies](https://docs.fenicsproject.org/dolfinx/main/python/installation.html)
- [Gmsh licensing](https://gmsh.info/#Licensing)
- [Gmsh license text](https://gmsh.info/LICENSE.txt)
