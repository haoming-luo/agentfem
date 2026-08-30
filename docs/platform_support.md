# Platform and Optional-Dependency Support

## First-release support policy

AgentFEM reports platforms by executable evidence, not by whether Python can
theoretically import one package.

| Route | 26 August 2026 level | Evidence and boundary |
| --- | --- | --- |
| Native Linux | CI verified | Full tests, two-rank MPI, wheel and release Demo gates run on Ubuntu. |
| Native macOS | Developer verified | Maintainer development and release verification run on Apple Silicon; AgentFEM verifies the MPI launcher against `mpi4py`. |
| Windows through WSL2 | Recommended Windows route | Uses the Linux FEniCSx/PETSc/MPI stack exercised by CI. Host GUI and filesystem integration remain local setup concerns. |
| Native Windows | Supported serial core | Hosted Windows CI exercises the installed wheel through DOLFINx native assembly and SciPy/PyAMG. Linear statics, heat, linear/explicit dynamics, results, campaigns, and agent tooling share the public API; PETSc nonlinear, exact MPC, and distributed routes are excluded. |

The current machine can produce a compact issue-report record:

```python
from agentfem import platforms

print(platforms.runtime_report().format())
```

This reports the operating-system route, Python and core-package versions,
the exact interpreter and imported AgentFEM directory, any installed-package
shadowing or stale installed-version metadata, the environment-matched MPI
launcher, and the availability of meshio, Gmsh, PyVista, PyTorch, and
`dolfinx_mpc`.

If `doctor` reports `AFM-MPI-LAUNCHER-RECOVERED`, the incompatible `PATH`
launcher has already been bypassed. Use `agentfem run --mpi N` for a project or
`agentfem mpi-run -n N -- ...` for another program. A PATH launcher from a
different MPI implementation can start ordinary processes but cannot safely
initialize the active `mpi4py` runtime. `MISMATCH` and `MISSING` are fail-closed
environment errors rather than numerical failures.

## Installed-wheel acceptance

Platform support can be recorded with the same release workflow on every
machine. From a clean checkout and a compatible FEniCSx environment:

```bash
python -m build
python -m pip install --no-deps --force-reinstall dist/*.whl
python release_gate.py --dist dist --smoke \
  --report agent-acceptance.json \
  --platform-report platform-acceptance.json
```

The platform report records the route, exact wheel hash, Python/FEniCSx/PETSc/
MPI identity, clean source commit and verified installed templates. GitHub's
platform-acceptance workflow produces Linux and macOS artifacts, including a
two-rank installed-wheel smoke, and aggregates them into a promotion snapshot.
WSL2 promotion
still requires the same command inside a real WSL2 environment; native Linux
or an ordinary Windows runner is not silently relabelled as Windows-through-
WSL2 evidence.

Promotion evidence is candidate-specific. The audit rejects a passed record
whose AgentFEM version or source commit differs from the checkout being
promoted, and platform evidence additionally rejects a dirty source tree.
Historical acceptance records remain useful release history but cannot be
reused to promote a later candidate.

### Real WSL2 acceptance

From Windows PowerShell, first prove that the selected distribution is WSL2:

```powershell
wsl --list --verbose
```

Then enter that distribution, keep the checkout in its Linux filesystem,
activate the FEniCSx environment, and run:

```bash
bash tools/run_wsl2_acceptance.sh wsl2-acceptance
```

The script checks for a `microsoft-standard-WSL2` kernel, builds one immutable
wheel/sdist pair, runs the installed release workflows, and performs a
two-rank MPI smoke. Internally it uses the fail-closed command:

```bash
python release_gate.py --dist DIST --smoke --mpi-ranks 2 \
  --require-platform wsl2 \
  --report agent-acceptance.json \
  --platform-report platform-acceptance-wsl2.json
```

Running that command on native Linux or WSL1 is an error, so its JSON cannot
be produced by relabelling ordinary Ubuntu evidence. The resulting platform
record contains the exact wheel hash, kernel route, distribution name when
available, MPI launcher and rank count, package versions, templates and
result-provenance checks.

Downloaded platform, extension and agent-trial artifacts can be audited
together without hand-maintaining a long argument list:

```bash
python promotion_gate.py \
  --evidence-directory promotion-evidence \
  --report promotion.json --require-complete
```

## Windows numerical runtimes

AgentFEM detects one of two FEniCSx numerical runtimes:

- `fenicsx-petsc`: full nonlinear, exact-MPC (when installed), and MPI route;
- `fenicsx-native-serial`: PETSc-free DOLFINx native assembly plus SciPy/PyAMG.

Both consume the same Study, Model, operators, constraints, Step, result, and
verification objects. Provider capabilities decide whether a procedure can be
lowered; application code does not branch on the operating system.

Runtime selection is normally automatic. CI or advanced diagnostics may set
`AGENTFEM_RUNTIME=fenicsx-petsc` or
`AGENTFEM_RUNTIME=fenicsx-native-serial`. An unknown name, or an explicitly
requested runtime whose required package is unavailable, fails before model
lowering instead of silently changing the numerical route.

The native linear policy is equally fail-closed. `direct_solver()` maps to
SciPy SuperLU through `spsolve`; iterative CG, GMRES, and BiCGStab support no
preconditioner, diagonal Jacobi, or the documented PyAMG aliases. Unsupported
PETSc factor packages and preconditioners are rejected rather than silently
ignored. Non-finite solutions, singular-matrix warnings, and residuals outside
the declared tolerance are recorded as non-converged solves.

`agentfem run --mpi N` also checks the active runtime before launching ranks.
The native serial profile therefore rejects `N > 1` immediately; the generic
`agentfem mpi-run` command remains available for unrelated MPI programs when
the local launcher is coherent.

Named project execution profiles are a separate operational layer. A `local`
profile can resolve automatically to the native Windows runtime, while a
`cluster` profile requests PETSc and MPI on the destination machine. The same
project can be transported in a verified `.afm` bundle; see
[Portable Projects and Execution Profiles](project_portability.md).

Use native Windows for the supported serial core described in `INSTALL.md`.
Use WSL2 when a Windows workstation needs the complete Linux
FEniCSx/PETSc/MPI stack. Keep WSL projects in the Linux filesystem for compile
and I/O performance. ParaView may run in WSLg or on Windows.

Native Windows promotion beyond the serial core requires:

1. a conda-forge PETSc/petsc4py or equivalent nonlinear provider;
2. a distributed Windows MPI contract exercised by AgentFEM;
3. exact MPC support or a verified alternative;
4. nonlinear and distributed benchmark evidence equal to the full runtime.

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
- [DOLFINx native assembly API](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.fem.html)
- [DOLFINx PyAMG demo](https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_pyamg.html)
- [Gmsh licensing](https://gmsh.info/#Licensing)
- [Gmsh license text](https://gmsh.info/LICENSE.txt)
