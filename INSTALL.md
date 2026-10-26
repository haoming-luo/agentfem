# Install AgentFEM

AgentFEM supports Linux, macOS, and Windows through WSL2. It depends on the
compiled FEniCSx/DOLFINx stack, so use conda-forge unless you already maintain
a compatible MPI/PETSc/DOLFINx build.

Windows users: start with the [three-step Windows route](#windows), without
cloning this repository. For an already downloaded compatible runtime bundle,
see the optional offline route below.

For new users who do not want to manage a scientific Python environment,
AgentFEM's release workflow also produces self-contained offline runtime
installers for Apple Silicon macOS and Windows through WSL2. See
[Offline Runtime Installers](docs/runtime_installers.md). These packages embed
the compatible finite-element stack and do not contact conda, PyPI, or GitHub
during installation.

## Recommended Environment

```bash
mamba create -n agentfem-env --override-channels -c conda-forge \
  python=3.11 fenics-dolfinx=0.11 c-compiler cxx-compiler agentfem
mamba activate agentfem-env
agentfem doctor
```

This installs AgentFEM and the compatible FEniCSx/PETSc/MPI foundation in one
environment.

The explicit Python and DOLFINx versions are the current release-tested
runtime contract. They also prevent an older solver stack from being selected
from a stale package index.

## Mirror Channel / 镜像通道

When the user explicitly asks to "use the mirror channel", or the canonical
source is unreachable, use one conda-forge mirror for the entire compiled
numerical stack. Do not infer the route from an IP address or VPN. This one-shot
command uses TUNA and leaves the user's global conda configuration unchanged:

```bash
mamba create -n agentfem-env --no-rc --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  python=3.11 fenics-dolfinx=0.11 c-compiler cxx-compiler agentfem
mamba activate agentfem-env
agentfem doctor
```

If `mamba` is unavailable, use conda's equivalent command (conda does not
accept mamba's `--no-rc` option and dependency resolution will usually be
slower):

```bash
conda create -n agentfem-env --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
  python=3.11 fenics-dolfinx=0.11 c-compiler cxx-compiler agentfem
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
`agentfem=0.3.3`, and retain the resulting environment specification with the
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

**WSL → Ubuntu → AgentFEM. No GitHub clone or AgentFEM installer is required.**
Already have working Ubuntu/WSL2? Skip straight to the
[mirror environment command](#mirror-channel--镜像通道) and [first solve](#smoke-test).

For a new machine using the domestic mirror route:

1. **Prepare WSL.** Let your agent inspect and do the authorised work.
   If it cannot obtain administrator permission, open Terminal (Administrator)
   and run the command below. Restart only if Windows asks.
2. **Add Ubuntu from TUNA.** Use the
   [checked, copy-safe image import](docs/windows_agent_installation.md#2-add-ubuntu-from-the-mirror-only-if-absent).
   This avoids the GitHub-hosted distribution-list download. The agent can
   prepare a normal account automatically; no shared password or daily root use.
3. **Install and verify.** Reuse/install Linux Miniforge from its verified
   [TUNA mirror](https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/),
   use the single-channel command above, then doctor → workspace protection →
   [installed first solve](#smoke-test).

```powershell
wsl --install --no-distribution --web-download
```

国内用户：不用先下载源码，不用默认进入“Windows 功能”勾选。
让智能体完成允许的操作；权限不足时，在管理员终端执行上面一条命令，
按系统提示重启，然后继续镜像安装。详细命令由
[Windows 简明指引](docs/windows_agent_installation.md)统一维护。

`--web-download` is a non-Store option, not a domestic mirror or a speed
guarantee. If its download fails, the Store platform route is
`wsl --install --no-distribution`. A denied permission or restart request is
not a reason to try another launcher. Standard `wsl --install` normally
installs Ubuntu; `--no-distribution` intentionally leaves that to the mirror.
Do not repeatedly retry `wsl --install -d Ubuntu` after metadata timeouts.
An installed Store App alone does not prove a registered, working WSL2 distro.

Existing environments are reused, not overwritten. A Store/bundle account
prompt stays private; imported Ubuntu can use automated non-root preparation.
Do not remove a distribution to repair an installation. Check
`agentfem workspace --json` before any planned removal; its
`protected_from_distribution_removal` must be true, and other Linux-only data
must also be backed up.

### Optional tools, not prerequisites

Already have a compatible published Complete Runtime ZIP? Extract it fully and
open `Install-AgentFEM.cmd`; it supplies its own Linux image, so do not also
download Ubuntu. Already have a trusted source checkout? Its
`packaging/runtime/setup_environment.py --channel tuna` can plan the Linux
environment and `--apply --json` can install/verify it with logs.
Neither a source checkout nor an offline bundle is required for ordinary users.
See the [optional installer reference](docs/windows_installer_reference.md)
for status, interrupted installs and explicit recovery.

The [Windows field reports](https://github.com/haoming-luo/agentfem/discussions/3)
record successful mirror installs. Package versions and mirror synchronization
change; query the selected channel rather than assuming an old report's
version is current. A successful basic solve on an older version does not
establish compatibility with a project requiring newer APIs.

Native Windows remains experimental. The supported Windows numerical runtime
is Linux through WSL2; Windows PowerShell installer tests do not certify a
native Windows FEniCSx/PETSc solver stack.

## Smoke Test

For an installed environment, no repository clone or downloaded example is
required. From a Linux/WSL terminal (or an activated macOS environment):

```bash
agentfem init --template static-solid ~/AgentFEMProjects/install-check
agentfem check --project ~/AgentFEMProjects/install-check --json
agentfem run --project ~/AgentFEMProjects/install-check --name first-solve
agentfem verify --project ~/AgentFEMProjects/install-check --json
```

Use a new directory, or inspect and reuse your own existing check project;
do not delete a user's project to repeat the check. For non-interactive agent
tools, prefix each command with `mamba run -n agentfem-env` using the installed
Mamba executable. Shell activation in one tool call is not guaranteed to carry
over to the next. A successful `doctor` alone is not a successful finite-element
solve: report the run and verification results separately.

For developers working from a repository checkout, the equivalent example is:

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

Keep the user's chosen mirror for later optional installs. Record the choice
in a local project/agent note, not global conda/pip configuration. On the TUNA
route, for example (inspect the plan before confirming package changes):

```bash
mamba install -n agentfem-env --no-rc --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge matplotlib
mamba run -n agentfem-env python -m pip install 'agentfem[mesh-formats]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Respect the project's AgentFEM version pin when selecting extras. Pip extras
are for an existing compatible solver environment, not a substitute for it.

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
