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

## Mirror Channel / 镜像通道

When the user explicitly asks to "use the mirror channel", or the canonical
source is unreachable, use one conda-forge mirror for the entire compiled
numerical stack. Do not infer the route from an IP address or VPN. This one-shot
command uses TUNA and leaves the user's global conda configuration unchanged:

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

### Start here

Choose **one** route; keep it through restarts. You do not need native Windows
Python, Docker, or a second Linux distribution to use a healthy existing WSL2
environment.

| Your starting point | Next step |
| --- | --- |
| A compatible Complete Runtime ZIP | Extract it fully and open `Install-AgentFEM.cmd`. It owns platform preparation, recovery and first-solve verification. |
| Ubuntu/WSL2 and Miniforge already work | Use the guided environment setup below; it reuses existing packages instead of reinstalling them. |
| New Windows computer, using the mirror route | Follow [Windows agent installation](docs/windows_agent_installation.md): prepare WSL, privately finish the Ubuntu account prompt, then use the guided setup. |

Windows may ask for administrator approval, a restart, or a Linux password.
These are distinct steps, not repeated failures. Save your work before restarting;
keep the same installer and options. Enter passwords only in the local account
prompt, never in an AI chat. An agent-host permission refusal needs its scoped
approval, not another shell or disabled security controls.

### Guided Ubuntu / mirror setup

From a trusted source checkout, run **inside Ubuntu** with its existing Python 3
and Linux Miniforge. This entry point does not need AgentFEM installed yet:

```bash
python3 packaging/runtime/setup_environment.py --channel tuna
python3 packaging/runtime/setup_environment.py --channel tuna --apply
```

The first command is a plan, with no package changes; the second creates a
missing environment or verifies the existing `agentfem-env`, protects the
conventional WSL project workspace, and runs a small installed-template solve.
It never upgrades or repairs an existing environment implicitly. If needed,
select Miniforge explicitly with `--manager ~/miniforge3/bin/mamba`, select an
existing environment with `--prefix /absolute/environment/path`, or request an
exact published package with `--version X.Y.Z`. Use `--channel conda-forge`
for the canonical source. These options do not change global conda settings.

If Miniforge is missing, first obtain its installer from the
[official distribution](https://github.com/conda-forge/miniforge#download),
verify the published checksum, and install it in your Linux account. The TUNA
package channel is not a promise of a Miniforge installer download mirror; do
not invent one. The setup command reports this prerequisite without downloading
or executing an unverified bootstrapper.

Keep the same command after interruption. A completed environment is reused,
each attempt retains its own logs and first-solve project under
`~/.local/state/agentfem/setup/`, and an incomplete environment is left intact
for inspection. Do not delete it or start another package transaction while
one is active. `--json` gives agents one structured final result and paths to
the detailed stage logs. A saved completion is not a new health check.

The setup does not modify Windows, perform automatic reboot, remove a distro,
or register startup tasks. Its small first-solve project/cache stay in Linux
storage; the ordinary project workspace is protected on Windows. Preserve
valuable Linux-only data before removing a distribution. No optional heavy
visualization or learning packages are installed by this core setup.

When a compatible published Windows Complete Runtime is available, open its
bundled `Install-AgentFEM.cmd` after fully extracting the ZIP. An explicitly
requested Ubuntu + mirror route remains supported; it does not require this
bundle or a development candidate package.
For AI-assisted installation and WorkBuddy permissions, follow
[Windows agent installation](docs/windows_agent_installation.md).
The Complete Runtime installer
prepares WSL without downloading Ubuntu and then imports the bundled AgentFEM
image. Existing WSL users may instead use Ubuntu, Miniforge, and the
Linux environment above; this uses the same package family exercised by
AgentFEM's Linux CI.

Agents can use the source installer's `-PrepareWslOnly` option to prepare only
the Windows platform before an Ubuntu + mirror installation. This does not
install the scientific software or replace existing distributions. See the
Windows agent guide for non-interactive handoffs and structured progress;
users still approve platform changes and enter their own account credentials.

The published v0.3.6 Windows Preview has one known first-solve issue: its
exported WSL image may not expose a C compiler to FEniCSx. Repair that existing
runtime in place, without touching project data, from its WSL terminal:

```bash
sudo apt-get update
sudo apt-get install -y gcc g++
agentfem doctor
```

The corrected runtime contract installs the toolchain in the exported root
filesystem and verifies a cold-cache finite-element solve after re-importing
the final `.wsl` artifact. The
[first successful Windows field report](https://github.com/haoming-luo/agentfem/discussions/3)
records a clean-host WSL2 + TUNA installation through a completed solve. TUNA
briefly lagged the release during that run and now carries AgentFEM 0.3.6.

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
