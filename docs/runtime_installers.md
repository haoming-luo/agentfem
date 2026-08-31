# Offline Runtime Installers

AgentFEM's normal Python package remains available through conda-forge and
PyPI. Runtime installers serve a different audience: a new user can install a
complete, compatible finite-element environment without solving packages or
cloning a repository.

## Runtime targets

| Host | Artifact | Status |
| --- | --- | --- |
| Windows 10/11 with WSL2 | `AgentFEM-Complete-<version>-WSL2-x86_64-preview-offline.zip` | Primary public Preview; accepted in the Linux image, pending real WSL2 evidence |
| Apple Silicon macOS | `AgentFEM-Complete-<version>-macOS-arm64-unsigned-preview.pkg` | Public Preview; installed and exercised on a clean Apple Silicon runner, explicitly not signed or notarized |
| Intel macOS | `.pkg` | Demand-driven follow-up |
| Native Windows | `.exe` | Experimental until the complete PETSc/MPI/MPC gate passes |

The runtime includes Python, FEniCSx/DOLFINx, PETSc, MPI, HDF5, a C compiler
for cold-cache form JIT, and the exact AgentFEM wheel associated with the
release. It does not replace the system Python or alter shell initialization.

On macOS each runtime is installed in a versioned, immutable prefix such as
`~/Library/agentfemruntime-0.3.0`. A new version is installed beside the old
one and the visible launcher is updated to the selected runtime. User projects
remain in `~/AgentFEMProjects`, outside every runtime.

The recommended **Complete** profile includes the Gmsh executable and Python
API, enabling CAD/STEP-to-mesh workflows immediately after installation. The
smaller **Core** profile omits Gmsh. PyTorch and desktop visualization remain
separate optional packs in both profiles.

Gmsh is an independent GPL-2.0-or-later program; aggregating it does not change
AgentFEM's Apache-2.0 source license. Every Complete release nevertheless
carries the Gmsh license and publishes the exact corresponding source and
conda-forge build recipe beside the installer. Noncommercial use does not
remove these redistribution obligations.

## macOS

AgentFEM 0.3.1 deliberately publishes an unsigned Preview and does not require
Apple Developer credentials. Its filename always contains `unsigned-preview`,
and the download page preserves that wording.

Verify the published SHA-256 checksum, open the package, and follow the normal
installer. macOS may block the first open because the package is not notarized;
use the one-time **Open Anyway** control in System Settings → Privacy &
Security after verifying the checksum. Do not disable Gatekeeper globally and
do not run an undocumented quarantine-removal command. After installation,
open `AgentFEM Terminal.command`; it starts the isolated runtime and performs a
health check.

The Preview is not described as signed, notarized, or trusted by Apple. The
workflow still installs and exercises the exact package on a clean GitHub-
hosted Apple Silicon runner before publishing it.

## Windows through WSL2

Unzip the Preview. For WSL 2.4.4 or newer, its `.wsl` image can be opened
directly in File Explorer or installed from PowerShell:

```powershell
wsl --install --from-file .\AgentFEM-Complete-<version>-WSL2-x86_64.wsl
```

The offline bundle includes `Install-AgentFEM.ps1`, which verifies the image,
refuses to overwrite an existing `AgentFEM` distribution, imports it, and
starts the first-use account setup. If WSL itself is absent, Windows must first
enable that operating-system feature; this may require administrator access,
a reboot, and Microsoft network access.

The imported distribution creates an `AgentFEM` Start-menu entry and Windows
Terminal profile. Simulation projects are kept in `~/AgentFEMProjects`, apart
from the immutable runtime.

## Integrity and mirrors

Every release publishes `SHA256SUMS`, `runtime-artifacts.json`, the exact
package lock, a software bill of materials, and third-party notices. Official
GitHub assets and the domestic mirror must contain byte-identical files.
Uncontrolled download proxies are not official distribution channels.

Runtime installation never makes a numerical result trusted by itself. The
installed runtime still attaches solver, environment, provenance, balance,
and verification evidence to each `SimulationResult`.

## Official download location

Accepted Preview installers are GitHub Release assets:

<https://github.com/haoming-luo/agentfem/releases/latest>

To avoid storing the same multi-gigabyte image twice, the public Windows asset
is the single offline ZIP containing the image, installer script, checksums,
SBOM, runtime manifest, Gmsh license, corresponding source, and conda-forge
recipe. The macOS asset is the unsigned Preview package plus its acceptance
record. A candidate from an Actions artifact is not a public Preview until the
workflow has produced and uploaded its acceptance evidence.
