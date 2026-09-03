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
`~/Library/agentfemruntime-0.3.1`. A new version is installed beside the old
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

Extract the entire Preview ZIP, open PowerShell in the extracted folder, and
run the bundled installer:

```powershell
powershell -ExecutionPolicy Bypass -File .\Install-AgentFEM.ps1
```

Do not run the script while browsing inside the compressed ZIP. The bundle's
`START-HERE.txt` records the same procedure for both people and AI agents.

For WSL 2.4.4 or newer, advanced users may instead install the `.wsl` image
directly:

```powershell
wsl --install --from-file .\AgentFEM-Complete-<version>-WSL2-x86_64.wsl
```

`Install-AgentFEM.ps1` checks that WSL 2.4.4 or newer is available, verifies
the image, imports it, and starts the first-use account setup. If an existing
`AgentFEM` distribution is present, the default remains a non-destructive
side-by-side installation named `AgentFEM-<version>`.

Users who want the normal stable name to move to the new runtime can request a
transactional replacement:

```powershell
powershell -ExecutionPolicy Bypass -File .\Install-AgentFEM.ps1 -Upgrade
```

This is not a blind in-place overwrite. The installer first imports the new
image under a temporary name and requires its embedded identity and
`agentfem doctor` check to pass. It then exports a complete snapshot of the old
distribution, separately archives the Linux user home, switches the stable
`AgentFEM` registration, restores the user files, and repeats the health check.
If a post-switch step fails, the previous full snapshot is imported again. The
recovery directory is retained under `%LOCALAPPDATA%\AgentFEM\Backups` unless
the user explicitly supplies `-RemoveBackupAfterSuccess`.

The distinction is required by WSL's lifecycle: WSL provides distribution
export/import and destructive unregister operations, but no atomic in-place
runtime replacement or rename. The installer is therefore the owner of the
backup, validation, switching, and rollback transaction. Users and agents must
not reproduce its internal `wsl --unregister` steps manually.
[Microsoft's WSL command reference](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)
defines these lifecycle operations and explicitly warns that unregistering a
distribution permanently removes its data.

If WSL itself is absent, Windows must first enable that operating-system
feature; this may require administrator access, a reboot, and Microsoft network
access. The bundled `START-HERE.txt` includes the `--web-download` recovery
path for a stalled Store download and the exact `wsl --status` / `wsl
--version` diagnostics.

The imported distribution creates a Start-menu entry and Windows Terminal
profile using its registered distribution name. The installer creates the
authoritative project workspace in Windows `Documents\AgentFEMProjects` and
exposes it inside Linux as `~/AgentFEMProjects`. Project inputs, result
manifests, field files, histories, and checkpoints therefore survive replacing
or unregistering the runtime distribution. Check the custody contract with:

```bash
agentfem workspace
agentfem workspace --json
```

The same mechanism applies to AgentFEM installed through Mamba in an existing
Ubuntu or other WSL distribution. One command migrates an existing conventional
workspace with a copy, content verification, retained pre-migration copy, and
only then a link switch:

```bash
agentfem workspace --protect
```

A different Windows location is equally valid, for example a large `D:` drive:

```bash
agentfem workspace --protect --path /mnt/d/AgentFEMProjects
```

Projects created below that workspace keep their default `outputs/` directory
beside the model, so the complete reproducible project remains one portable
Windows folder.

Use the Windows host workspace as the safe default. Linux-native storage can be
faster for very high-frequency temporary I/O, so advanced workflows may keep
recomputable scratch or caches inside WSL, but an accepted project's only copy
must not remain there. This follows Microsoft's guidance that cross-filesystem
I/O can be slower while also respecting its warning that unregistering a WSL
distribution permanently deletes its contents.

To remove the Complete Runtime, use the bundled fail-closed wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File .\Remove-AgentFEM.ps1
```

It verifies persistent project custody and exports a full recovery snapshot
before unregistering the distribution. It never deletes the Windows project
directory. Neither users nor agents should call `wsl --unregister AgentFEM`
directly.

Every project summary and execution record carries both `project_storage` and
`output_storage` custody metadata. A project or explicit output path remaining
inside a replaceable WSL distribution is therefore visible to people, agents,
and downstream verification rather than being a hidden installation detail.

## Integrity and mirrors

GitHub records a SHA-256 digest for every public release asset. The Windows
offline ZIP also contains `SHA256SUMS`, `runtime-artifacts.json`, the exact
package lock, a software bill of materials, and third-party notices. The macOS
package embeds its runtime identity, lock, software bill of materials, and
notices, while its acceptance record is published beside it. Official GitHub
assets and any project-operated mirror must contain byte-identical files.
Uncontrolled download proxies are not official distribution channels.

Each release also carries `distribution-manifest.json`. It separates artifact
identity from delivery: GitHub is the canonical route, while a project-
operated mainland China route may be preferred when reachable. Both URLs name
the same accepted file and advertise the same byte count and SHA-256. A
download with a different digest is rejected rather than installed. This
keeps a faster regional route from creating a second, scientifically distinct
runtime.

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
