# Offline Runtime Installers

AgentFEM's normal Python package remains available through conda-forge and
PyPI. Runtime installers serve a different audience: a new user can install a
complete, compatible finite-element environment without solving packages or
cloning a repository.

## Runtime targets

| Host | Artifact | Status |
| --- | --- | --- |
| Apple Silicon macOS | `AgentFEM-Complete-<version>-macOS-arm64.pkg` | Recommended native installer after signing and notarization |
| Windows 10/11 with WSL2 | `AgentFEM-Complete-<version>-WSL2-x86_64.wsl` | Recommended Windows route after real WSL2 acceptance |
| Intel macOS | `.pkg` | Demand-driven follow-up |
| Native Windows | `.exe` | Experimental until the complete PETSc/MPI/MPC gate passes |

The runtime includes Python, FEniCSx/DOLFINx, PETSc, MPI, HDF5, a C compiler
for cold-cache form JIT, and the exact AgentFEM wheel associated with the
release. It does not replace the system Python or alter shell initialization.

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

Download the signed and notarized `.pkg`, open it, and follow the standard
macOS installer. After installation, open `AgentFEM Terminal.command` in the
user Applications folder. It opens an isolated AgentFEM shell and performs a
health check on first launch.

Only an artifact whose filename does **not** contain `unsigned` is intended for
public distribution. The public package must pass:

```text
Developer ID signatures
→ Apple notarization
→ stapled ticket
→ Gatekeeper assessment
→ offline cold-cache solve
```

### Supplying the Apple Developer identity

The Apple Developer Program account holder should create both **Developer ID
Application** and **Developer ID Installer** certificates. Export the two
certificates together with their private keys from Keychain Access as one
password-protected `.p12`. Never commit or send that file through chat.

For a local release machine, import the `.p12`, then store notarization
credentials in the login Keychain:

```bash
xcrun notarytool store-credentials agentfem-notary \
  --apple-id "APPLE_ID" \
  --team-id "TEAM_ID" \
  --password "APP_SPECIFIC_PASSWORD"

export AGENTFEM_APP_IDENTITY="Developer ID Application: ... (TEAM_ID)"
export AGENTFEM_INSTALLER_IDENTITY="Developer ID Installer: ... (TEAM_ID)"
export AGENTFEM_NOTARY_PROFILE="agentfem-notary"
```

For GitHub Actions, add the following encrypted repository secrets:

| Secret | Content |
| --- | --- |
| `APPLE_DEVELOPER_CERTIFICATE_P12` | Base64-encoded `.p12` containing both Developer ID identities |
| `APPLE_DEVELOPER_CERTIFICATE_PASSWORD` | Password used while exporting the `.p12` |
| `APPLE_BUILD_KEYCHAIN_PASSWORD` | A new random password used only by the temporary CI Keychain |
| `APPLE_ID` | Apple Developer account email |
| `APPLE_TEAM_ID` | Ten-character Apple Developer Team ID |
| `APPLE_APP_PASSWORD` | Apple app-specific password used by `notarytool` |

The runtime workflow emits an explicitly suffixed `-unsigned.pkg` when these
credentials are absent. Such a candidate can be inspected locally but cannot
be promoted to the public download.

## Windows through WSL2

For WSL 2.4.4 or newer, the `.wsl` image can be opened directly in File
Explorer or installed from PowerShell:

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

Promoted installers are GitHub Release assets:

<https://github.com/haoming-luo/agentfem/releases/latest>

The download page must list the Complete installer, Core installer, checksums,
SBOM, runtime manifest, and—when Gmsh is included—the corresponding Gmsh
source and conda-forge recipe. A candidate from an Actions artifact is not an
official public installer until it passes the platform acceptance record.
