# AgentFEM Runtime installers

This directory defines the self-contained desktop runtimes distributed beside
the ordinary PyPI and conda-forge packages. The installers are a delivery
surface, not a second implementation of AgentFEM.

The first supported artifacts are offered in two profiles:

- `AgentFEM-Complete-<version>-macOS-arm64.pkg` and the corresponding WSL2
  bundle are the recommended one-click runtimes and include Gmsh;
- `AgentFEM-Core-<version>-...` is the lean solver runtime without Gmsh.

Both profiles contain a pinned FEniCSx/PETSc/MPI runtime and the exact
published AgentFEM wheel. Neither contains PyTorch, PyVista, or development
tools. Gmsh remains an independent GPL-2.0-or-later component even when it is
aggregated in the Complete installer; the release must carry its license and
make the corresponding source and build recipe available beside the binary.

## Build contracts

Build the macOS candidate on Apple Silicon:

```bash
python packaging/runtime/build_runtime.py macos \
  --profile complete \
  --output-dir dist/runtime
```

For reproducible CI or local release builds, set `CONSTRUCTOR_CONDA_EXE` to a
constructor-compatible `micromamba` or `conda-standalone` executable. Do not
point it at an ordinary activated `conda` command.

Build the WSL2 x86-64 image on a Docker-capable host:

```bash
python packaging/runtime/build_runtime.py wsl \
  --profile complete \
  --output-dir dist/runtime
```

Generate checksums and the machine-readable release manifest:

```bash
python packaging/runtime/build_runtime.py manifest \
  --output-dir dist/runtime
```

After installing a candidate on the target machine, run the cold-cache serial,
MPI, Gmsh (Complete only), project, and verification acceptance contract:

```bash
python packaging/runtime/verify_installed_runtime.py \
  --prefix "$HOME/Library/agentfemruntime-<version>" \
  --profile complete \
  --output dist/runtime/macos-arm64-acceptance.json
```

The ordinary cross-platform workflow deliberately emits an explicitly named
`unsigned-preview` package. It may be published only with its checksum and
local installation acceptance evidence; it must never be described as signed
or notarized, and installation guidance must not disable Gatekeeper globally.

The separate `Signed macOS runtime` workflow consumes release secrets without
recording them in the repository. It supplies
`AGENTFEM_APP_IDENTITY`, `AGENTFEM_INSTALLER_IDENTITY`, and
`AGENTFEM_NOTARY_PROFILE` to this same builder, then verifies the Developer ID
signature, Apple notarization ticket, Gatekeeper assessment, and installed
runtime before a formal `.pkg` can be uploaded.

## Release gates

An installer is promotable only when its evidence records all of the
following:

1. the clean installer-builder commit and exact embedded AgentFEM wheel;
2. the exact embedded wheel and SHA-256;
3. the complete conda package manifest and third-party licenses;
4. corresponding source and build recipes for every bundled GPL component;
5. a cold-cache JIT solve, not only an import test;
6. a two-rank MPI solve with the runtime-matched launcher;
7. `agentfem doctor`, project creation, execution, and verification;
8. an offline installation test on a clean target;
9. signature and notarization status for the public macOS package;
10. a real WSL2 acceptance record for the public `.wsl` image.

User projects live outside the runtime prefix. Replacing or uninstalling a
runtime must never remove project data.

macOS runtime prefixes are versioned and immutable. A later release installs
beside the prior environment and updates `AgentFEM Terminal.command` to the
new version. This makes upgrades non-destructive and keeps an older numerical
environment available for exact reproduction until the user elects to remove
it.

Windows offers both policies explicitly. The default installer path creates a
versioned side-by-side WSL distribution. `Install-AgentFEM.ps1 -Upgrade`
performs a controlled replacement of the stable `AgentFEM` distribution: it
accepts a temporary candidate, exports a full rollback snapshot, archives the
user home independently, switches the registration, restores the user files,
and repeats `agentfem doctor`. Any failure after the switch restores the old
snapshot. Every released installer script must pass PowerShell's parser in CI;
the replacement route additionally remains Preview until exercised on a real
Windows/WSL2 host.
