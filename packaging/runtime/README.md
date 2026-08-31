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
  --profile complete \
  --output dist/runtime/macos-arm64-acceptance.json
```

The macOS builder accepts these environment variables without recording their
values in the repository:

- `AGENTFEM_APP_IDENTITY`: `Developer ID Application: ...`
- `AGENTFEM_INSTALLER_IDENTITY`: `Developer ID Installer: ...`
- `AGENTFEM_NOTARY_PROFILE`: a Keychain profile created with `notarytool`

If the identities are absent, the builder emits an explicitly named unsigned
candidate. An unsigned candidate is suitable for local packaging tests but
must never be advertised as the public macOS installer.

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
