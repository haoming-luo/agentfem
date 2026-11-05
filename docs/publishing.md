# Publishing AgentFEM to PyPI

AgentFEM distributes a source archive and platform-specific wheels containing
native accelerators. It expects a compatible FEniCSx/DOLFINx stack first,
usually from conda-forge. A pure-Python fallback is not performance-equivalent
to the native candidate and must not be presented as the tested native build.
Build identity matters: equal development version strings do not imply equal
source, interfaces or accelerators.

## First-Time PyPI Setup

1. Create or log in to the PyPI account that will own `agentfem`.
2. Create a pending Trusted Publisher for:
   - owner: `haoming-luo`
   - repository: `agentfem`
   - workflow filename: `publish-pypi.yml`
   - environment name: `pypi`
3. Confirm the package name `agentfem` is available on PyPI.

Trusted Publishing is preferred because GitHub Actions can publish with OIDC
without storing a long-lived PyPI API token in repository secrets.

## Release Checklist

1. Update the version in `pyproject.toml`.
2. Update `__version__` in `src/agentfem/__init__.py`.
3. Update `CITATION.cff`, `CHANGELOG.md`, `docs/agentfem.json`, the versioned
   release contract and release notes together. Keep scientific maturity and
   open evidence gates explicit; a stable package does not promote every
   experimental workflow.
4. Run the smoke examples in a FEniCSx environment.
5. Commit the candidate and build from a clean packaging workspace into fresh
   output directories. Preserve existing artifacts; do not clear a user's
   build/output tree. The release gate rejects runtime-file identity drift.
6. Run the same release gate used by CI:

   ```bash
   reuse lint
   python -m build --sdist --outdir /tmp/agentfem-candidate-source
   python tools/build_platform_wheel.py \
     --source-dir /tmp/agentfem-candidate-source \
     --output /tmp/agentfem-candidate-dist
   python -m twine check /tmp/agentfem-candidate-dist/*.whl /tmp/agentfem-candidate-dist/*.tar.gz
   python release_gate.py --dist /tmp/agentfem-candidate-dist --smoke
   ```

   Use unique, previously absent directory names for each candidate. Add
   `--repair linux` to the platform build on Linux. Install build/test
   dependencies in the chosen environment first; do not replace the user's
   production installation. The build report is packaging evidence only.

7. Require the release PR's applicable CI, installed workflows and documented
   scientific gates to pass. Record any intentionally experimental exclusions.
8. Merge the candidate, then create and push the matching version tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The GitHub Actions workflow `.github/workflows/publish-pypi.yml` builds the
Linux wheel and source distribution exactly once. It then installs and verifies that
immutable wheel through the complete serial, MPI, checkpoint, project-template,
and release-workflow gates. Only those same artifacts are published to PyPI;
GitHub provenance attestation is added when repository visibility supports it.
The workflow also generates a REUSE-derived SPDX source record and attests it
beside the immutable distributions; it never rebuilds after verification.
The macOS native wheel is separately built from the same source archive and
tested on macOS. A local macOS check is not Windows/WSL host acceptance.
Confirm the publication job and the uploaded artifact hashes before announcing
availability. A merged PR or tag alone is not a completed PyPI release.

## Install Command for Users

After publication:

```bash
python -m pip install agentfem
```

For optional mesh format conversion:

```bash
python -m pip install "agentfem[mesh-formats]"
```
