# Publishing AgentFEM to PyPI

AgentFEM is distributed as a pure Python package. It expects users to install a
compatible FEniCSx/DOLFINx stack first, usually from conda-forge, then install
AgentFEM with pip.

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
2. Update `__version__` in `__init__.py`.
3. Run the smoke examples in a FEniCSx environment.
4. Run the same release gate used by CI:

   ```bash
   python -m pip install -e ".[dev]"
   python -m build
   python -m twine check dist/*
   python release_gate.py --dist dist --smoke
   ```

5. Commit the release changes.
6. Create and push a version tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The GitHub Actions workflow `.github/workflows/publish-pypi.yml` first repeats
the serial and MPI FEniCSx verification, checks that the tag matches both
version declarations, validates the actual wheel/sdist payload, and only then
publishes the tagged release to PyPI.

## Install Command for Users

After publication:

```bash
python -m pip install agentfem
```

For optional mesh format conversion:

```bash
python -m pip install "agentfem[mesh-formats]"
```
