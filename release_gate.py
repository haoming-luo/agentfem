"""Fail-fast release checks that do not hide behind an editable install.

The gate has two layers.  ``--dist`` checks source metadata and the actual
wheel/sdist payload without importing DOLFINx.  ``--smoke`` is run inside the
tested FEniCSx environment and exercises the small representative workflows
that define the first public release.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import re
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parent
REQUIRED_WHEEL_MEMBERS = (
    "agentfem/__init__.py",
    "agentfem/models.py",
    "agentfem/dependencies.py",
    "agentfem/platforms.py",
    "agentfem/provenance.py",
    "agentfem/results/core.py",
    "agentfem/results/execution.py",
    "agentfem/results/recovery.py",
    "agentfem/cli.py",
    "agentfem/project.py",
    "agentfem/extensions.py",
    "agentfem/upgrades.py",
    "agentfem/py.typed",
    "agentfem/templates/static-solid/case.py",
    "agentfem/templates/static-solid/agentfem.toml",
    "agentfem/templates/static-solid/AGENTS.md",
    "agentfem/templates/steady-heat/case.py",
    "agentfem/templates/structural-dynamics/case.py",
    "agentfem/verification.py",
    "agentfem/benchmarks/golden.py",
    "agentfem/constitutive/creep.py",
    "agentfem/datasets/torch.py",
    "agentfem/surrogates/pinn_torch.py",
    "agentfem/surrogates/training.py",
    "agentfem/knowledge/catalog.json",
    "agentfem/knowledge/benchmarks/creep_abaqus_constant_stress.json",
    "agentfem/knowledge/cards/integration_point_recovery.json",
    "agentfem/knowledge/cards/transient_checkpoint_portability.json",
    "agentfem/materials/data/steel_generic.json",
)
FORBIDDEN_DISTRIBUTION_PARTS = ("__pycache__",)
FORBIDDEN_DISTRIBUTION_SUFFIXES = (".pyc", ".pyo")
SMOKE_EXAMPLES = (
    "examples/static_elasticity_2d.py",
    "examples/transient_heat_2d.py",
    "examples/creep_hot_wall_assessment.py",
)
SMOKE_IDENTITY_FILES = (
    "__init__.py",
    "checkpointing.py",
    "problems.py",
    "results/core.py",
    "provenance.py",
)


def source_version() -> str:
    tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(item, ast.Name) and item.id == "__version__" for item in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
    raise RuntimeError("__init__.py must define a literal __version__.")


def project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("pyproject.toml must define project.version.")
    return match.group(1)


def check_versions(*, tag: str | None = None) -> str:
    source = source_version()
    project = project_version()
    if source != project:
        raise RuntimeError(
            f"Version mismatch: __version__={source!r}, project.version={project!r}."
        )
    if tag is not None:
        expected = tag.removeprefix("v")
        if expected != project:
            raise RuntimeError(
                f"Release tag {tag!r} does not match project version {project!r}."
            )
    return project


def check_dependency_boundaries() -> None:
    """Keep GPL Gmsh integration outside the Apache-2.0 core dependency set."""

    record = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = record["project"]
    required = {_requirement_name(item) for item in project.get("dependencies", ())}
    if "gmsh" in required:
        raise RuntimeError(
            "Gmsh must remain an optional integration, not an AgentFEM core dependency."
        )
    optional = project.get("optional-dependencies", {})
    gmsh_extra = {_requirement_name(item) for item in optional.get("gmsh", ())}
    if "gmsh" not in gmsh_extra:
        raise RuntimeError(
            "The optional 'gmsh' extra must declare the separately licensed Gmsh package."
        )


def _requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\s\[]", requirement.strip(), maxsplit=1)[0].lower()


def _archive_members(path: Path) -> set[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return set(archive.namelist())
    with tarfile.open(path, "r:gz") as archive:
        members = set()
        for name in archive.getnames():
            parts = name.split("/", 1)
            members.add(parts[1] if len(parts) == 2 else parts[0])
        return members


def check_distributions(directory: Path) -> Path:
    wheels = sorted(directory.glob("agentfem-*.whl"))
    sdists = sorted(directory.glob("agentfem-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError(
            f"Expected exactly one AgentFEM wheel and sdist in {directory}."
        )
    wheel_members = _archive_members(wheels[0])
    missing = [name for name in REQUIRED_WHEEL_MEMBERS if name not in wheel_members]
    if missing:
        raise RuntimeError(f"Wheel omits required runtime assets: {missing}.")
    forbidden = _forbidden_distribution_members(wheel_members)
    if forbidden:
        raise RuntimeError(
            "Wheel contains build-machine bytecode/cache artifacts: "
            f"{forbidden[:8]}."
        )
    sdist_members = _archive_members(sdists[0])
    for required in ("README.md", "LICENSE", "NOTICE", "pyproject.toml"):
        if required not in sdist_members:
            raise RuntimeError(f"Source distribution omits {required}.")
    return wheels[0]


def _forbidden_distribution_members(members) -> list[str]:
    """Return non-source cache artifacts that must never enter a release."""

    return sorted(
        name
        for name in members
        if any(part in FORBIDDEN_DISTRIBUTION_PARTS for part in Path(name).parts)
        or str(name).endswith(FORBIDDEN_DISTRIBUTION_SUFFIXES)
    )


def run_smoke(*, wheel: Path | None = None) -> None:
    """Exercise an installed distribution and reject same-version stale code."""

    with _smoke_environment(wheel) as environment:
        _check_installed_identity()
        environment["AGENTFEM_RELEASE_SMOKE"] = "1"
        environment["AGENTFEM_INSTALLED_SMOKE"] = "1"
        for example in SMOKE_EXAMPLES:
            subprocess.run(
                [sys.executable, example],
                cwd=ROOT,
                check=True,
                env=environment,
            )
        run_installed_project_smoke(environment=environment)


@contextmanager
def _smoke_environment(wheel: Path | None):
    if wheel is None:
        yield dict(os.environ)
        return
    with tempfile.TemporaryDirectory(prefix="agentfem-wheel-") as directory:
        target = Path(directory) / "site-packages"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target),
                str(wheel),
            ],
            check=True,
        )
        sys.path.insert(0, str(target))
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(target) if not existing else str(target) + os.pathsep + existing
        )
        try:
            yield environment
        finally:
            sys.path.remove(str(target))


def _check_installed_identity() -> None:
    distribution = metadata.distribution("agentfem")
    expected = check_versions()
    if distribution.version != expected:
        raise RuntimeError(
            f"Imported distribution is {distribution.version}, "
            f"expected tested source {expected}."
        )
    mismatches = []
    for relative in SMOKE_IDENTITY_FILES:
        source = ROOT / relative
        installed = Path(distribution.locate_file(Path("agentfem") / relative))
        if not installed.is_file() or _sha256(installed) != _sha256(source):
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(
            "Installed AgentFEM has the expected version number but different "
            f"code in {mismatches}. Install the wheel being tested or combine "
            "--dist with --smoke."
        )


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run_installed_project_smoke(*, environment=None) -> None:
    """Prove the wheel works without repository examples or source paths."""

    with tempfile.TemporaryDirectory(prefix="agentfem-installed-") as directory:
        project = Path(directory) / "case"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "agentfem.cli",
                "init",
                str(project),
                "--template",
                "static-solid",
            ],
            cwd=directory,
            check=True,
            env=environment,
        )
        upgrade = subprocess.run(
            [
                sys.executable,
                "-m",
                "agentfem.cli",
                "upgrade",
                "--project",
                str(project),
                "--json",
            ],
            cwd=directory,
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )
        upgrade_record = json.loads(upgrade.stdout)
        if upgrade_record["status"] != "current":
            raise RuntimeError(
                "Installed project template immediately requires migration: "
                f"{upgrade_record['findings']}"
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "agentfem.cli",
                "check",
                "--project",
                str(project),
                "--json",
            ],
            cwd=directory,
            check=True,
            env=environment,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "agentfem.cli",
                "run",
                "--project",
                str(project),
                "--run-id",
                "installed-smoke",
                "--json",
            ],
            cwd=directory,
            check=True,
            env=environment,
        )
        latest = project / "outputs" / "case" / "latest.json"
        if not latest.is_file():
            raise RuntimeError(
                "Installed project smoke did not publish the latest-run pointer."
            )
        record = json.loads(latest.read_text(encoding="utf-8"))
        result = Path(record["result_manifest"])
        if not result.is_file():
            raise RuntimeError(
                "Installed project smoke did not publish SimulationResult manifest."
            )
        verification = subprocess.run(
            [
                sys.executable,
                "-m",
                "agentfem.cli",
                "verify",
                str(result),
                "--json",
            ],
            cwd=directory,
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )
        if json.loads(verification.stdout)["status"] != "verified":
            raise RuntimeError("Installed project result provenance did not verify.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    version = check_versions(tag=options.tag)
    check_dependency_boundaries()
    wheel = None
    if options.dist is not None:
        wheel = check_distributions(options.dist)
    if options.smoke:
        run_smoke(wheel=wheel)
    print(f"AgentFEM {version} release gate passed.")


if __name__ == "__main__":
    main()
