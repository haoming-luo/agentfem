"""Fail-fast release checks that do not hide behind an editable install.

The gate has two layers.  ``--dist`` checks source metadata and the actual
wheel/sdist payload without importing DOLFINx.  ``--smoke`` is run inside the
tested FEniCSx environment and exercises the small representative workflows
that define the first public release.
"""

from __future__ import annotations

import argparse
import ast
from importlib import metadata
from pathlib import Path
import re
import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parent
REQUIRED_WHEEL_MEMBERS = (
    "agentfem/__init__.py",
    "agentfem/models.py",
    "agentfem/dependencies.py",
    "agentfem/platforms.py",
    "agentfem/results/core.py",
    "agentfem/constitutive/creep.py",
    "agentfem/datasets/torch.py",
    "agentfem/surrogates/pinn_torch.py",
    "agentfem/surrogates/training.py",
    "agentfem/knowledge/catalog.json",
    "agentfem/materials/data/steel_generic.json",
)
SMOKE_EXAMPLES = (
    "examples/static_elasticity_2d.py",
    "examples/transient_heat_2d.py",
    "examples/creep_hot_wall_assessment.py",
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


def check_distributions(directory: Path) -> None:
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
    sdist_members = _archive_members(sdists[0])
    for required in ("README.md", "LICENSE", "NOTICE", "pyproject.toml"):
        if required not in sdist_members:
            raise RuntimeError(f"Source distribution omits {required}.")


def run_smoke() -> None:
    installed = metadata.version("agentfem")
    expected = check_versions()
    if installed != expected:
        raise RuntimeError(
            f"Imported distribution is {installed}, expected tested source {expected}."
        )
    for example in SMOKE_EXAMPLES:
        environment = dict(os.environ)
        environment["AGENTFEM_RELEASE_SMOKE"] = "1"
        environment["AGENTFEM_INSTALLED_SMOKE"] = "1"
        subprocess.run(
            [sys.executable, example],
            cwd=ROOT,
            check=True,
            env=environment,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    version = check_versions(tag=options.tag)
    check_dependency_boundaries()
    if options.dist is not None:
        check_distributions(options.dist)
    if options.smoke:
        run_smoke()
    print(f"AgentFEM {version} release gate passed.")


if __name__ == "__main__":
    main()
