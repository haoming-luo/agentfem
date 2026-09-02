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
from datetime import datetime, timezone
from hashlib import sha256
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parent
SOURCE_PACKAGE = ROOT / "src" / "agentfem"
REQUIRED_WHEEL_MEMBERS = (
    "agentfem/__init__.py",
    "agentfem/_axisymmetric.py",
    "agentfem/_step_builders.py",
    "agentfem/models.py",
    "agentfem/dependencies.py",
    "agentfem/platforms.py",
    "agentfem/provenance.py",
    "agentfem/results/core.py",
    "agentfem/results/execution.py",
    "agentfem/results/lifecycle.py",
    "agentfem/results/recovery.py",
    "agentfem/step_providers.py",
    "agentfem/cli.py",
    "agentfem/feedback.py",
    "agentfem/feedback-endpoint.json",
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
    "agentfem/benchmarks/inelastic.py",
    "agentfem/constitutive/creep.py",
    "agentfem/fatigue_fracture.py",
    "agentfem/datasets/torch.py",
    "agentfem/learning/core.py",
    "agentfem/learning/execution.py",
    "agentfem/surrogates/pinn_torch.py",
    "agentfem/surrogates/training.py",
    "agentfem/knowledge/catalog.json",
    "agentfem/knowledge/benchmarks/creep_abaqus_constant_stress.json",
    "agentfem/knowledge/benchmarks/creep_nafems_r0027_test7.json",
    "agentfem/knowledge/benchmarks/axisymmetric_lame_cylinder.json",
    "agentfem/knowledge/benchmarks/j2_thick_cylinder_mpi.json",
    "agentfem/knowledge/benchmarks/chaboche_combined_hardening.json",
    "agentfem/knowledge/cards/integration_point_recovery.json",
    "agentfem/knowledge/cards/chaboche_global_plasticity.json",
    "agentfem/knowledge/cards/axisymmetric_solid.json",
    "agentfem/knowledge/cards/transient_checkpoint_portability.json",
    "agentfem/knowledge/decisions/0016-thin-model-facade-and-execution-policy.md",
    "agentfem/knowledge/decisions/0017-public-api-lifecycle.md",
    "agentfem/knowledge/decisions/0024-runtime-project-custody.md",
    "agentfem/materials/data/steel_generic.json",
)


def _source_identity() -> tuple[str | None, bool | None]:
    """Return the exact candidate commit and whether its checkout is dirty."""

    commit = os.environ.get("GITHUB_SHA")
    if not commit:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            commit = completed.stdout.strip() or None
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return commit, dirty


FORBIDDEN_DISTRIBUTION_PARTS = ("__pycache__",)
FORBIDDEN_DISTRIBUTION_SUFFIXES = (".pyc", ".pyo")
SMOKE_COMMANDS = (
    ("examples/static_elasticity_2d.py", ()),
    ("examples/axisymmetric_thick_cylinder.py", ()),
    ("examples/transient_heat_2d.py", ()),
    ("examples/wave_packet_inclusion_2d.py", ()),
    (
        "examples/abaqus_c3d10h_periodic_cell/case.py",
        (
            "--displacement",
            "0.0",
            "--video-format",
            "none",
            "--output",
            "{release_output}/c3d10h",
        ),
    ),
    ("examples/j2_plasticity_3d.py", ()),
    ("examples/chaboche_cyclic_3d.py", ()),
    ("examples/implicit_creep_relaxation_3d.py", ()),
    ("examples/creep_hot_wall_assessment.py", ()),
    ("examples/static_elasticity_surrogate_campaign.py", ()),
)
INSTALLED_PROJECT_TEMPLATES = (
    "static-solid",
    "steady-heat",
    "structural-dynamics",
)


def source_version() -> str:
    tree = ast.parse((SOURCE_PACKAGE / "__init__.py").read_text(encoding="utf-8"))
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


def release_contract_path(version: str | None = None) -> Path:
    """Return the immutable contract matching the candidate package version."""

    selected = source_version() if version is None else str(version)
    return SOURCE_PACKAGE / "release" / f"{selected}.json"


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
    version = check_versions()
    required_wheel_members = REQUIRED_WHEEL_MEMBERS + (
        f"agentfem/release/{version}.json",
    )
    missing = [name for name in required_wheel_members if name not in wheel_members]
    if missing:
        raise RuntimeError(f"Wheel omits required runtime assets: {missing}.")
    forbidden = _forbidden_distribution_members(wheel_members)
    if forbidden:
        raise RuntimeError(
            "Wheel contains build-machine bytecode/cache artifacts: "
            f"{forbidden[:8]}."
        )
    sdist_members = _archive_members(sdists[0])
    for required in (
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "NOTICE",
        "pyproject.toml",
        "release_gate.py",
        "promotion_gate.py",
        "build_docs.py",
        "build_knowledge.py",
        "tools/run_wsl2_acceptance.sh",
        "tools/agent_trial_acceptance.py",
        "tools/prepare_agent_trial.py",
        "services/reliability-collector/worker.js",
        "services/reliability-collector/worker.test.mjs",
        "services/reliability-collector/schema.sql",
        f"src/agentfem/release/{version}.json",
        "skills/agentfem/SKILL.md",
        "skills/agentfem/agents/openai.yaml",
        "skills/agentfem/references/workflow.md",
    ):
        if required not in sdist_members:
            raise RuntimeError(f"Source distribution omits {required}.")
    return wheels[0]


def check_release_contract(
    *,
    tag: str | None = None,
    source_root: Path | None = None,
) -> dict:
    """Validate the packaged scope and, when supplied, its source evidence."""

    contract_path = release_contract_path()
    if not contract_path.is_file():
        raise RuntimeError(
            "The candidate package has no matching release contract: "
            f"{contract_path.relative_to(ROOT)}."
        )
    record = json.loads(contract_path.read_text(encoding="utf-8"))
    if record.get("schema") != "agentfem.release-contract":
        raise RuntimeError("Release contract has an unknown schema.")
    target = record.get("target_version")
    if not isinstance(target, str) or not target:
        raise RuntimeError("Release contract must declare target_version.")
    if tag is not None and tag.removeprefix("v") != target:
        raise RuntimeError(
            f"Release contract targets {target!r}, not tag {tag!r}."
        )
    workflows = record.get("workflows")
    if not isinstance(workflows, list) or not workflows:
        raise RuntimeError("Release contract must declare gated workflows.")
    allowed = {"release", "engineering", "experimental"}
    identifiers = set()
    for workflow in workflows:
        identifier = workflow.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise RuntimeError("Every release workflow must have an id.")
        if identifier in identifiers:
            raise RuntimeError(f"Duplicate release workflow id {identifier!r}.")
        identifiers.add(identifier)
        if workflow.get("maturity") not in allowed:
            raise RuntimeError(
                f"Workflow {identifier!r} has an invalid maturity."
            )
        if source_root is not None:
            example = Path(source_root) / str(workflow.get("example", ""))
            if not example.is_file():
                raise RuntimeError(
                    f"Workflow {identifier!r} references missing example {example}."
                )
        if not workflow.get("evidence"):
            raise RuntimeError(
                f"Workflow {identifier!r} must name executable evidence."
            )
    gates = record.get("required_gates")
    if not isinstance(gates, list) or not gates:
        raise RuntimeError("Release contract must declare required_gates.")
    return record


def _forbidden_distribution_members(members) -> list[str]:
    """Return non-source cache artifacts that must never enter a release."""

    return sorted(
        name
        for name in members
        if any(part in FORBIDDEN_DISTRIBUTION_PARTS for part in Path(name).parts)
        or str(name).endswith(FORBIDDEN_DISTRIBUTION_SUFFIXES)
    )


def run_smoke(
    *,
    wheel: Path | None = None,
    mpi_ranks: int = 1,
) -> dict[str, object]:
    """Exercise an installed distribution and reject same-version stale code."""

    with _smoke_environment(wheel) as environment:
        _check_installed_identity(wheel=wheel)
        environment["AGENTFEM_RELEASE_SMOKE"] = "1"
        environment["AGENTFEM_INSTALLED_SMOKE"] = "1"
        environment["AGENTFEM_CAMPAIGN_SAMPLES"] = "4"
        environment["AGENTFEM_CAMPAIGN_RUN_ID"] = "release-smoke"
        acceptance = run_agent_entrypoint_smoke(environment=environment)
        with tempfile.TemporaryDirectory(prefix="agentfem-release-output-") as directory:
            for example, arguments in SMOKE_COMMANDS:
                selected = tuple(
                    value.format(release_output=directory) for value in arguments
                )
                subprocess.run(
                    [sys.executable, example, *selected],
                    cwd=ROOT,
                    check=True,
                    env=environment,
                )
            acceptance["templates"] = run_installed_project_smoke(
                environment=environment
            )
            if int(mpi_ranks) > 1:
                acceptance["mpi_smoke"] = run_installed_mpi_smoke(
                    ranks=int(mpi_ranks),
                    environment=environment,
                )
        print(json.dumps(acceptance, indent=2, sort_keys=True))
        return acceptance


def platform_acceptance(
    agent_acceptance: dict[str, object],
    *,
    wheel: Path | None,
) -> dict[str, object]:
    """Turn an installed release smoke into portable platform evidence."""

    fingerprint = agent_acceptance["runtime_fingerprint"]
    support = fingerprint["platform"]
    operating_system = fingerprint["operating_system"]
    system = str(operating_system["system"]).lower()
    route = str(support["route"]).lower()
    if "wsl2" in route:
        platform_id = "wsl2"
    elif "wsl1" in route:
        platform_id = "wsl1"
    elif system == "darwin":
        platform_id = "macos"
    elif system == "linux":
        platform_id = "linux"
    else:
        platform_id = system
    templates = agent_acceptance.get("templates", {})
    smoke_passed = bool(templates) and all(
        item.get("provenance") == "verified" for item in templates.values()
    )
    installed_wheel = wheel is not None and wheel.is_file()
    mpi_smoke = agent_acceptance.get("mpi_smoke")
    wsl = None
    route_passed = True
    if platform_id == "wsl2":
        kernel = " ".join(
            str(operating_system.get(name, ""))
            for name in ("release", "version")
        ).lower()
        wsl = {
            "distro_name": os.environ.get("WSL_DISTRO_NAME"),
            "interop_present": bool(os.environ.get("WSL_INTEROP")),
            "kernel_mentions_wsl2": "wsl2" in kernel
            or "microsoft-standard" in kernel,
        }
        route_passed = bool(
            wsl["kernel_mentions_wsl2"]
            and isinstance(mpi_smoke, dict)
            and mpi_smoke.get("status") == "passed"
            and int(mpi_smoke.get("rank_count", 0)) >= 2
        )
    source_commit, source_dirty = _source_identity()
    source_clean = bool(source_commit) and source_dirty is False
    record = {
        "schema": "agentfem.platform-acceptance",
        "schema_version": "0.1.0",
        "status": (
            "passed"
            if installed_wheel and smoke_passed and route_passed and source_clean
            else "failed"
        ),
        "platform_id": platform_id,
        "route": support["route"],
        "installed_wheel": installed_wheel,
        "wheel_sha256": _sha256(wheel) if installed_wheel else None,
        "release_smoke": "passed" if smoke_passed else "failed",
        "agentfem_version": agent_acceptance["agentfem_version"],
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ci_run_url": (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
            if all(
                name in os.environ
                for name in (
                    "GITHUB_SERVER_URL",
                    "GITHUB_REPOSITORY",
                    "GITHUB_RUN_ID",
                )
            )
            else None
        ),
        "python": fingerprint["python"],
        "machine": fingerprint["machine"],
        "operating_system": operating_system,
        "packages": fingerprint["packages"],
        "mpi": fingerprint["mpi"],
        "templates": templates,
        "mpi_smoke": mpi_smoke,
    }
    if wsl is not None:
        record["wsl"] = wsl
    return record


def require_platform_acceptance(
    record: dict[str, object],
    *,
    expected: str,
) -> None:
    """Reject evidence produced on a different or weaker platform route."""

    selected = str(expected).strip().lower()
    actual = str(record.get("platform_id", "")).lower()
    if record.get("status") != "passed" or actual != selected:
        raise RuntimeError(
            f"Required {selected!r} acceptance, received {actual!r} "
            f"with status {record.get('status')!r}."
        )
    if selected != "wsl2":
        return
    operating_system = record.get("operating_system", {})
    route = str(record.get("route", "")).lower()
    wsl = record.get("wsl", {})
    mpi = record.get("mpi_smoke") or {}
    if (
        str(operating_system.get("system", "")).lower() != "linux"
        or "wsl2" not in route
        or wsl.get("kernel_mentions_wsl2") is not True
    ):
        raise RuntimeError(
            "WSL2 acceptance requires a Linux runtime with a real "
            "microsoft-standard-WSL2 kernel."
        )
    if mpi.get("status") != "passed" or int(mpi.get("rank_count", 0)) < 2:
        raise RuntimeError("WSL2 acceptance requires a two-rank installed-wheel MPI smoke.")


def _installed_cli_json(arguments, *, environment, cwd) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "agentfem.cli", *arguments, "--json"],
        cwd=cwd,
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def run_agent_entrypoint_smoke(*, environment=None) -> dict[str, object]:
    """Verify the discovery surface an unfamiliar agent sees first."""

    with tempfile.TemporaryDirectory(prefix="agentfem-agent-entrypoint-") as directory:
        doctor = _installed_cli_json(
            ("doctor",), environment=environment, cwd=directory
        )
        workspace = _installed_cli_json(
            ("workspace",), environment=environment, cwd=directory
        )
        capabilities = _installed_cli_json(
            ("capabilities",), environment=environment, cwd=directory
        )
    if doctor.get("schema") != "agentfem.runtime-report":
        raise RuntimeError("Installed `agentfem doctor` returned an unknown schema.")
    if workspace.get("schema") != "agentfem.workspace":
        raise RuntimeError("Installed `agentfem workspace` returned an unknown schema.")
    if capabilities.get("schema") != "agentfem.capabilities":
        raise RuntimeError(
            "Installed `agentfem capabilities` returned an unknown schema."
        )
    evidence = capabilities.get("constitutive_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise RuntimeError("Installed capabilities omit constitutive evidence.")
    unsupported = [
        item["capability"]
        for item in evidence
        if not item.get("meets_declared_maturity", False)
    ]
    if unsupported:
        raise RuntimeError(
            "Constitutive claims outrun registered evidence: "
            f"{unsupported}."
        )
    return {
        "schema": "agentfem.agent-acceptance",
        "schema_version": "0.1.0",
        "agentfem_version": capabilities["agentfem_version"],
        "runtime": "passed",
        "workspace": workspace,
        "runtime_fingerprint": {
            "platform": doctor["platform"],
            "operating_system": doctor["operating_system"],
            "machine": doctor["machine"],
            "python": doctor["python"],
            "packages": doctor["packages"],
            "mpi": doctor["mpi"],
            "numerics": doctor["numerics"],
            "execution": doctor["execution"],
        },
        "capability_discovery": "passed",
        "declared_maturity_evidence": "passed",
        "templates": {},
    }


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


def _check_installed_identity(*, wheel: Path | None = None) -> None:
    """Prove that every runtime file in the tested wheel matches this source."""

    distribution = metadata.distribution("agentfem")
    expected = check_versions()
    if distribution.version != expected:
        raise RuntimeError(
            f"Imported distribution is {distribution.version}, "
            f"expected tested source {expected}."
        )
    imported = importlib.import_module("agentfem")
    installed_root = Path(imported.__file__).resolve().parent
    distribution_root = Path(
        distribution.locate_file("agentfem")
    ).resolve()
    if installed_root != distribution_root:
        raise RuntimeError(
            "Imported AgentFEM does not come from the selected installed "
            f"distribution: import={installed_root}, distribution={distribution_root}."
        )
    if wheel is not None and installed_root == SOURCE_PACKAGE.resolve():
        raise RuntimeError(
            "Wheel smoke imported the source checkout instead of the tested wheel."
        )

    members = (
        _archive_members(wheel)
        if wheel is not None
        else {
            str(item)
            for item in (distribution.files or ())
        }
    )
    runtime_members = sorted(
        name
        for name in members
        if name.startswith("agentfem/")
        and not name.endswith((".pyc", ".pyo"))
        and "__pycache__" not in Path(name).parts
    )
    mismatches = []
    for member in runtime_members:
        relative = Path(member).relative_to("agentfem")
        source = SOURCE_PACKAGE / relative
        installed = distribution_root / relative
        if (
            not source.is_file()
            or not installed.is_file()
            or _sha256(installed) != _sha256(source)
        ):
            mismatches.append(str(relative))
    if mismatches:
        raise RuntimeError(
            "Installed AgentFEM has the expected version number but different "
            f"runtime files in {mismatches[:12]}. Install the wheel being tested or combine "
            "--dist with --smoke."
        )


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run_installed_project_smoke(*, environment=None) -> dict[str, object]:
    """Prove every installed template works without repository source paths."""

    accepted = {}
    with tempfile.TemporaryDirectory(prefix="agentfem-installed-") as directory:
        for template in INSTALLED_PROJECT_TEMPLATES:
            selected_project = Path(directory) / template
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agentfem.cli",
                    "init",
                    str(selected_project),
                    "--template",
                    template,
                    "--name",
                    template,
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
                    str(selected_project),
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
                    f"Installed {template} template immediately requires migration: "
                    f"{upgrade_record['findings']}"
                )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agentfem.cli",
                    "check",
                    "--project",
                    str(selected_project),
                    "--json",
                ],
                cwd=directory,
                check=True,
                env=environment,
            )
            run_id = f"installed-{template}"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agentfem.cli",
                    "run",
                    "--project",
                    str(selected_project),
                    "--run-id",
                    run_id,
                    "--json",
                ],
                cwd=directory,
                check=True,
                env=environment,
            )
            latest = (
                selected_project
                / "outputs"
                / template
                / "latest.json"
            )
            if not latest.is_file():
                raise RuntimeError(
                    f"Installed {template} smoke did not publish latest.json."
                )
            record = json.loads(latest.read_text(encoding="utf-8"))
            result = Path(record["result_manifest"])
            if not result.is_file():
                raise RuntimeError(
                    f"Installed {template} smoke did not publish SimulationResult."
                )
            inspected = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agentfem.cli",
                    "inspect",
                    "--project",
                    str(selected_project),
                    "--json",
                ],
                cwd=directory,
                check=True,
                env=environment,
                capture_output=True,
                text=True,
            )
            if json.loads(inspected.stdout).get("run_id") != run_id:
                raise RuntimeError(
                    f"Installed {template} inspect did not resolve the latest run."
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
                raise RuntimeError(
                    f"Installed {template} result provenance did not verify."
                )
            accepted[template] = {
                "project_check": "passed",
                "upgrade": "current",
                "run": "completed",
                "inspect": "passed",
                "provenance": "verified",
            }
    return accepted


def run_installed_mpi_smoke(*, ranks: int, environment=None) -> dict[str, object]:
    """Exercise an installed wheel through the active environment's MPI launcher."""

    from agentfem import platforms

    selected = int(ranks)
    if selected < 2:
        raise ValueError("Installed MPI smoke requires at least two ranks.")
    launcher = platforms.runtime_report().mpi.get("recommended_launcher")
    if not launcher:
        raise RuntimeError("No MPI launcher matches the active mpi4py environment.")
    subprocess.run(
        [
            str(launcher),
            "-n",
            str(selected),
            sys.executable,
            str(ROOT / "examples" / "static_elasticity_2d.py"),
        ],
        cwd=ROOT,
        check=True,
        env=environment,
    )
    return {
        "status": "passed",
        "rank_count": selected,
        "launcher": str(launcher),
        "workflow": "examples/static_elasticity_2d.py",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--mpi-ranks",
        type=int,
        default=1,
        help="Also run the installed-wheel smoke with this MPI rank count.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write the installed-use acceptance record; requires --smoke.",
    )
    parser.add_argument(
        "--platform-report",
        type=Path,
        help="Write installed-wheel platform acceptance; requires --dist and --smoke.",
    )
    parser.add_argument(
        "--require-platform",
        choices=("linux", "macos", "wsl2"),
        help="Fail unless the platform report proves this exact route.",
    )
    options = parser.parse_args()
    if options.report is not None and not options.smoke:
        parser.error("--report requires --smoke")
    if options.platform_report is not None and (not options.smoke or options.dist is None):
        parser.error("--platform-report requires --dist and --smoke")
    if options.require_platform is not None and options.platform_report is None:
        parser.error("--require-platform requires --platform-report")
    if options.require_platform == "wsl2" and options.mpi_ranks < 2:
        parser.error("WSL2 acceptance requires --mpi-ranks 2 or greater")
    version = check_versions(tag=options.tag)
    check_dependency_boundaries()
    check_release_contract(tag=options.tag, source_root=ROOT)
    wheel = None
    if options.dist is not None:
        wheel = check_distributions(options.dist)
    if options.smoke:
        acceptance = run_smoke(wheel=wheel, mpi_ranks=options.mpi_ranks)
        if options.report is not None:
            options.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = options.report.with_suffix(options.report.suffix + ".tmp")
            temporary.write_text(
                json.dumps(acceptance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(options.report)
        if options.platform_report is not None:
            platform_record = platform_acceptance(acceptance, wheel=wheel)
            options.platform_report.parent.mkdir(parents=True, exist_ok=True)
            temporary = options.platform_report.with_suffix(
                options.platform_report.suffix + ".tmp"
            )
            temporary.write_text(
                json.dumps(platform_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(options.platform_report)
            if options.require_platform is not None:
                require_platform_acceptance(
                    platform_record,
                    expected=options.require_platform,
                )
    print(f"AgentFEM {version} release gate passed.")


if __name__ == "__main__":
    main()
