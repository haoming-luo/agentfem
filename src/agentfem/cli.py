"""Command-line product shell for installed AgentFEM projects."""

from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from importlib import resources
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

from mpi4py import MPI

from . import __version__
from . import extensions
from . import platforms
from . import provenance
from . import upgrades
from ._api_contract import CAPABILITIES_SCHEMA_VERSION, CLI_COMMANDS
from .mpi_runtime import audit_mpi_runtime, mpi_command
from .project import (
    PROJECT_FILENAME,
    ProjectConfig,
    RunContext,
    discover,
    new_run_id,
    protect_workspace,
    workspace_report,
)


TEMPLATE_PACKAGE = "agentfem.templates"


def _json(record) -> str:
    return json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False)


def _emit(record, *, as_json: bool, human: str | None = None) -> None:
    if MPI.COMM_WORLD.rank != 0:
        return
    print(_json(record) if as_json else (human or _json(record)), flush=True)


def _named_paths(values: list[str], *, option: str) -> dict[str, Path]:
    """Parse repeatable ``NAME=PATH`` CLI assets without guessing identity."""

    selected: dict[str, Path] = {}
    normalized: set[str] = set()
    for value in values:
        name, separator, path = str(value).partition("=")
        name = name.strip()
        path = path.strip()
        if not separator or not name or not path:
            raise ValueError(f"{option} requires NAME=PATH; received {value!r}.")
        key = name.upper()
        if key in normalized:
            raise ValueError(f"{option} material {name!r} was supplied twice.")
        normalized.add(key)
        selected[name] = Path(path).expanduser().resolve()
    return selected


def _templates() -> tuple[str, ...]:
    root = resources.files(TEMPLATE_PACKAGE)
    return tuple(
        sorted(
            item.name
            for item in root.iterdir()
            if item.is_dir() and (item / "case.py").is_file()
        )
    )


def _copy_template(template: str, target: Path, name: str, *, force: bool) -> None:
    if template not in _templates():
        raise ValueError(f"Unknown template {template!r}. Available: {_templates()}.")
    target.mkdir(parents=True, exist_ok=True)
    occupied = tuple(item for item in target.iterdir() if item.name != ".DS_Store")
    if occupied and not force:
        raise FileExistsError(
            f"Target directory is not empty: {target}. Pass --force to add missing files."
        )
    source = resources.files(TEMPLATE_PACKAGE) / template
    for item in source.iterdir():
        if not item.is_file():
            continue
        destination = target / item.name
        if destination.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {destination}.")
        text = (
            item.read_text(encoding="utf-8")
            .replace("{{PROJECT_NAME}}", name)
            .replace("{{AGENTFEM_VERSION}}", __version__)
        )
        destination.write_text(text, encoding="utf-8")
    (target / "outputs").mkdir(exist_ok=True)
    gitignore = target / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("outputs/\n__pycache__/\n", encoding="utf-8")


def _project(value: str | None) -> ProjectConfig:
    return discover(value)


def _check_project(project: ProjectConfig) -> dict[str, object]:
    errors = list(project.check())
    storage = project.summary()["storage"]
    warnings = tuple(
        item for item in (storage.get("warning"),) if isinstance(item, str) and item
    )
    syntax = "not_checked"
    if project.entrypoint.is_file():
        try:
            ast.parse(project.entrypoint.read_text(encoding="utf-8"), filename=str(project.entrypoint))
            syntax = "valid"
        except SyntaxError as exc:
            syntax = "invalid"
            errors.append(f"{exc.filename}:{exc.lineno}:{exc.offset}: {exc.msg}")
    upgrade = upgrades.inspect_project(project)
    for item in upgrade.errors:
        message = f"{item.code}: {item.message}"
        if not any(item.message in existing for existing in errors):
            errors.append(message)
    return {
        "schema": "agentfem.project-check",
        "schema_version": "0.1.0",
        "status": "passed" if not errors else "failed",
        "project": project.summary(),
        "syntax": syntax,
        "errors": errors,
        "warnings": warnings,
        "upgrade_status": upgrade.status,
        "upgrade_findings": tuple(
            item.as_dict(root=project.root) for item in upgrade.findings
        ),
    }


def _format_check(record: dict[str, object]) -> str:
    if record["status"] != "passed":
        return "Project check failed:\n" + "\n".join(record["errors"])
    text = "Project check passed."
    if record["upgrade_status"] != "current":
        count = len(record["upgrade_findings"])
        text += (
            f"\nUpgrade review recommended: {count} finding(s). "
            "Run `agentfem upgrade`."
        )
    if record.get("warnings"):
        text += "\n" + "\n".join(
            f"Warning: {warning}" for warning in record["warnings"]
        )
    return text


def _command_upgrade(args) -> int:
    project = _project(args.project)
    changed = ()
    if args.apply_safe:
        changed = upgrades.apply_safe_metadata(project)
        project = ProjectConfig.load(project.root)
    report = upgrades.inspect_project(project)
    record = report.summary()
    record["changed_files"] = tuple(str(path) for path in changed)
    if args.write_plan:
        plan_path = Path(args.write_plan).expanduser()
        if not plan_path.is_absolute():
            plan_path = project.root / plan_path
        report.write(plan_path)
        record["plan"] = str(plan_path.resolve())
    _emit(record, as_json=args.json, human=report.format())
    return 2 if report.errors else 0


def _command_workspace(args) -> int:
    """Inspect or protect the conventional installed-use project directory."""

    report = (
        protect_workspace(args.path)
        if args.protect
        else workspace_report(args.path)
    )
    record = report.summary()
    _emit(record, as_json=args.json, human=report.format())
    return 0 if report.protected or not report.under_wsl else 2


def _run_context(project: ProjectConfig, run_id: str | None, output: str | None) -> RunContext:
    comm = MPI.COMM_WORLD
    selected_id = comm.bcast(run_id or (new_run_id() if comm.rank == 0 else None), root=0)
    context = RunContext.create(project, run_id=selected_id, output_directory=output)
    prepare_error = None
    if comm.rank == 0:
        try:
            context.prepare()
        except Exception as exc:
            prepare_error = {
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
            }
    prepare_error = comm.bcast(prepare_error, root=0)
    if prepare_error is not None:
        raise RuntimeError(
            "Could not prepare the shared run directory: "
            f"{prepare_error['type']}: {prepare_error['message']}"
        )
    return context


def _rank_error(exc: Exception | None, *, rank: int) -> dict[str, object] | None:
    if exc is None:
        return None
    record = {
        "rank": int(rank),
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "stage": "case_execution",
    }
    report = getattr(exc, "report", None)
    if report is not None and hasattr(report, "as_dict"):
        validation = report.as_dict()
        record["stage"] = "model_preflight"
        record["validation"] = validation
        errors = tuple(validation.get("issues", ()))
        if errors:
            record["code"] = errors[0].get("code")
    return record


def _collect_rank_errors(comm, local_error) -> tuple[dict[str, object], ...]:
    """Collect per-rank failures before any rank can report completion."""

    return tuple(item for item in comm.allgather(local_error) if item is not None)


def _launch_mpi(args) -> int:
    child = [
        sys.executable,
        "-m",
        "agentfem.cli",
        "run",
    ]
    if args.project:
        child.extend(("--project", args.project))
    if args.run_id:
        child.extend(("--run-id", args.run_id))
    if args.output:
        child.extend(("--output", args.output))
    if args.json:
        child.append("--json")
    child.append("--inside-mpi")
    command = mpi_command(args.mpi, child)
    return subprocess.run(command, check=False).returncode


def _command_mpi_run(args) -> int:
    """Run an arbitrary command with the launcher verified by AgentFEM."""

    child = tuple(args.child_command)
    if child and child[0] == "--":
        child = child[1:]
    if not child:
        raise ValueError("mpi-run requires a child command after `--`.")
    command = mpi_command(args.ranks, child)
    audit = audit_mpi_runtime()
    if MPI.COMM_WORLD.rank == 0:
        print(
            f"AgentFEM MPI: {audit.family} launcher {audit.selected_launcher} "
            f"({args.ranks} ranks)",
            flush=True,
        )
    return subprocess.run(command, check=False).returncode


def _command_run(args) -> int:
    if args.mpi > 1 and MPI.COMM_WORLD.size == 1 and not args.inside_mpi:
        return _launch_mpi(args)
    project = _project(args.project)
    check = _check_project(project)
    if check["status"] != "passed":
        _emit(check, as_json=args.json, human="Project check failed:\n" + "\n".join(check["errors"]))
        return 2
    extensions.load_extensions(project.extensions)
    context = _run_context(project, args.run_id, args.output)
    output_storage = context.summary()["output_storage"]
    previous_environment = {key: os.environ.get(key) for key in context.environment()}
    os.environ.update(context.environment())
    previous_directory = Path.cwd()
    local_error = None
    try:
        os.chdir(project.root)
        if MPI.COMM_WORLD.rank == 0 and not args.json:
            print(
                f"AgentFEM run {context.run_id}\n"
                f"  project: {project.name}\n"
                f"  output: {context.output_directory}",
                flush=True,
            )
            printed_warnings = list(check.get("warnings", ()))
            for warning in printed_warnings:
                print(f"  warning: {warning}", flush=True)
            output_warning = output_storage.get("warning")
            if output_warning and output_warning not in printed_warnings:
                print(f"  warning: {output_warning}", flush=True)
        with ExitStack() as stack:
            if args.json:
                rank = MPI.COMM_WORLD.rank
                stdout_log = stack.enter_context(
                    context.artifact(f"logs/stdout.rank-{rank}.log").open(
                        "w", encoding="utf-8", buffering=1
                    )
                )
                stderr_log = stack.enter_context(
                    context.artifact(f"logs/stderr.rank-{rank}.log").open(
                        "w", encoding="utf-8", buffering=1
                    )
                )
                stack.enter_context(redirect_stdout(stdout_log))
                stack.enter_context(redirect_stderr(stderr_log))
            runpy.run_path(str(project.entrypoint), run_name="__main__")
    except Exception as exc:
        local_error = _rank_error(exc, rank=MPI.COMM_WORLD.rank)
    finally:
        os.chdir(previous_directory)
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    rank_errors = _collect_rank_errors(MPI.COMM_WORLD, local_error)
    if rank_errors:
        primary = rank_errors[0]
        error = {
            "type": primary["type"],
            "message": primary["message"],
            "rank": primary["rank"],
            "stage": primary.get("stage", "case_execution"),
            "code": primary.get("code"),
            "traceback": primary.get("traceback"),
            "validation": primary.get("validation"),
            "rank_errors": rank_errors,
        }
        if MPI.COMM_WORLD.rank == 0:
            context.write_execution("failed", error=error, structured_result=False)
        _emit(
            {**context.summary(), "status": "failed", "error": error},
            as_json=args.json,
            human=(
                f"Run failed on MPI rank {primary['rank']}: "
                f"{primary['type']}: {primary['message']}\n"
                f"  execution: {context.execution_path}"
            ),
        )
        return 1

    if MPI.COMM_WORLD.rank == 0 and not context.execution_path.is_file():
        context.write_execution(
            "completed",
            structured_result=context.manifest_path.is_file(),
        )
    record = (
        json.loads(context.execution_path.read_text(encoding="utf-8"))
        if MPI.COMM_WORLD.rank == 0
        else {}
    )
    _emit(
        record,
        as_json=args.json,
        human=(
            f"Run completed: {context.run_id}\n"
            f"  result: {context.manifest_path if context.manifest_path.is_file() else 'script completed without a published SimulationResult'}\n"
            f"  execution: {context.execution_path}"
        ),
    )
    return 0


def _command_inspect(args) -> int:
    selected = Path(args.path).expanduser().resolve() if args.path else None
    if selected is None:
        project = _project(args.project)
        selected = project.output_directory / project.name / "latest.json"
    if selected.is_dir():
        candidates = (selected / "execution.json", selected / "result.json", selected / "latest.json")
        selected = next((item for item in candidates if item.is_file()), selected)
    if not selected.is_file():
        raise FileNotFoundError(f"No inspectable AgentFEM record found at {selected}.")
    record = json.loads(selected.read_text(encoding="utf-8"))
    _emit(record, as_json=args.json, human=_format_record(record, selected))
    return 0


def _command_inspect_abaqus(args) -> int:
    from .mesh import inspect_abaqus_input

    report = inspect_abaqus_input(Path(args.path).expanduser().resolve())
    record = report.summary()
    if args.write:
        destination = Path(args.write).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_json(record) + "\n", encoding="utf-8")
        record["written_report"] = str(destination)
    _emit(record, as_json=args.json, human=report.text())
    return 0


def _command_inspect_user_material(args) -> int:
    from .constitutive.user_material import inspect_abaqus_user_material

    report = inspect_abaqus_user_material(
        Path(args.path).expanduser().resolve(), kind=args.kind
    )
    record = report.summary()
    if args.write:
        destination = Path(args.write).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_json(record) + "\n", encoding="utf-8")
        record["written_report"] = str(destination)
    _emit(record, as_json=args.json, human=report.format())
    return 2 if report.status == "manual_adaptation_required" else 0


def _command_migrate_abaqus(args) -> int:
    from .mesh import create_abaqus_migration_project

    record = create_abaqus_migration_project(
        Path(args.source).expanduser().resolve(),
        Path(args.destination).expanduser().resolve(),
        name=args.name,
        created_with=__version__,
        user_material_sources=_named_paths(
            args.user_material,
            option="--user-material",
        ),
    )
    _emit(
        record,
        as_json=args.json,
        human=(
            f"Created reviewable Abaqus migration project: {record['project']}\n"
            f"  plan: {record['migration_plan']}\n"
            f"  review: {record['migration_report']}\n"
            f"  native lowering: {record['native_lowering_status']}\n"
            "Next: run `agentfem check`, then review migration.json."
        ),
    )
    return 0


def _command_lower_abaqus(args) -> int:
    from .mesh import lower_abaqus_migration_project

    record = lower_abaqus_migration_project(
        Path(args.project).expanduser().resolve(),
        reviewed_by=args.reviewed_by,
        unit_system=args.unit_system,
        activate=args.activate,
        force=args.force,
    )
    _emit(
        record,
        as_json=args.json,
        human=(
            f"Created reviewed native Abaqus draft: {record['entrypoint']}\n"
            f"  decision: {record['lowering_record']}\n"
            f"  status: {record['status']}\n"
            + (
                "Next: run `agentfem check` and `agentfem run`."
                if args.activate
                else "Review the draft, then rerun with --activate --force."
            )
        ),
    )
    return 0


def _result_manifest_path(path: str | None, project_path: str | None) -> Path:
    """Resolve a result manifest from a file, run directory, or latest pointer."""

    selected = Path(path).expanduser().resolve() if path else None
    if selected is None:
        project = _project(project_path)
        selected = project.output_directory / project.name / "latest.json"
    if selected.is_dir():
        selected = selected / "result.json"
    for _ in range(3):
        if not selected.is_file():
            raise FileNotFoundError(
                f"No AgentFEM result manifest found at {selected}."
            )
        record = json.loads(selected.read_text(encoding="utf-8"))
        if record.get("schema") == "agentfem.simulation-result":
            return selected
        target = record.get("result_manifest")
        if target is None:
            raise ValueError(
                f"{selected} is not an AgentFEM result manifest or result pointer."
            )
        candidate = Path(str(target)).expanduser()
        selected = (
            candidate.resolve()
            if candidate.is_absolute()
            else (selected.parent / candidate).resolve()
        )
    raise ValueError("AgentFEM result pointer chain is unexpectedly deep.")


def _command_verify(args) -> int:
    path = _result_manifest_path(args.path, args.project)
    report = provenance.verify_manifest(path)
    _emit(report.summary(), as_json=args.json, human=report.format())
    return 0 if report.verified else 2


def _format_record(record: dict[str, object], path: Path) -> str:
    lines = [f"AgentFEM record: {path}"]
    for key in ("schema", "project", "run_id", "name", "status", "trust_level"):
        if key in record:
            lines.append(f"  {key}: {record[key]}")
    if "quantities" in record:
        lines.append(f"  quantities: {', '.join(record['quantities']) or '<none>'}")
    if "artifacts" in record:
        lines.append(f"  artifacts: {', '.join(record['artifacts']) or '<none>'}")
    return "\n".join(lines)


def _command_telemetry(args) -> int:
    from . import feedback as reliability

    if args.action == "on":
        selected = reliability.configure("basic")
        record = selected.summary()
        human = (
            "Anonymous reliability reporting is on. Models, meshes, parameters, "
            "source code, paths, and results are never included."
        )
    elif args.action == "off":
        selected = reliability.configure("off")
        record = selected.summary()
        human = "Anonymous reliability reporting is off."
    elif args.action == "flush":
        record = reliability.flush()
        human = f"Reliability delivery: {record['status']} ({record['sent']} sent)."
    elif args.action == "show-last":
        event = reliability.last_event()
        record = {
            "schema": "agentfem.reliability-preview",
            "schema_version": "0.1.0",
            "event": event,
        }
        human = _json(record) if event else "No queued reliability event."
    else:
        record = reliability.preferences().summary()
        human = (
            f"Anonymous reliability reporting: {record['mode']}\n"
            f"Delivery available: {record['delivery_available']}\n"
            f"Queued reports: {record['queue_size']}"
        )
    _emit(record, as_json=args.json, human=human)
    return 0


def _command_diagnose(args) -> int:
    from . import feedback as reliability

    record = reliability.diagnose(args.path, project=args.project)
    _emit(record, as_json=args.json, human=reliability.format_diagnosis(record))
    # The diagnosis command succeeded even when the execution it inspected
    # failed; reserve a non-zero status for an unavailable/invalid record.
    return 0


def _command_assist(args) -> int:
    from . import feedback as reliability

    record = reliability.create_support_directory(
        args.path,
        project=args.project,
        destination=args.output,
        force=args.force,
    )
    _emit(
        record,
        as_json=args.json,
        human=(
            f"Created private AgentFEM support task: {record['directory']}\n"
            "Give that directory to Codex or another AI agent. Nothing was uploaded."
        ),
    )
    return 0


def _command_feedback(args) -> int:
    from . import feedback as reliability

    if args.github:
        record = reliability.submit_github_issue(args.path, project=args.project)
        human = (
            f"Existing AgentFEM issue: {record['issue']['url']}"
            if record["status"] == "existing_issue"
            else f"Created AgentFEM issue: {record['url']}"
        )
    else:
        record = reliability.create_feedback_archive(
            args.path,
            project=args.project,
            destination=args.output,
        )
        human = (
            f"Created sanitized feedback archive: {record['archive']}\n"
            "Nothing was uploaded. Review it before sharing."
        )
    _emit(record, as_json=args.json, human=human)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentfem", description=__doc__)
    parser.add_argument("--version", action="version", version=f"AgentFEM {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Report the numerical runtime and optional integrations.")
    doctor.add_argument("--json", action="store_true")

    workspace = sub.add_parser(
        "workspace",
        help="Inspect or protect the project directory used by installed AgentFEM.",
    )
    workspace.add_argument("--path")
    workspace.add_argument(
        "--protect",
        action="store_true",
        help="On WSL, migrate ~/AgentFEMProjects to durable Windows storage.",
    )
    workspace.add_argument("--json", action="store_true")

    init = sub.add_parser("init", help="Create an installed-use AgentFEM project.")
    init.add_argument("path", nargs="?", default=".")
    init.add_argument("--template", default="static-solid")
    init.add_argument("--name")
    init.add_argument("--force", action="store_true")
    init.add_argument("--json", action="store_true")

    templates = sub.add_parser("templates", help="List version-matched installed project templates.")
    templates.add_argument("--json", action="store_true")

    check = sub.add_parser("check", help="Check project structure and Python syntax without solving.")
    check.add_argument("--project")
    check.add_argument("--json", action="store_true")

    upgrade = sub.add_parser(
        "upgrade",
        help="Inspect legacy patterns and produce a safe, agent-readable migration plan.",
    )
    upgrade.add_argument("--project")
    upgrade.add_argument("--apply-safe", action="store_true")
    upgrade.add_argument("--write-plan")
    upgrade.add_argument("--json", action="store_true")

    run = sub.add_parser("run", help="Run a project with a standard output and evidence contract.")
    run.add_argument("--project")
    run.add_argument("--run-id")
    run.add_argument("--output")
    run.add_argument("--mpi", type=int, default=1)
    run.add_argument("--json", action="store_true")
    run.add_argument("--inside-mpi", action="store_true", help=argparse.SUPPRESS)

    mpi_run = sub.add_parser(
        "mpi-run",
        help="Run a command with an MPI launcher verified against the active environment.",
    )
    mpi_run.add_argument("-n", "--ranks", type=int, required=True)
    mpi_run.add_argument("child_command", nargs=argparse.REMAINDER)

    inspect = sub.add_parser("inspect", help="Summarize a result, execution, or latest-run record.")
    inspect.add_argument("path", nargs="?")
    inspect.add_argument("--project")
    inspect.add_argument("--json", action="store_true")

    inspect_abaqus = sub.add_parser(
        "inspect-abaqus",
        help="Inventory an Abaqus input deck before migration or conversion.",
    )
    inspect_abaqus.add_argument("path")
    inspect_abaqus.add_argument("--write")
    inspect_abaqus.add_argument("--json", action="store_true")

    inspect_user_material = sub.add_parser(
        "inspect-user-material",
        help="Classify an Abaqus UMAT/UHYPER source before adapter development.",
    )
    inspect_user_material.add_argument("path")
    inspect_user_material.add_argument(
        "--kind", choices=("auto", "UMAT", "UHYPER"), default="auto"
    )
    inspect_user_material.add_argument("--write")
    inspect_user_material.add_argument("--json", action="store_true")

    migrate_abaqus = sub.add_parser(
        "migrate-abaqus",
        help="Create a fail-closed AgentFEM project from an Abaqus input deck.",
    )
    migrate_abaqus.add_argument("source")
    migrate_abaqus.add_argument("destination")
    migrate_abaqus.add_argument("--name")
    migrate_abaqus.add_argument(
        "--user-material",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Associate a deck *USER MATERIAL name with a UMAT/UHYPER source; "
            "repeat for multiple materials. Sources are inspected and bundled, "
            "never activated automatically."
        ),
    )
    migrate_abaqus.add_argument("--json", action="store_true")

    lower_abaqus = sub.add_parser(
        "lower-abaqus",
        help="Create a reviewed native draft from an eligible Abaqus migration.",
    )
    lower_abaqus.add_argument("project")
    lower_abaqus.add_argument("--reviewed-by", required=True)
    lower_abaqus.add_argument("--unit-system", required=True)
    lower_abaqus.add_argument("--activate", action="store_true")
    lower_abaqus.add_argument("--force", action="store_true")
    lower_abaqus.add_argument("--json", action="store_true")

    verify = sub.add_parser(
        "verify",
        help="Check a result manifest and the integrity of its registered artifacts.",
    )
    verify.add_argument("path", nargs="?")
    verify.add_argument("--project")
    verify.add_argument("--json", action="store_true")

    capabilities = sub.add_parser("capabilities", help="Return machine-readable runtime capabilities.")
    capabilities.add_argument("--json", action="store_true")
    extension_command = sub.add_parser(
        "extensions",
        help="List installed extensions or explicitly activate selected packages.",
    )
    extension_command.add_argument("--load", action="append", default=[])
    extension_command.add_argument("--json", action="store_true")

    telemetry = sub.add_parser(
        "telemetry",
        help="Inspect or control privacy-bounded anonymous reliability reports.",
    )
    telemetry.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=("status", "on", "off", "show-last", "flush"),
    )
    telemetry.add_argument("--json", action="store_true")

    diagnose = sub.add_parser(
        "diagnose",
        help="Explain the latest failed run locally without uploading evidence.",
    )
    diagnose.add_argument("path", nargs="?")
    diagnose.add_argument("--project")
    diagnose.add_argument("--json", action="store_true")

    assist = sub.add_parser(
        "assist",
        help="Create a private sanitized task for Codex or another AI agent.",
    )
    assist.add_argument("path", nargs="?")
    assist.add_argument("--project")
    assist.add_argument("--output")
    assist.add_argument("--force", action="store_true")
    assist.add_argument("--json", action="store_true")

    feedback_command = sub.add_parser(
        "feedback",
        help="Create a sanitized support archive or explicitly submit a GitHub issue.",
    )
    feedback_command.add_argument("path", nargs="?")
    feedback_command.add_argument("--project")
    feedback_command.add_argument("--output")
    feedback_command.add_argument("--github", action="store_true")
    feedback_command.add_argument("--json", action="store_true")
    return parser


def _dispatch(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = platforms.runtime_report()
            _emit(report.summary(), as_json=args.json, human=report.format())
            return 0
        if args.command == "workspace":
            return _command_workspace(args)
        if args.command == "templates":
            record = {"schema": "agentfem.templates", "schema_version": "0.1.0", "templates": _templates()}
            _emit(record, as_json=args.json, human="Available templates:\n" + "\n".join(f"  - {item}" for item in _templates()))
            return 0
        if args.command == "init":
            target = Path(args.path).expanduser().resolve()
            name = args.name or target.name
            _copy_template(args.template, target, name, force=args.force)
            project = ProjectConfig.load(target / PROJECT_FILENAME)
            record = {"schema": "agentfem.project-created", "schema_version": "0.1.0", "template": args.template, "project": project.summary()}
            _emit(record, as_json=args.json, human=f"Created AgentFEM project: {target}\nNext: cd {target} && agentfem check && agentfem run")
            return 0
        if args.command == "check":
            record = _check_project(_project(args.project))
            _emit(record, as_json=args.json, human=_format_check(record))
            return 0 if record["status"] == "passed" else 2
        if args.command == "upgrade":
            return _command_upgrade(args)
        if args.command == "run":
            if args.mpi <= 0:
                parser.error("--mpi must be positive")
            return _command_run(args)
        if args.command == "mpi-run":
            if args.ranks <= 0:
                parser.error("--ranks must be positive")
            return _command_mpi_run(args)
        if args.command == "inspect":
            return _command_inspect(args)
        if args.command == "inspect-abaqus":
            return _command_inspect_abaqus(args)
        if args.command == "inspect-user-material":
            return _command_inspect_user_material(args)
        if args.command == "migrate-abaqus":
            return _command_migrate_abaqus(args)
        if args.command == "lower-abaqus":
            return _command_lower_abaqus(args)
        if args.command == "verify":
            return _command_verify(args)
        if args.command == "telemetry":
            return _command_telemetry(args)
        if args.command == "diagnose":
            return _command_diagnose(args)
        if args.command == "assist":
            return _command_assist(args)
        if args.command == "feedback":
            return _command_feedback(args)
        if args.command == "extensions":
            extensions.load_extensions(args.load)
            record = extensions.extension_status()
            installed = record["installed"]
            human = "Installed AgentFEM extensions:\n" + (
                "\n".join(
                    f"  - {item['name']} "
                    f"({item['distribution'] or 'unknown package'})"
                    for item in installed
                )
                if installed
                else "  <none>"
            )
            _emit(record, as_json=args.json, human=human)
            return 0
        if args.command == "capabilities":
            from . import benchmarks, constitutive, models, public_api
            from ._architecture_contract import ownership_contract
            public_modules = {
                level: public_api(level)
                for level in ("core", "advanced", "expert")
            }
            record = {
                "schema": "agentfem.capabilities",
                "schema_version": CAPABILITIES_SCHEMA_VERSION,
                "agentfem_version": __version__,
                "commands": CLI_COMMANDS,
                # Keep the flat inventory for 0.2.0 clients while offering a
                # progressive contract to new CLIs, GUIs, and agents.
                "public_modules": public_api(),
                "public_api": public_modules,
                "model_api": {
                    level: models.model_api(level)
                    for level in ("core", "advanced", "compatibility")
                },
                "model_api_contract": models.model_api_contract(),
                "ownership_contract": ownership_contract(),
                "templates": _templates(),
                "runtime": platforms.runtime_report().summary(),
                "constitutive": tuple(
                    item.as_dict() for item in constitutive.capabilities()
                ),
                "constitutive_evidence": tuple(
                    item.as_dict()
                    for item in benchmarks.audit_capability_evidence()
                ),
                "step_providers": tuple(
                    item.summary() for item in models.step_providers()
                ),
                "extensions": extensions.extension_status(),
            }
            _emit(record, as_json=args.json, human=_json(record))
            return 0
    except (OSError, FileExistsError, ValueError, RuntimeError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        details = getattr(exc, "details", None)
        if callable(details):
            error["details"] = details()
        _emit(
            {
                "schema": "agentfem.cli-error",
                "schema_version": "0.1.0",
                "status": "failed",
                "error": error,
            },
            as_json=getattr(args, "json", False),
            human=f"AgentFEM: {exc}",
        )
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the product shell and observe reliability without owning the solve."""

    from . import feedback as reliability

    selected_argv = list(sys.argv[1:] if argv is None else argv)
    parsed = build_parser().parse_args(selected_argv)
    as_json = bool(getattr(parsed, "json", False))
    if parsed.command not in {"telemetry", "diagnose", "assist", "feedback"}:
        reliability.show_notice_once(as_json=as_json)
    started = time.monotonic()
    exit_code = _dispatch(selected_argv)
    # ``agentfem run --mpi N`` is a launcher. The rank-zero child records the
    # actual solve, preventing one user action from producing two events.
    launcher_only = (
        parsed.command == "run"
        and getattr(parsed, "mpi", 1) > 1
        and not getattr(parsed, "inside_mpi", False)
    )
    if not launcher_only:
        observation = reliability.observe_cli(
            parsed.command,
            exit_code,
            duration_seconds=time.monotonic() - started,
            project=getattr(parsed, "project", None),
        )
        escalation = observation.get("escalation") if observation else None
        if (
            not as_json
            and isinstance(escalation, dict)
            and escalation.get("suggest_agent_now")
        ):
            print(
                "This failure has repeated three times. Create a private task for "
                "Codex with: agentfem assist --force",
                file=sys.stderr,
                flush=True,
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
