"""Installed-project and run-context support for AgentFEM applications.

The project layer is deliberately thinner than the scientific model layer.  A
``case.py`` file remains ordinary Python and the source of modeling truth;
``agentfem.toml`` only records operational information needed by terminals,
IDEs, GUI clients, and agents to run the case consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import subprocess
import tomllib
from typing import Mapping


PROJECT_FILENAME = "agentfem.toml"
PROJECT_SCHEMA = "agentfem.project"
CURRENT_PROJECT_SCHEMA_VERSION = "0.2.0"
RUN_SCHEMA = "agentfem.run"

_ENV_PROJECT_ROOT = "AGENTFEM_PROJECT_ROOT"
_ENV_PROJECT_NAME = "AGENTFEM_PROJECT_NAME"
_ENV_OUTPUT_DIR = "AGENTFEM_OUTPUT_DIR"
_ENV_RUN_ID = "AGENTFEM_RUN_ID"
_ENV_MANIFEST = "AGENTFEM_RESULT_MANIFEST"
_ENV_EXECUTION = "AGENTFEM_EXECUTION_RECORD"
_ENV_PROJECTS_HOME = "AGENTFEM_PROJECTS_HOME"
_WORKSPACE_NAME = "AgentFEMProjects"


def _safe_name(value: str, *, label: str) -> str:
    selected = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-.")
    if not selected:
        raise ValueError(f"{label} must contain at least one letter or number.")
    return selected


def new_run_id(now: datetime | None = None) -> str:
    """Return a sortable, collision-resistant identifier for one execution."""

    selected = now or datetime.now(timezone.utc)
    stamp = selected.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def _running_under_wsl() -> bool:
    """Return whether the current Linux process belongs to WSL."""

    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    return platform.system() == "Linux" and "microsoft" in platform.release().lower()


def _is_windows_host_path(path: Path) -> bool:
    """Recognize a Windows drive mounted through WSL's conventional DrvFS path."""

    parts = path.absolute().parts
    return (
        len(parts) >= 4
        and parts[0] == "/"
        and parts[1] == "mnt"
        and len(parts[2]) == 1
        and parts[2].isalpha()
    )


def _windows_path_to_wsl(value: str) -> Path:
    completed = subprocess.run(
        ("wslpath", "-a", "-u", value),
        check=False,
        capture_output=True,
        text=True,
    )
    selected = completed.stdout.strip()
    if completed.returncode != 0 or not selected:
        raise RuntimeError(f"Could not translate Windows path into WSL: {value}")
    return Path(selected)


def _windows_documents_from_wsl() -> Path | None:
    """Discover the host Documents folder without assuming an English home path."""

    try:
        completed = subprocess.run(
            (
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Environment]::GetFolderPath('MyDocuments')",
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    selected = completed.stdout.replace("\x00", "").strip()
    if completed.returncode != 0 or not selected:
        return None
    try:
        return _windows_path_to_wsl(selected)
    except (OSError, RuntimeError):
        return None


def recommended_workspace(path: str | Path | None = None) -> Path:
    """Return the cross-installation project root recommended on this host.

    An explicit path or ``AGENTFEM_PROJECTS_HOME`` always wins.  A normal
    Linux/macOS installation uses a directory in the user's home.  Under WSL,
    AgentFEM discovers the Windows Documents directory so projects survive
    removal of the Linux distribution.
    """

    selected = path if path is not None else os.environ.get(_ENV_PROJECTS_HOME)
    if selected:
        raw = str(selected).strip()
        if _running_under_wsl() and re.match(r"^[A-Za-z]:[\\/]", raw):
            return _windows_path_to_wsl(raw).expanduser().absolute()
        return Path(raw).expanduser().absolute()
    if _running_under_wsl():
        documents = _windows_documents_from_wsl()
        if documents is not None:
            return (documents / _WORKSPACE_NAME).absolute()
    return (Path.home() / _WORKSPACE_NAME).absolute()


def _tree_fingerprint(root: Path) -> tuple[tuple[str, int, str], ...]:
    """Return a content identity used to verify a project-directory migration."""

    records: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(
                f"Workspace migration does not copy symbolic links safely: {path}"
            )
        if path.is_dir():
            records.append((relative + "/", 0, "directory"))
            continue
        if not path.is_file():
            raise ValueError(f"Workspace contains an unsupported special file: {path}")
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append((relative, path.stat().st_size, digest.hexdigest()))
    return tuple(records)


@dataclass(frozen=True)
class WorkspaceReport:
    """Custody report for the conventional installed-use project directory."""

    link: Path
    target: Path
    under_wsl: bool
    storage: str
    protected: bool
    exists: bool
    source: str
    retained_source: Path | None = None

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.workspace",
            "schema_version": "0.1.0",
            "link": str(self.link),
            "target": str(self.target),
            "under_wsl": self.under_wsl,
            "storage": self.storage,
            "protected_from_distribution_removal": self.protected,
            "exists": self.exists,
            "source": self.source,
            "retained_source": (
                None if self.retained_source is None else str(self.retained_source)
            ),
            "warning": (
                None
                if self.protected
                else "Projects stored inside this WSL distribution are deleted if the distribution is unregistered."
            ),
        }

    def format(self) -> str:
        status = "protected" if self.protected else "not protected"
        lines = [
            f"AgentFEM workspace: {status}",
            f"  projects: {self.link}",
            f"  storage: {self.storage}",
            f"  target: {self.target}",
        ]
        if self.retained_source is not None:
            lines.append(f"  retained pre-migration copy: {self.retained_source}")
        if not self.protected:
            lines.append(
                "  warning: this directory is removed with its WSL distribution"
            )
            lines.append("  fix: agentfem workspace --protect")
        return "\n".join(lines)


def workspace_report(
    path: str | Path | None = None,
    *,
    link: str | Path | None = None,
) -> WorkspaceReport:
    """Inspect project custody without changing the filesystem."""

    conventional = Path(link or (Path.home() / _WORKSPACE_NAME)).expanduser().absolute()
    under_wsl = _running_under_wsl()
    target = (
        conventional.resolve(strict=False)
        if path is None and conventional.is_symlink()
        else recommended_workspace(path)
    )
    resolved = conventional.resolve() if conventional.exists() else conventional
    host_storage = under_wsl and _is_windows_host_path(resolved)
    protected = not under_wsl or host_storage
    source = (
        "explicit"
        if path is not None
        else ("environment" if os.environ.get(_ENV_PROJECTS_HOME) else "detected")
    )
    return WorkspaceReport(
        link=conventional,
        target=target,
        under_wsl=under_wsl,
        storage=(
            "windows_host"
            if host_storage
            else ("wsl_distribution" if under_wsl else "native_user_home")
        ),
        protected=protected,
        exists=conventional.exists(),
        source=source,
    )


def storage_custody(path: str | Path) -> dict[str, object]:
    """Describe whether a project path survives removal of its runtime."""

    selected = Path(path).expanduser().absolute()
    resolved = selected.resolve(strict=False)
    under_wsl = _running_under_wsl()
    host_storage = under_wsl and _is_windows_host_path(resolved)
    protected = not under_wsl or host_storage
    return {
        "path": str(selected),
        "resolved_path": str(resolved),
        "storage": (
            "windows_host"
            if host_storage
            else ("wsl_distribution" if under_wsl else "native_filesystem")
        ),
        "protected_from_distribution_removal": protected,
        "warning": (
            None
            if protected
            else "This project is inside a WSL distribution and must be exported before that distribution is unregistered."
        ),
    }


def protect_workspace(
    path: str | Path | None = None,
    *,
    link: str | Path | None = None,
) -> WorkspaceReport:
    """Create or migrate the conventional workspace to durable host storage.

    Existing project files are copied and content-verified before the original
    directory is renamed.  The pre-migration copy is deliberately retained;
    AgentFEM never trades user data for cleanup convenience.
    """

    conventional = Path(link or (Path.home() / _WORKSPACE_NAME)).expanduser().absolute()
    target = (
        conventional.resolve(strict=False)
        if path is None and conventional.is_symlink()
        else recommended_workspace(path)
    )
    under_wsl = _running_under_wsl()
    if under_wsl and not _is_windows_host_path(target):
        raise ValueError(
            "A protected WSL workspace must be on a mounted Windows drive "
            "such as /mnt/c/Users/.../Documents/AgentFEMProjects."
        )
    if target.exists() and not target.is_dir():
        raise ValueError(f"Persistent workspace target is not a directory: {target}")
    if not under_wsl:
        target.mkdir(parents=True, exist_ok=True)
        return WorkspaceReport(
            link=target,
            target=target,
            under_wsl=False,
            storage="native_user_home",
            protected=True,
            exists=True,
            source="configured",
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    conventional.parent.mkdir(parents=True, exist_ok=True)
    if conventional.is_symlink():
        resolved = conventional.resolve()
        if resolved != target.resolve():
            raise ValueError(
                f"{conventional} already points to {resolved}; refusing to replace it."
            )
        target.mkdir(parents=True, exist_ok=True)
        return WorkspaceReport(
            link=conventional,
            target=target,
            under_wsl=True,
            storage="windows_host",
            protected=True,
            exists=True,
            source="configured",
        )
    if conventional.exists() and not conventional.is_dir():
        raise ValueError(f"Workspace path is not a directory: {conventional}")

    retained: Path | None = None
    if conventional.is_dir():
        source_identity = _tree_fingerprint(conventional)
        if target.exists() and any(target.iterdir()):
            if _tree_fingerprint(target) != source_identity:
                raise FileExistsError(
                    f"Persistent target is not empty and differs from {conventional}: {target}"
                )
        else:
            staging = target.parent / f".{target.name}.import-{secrets.token_hex(4)}"
            shutil.copytree(conventional, staging, copy_function=shutil.copyfile)
            if _tree_fingerprint(staging) != source_identity:
                raise RuntimeError(
                    "Workspace copy verification failed; original files remain "
                    "untouched."
                )
            if target.exists():
                target.rmdir()
            staging.replace(target)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        retained = conventional.with_name(
            f"{conventional.name}.retained-{stamp}-{secrets.token_hex(2)}"
        )
        conventional.rename(retained)
        try:
            conventional.symlink_to(target, target_is_directory=True)
        except Exception:
            retained.rename(conventional)
            raise
    else:
        target.mkdir(parents=True, exist_ok=True)
        conventional.symlink_to(target, target_is_directory=True)

    return WorkspaceReport(
        link=conventional,
        target=target,
        under_wsl=True,
        storage="windows_host",
        protected=True,
        exists=True,
        source="configured",
        retained_source=retained,
    )


@dataclass(frozen=True)
class ProjectConfig:
    """Operational metadata for an AgentFEM case directory."""

    root: Path
    name: str
    entrypoint: Path
    output_directory: Path
    extensions: tuple[str, ...] = ()
    schema_version: str = CURRENT_PROJECT_SCHEMA_VERSION
    created_with: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        selected = Path(path).expanduser().resolve()
        config_path = selected / PROJECT_FILENAME if selected.is_dir() else selected
        if not config_path.is_file():
            raise FileNotFoundError(f"AgentFEM project file not found: {config_path}")
        record = tomllib.loads(config_path.read_text(encoding="utf-8"))
        project = record.get("project", {})
        run = record.get("run", {})
        extension_record = record.get("extensions", {})
        root = config_path.parent
        name = _safe_name(project.get("name", root.name), label="project.name")
        entrypoint = root / project.get("entrypoint", "case.py")
        output_directory = root / run.get("output_directory", "outputs")
        required_extensions = extension_record.get("required", [])
        if not isinstance(required_extensions, list) or not all(
            isinstance(item, str) and item.strip() for item in required_extensions
        ):
            raise ValueError("extensions.required must be an array of non-empty names.")
        return cls(
            root=root,
            name=name,
            entrypoint=entrypoint.resolve(),
            output_directory=output_directory.resolve(),
            extensions=tuple(required_extensions),
            schema_version=str(
                project.get("schema_version", CURRENT_PROJECT_SCHEMA_VERSION)
            ),
            created_with=(
                None
                if project.get("created_with") is None
                else str(project["created_with"])
            ),
        )

    def check(self) -> tuple[str, ...]:
        """Return operational project errors without executing the case."""

        errors = []
        if not self.entrypoint.is_file():
            errors.append(f"Entrypoint does not exist: {self.entrypoint}")
        elif self.entrypoint.suffix != ".py":
            errors.append("The initial project runner supports Python entrypoints only.")
        try:
            self.entrypoint.relative_to(self.root)
        except ValueError:
            errors.append("project.entrypoint must remain inside the project directory.")
        try:
            self.output_directory.relative_to(self.root)
        except ValueError:
            errors.append(
                "run.output_directory must remain inside the project directory."
            )
        selected_schema = _numeric_version(self.schema_version)
        current_schema = _numeric_version(CURRENT_PROJECT_SCHEMA_VERSION)
        if selected_schema is None:
            errors.append(
                f"project.schema_version {self.schema_version!r} is not a valid "
                "numeric version."
            )
        elif selected_schema > current_schema:
            errors.append(
                f"Project schema {self.schema_version} is newer than this runtime "
                f"supports ({CURRENT_PROJECT_SCHEMA_VERSION})."
            )
        if self.extensions:
            from .extensions import missing_extensions

            missing = missing_extensions(self.extensions)
            if missing:
                errors.append(
                    "Required AgentFEM extensions are not installed: "
                    + ", ".join(missing)
                )
        return tuple(errors)

    def summary(self) -> dict[str, object]:
        return {
            "schema": PROJECT_SCHEMA,
            "schema_version": self.schema_version,
            "created_with": self.created_with,
            "name": self.name,
            "root": str(self.root),
            "entrypoint": str(self.entrypoint),
            "output_directory": str(self.output_directory),
            "extensions": self.extensions,
            "storage": storage_custody(self.root),
        }


def discover(start: str | Path | None = None) -> ProjectConfig:
    """Find the nearest ``agentfem.toml`` from ``start`` upward."""

    selected = Path.cwd() if start is None else Path(start).expanduser()
    selected = selected.resolve()
    if selected.is_file() and selected.name == PROJECT_FILENAME:
        return ProjectConfig.load(selected)
    current = selected.parent if selected.is_file() else selected
    for candidate in (current, *current.parents):
        project_file = candidate / PROJECT_FILENAME
        if project_file.is_file():
            return ProjectConfig.load(project_file)
    raise FileNotFoundError(
        f"No {PROJECT_FILENAME} found from {selected}. "
        "Run `agentfem init` or pass --project explicitly."
    )


def _numeric_version(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$", str(value))
    return None if match is None else tuple(int(item) for item in match.groups())


@dataclass(frozen=True)
class RunContext:
    """Filesystem and identity contract shared by scripts, CLIs, GUIs, and agents."""

    project_root: Path
    project_name: str
    run_id: str
    output_directory: Path
    manifest_path: Path
    execution_path: Path

    @classmethod
    def create(
        cls,
        project: ProjectConfig,
        *,
        run_id: str | None = None,
        output_directory: str | Path | None = None,
    ) -> "RunContext":
        selected_id = _safe_name(run_id or new_run_id(), label="run_id")
        root = (
            project.output_directory
            if output_directory is None
            else Path(output_directory).expanduser().resolve()
        )
        run_directory = root / project.name / selected_id
        return cls(
            project_root=project.root,
            project_name=project.name,
            run_id=selected_id,
            output_directory=run_directory,
            manifest_path=run_directory / "result.json",
            execution_path=run_directory / "execution.json",
        )

    @classmethod
    def from_environment(cls) -> "RunContext | None":
        root = os.environ.get(_ENV_PROJECT_ROOT)
        project_name = os.environ.get(_ENV_PROJECT_NAME)
        output = os.environ.get(_ENV_OUTPUT_DIR)
        run_id = os.environ.get(_ENV_RUN_ID)
        manifest = os.environ.get(_ENV_MANIFEST)
        execution = os.environ.get(_ENV_EXECUTION)
        if not all((root, project_name, output, run_id, manifest, execution)):
            return None
        project_root = Path(root).resolve()
        return cls(
            project_root=project_root,
            project_name=_safe_name(project_name, label="project_name"),
            run_id=_safe_name(run_id, label="run_id"),
            output_directory=Path(output).resolve(),
            manifest_path=Path(manifest).resolve(),
            execution_path=Path(execution).resolve(),
        )

    def prepare(self) -> "RunContext":
        self.output_directory.mkdir(parents=True, exist_ok=True)
        return self

    def artifact(self, name: str | Path) -> Path:
        """Resolve an artifact path inside this run directory."""

        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Run artifacts must be relative paths inside the run directory.")
        path = (self.output_directory / relative).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def environment(self) -> dict[str, str]:
        return {
            _ENV_PROJECT_ROOT: str(self.project_root),
            _ENV_PROJECT_NAME: self.project_name,
            _ENV_OUTPUT_DIR: str(self.output_directory),
            _ENV_RUN_ID: self.run_id,
            _ENV_MANIFEST: str(self.manifest_path),
            _ENV_EXECUTION: str(self.execution_path),
        }

    def summary(self) -> dict[str, object]:
        from .extensions import loaded_extensions

        return {
            "schema": RUN_SCHEMA,
            "schema_version": "0.1.0",
            "project": self.project_name,
            "run_id": self.run_id,
            "project_root": str(self.project_root),
            "output_directory": str(self.output_directory),
            "result_manifest": str(self.manifest_path),
            "execution_record": str(self.execution_path),
            "project_storage": storage_custody(self.project_root),
            "output_storage": storage_custody(self.output_directory),
            "extensions": tuple(
                item.as_dict() for item in loaded_extensions()
            ),
        }

    def write_execution(
        self,
        status: str,
        *,
        result_manifest: str | Path | None = None,
        error: Mapping[str, object] | None = None,
        structured_result: bool | None = None,
    ) -> Path:
        record = {
            **self.summary(),
            "status": str(status),
            "stage": (
                "completed"
                if error is None and str(status) == "completed"
                else (None if error is None else error.get("stage", "case_execution"))
            ),
            "structured_result": (
                self.manifest_path.is_file()
                if structured_result is None
                else bool(structured_result)
            ),
            "result_manifest": str(result_manifest or self.manifest_path),
            "error": None if error is None else dict(error),
        }
        _write_json(self.execution_path, record)
        self._write_latest_pointer(status)
        return self.execution_path

    def publish(self, result, *, include_histories: bool = True) -> Path:
        """Publish a ``SimulationResult`` through the stable run contract."""

        if not hasattr(result, "write_manifest"):
            raise TypeError("RunContext.publish requires a SimulationResult-like object.")
        result.metadata.setdefault("run", self.summary())
        path = result.write_manifest(
            self.manifest_path,
            include_histories=include_histories,
            relative_artifacts=True,
        )
        self.write_execution("completed", result_manifest=path, structured_result=True)
        return path

    def _write_latest_pointer(self, status: str) -> None:
        pointer = self.output_directory.parent / "latest.json"
        _write_json(
            pointer,
            {
                "schema": "agentfem.latest-run",
                "schema_version": "0.1.0",
                "project": self.project_name,
                "run_id": self.run_id,
                "status": status,
                "execution_record": str(self.execution_path),
                "result_manifest": str(self.manifest_path),
            },
        )


def current_run(
    *,
    project_root: str | Path | None = None,
    project_name: str | None = None,
) -> RunContext:
    """Return the CLI-provided context or create one for direct Python use."""

    active = RunContext.from_environment()
    if active is not None:
        return active.prepare()
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    try:
        config = discover(root)
    except FileNotFoundError:
        config = ProjectConfig(
            root=root,
            name=_safe_name(project_name or root.name, label="project_name"),
            entrypoint=root / "case.py",
            output_directory=root / "outputs",
        )
    return RunContext.create(config).prepare()


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "CURRENT_PROJECT_SCHEMA_VERSION",
    "PROJECT_FILENAME",
    "PROJECT_SCHEMA",
    "RUN_SCHEMA",
    "ProjectConfig",
    "RunContext",
    "WorkspaceReport",
    "current_run",
    "discover",
    "new_run_id",
    "protect_workspace",
    "recommended_workspace",
    "storage_custody",
    "workspace_report",
]
