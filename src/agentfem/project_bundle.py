"""Deterministic, integrity-checked transport for AgentFEM projects.

An ``.afm`` bundle carries scientific inputs and project code, never outputs,
credentials, caches, or machine-specific execution state.  It is a ZIP
container with a versioned manifest, but callers must use this module rather
than treating arbitrary ZIP members as trusted paths.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import ast
from fnmatch import fnmatch
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile
from typing import Iterator
import zipfile

from .project import PROJECT_FILENAME, ProjectConfig


BUNDLE_SCHEMA = "agentfem.project-bundle"
BUNDLE_SCHEMA_VERSION = "0.1.0"
BUNDLE_MANIFEST = "agentfem.bundle.json"
BUNDLE_SUFFIX = ".afm"

_IGNORED_NAMES = {
    ".DS_Store",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    "__pycache__",
}
_SECRET_PATTERNS = (
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    "credentials.json",
    "secrets.toml",
)


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(name: str) -> PurePosixPath:
    if "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe AgentFEM bundle member: {name!r}.")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe AgentFEM bundle member: {name!r}.")
    if path.parts[0].endswith(":"):
        raise ValueError(f"Unsafe AgentFEM bundle drive path: {name!r}.")
    return path


def _ignore_patterns(root: Path) -> tuple[str, ...]:
    path = root / ".agentfemignore"
    if not path.is_file():
        return ()
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _ignored(relative: Path, *, patterns: tuple[str, ...], output: Path | None) -> bool:
    if any(part in _IGNORED_NAMES for part in relative.parts):
        return True
    logical = relative.as_posix()
    if output is not None and (relative == output or output in relative.parents):
        return True
    if any(fnmatch(logical, pattern) or fnmatch(relative.name, pattern) for pattern in _SECRET_PATTERNS):
        return True
    return any(fnmatch(logical, pattern) for pattern in patterns)


@dataclass(frozen=True)
class BundleReport:
    """Verified contents and identity of one portable project bundle."""

    path: Path
    project: dict[str, object]
    files: tuple[dict[str, object], ...]
    bundle_sha256: str
    excluded: tuple[str, ...] = ()

    def summary(self) -> dict[str, object]:
        return {
            "schema": BUNDLE_SCHEMA,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "status": "verified",
            "path": str(self.path),
            "bundle_sha256": self.bundle_sha256,
            "project": dict(self.project),
            "file_count": len(self.files),
            "files": self.files,
            "excluded": self.excluded,
        }

    def source_summary(self) -> dict[str, object]:
        return {
            "kind": "agentfem_project_bundle",
            "name": self.path.name,
            "sha256": self.bundle_sha256,
            "schema_version": BUNDLE_SCHEMA_VERSION,
        }


def pack_project(
    project: ProjectConfig | str | Path,
    destination: str | Path | None = None,
) -> BundleReport:
    """Create a deterministic project bundle after fail-closed path checks."""

    config = project if isinstance(project, ProjectConfig) else ProjectConfig.load(project)
    errors = config.check()
    if errors:
        raise ValueError("Cannot pack an invalid AgentFEM project: " + "; ".join(errors))
    target = (
        config.root.parent / f"{config.name}{BUNDLE_SUFFIX}"
        if destination is None
        else Path(destination).expanduser().resolve()
    )
    if target.suffix.lower() != BUNDLE_SUFFIX:
        target = target.with_suffix(BUNDLE_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        output_relative = config.output_directory.relative_to(config.root)
    except ValueError:
        output_relative = None
    patterns = _ignore_patterns(config.root)
    absolute_inputs: list[str] = []
    for source in sorted(config.root.rglob("*.py")):
        relative = source.relative_to(config.root)
        if _ignored(relative, patterns=patterns, output=output_relative):
            continue
        absolute_inputs.extend(
            f"{relative.as_posix()}:{literal}"
            for literal in _obvious_absolute_inputs(source)
        )
    if absolute_inputs:
        raise ValueError(
            "AFM-PROJECT-PORTABILITY-001: project code contains obvious "
            "machine-absolute input paths: "
            + ", ".join(absolute_inputs)
            + ". Copy scientific assets into the project and use paths relative "
            "to Path(__file__)."
        )
    files: list[tuple[str, Path]] = []
    excluded: list[str] = []
    for path in sorted(config.root.rglob("*")):
        relative = path.relative_to(config.root)
        if _ignored(relative, patterns=patterns, output=output_relative):
            if path.is_file():
                excluded.append(relative.as_posix())
            continue
        if path.is_symlink():
            raise ValueError(
                f"Project bundles reject symbolic links; replace {relative} with "
                "an ordinary project file."
            )
        if not path.is_file() or path.resolve() == target:
            continue
        files.append((relative.as_posix(), path))
    names = {name for name, _ in files}
    if PROJECT_FILENAME not in names:
        raise ValueError(f"Project bundle is missing {PROJECT_FILENAME}.")
    entrypoint = config.entrypoint.relative_to(config.root).as_posix()
    if entrypoint not in names:
        raise ValueError(f"Project bundle is missing entrypoint {entrypoint!r}.")

    file_records = tuple(
        {
            "path": name,
            "size_bytes": path.stat().st_size,
            "sha256": _file_digest(path),
        }
        for name, path in files
    )
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "project": {
            "name": config.name,
            "entrypoint": entrypoint,
            "project_schema_version": config.schema_version,
            "required_extensions": config.extensions,
        },
        "files": file_records,
        "excluded": tuple(sorted(excluded)),
    }
    temporary = target.with_name(target.name + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, path in files:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with path.open("rb") as source, archive.open(
                    info, "w", force_zip64=True
                ) as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            info = zipfile.ZipInfo(
                BUNDLE_MANIFEST, date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _json_bytes(manifest))
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return inspect_bundle(target)


def inspect_bundle(path: str | Path) -> BundleReport:
    """Validate every path, size, and digest before a bundle can be used."""

    selected = Path(path).expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"AgentFEM project bundle not found: {selected}")
    with zipfile.ZipFile(selected, "r") as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise ValueError("AgentFEM project bundle contains duplicate members.")
        for name in members:
            _safe_member(name)
        if BUNDLE_MANIFEST not in members:
            raise ValueError("AgentFEM project bundle has no manifest.")
        manifest = json.loads(archive.read(BUNDLE_MANIFEST))
        if manifest.get("schema") != BUNDLE_SCHEMA:
            raise ValueError("Unsupported AgentFEM project bundle schema.")
        if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise ValueError("Unsupported AgentFEM project bundle schema version.")
        records = manifest.get("files")
        if not isinstance(records, list):
            raise ValueError("AgentFEM project bundle file inventory is invalid.")
        expected_names: set[str] = set()
        verified: list[dict[str, object]] = []
        for raw in records:
            if not isinstance(raw, dict):
                raise ValueError("AgentFEM project bundle file record is invalid.")
            name = _safe_member(str(raw.get("path", ""))).as_posix()
            if name == BUNDLE_MANIFEST or name in expected_names:
                raise ValueError("AgentFEM project bundle inventory is ambiguous.")
            expected_names.add(name)
            if name not in members:
                raise ValueError(f"AgentFEM project bundle is missing {name!r}.")
            info = archive.getinfo(name)
            if info.file_size != int(raw.get("size_bytes", -1)):
                raise ValueError(f"AgentFEM project bundle size differs for {name!r}.")
            if _archive_member_digest(archive, name) != str(raw.get("sha256", "")):
                raise ValueError(f"AgentFEM project bundle digest differs for {name!r}.")
            verified.append(dict(raw))
        extras = set(members) - expected_names - {BUNDLE_MANIFEST}
        if extras:
            raise ValueError(
                "AgentFEM project bundle contains unregistered members: "
                + ", ".join(sorted(extras))
            )
        project = manifest.get("project")
        if not isinstance(project, dict):
            raise ValueError("AgentFEM project bundle project record is invalid.")
        if PROJECT_FILENAME not in expected_names:
            raise ValueError(f"AgentFEM project bundle lacks {PROJECT_FILENAME}.")
    return BundleReport(
        path=selected,
        project=dict(project),
        files=tuple(verified),
        bundle_sha256=_file_digest(selected),
        excluded=tuple(str(item) for item in manifest.get("excluded", ())),
    )


def unpack_bundle(
    path: str | Path,
    destination: str | Path,
    *,
    force: bool = False,
) -> BundleReport:
    """Materialize a verified bundle without using unsafe ZIP extraction."""

    report = inspect_bundle(path)
    target = Path(destination).expanduser().resolve()
    if target.exists() and any(target.iterdir()) and not force:
        raise FileExistsError(
            f"Bundle destination is not empty: {target}. Pass --force to add files."
        )
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(report.path, "r") as archive:
        for record in report.files:
            relative = _safe_member(str(record["path"]))
            output = target.joinpath(*relative.parts)
            if output.exists() and not force:
                raise FileExistsError(f"Refusing to overwrite {output}.")
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(relative.as_posix(), "r") as source, output.open(
                "wb"
            ) as destination_stream:
                shutil.copyfileobj(source, destination_stream, length=1024 * 1024)
    config = ProjectConfig.load(target)
    errors = config.check()
    if errors:
        raise ValueError("Unpacked AgentFEM project is invalid: " + "; ".join(errors))
    return report


@contextmanager
def materialize_bundle(path: str | Path) -> Iterator[tuple[Path, BundleReport]]:
    """Yield a temporary verified project root for direct bundle execution."""

    with tempfile.TemporaryDirectory(prefix="agentfem-bundle-") as directory:
        root = Path(directory)
        report = unpack_bundle(path, root)
        yield root, report


def _json_bytes(record: dict[str, object]) -> bytes:
    return (
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _archive_member_digest(archive: zipfile.ZipFile, name: str) -> str:
    digest = sha256()
    with archive.open(name, "r") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _obvious_absolute_inputs(entrypoint: Path) -> tuple[str, ...]:
    """Find common literal file inputs that cannot survive project transfer."""

    tree = ast.parse(entrypoint.read_text(encoding="utf-8"), filename=str(entrypoint))
    selected: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func).lower()
        path_like = (
            name in {"path", "open", "load", "loadtxt", "genfromtxt"}
            or name.startswith("read")
            or name.endswith(("_file", "_mesh", "_table"))
        )
        if not path_like:
            continue
        values = list(node.args[:1]) + [
            item.value
            for item in node.keywords
            if item.arg in {"path", "file", "filename", "source"}
        ]
        for value in values:
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            literal = value.value
            # ``Path`` follows the host OS and therefore misses POSIX paths on
            # Windows (and can miss Windows paths on POSIX).  A portable bundle
            # must recognize both syntaxes independently of where it is packed.
            if (
                PurePosixPath(literal).is_absolute()
                or PureWindowsPath(literal).is_absolute()
            ):
                selected.add(literal)
    return tuple(sorted(selected))


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


__all__ = [
    "BUNDLE_MANIFEST",
    "BUNDLE_SCHEMA",
    "BUNDLE_SCHEMA_VERSION",
    "BUNDLE_SUFFIX",
    "BundleReport",
    "inspect_bundle",
    "materialize_bundle",
    "pack_project",
    "unpack_bundle",
]
