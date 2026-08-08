"""Versioned external scientific data and dependency-free XLSX inspection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
import json
from pathlib import Path
import posixpath
import re
from typing import Iterable, Mapping
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np


_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REFERENCE = re.compile(r"([A-Z]+)([0-9]+)$")
_MANIFEST_SCHEMA = "agentfem.external-scientific-dataset"
_MANIFEST_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class ExternalFile:
    """One immutable file identity in a public scientific dataset."""

    path: str
    size: int
    sha256: str
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        path = str(self.path).strip()
        digest = str(self.sha256).strip().lower()
        roles = tuple(str(value).strip() for value in self.roles if str(value).strip())
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("External file paths must be nonempty relative paths.")
        if int(self.size) <= 0:
            raise ValueError("External file size must be positive.")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("External file sha256 must contain 64 hexadecimal digits.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "roles", roles)

    def summary(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "roles": self.roles,
        }


@dataclass(frozen=True)
class ExternalDatasetAudit:
    """Local evidence that downloaded public data matches its manifest."""

    dataset_id: str
    root: Path
    required_roles: tuple[str, ...]
    checked_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    size_mismatches: tuple[str, ...]
    digest_mismatches: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not (
            self.missing_files
            or self.size_mismatches
            or self.digest_mismatches
        )

    def require(self) -> "ExternalDatasetAudit":
        if not self.accepted:
            raise RuntimeError(
                "External scientific dataset audit failed: "
                f"missing={self.missing_files}, "
                f"size_mismatches={self.size_mismatches}, "
                f"digest_mismatches={self.digest_mismatches}."
            )
        return self

    def summary(self) -> dict[str, object]:
        return {
            "kind": "external_scientific_dataset_audit",
            "dataset_id": self.dataset_id,
            "root": str(self.root),
            "required_roles": self.required_roles,
            "checked_files": self.checked_files,
            "missing_files": self.missing_files,
            "size_mismatches": self.size_mismatches,
            "digest_mismatches": self.digest_mismatches,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class ExternalDatasetManifest:
    """Versioned public dataset identity, scope, and local audit policy."""

    dataset_id: str
    title: str
    doi: str
    version_id: str
    license: str
    landing_page: str
    default_required_roles: tuple[str, ...]
    files: tuple[ExternalFile, ...]
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        files = tuple(self.files)
        if not files or len({item.path for item in files}) != len(files):
            raise ValueError("External dataset files must be nonempty and unique.")
        object.__setattr__(self, "files", files)
        object.__setattr__(
            self,
            "default_required_roles",
            tuple(str(value) for value in self.default_required_roles),
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def read(cls, path: str | Path) -> "ExternalDatasetManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ExternalDatasetManifest":
        schema = str(data.get("schema", ""))
        version = str(data.get("schema_version", ""))
        if schema != _MANIFEST_SCHEMA:
            raise ValueError(
                f"Unsupported external dataset schema {schema!r}; "
                f"expected {_MANIFEST_SCHEMA!r}."
            )
        if version != _MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported external dataset schema version {version!r}; "
                f"expected {_MANIFEST_SCHEMA_VERSION!r}."
            )
        return cls(
            dataset_id=data["dataset_id"],
            title=data["title"],
            doi=data["doi"],
            version_id=str(data["version_id"]),
            license=data["license"],
            landing_page=data["landing_page"],
            default_required_roles=tuple(data.get("default_required_roles", ())),
            files=tuple(
                ExternalFile(
                    path=item["path"],
                    size=item["size"],
                    sha256=item["sha256"],
                    roles=tuple(item.get("roles", ())),
                )
                for item in data["files"]
            ),
            metadata=data.get("metadata"),
        )

    def files_for_roles(self, roles: Iterable[str] = ()) -> tuple[ExternalFile, ...]:
        selected_roles = tuple(str(value) for value in roles)
        if not selected_roles:
            selected_roles = self.default_required_roles
        required = set(selected_roles)
        available = {role for item in self.files for role in item.roles}
        unknown = required - available
        if unknown:
            raise ValueError(
                f"External dataset {self.dataset_id!r} has no files for roles "
                f"{tuple(sorted(unknown))}."
            )
        return tuple(
            item for item in self.files if required.intersection(item.roles)
        )

    def audit(
        self,
        root: str | Path,
        *,
        roles: Iterable[str] = (),
        verify_hashes: bool = True,
    ) -> ExternalDatasetAudit:
        location = Path(root)
        selected_roles = tuple(str(value) for value in roles)
        if not selected_roles:
            selected_roles = self.default_required_roles
        selected_files = self.files_for_roles(selected_roles)
        if not selected_files:
            raise ValueError(
                f"No external files match required roles {selected_roles!r}."
            )
        missing = []
        sizes = []
        digests = []
        checked = []
        for item in selected_files:
            path = location / item.path
            if not path.is_file():
                missing.append(item.path)
                continue
            checked.append(item.path)
            if path.stat().st_size != item.size:
                sizes.append(item.path)
                continue
            if verify_hashes and _file_sha256(path) != item.sha256:
                digests.append(item.path)
        return ExternalDatasetAudit(
            dataset_id=self.dataset_id,
            root=location.resolve(),
            required_roles=selected_roles,
            checked_files=tuple(checked),
            missing_files=tuple(missing),
            size_mismatches=tuple(sizes),
            digest_mismatches=tuple(digests),
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "external_scientific_dataset_manifest",
            "dataset_id": self.dataset_id,
            "title": self.title,
            "doi": self.doi,
            "version_id": self.version_id,
            "license": self.license,
            "landing_page": self.landing_page,
            "default_required_roles": self.default_required_roles,
            "file_count": len(self.files),
            "total_size": sum(item.size for item in self.files),
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class SpreadsheetSheet:
    """Rectangular values from one XLSX worksheet."""

    name: str
    rows: tuple[tuple[object | None, ...], ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return max((len(row) for row in self.rows), default=0)

    def array(self, *, fill_value=None) -> np.ndarray:
        values = np.full(
            (self.row_count, self.column_count),
            fill_value,
            dtype=object,
        )
        for row_index, row in enumerate(self.rows):
            values[row_index, : len(row)] = row
        return values

    def numeric_block(
        self,
        *,
        start_row: int = 0,
        start_column: int = 0,
        stop_row: int | None = None,
        stop_column: int | None = None,
    ) -> np.ndarray:
        """Return a strict finite numeric sub-table for research comparison."""

        raw = self.array(fill_value=np.nan)[
            int(start_row) : None if stop_row is None else int(stop_row),
            int(start_column) : None if stop_column is None else int(stop_column),
        ]
        try:
            values = raw.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Worksheet {self.name!r} selection is not purely numeric."
            ) from exc
        if values.size == 0 or np.any(~np.isfinite(values)):
            raise ValueError(
                f"Worksheet {self.name!r} numeric selection is empty or nonfinite."
            )
        return values

    def summary(self) -> dict[str, object]:
        values = self.array()
        nonempty = sum(value is not None for value in values.reshape(-1))
        numeric = sum(
            isinstance(value, (int, float, np.integer, np.floating))
            for value in values.reshape(-1)
            if value is not None
        )
        return {
            "name": self.name,
            "rows": self.row_count,
            "columns": self.column_count,
            "nonempty_cells": nonempty,
            "numeric_cells": numeric,
        }


@dataclass(frozen=True)
class SpreadsheetWorkbook:
    """Dependency-free, read-only representation of one XLSX workbook."""

    path: Path
    sheets: tuple[SpreadsheetSheet, ...]

    def sheet(self, name: str) -> SpreadsheetSheet:
        for sheet in self.sheets:
            if sheet.name == name:
                return sheet
        raise KeyError(
            f"Workbook does not contain sheet {name!r}; "
            f"available={tuple(sheet.name for sheet in self.sheets)}."
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "xlsx_workbook",
            "path": str(self.path),
            "sheets": [sheet.summary() for sheet in self.sheets],
        }


def science_supershear_dryad_manifest() -> ExternalDatasetManifest:
    """Return the pinned CC0 Dryad v7 manifest for Science 2023."""

    location = resources.files("agentfem.knowledge").joinpath(
        "external_data/science_supershear_dryad_v7.json"
    )
    return ExternalDatasetManifest.from_mapping(
        json.loads(location.read_text(encoding="utf-8"))
    )


def science_supershear_v5_research_task() -> dict[str, object]:
    """Return the installed machine-readable V5 research handoff."""

    location = resources.files("agentfem.knowledge").joinpath(
        "research_tasks/science_supershear_v5.json"
    )
    task = json.loads(location.read_text(encoding="utf-8"))
    if task.get("schema") != "agentfem.research-task":
        raise ValueError("Unsupported AgentFEM research-task schema.")
    if task.get("schema_version") != "0.1.0":
        raise ValueError("Unsupported AgentFEM research-task schema version.")
    if not task.get("work_packages") or not task.get("stop_conditions"):
        raise ValueError("Research task requires work packages and stop conditions.")
    return task


def read_xlsx_workbook(path: str | Path) -> SpreadsheetWorkbook:
    """Read values and cached formula results from an XLSX without pandas."""

    location = Path(path)
    if location.suffix.lower() != ".xlsx":
        raise ValueError("read_xlsx_workbook requires an .xlsx file.")
    with ZipFile(location) as archive:
        shared = _shared_strings(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall(f"{{{_PACKAGE_REL_NS}}}Relationship")
        }
        sheets = []
        for item in workbook.findall(f".//{{{_MAIN_NS}}}sheet"):
            relation = item.attrib[f"{{{_REL_NS}}}id"]
            target = targets[relation].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            target = posixpath.normpath(target)
            if target.startswith("../") or target not in archive.namelist():
                raise ValueError(
                    f"Worksheet relationship resolves outside the XLSX archive: {target!r}."
                )
            sheets.append(
                SpreadsheetSheet(
                    name=item.attrib["name"],
                    rows=_worksheet_rows(archive, target, shared),
                )
            )
    return SpreadsheetWorkbook(path=location.resolve(), sheets=tuple(sheets))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _shared_strings(archive: ZipFile) -> tuple[str, ...]:
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return ()
    root = ET.fromstring(archive.read(name))
    return tuple(
        "".join(node.text or "" for node in item.findall(f".//{{{_MAIN_NS}}}t"))
        for item in root.findall(f"{{{_MAIN_NS}}}si")
    )


def _worksheet_rows(
    archive: ZipFile,
    target: str,
    shared: tuple[str, ...],
) -> tuple[tuple[object | None, ...], ...]:
    root = ET.fromstring(archive.read(target))
    sparse: dict[int, dict[int, object | None]] = {}
    maximum_row = -1
    maximum_column = -1
    for cell in root.findall(f".//{{{_MAIN_NS}}}c"):
        reference = cell.attrib.get("r")
        if reference is None:
            continue
        column, row = _cell_indices(reference)
        value = _cell_value(cell, shared)
        sparse.setdefault(row, {})[column] = value
        maximum_row = max(maximum_row, row)
        maximum_column = max(maximum_column, column)
    rows = []
    for row in range(maximum_row + 1):
        values = [
            sparse.get(row, {}).get(column)
            for column in range(maximum_column + 1)
        ]
        while values and values[-1] is None:
            values.pop()
        rows.append(tuple(values))
    while rows and not rows[-1]:
        rows.pop()
    return tuple(rows)


def _cell_indices(reference: str) -> tuple[int, int]:
    match = _CELL_REFERENCE.fullmatch(reference.upper())
    if match is None:
        raise ValueError(f"Unsupported XLSX cell reference {reference!r}.")
    letters, row = match.groups()
    column = 0
    for letter in letters:
        column = 26 * column + ord(letter) - ord("A") + 1
    return column - 1, int(row) - 1


def _cell_value(cell, shared: tuple[str, ...]):
    cell_type = cell.attrib.get("t", "n")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.findall(f".//{{{_MAIN_NS}}}t")
        )
    value_node = cell.find(f"{{{_MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared[int(raw)]
    if cell_type in {"str", "e"}:
        return raw
    if cell_type == "b":
        return bool(int(raw))
    number = float(raw)
    return int(number) if number.is_integer() else number


__all__ = [
    "ExternalDatasetAudit",
    "ExternalDatasetManifest",
    "ExternalFile",
    "SpreadsheetSheet",
    "SpreadsheetWorkbook",
    "read_xlsx_workbook",
    "science_supershear_dryad_manifest",
    "science_supershear_v5_research_task",
]
