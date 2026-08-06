"""External CAE mesh-format conversion helpers.

These helpers use ``meshio`` as an optional dependency. AgentFEM core mesh
operations do not require meshio, but Abaqus, NASTRAN, COMSOL-exported meshes,
and many neutral formats are best handled through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from agentfem import dependencies


SUPPORTED_EXTERNAL_FORMATS = {
    ".inp": "Abaqus input",
    ".cdb": "ANSYS CDB archive when supported by meshio",
    ".bdf": "NASTRAN bulk data",
    ".fem": "NASTRAN/OptiStruct bulk data",
    ".nas": "NASTRAN bulk data",
    ".msh": "Gmsh mesh",
    ".e": "Exodus mesh",
    ".exo": "Exodus mesh",
    ".med": "MED/Salome mesh",
    ".mphtxt": "COMSOL text mesh when supported by meshio",
    ".vtk": "VTK legacy mesh",
    ".vtu": "VTK XML unstructured grid",
    ".xdmf": "XDMF mesh",
}


@dataclass(frozen=True)
class MeshConversionResult:
    """Paths written by an external mesh conversion."""

    mesh_path: Path
    cell_type: str
    source_path: Path
    manifest_path: Path | None = None
    region_tags: dict[str, int] = field(default_factory=dict)
    facet_path: Path | None = None
    boundary_tags: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    source_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CellBlockSummary:
    """One source-mesh cell block."""

    cell_type: str
    count: int


@dataclass(frozen=True)
class ExternalMeshSummary:
    """Format-independent inventory before conversion."""

    source_path: Path
    point_count: int
    geometric_dimension: int
    cell_blocks: tuple[CellBlockSummary, ...]
    cell_sets: dict[str, dict[str, int]]
    point_sets: dict[str, int]
    point_data: tuple[str, ...]
    cell_data: tuple[str, ...]

    @property
    def cell_types(self) -> tuple[str, ...]:
        return tuple(block.cell_type for block in self.cell_blocks)

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "point_count": self.point_count,
            "geometric_dimension": self.geometric_dimension,
            "cell_blocks": [
                {"cell_type": block.cell_type, "count": block.count}
                for block in self.cell_blocks
            ],
            "cell_sets": self.cell_sets,
            "point_sets": self.point_sets,
            "point_data": self.point_data,
            "cell_data": self.cell_data,
        }


@dataclass(frozen=True)
class ExternalMeshBundle:
    """Explicit collection of single-topology solver-domain conversions."""

    source_path: Path
    conversions: tuple[MeshConversionResult, ...]
    manifest_path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "manifest_path": str(self.manifest_path),
            "domains": tuple(
                {
                    "cell_type": item.cell_type,
                    "mesh_path": str(item.mesh_path),
                    "manifest_path": str(item.manifest_path),
                }
                for item in self.conversions
            ),
        }


def require_meshio():
    """Import meshio or raise an actionable optional-dependency error."""

    return dependencies.require(
        "meshio",
        extra="mesh-formats",
        capability="External CAE mesh conversion",
    )


def describe_supported_external_formats() -> dict[str, str]:
    """Return common external mesh formats handled through meshio."""

    return dict(SUPPORTED_EXTERNAL_FORMATS)


def detect_mesh_format(path: str | Path) -> str:
    """Return a human-readable mesh format label from a file extension."""

    suffix = Path(path).suffix.lower()
    return SUPPORTED_EXTERNAL_FORMATS.get(suffix, "meshio-supported mesh format")


def inspect_external_mesh(
    path: str | Path,
    *,
    input_format: str | None = None,
) -> ExternalMeshSummary:
    """Read and inventory an external mesh without converting it.

    The inventory makes element blocks and named Abaqus/NASTRAN-style sets
    visible before AgentFEM chooses a DOLFINx cell topology.
    """

    meshio = require_meshio()
    source_path = Path(path)
    return summarize_external_mesh(
        source_path,
        meshio.read(source_path, file_format=input_format),
    )


def summarize_external_mesh(
    source_path: str | Path,
    source_mesh,
) -> ExternalMeshSummary:
    """Summarize an already-read meshio-like mesh."""

    points = np.asarray(source_mesh.points)
    geometric_dimension = int(points.shape[1]) if points.ndim == 2 else 0
    cell_sets: dict[str, dict[str, int]] = {}
    for name, values_by_block in getattr(source_mesh, "cell_sets", {}).items():
        counts: dict[str, int] = {}
        for index, values in enumerate(values_by_block):
            if index >= len(source_mesh.cells) or values is None:
                continue
            cell_type = source_mesh.cells[index].type
            counts[cell_type] = counts.get(cell_type, 0) + int(
                np.asarray(values).size
            )
        cell_sets[str(name)] = counts
    point_sets = {
        str(name): int(np.asarray(values).size)
        for name, values in getattr(source_mesh, "point_sets", {}).items()
    }
    return ExternalMeshSummary(
        source_path=Path(source_path),
        point_count=int(points.shape[0]),
        geometric_dimension=geometric_dimension,
        cell_blocks=tuple(
            CellBlockSummary(block.type, int(np.asarray(block.data).shape[0]))
            for block in source_mesh.cells
        ),
        cell_sets=cell_sets,
        point_sets=point_sets,
        point_data=tuple(sorted(getattr(source_mesh, "point_data", {}))),
        cell_data=tuple(sorted(getattr(source_mesh, "cell_data", {}))),
    )


def convert_topology_bundle(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    cell_types=None,
    prune_z: bool = False,
    input_format: str | None = None,
) -> ExternalMeshBundle:
    """Convert every requested source topology into an explicit domain file.

    DOLFINx mixed-topology support is still incomplete, so AgentFEM does not
    merge unlike top-dimensional cells into one opaque solver mesh.  The
    bundle preserves each block and makes domain selection an explicit model
    decision.
    """

    source = Path(input_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    inventory = inspect_external_mesh(source, input_format=input_format)
    selected = tuple(dict.fromkeys(cell_types or inventory.cell_types))
    conversions = tuple(
        convert_to_xdmf(
            source,
            output / f"{source.stem}_{cell_type}.xdmf",
            cell_type=cell_type,
            prune_z=prune_z,
            input_format=input_format,
        )
        for cell_type in selected
    )
    manifest_path = output / f"{source.stem}.mesh-bundle.json"
    manifest = {
        "schema": "agentfem.mesh-conversion-bundle",
        "schema_version": "0.1.0",
        "source": str(source),
        "source_fingerprint": _source_fingerprint(source),
        "domains": [
            {
                "cell_type": item.cell_type,
                "mesh_path": str(item.mesh_path),
                "manifest_path": str(item.manifest_path),
            }
            for item in conversions
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ExternalMeshBundle(source, conversions, manifest_path)


def convert_to_xdmf(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    cell_type: str | None = None,
    prune_z: bool = False,
    manifest_path: str | Path | None = None,
    facet_type: str | None = None,
    facet_output_path: str | Path | None = None,
    input_format: str | None = None,
    source_metadata: dict[str, object] | None = None,
    conversion_warnings: tuple[str, ...] = (),
) -> MeshConversionResult:
    """Convert external volume/cell and optional boundary blocks to XDMF.

    DOLFINx XDMF import expects one topological cell type per grid. Pass
    ``facet_type`` (for example ``"line"`` beside ``"triangle"`` or
    ``"triangle"`` beside ``"tetra"``) to write a second XDMF file carrying
    named boundary-set tags.
    """

    meshio = require_meshio()
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else input_path.with_suffix(".xdmf")

    source_mesh = meshio.read(input_path, file_format=input_format)
    inspection = summarize_external_mesh(input_path, source_mesh)
    selected_type = cell_type or _first_supported_cell_type(source_mesh.cells)
    cells = _cells_of_type(source_mesh.cells, selected_type)
    points = np.asarray(source_mesh.points)
    if prune_z:
        points = points[:, :2]

    cell_data = _cell_data_for_type(
        source_mesh.cell_data,
        source_mesh.cells,
        selected_type,
    )
    region_tags, set_members, overlaps = _cell_set_tags(
        source_mesh,
        selected_type,
        len(cells),
    )
    if region_tags:
        cell_data["agentfem_region"] = [
            _region_tag_array(set_members, region_tags, len(cells))
        ]
    converted = meshio.Mesh(
        points=points,
        cells=[(selected_type, cells)],
        cell_data=cell_data,
        point_data=dict(source_mesh.point_data),
        field_data=dict(source_mesh.field_data),
    )
    meshio.write(output_path, converted)

    selected_facet_path = None
    boundary_tags: dict[str, int] = {}
    boundary_members: dict[str, np.ndarray] = {}
    boundary_overlaps: dict[int, tuple[str, ...]] = {}
    if facet_type is not None:
        if facet_type == selected_type:
            raise ValueError("facet_type must differ from the selected cell_type.")
        facet_cells = _cells_of_type(source_mesh.cells, facet_type)
        facet_data = _cell_data_for_type(
            source_mesh.cell_data,
            source_mesh.cells,
            facet_type,
        )
        boundary_tags, boundary_members, boundary_overlaps = _cell_set_tags(
            source_mesh,
            facet_type,
            len(facet_cells),
        )
        if boundary_tags:
            facet_data["agentfem_boundary"] = [
                _region_tag_array(
                    boundary_members,
                    boundary_tags,
                    len(facet_cells),
                )
            ]
        selected_facet_path = (
            Path(facet_output_path)
            if facet_output_path is not None
            else output_path.with_name(f"{output_path.stem}_facets.xdmf")
        )
        meshio.write(
            selected_facet_path,
            meshio.Mesh(
                points=points,
                cells=[(facet_type, facet_cells)],
                cell_data=facet_data,
                point_data=dict(source_mesh.point_data),
                field_data=dict(source_mesh.field_data),
            ),
        )

    ignored_types = tuple(
        block.cell_type
        for block in inspection.cell_blocks
        if block.cell_type not in {selected_type, facet_type}
    )
    warnings = list(conversion_warnings)
    if ignored_types:
        warnings.append(
            "Only the selected topological cell type was written; omitted "
            f"blocks: {ignored_types!r}. Lower-dimensional boundary blocks "
            "must be converted separately before they can become facet tags."
        )
    if overlaps:
        warnings.append(
            "Overlapping source cell sets cannot be represented by one DOLFINx "
            "MeshTags value per cell. The first alphabetic set owns the XDMF "
            f"tag for overlapping cells; full memberships remain in the manifest: {overlaps!r}."
        )
    if boundary_overlaps:
        warnings.append(
            "Overlapping boundary sets cannot be represented by one DOLFINx "
            "MeshTags value per facet. The first alphabetic set owns the XDMF "
            "tag; full memberships remain in the manifest: "
            f"{boundary_overlaps!r}."
        )
    selected_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else output_path.with_suffix(".mesh.json")
    )
    source_summary = inspection.as_dict()
    source_summary["fingerprint"] = _source_fingerprint(input_path)
    source_summary["conversion_options"] = {
        "cell_type": selected_type,
        "facet_type": facet_type,
        "prune_z": bool(prune_z),
        "input_format": input_format,
    }
    if source_metadata:
        source_summary["format_metadata"] = source_metadata
    manifest = {
        "schema": "agentfem.mesh-conversion",
        "schema_version": "0.1.0",
        "source": source_summary,
        "output": {
            "mesh_path": str(output_path),
            "selected_cell_type": selected_type,
            "cell_count": int(len(cells)),
            "region_data_name": "agentfem_region" if region_tags else None,
            "region_tags": region_tags,
            "cell_set_members": {
                name: values.tolist() for name, values in set_members.items()
            },
            "facet_path": (
                None if selected_facet_path is None else str(selected_facet_path)
            ),
            "selected_facet_type": facet_type,
            "boundary_data_name": (
                "agentfem_boundary" if boundary_tags else None
            ),
            "boundary_tags": boundary_tags,
            "boundary_set_members": {
                name: values.tolist()
                for name, values in boundary_members.items()
            },
        },
        "warnings": warnings,
    }
    selected_manifest.parent.mkdir(parents=True, exist_ok=True)
    selected_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return MeshConversionResult(
        output_path,
        selected_type,
        input_path,
        selected_manifest,
        region_tags,
        selected_facet_path,
        boundary_tags,
        tuple(warnings),
        dict(source_metadata or {}),
    )


def convert_abaqus_inp_to_xdmf(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    cell_type: str | None = None,
    prune_z: bool = False,
    facet_type: str | None = None,
    facet_output_path: str | Path | None = None,
) -> MeshConversionResult:
    """Convert an Abaqus ``.inp`` mesh to XDMF through meshio."""

    from . import abaqus

    definitions = abaqus.read_element_definitions(input_path)
    semantics = abaqus.read_model_semantics(input_path)
    abaqus_metadata = {
        "abaqus_element_definitions": [item.summary() for item in definitions],
        "abaqus_model_semantics": semantics.summary(),
    }
    warnings = ()
    if any(item.is_hybrid for item in definitions):
        warnings = (
            "Abaqus hybrid element topology was converted, but XDMF does not "
            "encode independent pressure variables. Select a verified hybrid/mixed "
            "solution formulation; tetra10 geometry alone is not C3D10H equivalence.",
        )

    return convert_to_xdmf(
        input_path,
        output_path,
        cell_type=cell_type,
        prune_z=prune_z,
        facet_type=facet_type,
        facet_output_path=facet_output_path,
        input_format="abaqus",
        source_metadata=abaqus_metadata,
        conversion_warnings=warnings,
    )


def reusable_conversion(
    source_path: str | Path,
    output_path: str | Path,
    *,
    cell_type: str | None,
) -> MeshConversionResult | None:
    """Reuse XDMF only when source hash, options, and heavy data all match."""

    source_path, output_path = Path(source_path), Path(output_path)
    manifest_path = output_path.with_suffix(".mesh.json")
    if not manifest_path.exists() or not output_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source, output = manifest["source"], manifest["output"]
        options = source["conversion_options"]
        expected_type = cell_type or output["selected_cell_type"]
        if source["fingerprint"] != _source_fingerprint(source_path):
            return None
        if options != {
            "cell_type": expected_type,
            "facet_type": None,
            "prune_z": False,
            "input_format": "abaqus",
        }:
            return None
        if not output_path.with_suffix(".h5").exists():
            return None
        metadata = dict(source.get("format_metadata", {}))
        return MeshConversionResult(
            mesh_path=output_path,
            cell_type=str(output["selected_cell_type"]),
            source_path=source_path,
            manifest_path=manifest_path,
            region_tags={str(k): int(v) for k, v in output.get("region_tags", {}).items()},
            warnings=tuple(manifest.get("warnings", ())),
            source_metadata=metadata,
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def convert_nastran_to_xdmf(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    cell_type: str | None = None,
    prune_z: bool = False,
    facet_type: str | None = None,
    facet_output_path: str | Path | None = None,
) -> MeshConversionResult:
    """Convert a NASTRAN ``.bdf`` or ``.nas`` mesh to XDMF through meshio."""

    return convert_to_xdmf(
        input_path,
        output_path,
        cell_type=cell_type,
        prune_z=prune_z,
        facet_type=facet_type,
        facet_output_path=facet_output_path,
    )


def convert_comsol_export_to_xdmf(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    cell_type: str | None = None,
    prune_z: bool = False,
    facet_type: str | None = None,
    facet_output_path: str | Path | None = None,
) -> MeshConversionResult:
    """Convert a COMSOL-exported mesh when the export format is meshio-readable."""

    return convert_to_xdmf(
        input_path,
        output_path,
        cell_type=cell_type,
        prune_z=prune_z,
        facet_type=facet_type,
        facet_output_path=facet_output_path,
    )


def _first_supported_cell_type(cells: list[Any]) -> str:
    if not cells:
        raise ValueError("The source mesh does not contain any cell blocks.")
    preferred = (
        "hexahedron",
        "hexahedron20",
        "hexahedron27",
        "tetra",
        "tetra10",
        "wedge",
        "pyramid",
        "quad",
        "quad8",
        "quad9",
        "triangle",
        "triangle6",
        "line",
        "line3",
        "vertex",
    )
    available = [block.type for block in cells]
    for cell_type in preferred:
        if cell_type in available:
            return cell_type
    return available[0]


def _source_fingerprint(path: str | Path) -> dict[str, object]:
    """Return the immutable identity used to invalidate mesh conversions."""

    selected = Path(path)
    if not selected.exists():
        return {"algorithm": "sha256", "digest": None, "size": None}
    digest = sha256()
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "size": int(selected.stat().st_size),
    }


def _cells_of_type(cells: list[Any], cell_type: str) -> np.ndarray:
    selected = [np.asarray(block.data) for block in cells if block.type == cell_type]
    if not selected:
        available = ", ".join(block.type for block in cells)
        raise ValueError(
            f"Cell type `{cell_type}` is not present in the source mesh. "
            f"Available cell types: {available}"
        )
    return np.vstack(selected)


def _cell_data_for_type(
    cell_data: dict[str, list[np.ndarray]],
    cells: list[Any],
    cell_type: str,
) -> dict[str, list[np.ndarray]]:
    if not cell_data:
        return {}
    block_indices = [index for index, block in enumerate(cells) if block.type == cell_type]
    selected: dict[str, list[np.ndarray]] = {}
    for name, values_by_block in cell_data.items():
        arrays = [
            np.asarray(values_by_block[index])
            for index in block_indices
            if index < len(values_by_block)
        ]
        if arrays:
            selected[name] = [np.concatenate(arrays)]
    return selected


def _cell_set_tags(source_mesh, cell_type: str, count: int):
    offsets: dict[int, int] = {}
    current = 0
    for index, block in enumerate(source_mesh.cells):
        if block.type == cell_type:
            offsets[index] = current
            current += len(block.data)
    members: dict[str, np.ndarray] = {}
    for name in sorted(getattr(source_mesh, "cell_sets", {})):
        values_by_block = source_mesh.cell_sets[name]
        selected = []
        for index, offset in offsets.items():
            if index >= len(values_by_block) or values_by_block[index] is None:
                continue
            local = np.asarray(values_by_block[index], dtype=np.int64).reshape(-1)
            if np.any(local < 0) or np.any(local >= len(source_mesh.cells[index].data)):
                raise ValueError(
                    f"Cell set {name!r} contains invalid indices for block {index}."
                )
            selected.append(local + offset)
        if selected:
            members[str(name)] = np.unique(np.concatenate(selected))
    tags = {name: index + 1 for index, name in enumerate(members)}
    ownership = np.zeros(count, dtype=np.int32)
    overlaps: dict[int, tuple[str, ...]] = {}
    memberships_by_cell: dict[int, list[str]] = {}
    for name, indices in members.items():
        tag = tags[name]
        for cell in indices:
            memberships_by_cell.setdefault(int(cell), []).append(name)
            if ownership[cell] == 0:
                ownership[cell] = tag
    for cell, names in memberships_by_cell.items():
        if len(names) > 1:
            overlaps[cell] = tuple(names)
    return tags, members, overlaps


def _region_tag_array(
    members: dict[str, np.ndarray],
    tags: dict[str, int],
    count: int,
) -> np.ndarray:
    values = np.zeros(count, dtype=np.int32)
    for name in sorted(members):
        indices = members[name]
        unassigned = values[indices] == 0
        values[indices[unassigned]] = tags[name]
    return values
