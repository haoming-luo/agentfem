"""External CAE mesh-format conversion helpers.

These helpers use ``meshio`` as an optional dependency. AgentFEM core mesh
operations do not require meshio, but Abaqus, NASTRAN, COMSOL-exported meshes,
and many neutral formats are best handled through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_EXTERNAL_FORMATS = {
    ".inp": "Abaqus input",
    ".bdf": "NASTRAN bulk data",
    ".nas": "NASTRAN bulk data",
    ".msh": "Gmsh mesh",
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


def require_meshio():
    """Import meshio or raise an actionable optional-dependency error."""

    try:
        import meshio
    except ImportError as exc:
        raise ImportError(
            "External CAE mesh conversion requires the optional dependency "
            "`meshio`. Install it with `pip install meshio` or conda-forge."
        ) from exc
    return meshio


def describe_supported_external_formats() -> dict[str, str]:
    """Return common external mesh formats handled through meshio."""

    return dict(SUPPORTED_EXTERNAL_FORMATS)


def detect_mesh_format(path: str | Path) -> str:
    """Return a human-readable mesh format label from a file extension."""

    suffix = Path(path).suffix.lower()
    return SUPPORTED_EXTERNAL_FORMATS.get(suffix, "meshio-supported mesh format")


def convert_to_xdmf(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    cell_type: str | None = None,
    prune_z: bool = False,
) -> MeshConversionResult:
    """Convert an external mesh file to single-cell-type XDMF.

    DOLFINx XDMF mesh import expects a single topological cell type. If the
    source contains multiple cell blocks, pass ``cell_type`` explicitly, such
    as ``"triangle"``, ``"quad"``, ``"tetra"``, or ``"hexahedron"``.
    """

    meshio = require_meshio()
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else input_path.with_suffix(".xdmf")

    source_mesh = meshio.read(input_path)
    selected_type = cell_type or _first_supported_cell_type(source_mesh.cells)
    cells = _cells_of_type(source_mesh.cells, selected_type)
    points = np.asarray(source_mesh.points)
    if prune_z:
        points = points[:, :2]

    cell_data = _cell_data_for_type(source_mesh.cell_data, source_mesh.cells, selected_type)
    converted = meshio.Mesh(
        points=points,
        cells=[(selected_type, cells)],
        cell_data=cell_data,
        point_data=dict(source_mesh.point_data),
        field_data=dict(source_mesh.field_data),
    )
    meshio.write(output_path, converted)
    return MeshConversionResult(output_path, selected_type, input_path)


def convert_abaqus_inp_to_xdmf(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    cell_type: str | None = None,
    prune_z: bool = False,
) -> MeshConversionResult:
    """Convert an Abaqus ``.inp`` mesh to XDMF through meshio."""

    return convert_to_xdmf(
        input_path,
        output_path,
        cell_type=cell_type,
        prune_z=prune_z,
    )


def convert_nastran_to_xdmf(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    cell_type: str | None = None,
    prune_z: bool = False,
) -> MeshConversionResult:
    """Convert a NASTRAN ``.bdf`` or ``.nas`` mesh to XDMF through meshio."""

    return convert_to_xdmf(
        input_path,
        output_path,
        cell_type=cell_type,
        prune_z=prune_z,
    )


def convert_comsol_export_to_xdmf(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    cell_type: str | None = None,
    prune_z: bool = False,
) -> MeshConversionResult:
    """Convert a COMSOL-exported mesh when the export format is meshio-readable."""

    return convert_to_xdmf(
        input_path,
        output_path,
        cell_type=cell_type,
        prune_z=prune_z,
    )


def _first_supported_cell_type(cells: list[Any]) -> str:
    if not cells:
        raise ValueError("The source mesh does not contain any cell blocks.")
    preferred = (
        "hexahedron",
        "tetra",
        "quad",
        "triangle",
        "line",
        "vertex",
    )
    available = [block.type for block in cells]
    for cell_type in preferred:
        if cell_type in available:
            return cell_type
    return available[0]


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
