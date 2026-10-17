# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Unsmoothed accepted rigid-contact point export for bounded serial J2 Steps."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


_ARRAY_FIELDS = (
    "point_ids", "slave_positions", "master_closest_points",
    "surface_normals", "signed_gaps", "reference_weights",
    "normal_pressures", "penetrations", "active", "entity_ids",
    "surface_friction_tractions", "accepted_elastic_tangential_slips",
    "cumulative_friction_dissipation_densities",
)
_VECTOR_FIELDS = {
    "slave_positions", "master_closest_points", "surface_normals",
    "surface_friction_tractions", "accepted_elastic_tangential_slips",
}
_INTEGER_FIELDS = {"point_ids", "entity_ids"}


def _vtk_data_array(parent, name: str, values: np.ndarray) -> None:
    """Write one exact, unsmoothed numeric point array to a VTK XML view."""

    array = np.asarray(values)
    if array.dtype.kind == "b":
        vtk_type = "UInt8"
        flat = array.astype(np.uint8, copy=False).ravel()
    elif array.dtype.kind in "iu":
        vtk_type = "Int64"
        flat = array.astype(np.int64, copy=False).ravel()
    else:
        vtk_type = "Float64"
        flat = array.astype(np.float64, copy=False).ravel()
    attributes = {
        "type": vtk_type,
        "Name": name,
        "NumberOfComponents": str(array.shape[1] if array.ndim == 2 else 1),
        "format": "ascii",
    }
    element = ET.SubElement(parent, "DataArray", attributes)
    if vtk_type == "Float64":
        element.text = " ".join(format(float(value), ".17g") for value in flat)
    else:
        element.text = " ".join(str(int(value)) for value in flat)


def _contact_points_vtp(arrays: dict[str, np.ndarray]) -> str:
    """Provide a vertex-only ParaView view of the accepted raw samples."""

    count = len(arrays["point_ids"])
    root = ET.Element(
        "VTKFile", type="PolyData", version="1.0", byte_order="LittleEndian",
    )
    polydata = ET.SubElement(root, "PolyData")
    piece = ET.SubElement(
        polydata, "Piece",
        NumberOfPoints=str(count), NumberOfVerts=str(count),
        NumberOfLines="0", NumberOfStrips="0", NumberOfPolys="0",
    )
    point_data = ET.SubElement(
        piece, "PointData", Scalars="normal_pressures",
        Vectors="surface_friction_tractions",
    )
    for name, values in arrays.items():
        if name != "slave_positions":
            _vtk_data_array(point_data, name, values)
    _vtk_data_array(
        ET.SubElement(piece, "Points"),
        "current_slave_positions", arrays["slave_positions"],
    )
    vertices = ET.SubElement(piece, "Verts")
    _vtk_data_array(vertices, "connectivity", np.arange(count, dtype=np.int64))
    _vtk_data_array(vertices, "offsets", np.arange(1, count + 1, dtype=np.int64))
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
        root, encoding="unicode",
    ) + "\n"


def export_accepted_contact_points(
    step, directory, *, units=None, result=None, vtk_view=False,
) -> dict[str, Path]:
    """Archive accepted quadrature samples without smoothing or advancing State.

    The coordinate is the Step's normalized load factor, not physical time.
    Point IDs are stable only together with the named contact pair and geometry
    fingerprint. The NPZ is a scientific-data view, **not** a restart archive.
    No contact pressure is mislabeled as a Cauchy stress or nodal field.
    ``vtk_view=True`` additionally writes vertex-only PolyData at the current
    slave positions for ParaView; it does not interpolate or smooth the samples.
    """

    from ..checkpointing import atomic_savez, atomic_write_text
    from .. import __version__
    from .core import SimulationResult

    if result is not None and not isinstance(result, SimulationResult):
        raise TypeError("AFM-CONTACT-EXPORT-RESULT: expected SimulationResult.")
    if not isinstance(vtk_view, bool):
        raise TypeError("AFM-CONTACT-EXPORT-VIEW: vtk_view must be a boolean.")
    if result is not None and (
        {"accepted_contact_points", "accepted_contact_points_metadata",
         "accepted_contact_points_vtp"}
        & set(result.artifacts)
        or {"rigid_contact_normal_pressure_raw", "rigid_contact_friction_traction_raw"}
        & set(result.fields)
    ):
        raise ValueError("AFM-CONTACT-EXPORT-RESULT: contact fields already attached.")

    provider = getattr(step, "contact_provider", None)
    if provider is None:
        raise ValueError("AFM-CONTACT-EXPORT: the Step has no rigid contact provider.")
    domain = step.solution.function_space.mesh
    if domain.comm.size != 1:
        raise NotImplementedError("AFM-CONTACT-EXPORT-MPI: point export is serial only.")
    factor = float(step.accepted_load_factor)
    if (
        abs(float(step.state_transaction.accepted_factor) - factor) > 1.0e-12
        or not np.allclose(
            step.solution.x.array, step.accepted_solution.x.array,
            rtol=0.0, atol=1.0e-12,
        )
    ):
        raise RuntimeError(
            "AFM-CONTACT-EXPORT-STATE: export requires a fully accepted displacement "
            "and material State."
        )
    providers = tuple(getattr(provider, "providers", (provider,)))
    if not providers or any(not hasattr(item, "accepted_point_samples") for item in providers):
        raise TypeError("AFM-CONTACT-EXPORT: unsupported contact provider.")
    if len({item.name for item in providers}) != len(providers):
        raise ValueError("AFM-CONTACT-EXPORT-IDENTITY: pair names must be unique.")

    samples = [item.accepted_point_samples() for item in providers]
    for provider_item, item in zip(providers, samples, strict=True):
        expected_factor = factor if provider_item.has_prescribed_motion else 0.0
        if abs(float(item["load_factor"]) - expected_factor) > 1.0e-12:
            raise RuntimeError("AFM-CONTACT-EXPORT-POSE: tool pose is not accepted.")
        count = len(item["point_ids"])
        for field in _ARRAY_FIELDS:
            array = np.asarray(item[field])
            expected_shape = (count, 3) if field in _VECTOR_FIELDS else (count,)
            expected_kind = (
                "iu" if field in _INTEGER_FIELDS else
                "b" if field == "active" else "fiu"
            )
            if (
                array.shape != expected_shape
                or array.dtype.kind not in expected_kind
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(f"AFM-CONTACT-EXPORT-ARRAY: invalid {field}.")
        if np.unique(item["point_ids"]).size != count:
            raise ValueError("AFM-CONTACT-EXPORT-IDENTITY: duplicate point ID in pair.")

    selected_units = {} if units is None else dict(units)
    encoded_units = json.loads(json.dumps(selected_units, allow_nan=False))
    metadata = {
        "schema": "agentfem.accepted-contact-points.v1",
        "agentfem_version": __version__,
        "step_name": step.name,
        "coordinate_name": "load_factor",
        "coordinate_value": factor,
        "complete": factor >= 1.0 - 1.0e-12,
        "location": "reference_boundary_quadrature",
        "pressure_measure": "penalty_force_per_reference_area_not_cauchy_pressure",
        "smoothing": "none",
        "gap_sign": "positive_admissible_negative_penetration",
        "friction_traction_sign": "force_on_rigid_surface_per_reference_area",
        "units": encoded_units,
        "restart_archive": False,
        "scientifically_validated": False,
        "vtk_view": "current_slave_positions_vertex_only" if vtk_view else None,
        "pairs": [
            {
                "pair_index": index,
                "constraint_name": item["constraint_name"],
                "surface_name": item["surface_name"],
                "geometry_fingerprint": item["geometry_fingerprint"],
                "trace_source": item["trace_source"],
                "point_count": len(item["point_ids"]),
            }
            for index, item in enumerate(samples)
        ],
    }
    encoded = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    selected = Path(directory)
    paths = {
        "contact_points": selected / "accepted_contact_points.npz",
        "contact_point_metadata": selected / "accepted_contact_points.json",
    }
    if vtk_view:
        paths["contact_point_view"] = selected / "accepted_contact_points.vtp"
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("AFM-CONTACT-EXPORT-EXISTS: use a fresh output directory.")
    arrays = {
        field: np.concatenate([np.asarray(item[field]) for item in samples], axis=0)
        for field in _ARRAY_FIELDS
    }
    arrays["pair_index"] = np.concatenate([
        np.full(len(item["point_ids"]), index, dtype=np.int32)
        for index, item in enumerate(samples)
    ])
    vtk_content = _contact_points_vtp(arrays) if vtk_view else None
    atomic_savez(paths["contact_points"], **arrays)
    atomic_write_text(paths["contact_point_metadata"], encoded)
    if vtk_content is not None:
        atomic_write_text(paths["contact_point_view"], vtk_content)
    if result is not None:
        result.add_artifact("accepted_contact_points", paths["contact_points"])
        result.add_artifact(
            "accepted_contact_points_metadata", paths["contact_point_metadata"],
        )
        if vtk_view:
            result.add_artifact(
                "accepted_contact_points_vtp", paths["contact_point_view"],
            )
        for name, key, description in (
            (
                "rigid_contact_normal_pressure_raw", "normal_pressures",
                "Positive penalty force per reference boundary area; no smoothing.",
            ),
            (
                "rigid_contact_friction_traction_raw", "surface_friction_tractions",
                "Tangential force on the rigid surface per reference area; no smoothing.",
            ),
        ):
            result.add_field(
                name, artifact=paths["contact_points"],
                location="reference_boundary_quadrature", description=description,
                processing={
                    "method": "accepted_raw_contact_point_export",
                    "npz_key": key,
                    "pair_index_key": "pair_index",
                    "point_id_key": "point_ids",
                    "smoothing": "none",
                },
            )
    return paths


__all__ = ["export_accepted_contact_points"]
