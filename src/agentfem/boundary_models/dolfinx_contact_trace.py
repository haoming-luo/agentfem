# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reviewed DOLFINx adapter for the backend-neutral contact trace contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_trace import ContactTrace, ContactTraceEvaluation
from .dolfinx_adapter import (
    _collective_error,
    dolfinx_boundary_region_triangle_partition,
)
from .rigid import _readonly_array


_TRIANGLE_DEGREE_TWO = np.asarray(
    (
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    ),
    dtype=float,
)

_QUADRILATERAL_DEGREE_THREE = np.asarray(
    tuple(
        (0.5 + xi / (2.0 * np.sqrt(3.0)), 0.5 + eta / (2.0 * np.sqrt(3.0)))
        for eta in (-1.0, 1.0)
        for xi in (-1.0, 1.0)
    ),
    dtype=float,
)


def _quadrilateral_shape_values(r: float, s: float) -> np.ndarray:
    return np.asarray(
        ((1.0 - r) * (1.0 - s), r * (1.0 - s), (1.0 - r) * s, r * s),
        dtype=float,
    )


def _quadrilateral_shape_derivatives(r: float, s: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((-(1.0 - s), 1.0 - s, -s, s), dtype=float),
        np.asarray((-(1.0 - r), -r, 1.0 - r, r), dtype=float),
    )


def _quadrilateral_boundary_identity(region):
    """Return owned Q1 face geometry and partition-independent facet IDs."""

    domain = region.domain
    comm = domain.comm
    local_error = None
    owned = None
    facet_geometry = None
    records = None
    try:
        from dolfinx import mesh as mesh_api

        facet_dimension = int(domain.topology.dim) - 1
        domain.topology.create_entities(facet_dimension)
        domain.topology.create_entity_permutations()
        owned_exterior = np.asarray(
            mesh_api.exterior_facet_indices(domain.topology), dtype=np.int32
        )
        tagged = np.asarray(region.facet_tags.find(int(region.tag)), dtype=np.int32)
        facet_map = domain.topology.index_map(facet_dimension)
        owned = tagged[tagged < int(facet_map.size_local)]
        exterior = set(int(value) for value in owned_exterior)
        invalid = tuple(int(value) for value in owned if int(value) not in exterior)
        if invalid:
            raise ValueError(
                "contact facets must be owned exterior facets; first invalid "
                f"local index is {invalid[0]}"
            )
        facet_geometry = mesh_api.entities_to_geometry(
            domain,
            facet_dimension,
            owned,
            permute=True,
        )
        if facet_geometry.shape != (owned.size, 4):
            raise NotImplementedError(
                "hexahedral contact facets must map to four geometry vertices"
            )
        global_vertices = np.asarray(
            domain.geometry.input_global_indices,
            dtype=np.int64,
        )
        coordinates = np.asarray(domain.geometry.x, dtype=float)[:, :3]
        records = tuple(
            (
                tuple(sorted(int(value) for value in global_vertices[dofs])),
                tuple(
                    np.asarray(coordinates[dofs], dtype="<f8")[
                        np.argsort(global_vertices[dofs], kind="stable")
                    ].reshape(-1)
                ),
            )
            for dofs in facet_geometry
        )
        if any(len(set(key)) != 4 for key, _ in records):
            raise ValueError("a quadrilateral contact facet repeats a global vertex")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _collective_error(comm, local_error, context="extract owned exterior quadrilaterals")

    gathered = tuple(comm.allgather(records))
    canonical = tuple(sorted(record for group in gathered for record in group))
    keys = tuple(record[0] for record in canonical)
    if not keys:
        raise ValueError("DOLFINx boundary adapter selected no exterior facets.")
    if len(set(keys)) != len(keys):
        raise ValueError(
            "DOLFINx exterior facet ownership is not globally unique by vertex ID."
        )
    coordinates = np.asarray(
        [value for _, values in canonical for value in values], dtype=float
    ).reshape((-1, 3))
    scale = float(
        np.linalg.norm(np.max(coordinates, axis=0) - np.min(coordinates, axis=0))
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("DOLFINx exterior quadrilateral surface has zero scale.")
    facet_id_by_key = {key: index for index, key in enumerate(keys)}
    local_ids = np.asarray(
        [facet_id_by_key[key] for key, _ in records], dtype=np.int64
    )
    return owned, facet_geometry, local_ids, len(keys), scale


def _matching_dof_block(
    point: np.ndarray,
    dof_coordinates: np.ndarray,
    *,
    tolerance: float,
) -> int:
    distances = np.linalg.norm(dof_coordinates - point, axis=1)
    matches = np.flatnonzero(distances <= tolerance)
    if matches.size != 1:
        raise ValueError(
            "each contact vertex must match exactly one CG1 displacement block; "
            f"found {matches.size} matches"
        )
    return int(matches[0])


@dataclass(frozen=True, eq=False)
class DolfinxContactTraceAdapter:
    """Bind one reviewed DOLFINx CG1 boundary trace to current displacement."""

    trace: ContactTrace
    function_space: object
    reference_nodal_positions: object
    boundary_name: str
    facet_topology: str = "triangle"
    quadrature_rule: str = "triangle_degree_two_three_point"

    def __post_init__(self) -> None:
        if not isinstance(self.trace, ContactTrace):
            raise TypeError("DOLFINx contact adapter requires ContactTrace.")
        reference = np.asarray(self.reference_nodal_positions, dtype=float)
        if (
            reference.ndim != 2
            or reference.shape[1] != 3
            or not np.all(np.isfinite(reference))
        ):
            raise ValueError(
                "DOLFINx contact reference positions must have shape (nodes, 3)."
            )
        expected = int(
            self.function_space.dofmap.index_map.size_local
            + self.function_space.dofmap.index_map.num_ghosts
        )
        if reference.shape[0] != expected:
            raise ValueError(
                "DOLFINx contact reference positions differ from the dof map."
            )
        if not str(self.boundary_name).strip():
            raise ValueError("DOLFINx contact adapter requires a boundary name.")
        if self.facet_topology not in {"triangle", "quadrilateral"}:
            raise ValueError("DOLFINx contact adapter has unsupported facet topology.")
        object.__setattr__(
            self,
            "reference_nodal_positions",
            _readonly_array(reference),
        )
        object.__setattr__(self, "boundary_name", str(self.boundary_name))

    @property
    def communicator(self):
        return self.function_space.mesh.comm

    def current_nodal_positions(self, displacement) -> np.ndarray:
        """Return local-plus-ghost current positions after a forward scatter."""

        if displacement.function_space is not self.function_space:
            raise ValueError(
                "Contact displacement must use the adapter function-space instance."
            )
        displacement.x.scatter_forward()
        values = np.asarray(displacement.x.array, dtype=float)
        expected = self.reference_nodal_positions.shape[0] * 3
        if values.size != expected:
            raise ValueError(
                "Contact displacement storage is incompatible with blocked CG1."
            )
        return self.reference_nodal_positions + values.reshape((-1, 3))

    def evaluate(self, displacement) -> ContactTraceEvaluation:
        """Evaluate current slave quadrature positions for contact search."""

        return self.trace.evaluate(self.current_nodal_positions(displacement))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dolfinx_contact_trace_adapter",
            "boundary": self.boundary_name,
            "function_space": "continuous_blocked_vector_cg1",
            "spatial_dimension": 3,
            "facet_topology": self.facet_topology,
            "quadrature_rule": self.quadrature_rule,
            "measure_configuration": self.trace.measure_configuration,
            "point_count_local": self.trace.point_count,
            "rank": int(self.communicator.rank),
            "rank_count": int(self.communicator.size),
            "linearization": "not_provided",
        }


def dolfinx_boundary_region_contact_trace(
    region,
    function_space,
) -> DolfinxContactTraceAdapter:
    """Adapt a tagged tetrahedral or hexahedral CG1 boundary trace.

    The first production hand-off is intentionally narrow and fail-closed. It
    It supports three-dimensional first-order tetrahedral or hexahedral meshes,
    owned exterior facets, and a continuous blocked vector CG1 displacement
    space. Positive reference-area quadrature is emitted per facet.
    """

    if region is None or not hasattr(region, "domain"):
        raise TypeError("Contact trace adaptation requires an AgentFEM BoundaryRegion.")
    domain = region.domain
    comm = domain.comm
    cell_type = str(domain.topology.cell_type)
    if cell_type == "CellType.tetrahedron":
        facet_topology = "triangle"
        partition = dolfinx_boundary_region_triangle_partition(region)
        quadrilateral_identity = None
    elif cell_type == "CellType.hexahedron":
        facet_topology = "quadrilateral"
        partition = None
        quadrilateral_identity = _quadrilateral_boundary_identity(region)
    else:
        raise NotImplementedError(
            "contact traces support first-order tetrahedral or hexahedral meshes"
        )
    local_error = None
    adapter = None
    try:
        from dolfinx import mesh as mesh_api

        if function_space.mesh is not domain:
            raise ValueError("contact function space must belong to the boundary mesh")
        if tuple(function_space.element.value_shape) != (3,):
            raise NotImplementedError(
                "only three-component displacement fields are supported"
            )
        if int(function_space.dofmap.index_map_bs) != 3:
            raise NotImplementedError(
                "only blocked three-component displacement spaces are supported"
            )
        element = function_space.element.basix_element
        if (
            int(element.degree) != 1
            or bool(element.discontinuous)
            or str(element.family) != "ElementFamily.P"
            or str(element.map_type) != "MapType.identity"
        ):
            raise NotImplementedError(
                "only continuous identity-mapped CG1 displacement is supported"
            )

        facet_dimension = int(domain.topology.dim) - 1
        if facet_topology == "triangle":
            tagged = np.asarray(region.facet_tags.find(int(region.tag)), dtype=np.int32)
            facet_map = domain.topology.index_map(facet_dimension)
            owned = tagged[tagged < int(facet_map.size_local)]
            facet_geometry = mesh_api.entities_to_geometry(
                domain,
                facet_dimension,
                owned,
                permute=True,
            )
            if facet_geometry.shape != (owned.size, 3):
                raise NotImplementedError(
                    "tetrahedral contact facets must map to three geometry vertices"
                )
            local_surface = partition.local_surface
            local_facet_ids = (
                np.empty((0,), dtype=np.int64)
                if local_surface is None
                else np.asarray(local_surface.facet_ids, dtype=np.int64)
            )
            global_facet_count = partition.global_facet_count
            global_scale = partition.global_scale
            quadrature = _TRIANGLE_DEGREE_TWO
            nodes_per_facet = 3
            points_per_facet = 3
            quadrature_rule = "triangle_degree_two_three_point"
        else:
            (
                owned,
                facet_geometry,
                local_facet_ids,
                global_facet_count,
                global_scale,
            ) = (
                quadrilateral_identity
            )
            quadrature = _QUADRILATERAL_DEGREE_THREE
            nodes_per_facet = 4
            points_per_facet = 4
            quadrature_rule = "quadrilateral_gauss_degree_three_four_point"
        if local_facet_ids.shape != (owned.size,):
            raise RuntimeError("contact facet identity differs from owned facets")
        if global_facet_count > np.iinfo(np.int64).max // points_per_facet:
            raise OverflowError("contact quadrature point identity exceeds int64")

        coordinates = np.asarray(domain.geometry.x, dtype=float)[:, :3]
        global_vertices = np.asarray(
            domain.geometry.input_global_indices,
            dtype=np.int64,
        )
        dof_coordinates = np.asarray(
            function_space.tabulate_dof_coordinates(),
            dtype=float,
        )[:, :3]
        tolerance = max(
            float(global_scale) * 1.0e-12,
            np.finfo(float).eps * float(global_scale) * 1024.0,
        )

        point_ids = []
        node_ids = []
        shape_values = []
        weights = []
        for facet_id, geometry_dofs in zip(
            local_facet_ids,
            facet_geometry,
            strict=True,
        ):
            if facet_topology == "triangle":
                order = np.argsort(global_vertices[geometry_dofs], kind="stable")
                points = coordinates[geometry_dofs[order]]
            else:
                points = coordinates[geometry_dofs]
            blocks = tuple(
                _matching_dof_block(
                    point,
                    dof_coordinates,
                    tolerance=tolerance,
                )
                for point in points
            )
            if facet_topology == "triangle":
                area = 0.5 * float(
                    np.linalg.norm(
                        np.cross(points[1] - points[0], points[2] - points[0])
                    )
                )
                if not np.isfinite(area) or area <= 0.0:
                    raise ValueError("contact trace contains a degenerate triangle")
                facet_shape_values = quadrature
                facet_weights = np.full((3,), area / 3.0, dtype=float)
            else:
                facet_shape_values = []
                facet_weights = []
                for r, s in quadrature:
                    values = _quadrilateral_shape_values(float(r), float(s))
                    derivative_r, derivative_s = _quadrilateral_shape_derivatives(
                        float(r), float(s)
                    )
                    tangent_r = derivative_r @ points
                    tangent_s = derivative_s @ points
                    jacobian = float(np.linalg.norm(np.cross(tangent_r, tangent_s)))
                    if not np.isfinite(jacobian) or jacobian <= 0.0:
                        raise ValueError(
                            "contact trace contains a degenerate quadrilateral"
                        )
                    facet_shape_values.append(values)
                    facet_weights.append(0.25 * jacobian)
            for quadrature_index, (values, weight) in enumerate(
                zip(facet_shape_values, facet_weights, strict=True)
            ):
                point_ids.append(
                    int(facet_id) * points_per_facet + quadrature_index
                )
                node_ids.append(blocks)
                shape_values.append(tuple(float(value) for value in values))
                weights.append(float(weight))

        trace = ContactTrace(
            point_ids=np.asarray(point_ids, dtype=np.int64),
            node_ids=np.asarray(node_ids, dtype=np.int64).reshape(
                (-1, nodes_per_facet)
            ),
            shape_values=np.asarray(shape_values, dtype=float).reshape(
                (-1, nodes_per_facet)
            ),
            weights=np.asarray(weights, dtype=float),
            measure_configuration="reference",
            source=(
                "dolfinx_owned_exterior_"
                f"{facet_topology}s_continuous_blocked_vector_cg1"
            ),
        )
        adapter = DolfinxContactTraceAdapter(
            trace=trace,
            function_space=function_space,
            reference_nodal_positions=dof_coordinates,
            boundary_name=str(region.name),
            facet_topology=facet_topology,
            quadrature_rule=quadrature_rule,
        )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _collective_error(comm, local_error, context="build the contact trace")
    if adapter is None:  # pragma: no cover - collective error guards this path
        raise RuntimeError("DOLFINx contact trace collective validation failed.")
    return adapter


__all__ = [
    "DolfinxContactTraceAdapter",
    "dolfinx_boundary_region_contact_trace",
]
