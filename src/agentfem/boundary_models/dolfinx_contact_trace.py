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
            "facet_topology": "triangle",
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
    """Adapt a tagged tetrahedral boundary and blocked vector CG1 space.

    The first production hand-off is intentionally narrow and fail-closed. It
    supports a three-dimensional first-order tetrahedral mesh, owned exterior
    triangular facets, and a continuous blocked vector CG1 displacement space.
    Three positive reference-area quadrature points are emitted per facet.
    """

    if region is None or not hasattr(region, "domain"):
        raise TypeError("Contact trace adaptation requires an AgentFEM BoundaryRegion.")
    domain = region.domain
    comm = domain.comm
    partition = dolfinx_boundary_region_triangle_partition(region)
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
                "contact facets must map to exactly three geometry vertices"
            )
        local_surface = partition.local_surface
        local_facet_ids = (
            np.empty((0,), dtype=np.int64)
            if local_surface is None
            else np.asarray(local_surface.facet_ids, dtype=np.int64)
        )
        if local_facet_ids.shape != (owned.size,):
            raise RuntimeError(
                "contact facet identity differs from the boundary partition"
            )
        if partition.global_facet_count > np.iinfo(np.int64).max // 3:
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
            float(partition.global_scale) * 1.0e-12,
            np.finfo(float).eps * float(partition.global_scale) * 1024.0,
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
            order = np.argsort(global_vertices[geometry_dofs], kind="stable")
            points = coordinates[geometry_dofs[order]]
            blocks = tuple(
                _matching_dof_block(
                    point,
                    dof_coordinates,
                    tolerance=tolerance,
                )
                for point in points
            )
            area = 0.5 * float(
                np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
            )
            if not np.isfinite(area) or area <= 0.0:
                raise ValueError("contact trace contains a degenerate triangle")
            for quadrature_index, barycentric in enumerate(_TRIANGLE_DEGREE_TWO):
                point_ids.append(int(facet_id) * 3 + quadrature_index)
                node_ids.append(blocks)
                shape_values.append(tuple(float(value) for value in barycentric))
                weights.append(area / 3.0)

        trace = ContactTrace(
            point_ids=np.asarray(point_ids, dtype=np.int64),
            node_ids=np.asarray(node_ids, dtype=np.int64).reshape((-1, 3)),
            shape_values=np.asarray(shape_values, dtype=float).reshape((-1, 3)),
            weights=np.asarray(weights, dtype=float),
            measure_configuration="reference",
            source=("dolfinx_owned_exterior_triangles_continuous_blocked_vector_cg1"),
        )
        adapter = DolfinxContactTraceAdapter(
            trace=trace,
            function_space=function_space,
            reference_nodal_positions=dof_coordinates,
            boundary_name=str(region.name),
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
