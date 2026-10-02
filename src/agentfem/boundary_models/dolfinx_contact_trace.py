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


def _quadrilateral_shape_derivatives(
    r: float, s: float
) -> tuple[np.ndarray, np.ndarray]:
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
    _collective_error(
        comm, local_error, context="extract owned exterior quadrilaterals"
    )

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
    local_ids = np.asarray([facet_id_by_key[key] for key, _ in records], dtype=np.int64)
    return owned, facet_geometry, local_ids, len(keys), scale


def _higher_order_boundary_identity(region, *, corner_count: int):
    """Return owned high-order face closures and vertex-based stable IDs."""

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
        domain.topology.create_connectivity(facet_dimension, domain.topology.dim)
        domain.topology.create_connectivity(domain.topology.dim, facet_dimension)
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
        if facet_geometry.ndim != 2 or facet_geometry.shape[1] < corner_count:
            raise NotImplementedError(
                "contact facets do not expose the required geometry vertices"
            )
        global_vertices = np.asarray(
            domain.geometry.input_global_indices,
            dtype=np.int64,
        )
        coordinates = np.asarray(domain.geometry.x, dtype=float)[:, :3]
        corner_geometry = facet_geometry[:, :corner_count]
        records = tuple(
            (
                tuple(sorted(int(value) for value in global_vertices[dofs])),
                tuple(
                    np.asarray(coordinates[dofs], dtype="<f8")[
                        np.argsort(global_vertices[dofs], kind="stable")
                    ].reshape(-1)
                ),
            )
            for dofs in corner_geometry
        )
        if any(len(set(key)) != corner_count for key, _ in records):
            raise ValueError("a high-order contact facet repeats a global vertex")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _collective_error(comm, local_error, context="extract high-order exterior facets")

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
        raise ValueError("DOLFINx exterior high-order surface has zero scale.")
    facet_id_by_key = {key: index for index, key in enumerate(keys)}
    local_ids = np.asarray([facet_id_by_key[key] for key, _ in records], dtype=np.int64)
    return owned, facet_geometry, local_ids, len(keys), scale


def _reference_facet_quadrature(cell_type, local_facet: int, *, degree: int):
    """Map positive facet quadrature into one Basix reference cell."""

    import basix

    facet_vertices = basix.cell.topology(cell_type)[2][int(local_facet)]
    reference_vertices = np.asarray(basix.cell.geometry(cell_type), dtype=float)[
        facet_vertices
    ]
    if len(facet_vertices) == 3:
        points, weights = basix.make_quadrature(
            basix.CellType.triangle,
            int(degree),
        )
        barycentric = np.column_stack((1.0 - points[:, 0] - points[:, 1], points))
        cell_points = barycentric @ reference_vertices
        tangent_r = reference_vertices[1] - reference_vertices[0]
        tangent_s = reference_vertices[2] - reference_vertices[0]
        reference_tangents = np.broadcast_to(
            np.stack((tangent_r, tangent_s), axis=1),
            (points.shape[0], reference_vertices.shape[1], 2),
        )
        topology = "triangle"
    elif len(facet_vertices) == 4:
        points, weights = basix.make_quadrature(
            basix.CellType.quadrilateral,
            int(degree),
        )
        values = np.asarray(
            [_quadrilateral_shape_values(float(r), float(s)) for r, s in points]
        )
        cell_points = values @ reference_vertices
        reference_tangents = np.empty(
            (points.shape[0], reference_vertices.shape[1], 2), dtype=float
        )
        for index, (r, s) in enumerate(points):
            derivative_r, derivative_s = _quadrilateral_shape_derivatives(
                float(r), float(s)
            )
            reference_tangents[index, :, 0] = derivative_r @ reference_vertices
            reference_tangents[index, :, 1] = derivative_s @ reference_vertices
        topology = "quadrilateral"
    else:  # pragma: no cover - guarded by supported volume topologies
        raise NotImplementedError("unsupported high-order contact facet topology")
    return topology, cell_points, np.asarray(weights, dtype=float), reference_tangents


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
    """Bind one reviewed DOLFINx CG1/CG2 boundary trace to displacement."""

    trace: ContactTrace
    function_space: object
    reference_nodal_positions: object
    boundary_name: str
    facet_topology: str = "triangle"
    quadrature_rule: str = "triangle_degree_two_three_point"
    interpolation_degree: int = 1
    geometry_degree: int = 1

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
        if int(self.interpolation_degree) not in {1, 2}:
            raise ValueError("DOLFINx contact interpolation degree must be one or two.")
        if int(self.geometry_degree) not in {1, 2}:
            raise ValueError("DOLFINx contact geometry degree must be one or two.")
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
                "Contact displacement storage is incompatible with blocked CG1/CG2."
            )
        return self.reference_nodal_positions + values.reshape((-1, 3))

    def evaluate(self, displacement) -> ContactTraceEvaluation:
        """Evaluate current slave quadrature positions for contact search."""

        return self.trace.evaluate(self.current_nodal_positions(displacement))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dolfinx_contact_trace_adapter",
            "boundary": self.boundary_name,
            "function_space": (
                f"continuous_blocked_vector_cg{int(self.interpolation_degree)}"
            ),
            "spatial_dimension": 3,
            "facet_topology": self.facet_topology,
            "quadrature_rule": self.quadrature_rule,
            "interpolation_degree": int(self.interpolation_degree),
            "geometry_degree": int(self.geometry_degree),
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
    """Adapt a tagged tetrahedral or hexahedral CG1/CG2 boundary trace.

    The production hand-off remains intentionally bounded and fail-closed. It
    supports first- or second-order coordinate geometry, owned exterior
    facets, and a continuous blocked vector CG1 or CG2 displacement space.
    Positive reference-area quadrature is emitted per facet.
    """

    if region is None or not hasattr(region, "domain"):
        raise TypeError("Contact trace adaptation requires an AgentFEM BoundaryRegion.")
    domain = region.domain
    comm = domain.comm
    cell_type = str(domain.topology.cell_type)
    element = function_space.element.basix_element
    interpolation_degree = int(element.degree)
    geometry_degree = int(domain.geometry.cmaps[0].degree)
    high_order = interpolation_degree == 2 or geometry_degree == 2
    if cell_type == "CellType.tetrahedron":
        facet_topology = "triangle"
        partition = (
            None if high_order else dolfinx_boundary_region_triangle_partition(region)
        )
        quadrilateral_identity = None
        high_order_identity = (
            _higher_order_boundary_identity(region, corner_count=3)
            if high_order
            else None
        )
    elif cell_type == "CellType.hexahedron":
        facet_topology = "quadrilateral"
        partition = None
        quadrilateral_identity = (
            None if high_order else _quadrilateral_boundary_identity(region)
        )
        high_order_identity = (
            _higher_order_boundary_identity(region, corner_count=4)
            if high_order
            else None
        )
    else:
        raise NotImplementedError(
            "contact traces support tetrahedral or hexahedral meshes"
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
        if (
            interpolation_degree not in {1, 2}
            or bool(element.discontinuous)
            or str(element.family) != "ElementFamily.P"
            or str(element.map_type) != "MapType.identity"
        ):
            raise NotImplementedError(
                "only continuous identity-mapped CG1/CG2 displacement is supported"
            )
        if geometry_degree not in {1, 2}:
            raise NotImplementedError(
                "only first- or second-order coordinate geometry is supported"
            )
        if high_order:
            import basix

            expected_geometry_element = basix.create_element(
                basix.ElementFamily.P,
                domain.basix_cell(),
                geometry_degree,
                lagrange_variant=basix.LagrangeVariant(
                    domain.geometry.cmaps[0].variant
                ),
            )
            if int(domain.geometry.cmaps[0].dim) != int(expected_geometry_element.dim):
                raise NotImplementedError(
                    "high-order contact requires complete Lagrange P1/P2 "
                    "coordinate geometry; serendipity geometry is not yet supported"
                )

        facet_dimension = int(domain.topology.dim) - 1
        if facet_topology == "triangle" and not high_order:
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
        elif facet_topology == "quadrilateral" and not high_order:
            (
                owned,
                facet_geometry,
                local_facet_ids,
                global_facet_count,
                global_scale,
            ) = quadrilateral_identity
            quadrature = _QUADRILATERAL_DEGREE_THREE
            nodes_per_facet = 4
            points_per_facet = 4
            quadrature_rule = "quadrilateral_gauss_degree_three_four_point"
        else:
            (
                owned,
                facet_geometry,
                local_facet_ids,
                global_facet_count,
                global_scale,
            ) = high_order_identity
            import basix

            facet_cell_type = (
                basix.CellType.triangle
                if facet_topology == "triangle"
                else basix.CellType.quadrilateral
            )
            quadrature_degree = 2 * max(interpolation_degree, geometry_degree)
            reference_points, _ = basix.make_quadrature(
                facet_cell_type,
                quadrature_degree,
            )
            nodes_per_facet = int(element.dim)
            points_per_facet = int(reference_points.shape[0])
            quadrature_rule = (
                f"{facet_topology}_basix_positive_degree_{quadrature_degree}_"
                f"{points_per_facet}_point"
            )
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
        if high_order:
            import basix

            cell_basix_type = domain.basix_cell()
            geometry_element = expected_geometry_element
            facet_to_cell = domain.topology.connectivity(
                facet_dimension,
                domain.topology.dim,
            )
            cell_to_facet = domain.topology.connectivity(
                domain.topology.dim,
                facet_dimension,
            )
            geometry_dofmap = domain.geometry.dofmaps[0]

        for facet, facet_id, geometry_dofs in zip(
            owned,
            local_facet_ids,
            facet_geometry,
            strict=True,
        ):
            if high_order:
                adjacent = np.asarray(facet_to_cell.links(int(facet)), dtype=np.int32)
                if adjacent.size != 1:
                    raise ValueError(
                        "an exterior contact facet must have exactly one adjacent cell"
                    )
                cell = int(adjacent[0])
                local_matches = np.flatnonzero(
                    np.asarray(cell_to_facet.links(cell), dtype=np.int32) == int(facet)
                )
                if local_matches.size != 1:
                    raise ValueError(
                        "contact facet does not have one local cell-facet identity"
                    )
                (
                    evaluated_topology,
                    cell_points,
                    reference_weights,
                    reference_tangents,
                ) = _reference_facet_quadrature(
                    cell_basix_type,
                    int(local_matches[0]),
                    degree=quadrature_degree,
                )
                if evaluated_topology != facet_topology:
                    raise RuntimeError("contact facet topology changed during lowering")
                basis = element.tabulate(0, cell_points)[0, :, :, 0]
                if basis.shape != (points_per_facet, nodes_per_facet):
                    raise RuntimeError("contact displacement basis has wrong shape")
                geometry_basis = geometry_element.tabulate(1, cell_points)
                cell_geometry = coordinates[geometry_dofmap[cell]]
                jacobian = np.empty((points_per_facet, 3, 3), dtype=float)
                for derivative in range(3):
                    jacobian[:, :, derivative] = (
                        geometry_basis[derivative + 1, :, :, 0] @ cell_geometry
                    )
                physical_tangents = np.einsum(
                    "qij,qjk->qik",
                    jacobian,
                    reference_tangents,
                )
                surface_jacobian = np.linalg.norm(
                    np.cross(
                        physical_tangents[:, :, 0],
                        physical_tangents[:, :, 1],
                    ),
                    axis=1,
                )
                if not np.all(np.isfinite(surface_jacobian)) or np.any(
                    surface_jacobian <= 0.0
                ):
                    raise ValueError("contact trace contains a degenerate curved facet")
                blocks = tuple(
                    int(value) for value in function_space.dofmap.cell_dofs(cell)
                )
                facet_shape_values = basis
                facet_weights = reference_weights * surface_jacobian
            elif facet_topology == "triangle":
                order = np.argsort(global_vertices[geometry_dofs], kind="stable")
                points = coordinates[geometry_dofs[order]]
            else:
                points = coordinates[geometry_dofs]
            if not high_order:
                blocks = tuple(
                    _matching_dof_block(
                        point,
                        dof_coordinates,
                        tolerance=tolerance,
                    )
                    for point in points
                )
            if facet_topology == "triangle" and not high_order:
                area = 0.5 * float(
                    np.linalg.norm(
                        np.cross(points[1] - points[0], points[2] - points[0])
                    )
                )
                if not np.isfinite(area) or area <= 0.0:
                    raise ValueError("contact trace contains a degenerate triangle")
                facet_shape_values = quadrature
                facet_weights = np.full((3,), area / 3.0, dtype=float)
            elif not high_order:
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
                point_ids.append(int(facet_id) * points_per_facet + quadrature_index)
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
                f"{facet_topology}s_continuous_blocked_vector_"
                f"cg{interpolation_degree}"
            ),
        )
        adapter = DolfinxContactTraceAdapter(
            trace=trace,
            function_space=function_space,
            reference_nodal_positions=dof_coordinates,
            boundary_name=str(region.name),
            facet_topology=facet_topology,
            quadrature_rule=quadrature_rule,
            interpolation_degree=interpolation_degree,
            geometry_degree=geometry_degree,
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
