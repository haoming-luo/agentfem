# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed DOLFINx boundary adapters for reviewed rigid surfaces."""

from __future__ import annotations

import hashlib

import numpy as np

from .rigid import TriangulatedRigidSurface, _finite_nonnegative
from .search import TriangleSurfacePartition, _facet_identity_fingerprint


def _collective_error(comm, error: str | None, *, context: str) -> None:
    errors = tuple(comm.allgather(error))
    if any(value is not None for value in errors):
        details = "; ".join(
            f"rank {rank}: {value}"
            for rank, value in enumerate(errors)
            if value is not None
        )
        raise ValueError(f"DOLFINx boundary adapter failed to {context}: {details}")


def _distributed_geometry_fingerprint(
    records: tuple[tuple[tuple[int, int, int], tuple[float, ...]], ...],
    *,
    tolerance: float,
    ambiguity_tolerance: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"agentfem.dolfinx_exterior_triangle_partition.v1")
    digest.update(np.asarray((len(records),), dtype="<i8").tobytes())
    for vertex_ids, coordinates in records:
        digest.update(np.asarray(vertex_ids, dtype="<i8").tobytes())
        digest.update(np.asarray(coordinates, dtype="<f8").tobytes())
    digest.update(
        np.asarray((tolerance, ambiguity_tolerance), dtype="<f8").tobytes()
    )
    return digest.hexdigest()


def _local_surface_from_records(
    records: list[tuple[tuple[int, int, int], tuple[int, int, int], np.ndarray]],
    *,
    facet_id_by_key: dict[tuple[int, int, int], int],
    name: str,
    rank: int,
    tolerance: float,
    ambiguity_tolerance: float,
) -> TriangulatedRigidSurface | None:
    if not records:
        return None
    local_vertex_coordinates: dict[int, np.ndarray] = {}
    for _, oriented_ids, points in records:
        for vertex_id, point in zip(oriented_ids, points, strict=True):
            existing = local_vertex_coordinates.get(vertex_id)
            if existing is not None and not np.array_equal(existing, point):
                raise ValueError(
                    f"global input vertex {vertex_id} has inconsistent coordinates"
                )
            local_vertex_coordinates[vertex_id] = point
    ordered_vertex_ids = tuple(sorted(local_vertex_coordinates))
    vertex_index = {value: index for index, value in enumerate(ordered_vertex_ids)}
    return TriangulatedRigidSurface(
        vertices=np.asarray(
            [local_vertex_coordinates[value] for value in ordered_vertex_ids],
            dtype=float,
        ),
        triangles=np.asarray(
            [
                tuple(vertex_index[value] for value in oriented_ids)
                for _, oriented_ids, _ in records
            ],
            dtype=np.int64,
        ).reshape(-1, 3),
        facet_ids=np.asarray(
            [facet_id_by_key[key] for key, _, _ in records], dtype=np.int64
        ),
        tolerance=tolerance,
        ambiguity_tolerance=ambiguity_tolerance,
        name=f"{name}__rank_{rank}_owned",
    )


def dolfinx_exterior_triangle_partition(
    domain,
    *,
    facets=None,
    name: str = "dolfinx_exterior_triangle_surface",
    tolerance: float | None = None,
    ambiguity_tolerance: float | None = None,
) -> TriangleSurfacePartition:
    """Build a stable distributed triangle partition from owned exterior facets.

    The first adapter is deliberately narrow: a three-dimensional, first-order
    tetrahedral DOLFINx mesh and owned exterior facets.  It orients every
    triangle away from its adjacent volume cell, derives partition-independent
    facet identities from input-global vertex IDs, and retains only rank-local
    geometry.  Unsupported topology or geometry order is rejected collectively.
    """

    comm = domain.comm
    local_error = None
    local_records: list[
        tuple[tuple[int, int, int], tuple[int, int, int], np.ndarray]
    ] = []
    try:
        from dolfinx import mesh as mesh_api

        topology = domain.topology
        geometry = domain.geometry
        topological_dimension = int(topology.dim)
        geometric_dimension = int(geometry.dim)
        if topological_dimension != 3 or geometric_dimension != 3:
            raise NotImplementedError(
                "only three-dimensional volume meshes are supported"
            )
        if str(topology.cell_type) != "CellType.tetrahedron":
            raise NotImplementedError(
                "only tetrahedral volume topology is supported"
            )
        coordinate_maps = getattr(geometry, "cmaps", ())
        if len(coordinate_maps) != 1 or int(coordinate_maps[0].degree) != 1:
            raise NotImplementedError(
                "only first-order coordinate geometry is supported"
            )
        facet_dimension = topological_dimension - 1
        topology.create_entities(facet_dimension)
        topology.create_connectivity(facet_dimension, topological_dimension)
        topology.create_entity_permutations()
        owned_exterior = np.asarray(
            mesh_api.exterior_facet_indices(topology), dtype=np.int32
        )
        if facets is None:
            selected = owned_exterior
        else:
            selected_raw = np.asarray(facets)
            if selected_raw.ndim != 1:
                raise ValueError("facets must be a one-dimensional local index array")
            selected = selected_raw.astype(np.int32)
            if not np.array_equal(selected_raw, selected):
                raise ValueError("facet indices must be integers")
            if np.unique(selected).size != selected.size:
                raise ValueError("facet indices must be unique on each rank")
            owned = set(int(value) for value in owned_exterior)
            invalid = [int(value) for value in selected if int(value) not in owned]
            if invalid:
                raise ValueError(
                    "selected facets must be owned exterior facets; first invalid "
                    f"local index is {invalid[0]}"
                )

        facet_geometry = mesh_api.entities_to_geometry(
            domain, facet_dimension, selected, permute=True
        )
        if facet_geometry.shape != (selected.size, 3):
            raise NotImplementedError(
                "exterior facets must map to exactly three geometry nodes"
            )
        input_global_indices = np.asarray(
            geometry.input_global_indices, dtype=np.int64
        )
        coordinates = np.asarray(geometry.x, dtype=np.float64)[:, :3]
        facet_to_cell = topology.connectivity(
            facet_dimension, topological_dimension
        )
        for facet, geometry_dofs in zip(selected, facet_geometry, strict=True):
            adjacent_cells = np.asarray(
                facet_to_cell.links(int(facet)), dtype=np.int32
            )
            if adjacent_cells.size != 1:
                raise ValueError(
                    "an exterior facet must have exactly one adjacent volume cell"
                )
            cell_geometry = mesh_api.entities_to_geometry(
                domain,
                topological_dimension,
                np.asarray((adjacent_cells[0],), dtype=np.int32),
                permute=True,
            )[0]
            if cell_geometry.size != 4:
                raise NotImplementedError(
                    "a first-order tetrahedron must map to four geometry nodes"
                )
            oriented_dofs = np.asarray(geometry_dofs, dtype=np.int32).copy()
            facet_points = coordinates[oriented_dofs]
            cell_center = np.mean(coordinates[cell_geometry], axis=0)
            facet_center = np.mean(facet_points, axis=0)
            area_vector = np.cross(
                facet_points[1] - facet_points[0],
                facet_points[2] - facet_points[0],
            )
            orientation = float(np.dot(area_vector, cell_center - facet_center))
            if not np.isfinite(orientation) or orientation == 0.0:
                raise ValueError("cannot determine an outward facet orientation")
            if orientation > 0.0:
                oriented_dofs[[1, 2]] = oriented_dofs[[2, 1]]
            oriented_ids = tuple(
                int(value) for value in input_global_indices[oriented_dofs]
            )
            key = tuple(sorted(oriented_ids))
            if len(set(key)) != 3:
                raise ValueError("an exterior triangle has repeated global vertex IDs")
            local_records.append((key, oriented_ids, coordinates[oriented_dofs].copy()))
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _collective_error(comm, local_error, context="extract owned exterior triangles")

    canonical_local = tuple(
        (
            key,
            tuple(
                np.asarray(points, dtype=np.float64)[
                    [oriented_ids.index(value) for value in key]
                ].reshape(-1)
            ),
        )
        for key, oriented_ids, points in local_records
    )
    gathered = tuple(comm.allgather(canonical_local))
    canonical_global = tuple(sorted(record for group in gathered for record in group))
    keys = tuple(record[0] for record in canonical_global)
    if not keys:
        raise ValueError("DOLFINx boundary adapter selected no exterior facets.")
    if len(set(keys)) != len(keys):
        raise ValueError(
            "DOLFINx exterior facet ownership is not globally unique by vertex ID."
        )
    all_coordinates = np.asarray(
        [value for _, coordinates in canonical_global for value in coordinates],
        dtype=float,
    ).reshape(-1, 3)
    scale = float(
        np.linalg.norm(np.max(all_coordinates, axis=0) - np.min(all_coordinates, axis=0))
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("DOLFINx exterior triangle surface has zero global scale.")
    selected_tolerance = (
        max(scale * 1.0e-12, np.finfo(float).eps * scale * 1024.0)
        if tolerance is None
        else _finite_nonnegative(tolerance, name="Surface tolerance")
    )
    if selected_tolerance <= 0.0:
        raise ValueError("Surface tolerance must be positive.")
    selected_ambiguity = (
        selected_tolerance
        if ambiguity_tolerance is None
        else _finite_nonnegative(
            ambiguity_tolerance, name="Projection ambiguity tolerance"
        )
    )
    if selected_ambiguity <= 0.0:
        raise ValueError("Projection ambiguity tolerance must be positive.")

    facet_id_by_key = {key: index for index, key in enumerate(keys)}
    local_error = None
    local_surface = None
    try:
        local_surface = _local_surface_from_records(
            local_records,
            facet_id_by_key=facet_id_by_key,
            name=str(name),
            rank=int(comm.rank),
            tolerance=selected_tolerance,
            ambiguity_tolerance=selected_ambiguity,
        )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _collective_error(comm, local_error, context="validate local oriented shards")
    facet_ids = tuple(range(len(keys)))
    return TriangleSurfacePartition(
        local_surface=local_surface,
        global_geometry_fingerprint=_distributed_geometry_fingerprint(
            canonical_global,
            tolerance=selected_tolerance,
            ambiguity_tolerance=selected_ambiguity,
        ),
        global_facet_identity_fingerprint=_facet_identity_fingerprint(facet_ids),
        global_surface_name=str(name),
        global_facet_count=len(keys),
        global_scale=scale,
        global_ambiguity_tolerance=selected_ambiguity,
        rank=int(comm.rank),
        rank_count=int(comm.size),
        ownership_method="dolfinx_owned_exterior_facets_by_input_vertex_identity",
    )


def dolfinx_tagged_exterior_triangle_partition(
    domain,
    facet_tags,
    *,
    tag: int,
    name: str | None = None,
    tolerance: float | None = None,
    ambiguity_tolerance: float | None = None,
) -> TriangleSurfacePartition:
    """Build a distributed triangle partition from one DOLFINx facet tag."""

    facet_dimension = int(domain.topology.dim) - 1
    if int(facet_tags.dim) != facet_dimension:
        raise ValueError(
            "Tagged exterior triangle adaptation requires MeshTags on "
            f"dimension {facet_dimension}, received {facet_tags.dim}."
        )
    selected_tag = int(tag)
    tagged = np.asarray(facet_tags.find(selected_tag), dtype=np.int32)
    facet_map = domain.topology.index_map(facet_dimension)
    if facet_map is None:
        domain.topology.create_entities(facet_dimension)
        facet_map = domain.topology.index_map(facet_dimension)
    owned = tagged[tagged < int(facet_map.size_local)]
    present = any(domain.comm.allgather(bool(owned.size)))
    if not present:
        available_local = tuple(int(value) for value in np.unique(facet_tags.values))
        available = sorted(
            set().union(*domain.comm.allgather(available_local))
        )
        raise ValueError(
            f"Facet tag {selected_tag} is absent; available tags are {available}."
        )
    return dolfinx_exterior_triangle_partition(
        domain,
        facets=owned,
        name=(
            f"dolfinx_exterior_facet_tag_{selected_tag}"
            if name is None
            else str(name)
        ),
        tolerance=tolerance,
        ambiguity_tolerance=ambiguity_tolerance,
    )


def dolfinx_boundary_region_triangle_partition(
    region,
    *,
    name: str | None = None,
    tolerance: float | None = None,
    ambiguity_tolerance: float | None = None,
) -> TriangleSurfacePartition:
    """Adapt an AgentFEM named boundary region into the search contract."""

    if region is None or not hasattr(region, "domain"):
        raise TypeError("Boundary adaptation requires an AgentFEM BoundaryRegion.")
    if getattr(region, "facet_tags", None) is None:
        raise ValueError(
            "Boundary adaptation requires canonical facet tags; construct the "
            "region with mesh.boundary or mesh.tagged_boundary_region."
        )
    return dolfinx_tagged_exterior_triangle_partition(
        region.domain,
        region.facet_tags,
        tag=int(region.tag),
        name=str(region.name) if name is None else str(name),
        tolerance=tolerance,
        ambiguity_tolerance=ambiguity_tolerance,
    )


__all__ = [
    "dolfinx_boundary_region_triangle_partition",
    "dolfinx_exterior_triangle_partition",
    "dolfinx_tagged_exterior_triangle_partition",
]
