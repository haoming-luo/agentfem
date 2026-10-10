# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Extract fixed-reference traces from the backend's boundary and DOF topology."""

import basix
import numpy as np


def reference_trace_from_boundary(displacement, boundary, *, tolerance):
    from .fields import unwrap
    from .interfaces import reference_trace
    from .provenance import collective_call, collective_canonical_record

    space = unwrap(displacement).function_space
    comm = space.mesh.comm
    xyz, cells, nodes, topology = collective_call(
        lambda: _local_trace(displacement, boundary), comm=comm,
        label="Reference interface extraction",
    )
    collective_canonical_record(
        {"topology": topology, "tag": int(boundary.tag), "tolerance": float(tolerance)},
        comm=comm, label="Reference trace description")
    if comm.size > 1:
        index = space.dofmap.index_map
        ids = index.local_to_global(nodes.astype(np.int32))
        pieces = comm.allgather((ids, xyz, ids[cells]))
        coordinates, faces = {}, []
        for global_ids, points, global_faces in pieces:
            coordinates.update(zip(global_ids.tolist(), points, strict=True))
            faces.extend(global_faces.tolist())
        if not faces:
            raise ValueError("Interface boundary selects no facets globally.")
        ordered = sorted(coordinates)
        position = {node: i for i, node in enumerate(ordered)}
        xyz = np.array([coordinates[node] for node in ordered])
        cells = np.array([[position[node] for node in face] for face in
                          sorted(faces, key=lambda face: tuple(sorted(face)))], dtype=np.int32)
        local_ids = index.local_to_global(np.arange(index.size_local + index.num_ghosts, dtype=np.int32))
        local = {int(node): i for i, node in enumerate(local_ids)}
        nodes = np.array([local.get(node, -1) for node in ordered], dtype=np.int32)
    elif not len(cells):
        raise ValueError("Interface boundary selects no facets.")
    trace = reference_trace(xyz, cells, topology=topology, tolerance=tolerance)
    nodes.setflags(write=False)
    return trace, nodes


def _local_trace(displacement, boundary):
    """Return outward-oriented trace and its independent displacement block map.

    Boundary labels must identify one side explicitly. Coordinate matching over
    the whole mesh would collapse coincident but independent interface nodes.
    """
    from .fields import unwrap

    space = unwrap(displacement).function_space
    domain = space.mesh
    if boundary.domain is not domain:
        raise ValueError(
            "Interface boundary and displacement belong to different meshes."
        )
    element = space.element.basix_element
    topology = element.cell_type.name
    if (
        domain.topology.dim != 3
        or space.dofmap.index_map_bs != 3
        or element.degree != 1
        or element.discontinuous
        or topology not in {"tetrahedron", "hexahedron"}
    ):
        raise NotImplementedError(
            "Reference boundary traces require blocked continuous P1 tetrahedra or Q1 hexahedra."
        )
    maps = getattr(domain.geometry, "cmaps", None)
    coordinate_map = maps[0] if maps is not None else domain.geometry.cmap
    if coordinate_map.degree != 1:
        raise NotImplementedError(
            "Reference boundary extraction requires degree-one geometry."
        )
    if boundary.facet_tags is None or boundary.facet_tags.dim != 2:
        raise ValueError("Interface boundary must carry surface facet tags.")
    facets = np.asarray(boundary.facet_tags.find(int(boundary.tag)), dtype=np.int32)
    facets = facets[facets < domain.topology.index_map(2).size_local]
    domain.topology.create_connectivity(2, 3)
    domain.topology.create_connectivity(3, 2)
    f2c = domain.topology.connectivity(2, 3)
    c2f = domain.topology.connectivity(3, 2)
    reference_facets = basix.cell.topology(element.cell_type)[2]
    coordinates = space.tabulate_dof_coordinates()
    faces = []
    for facet in facets:
        adjacent = f2c.links(int(facet))
        if len(adjacent) != 1:
            raise ValueError(
                "Interface traces require exterior facets with independent DOFs; split a bonded internal interface first."
            )
        cell = int(adjacent[0])
        local = np.flatnonzero(c2f.links(cell) == facet)
        if len(local) != 1:
            raise ValueError("Interface facet has no unique parent-cell identity.")
        vertices = reference_facets[int(local[0])]
        local_dofs = [element.entity_dofs[0][vertex][0] for vertex in vertices]
        cell_dofs = space.dofmap.cell_dofs(cell)
        nodes = cell_dofs[local_dofs]
        if len(nodes) == 4:
            nodes = nodes[[0, 1, 3, 2]]  # Basix tensor-product -> cyclic perimeter
        points = coordinates[nodes]
        normal = np.cross(points[1] - points[0], points[-1] - points[0])
        outward = points.mean(axis=0) - coordinates[cell_dofs].mean(axis=0)
        direction = float(normal @ outward)
        if not np.isfinite(direction) or direction == 0:
            raise ValueError(
                "Interface facet has degenerate or ambiguous outward orientation."
            )
        faces.append(nodes if direction > 0 else nodes[::-1])
    width = 3 if topology == "tetrahedron" else 4
    nodes, inverse = np.unique(np.asarray(faces, dtype=np.int32).reshape(-1, width), return_inverse=True)
    return (coordinates[nodes], inverse.reshape(-1, width), nodes,
            "triangle" if topology == "tetrahedron" else "quadrilateral")
