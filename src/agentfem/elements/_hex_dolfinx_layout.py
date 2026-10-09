# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared Q1 cell/DOF layout, leaving mesh and communication with DOLFINx."""

import numpy as np


def prepare_layout(displacement, chunk_size):
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer.")
    space = displacement.function_space
    element = space.element.basix_element
    if (
        space.dofmap.index_map_bs != 3
        or element.degree != 1
        or element.discontinuous
        or element.cell_type.name != "hexahedron"
    ):
        raise ValueError(
            "Uniform Hex8 requires continuous blocked vector Q1 hexahedra."
        )
    maps = getattr(space.mesh.geometry, "cmaps", None)
    coordinate_map = maps[0] if maps is not None else space.mesh.geometry.cmap
    if coordinate_map.degree != 1:
        raise ValueError("Uniform Hex8 requires trilinear geometry.")
    count = space.mesh.topology.index_map(3).size_local
    cell_nodes = np.array(
        [space.dofmap.cell_dofs(k) for k in range(count)], dtype=np.int32
    ).reshape(-1, 8)
    cell_nodes.setflags(write=False)
    plans = []
    for start in range(0, count, chunk_size):
        nodes, inverse = np.unique(
            cell_nodes[start : start + chunk_size].ravel(), return_inverse=True
        )
        plans.append((nodes, inverse.astype(np.int32)))
    coordinates = space.tabulate_dof_coordinates()[cell_nodes]
    return cell_nodes, coordinates, plans
