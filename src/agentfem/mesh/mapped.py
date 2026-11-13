# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Tensor-product connectivity and explicit parametric geometry maps."""
from itertools import product

import basix
import numpy as np
from mpi4py import MPI


def _grid_arrays(axes, mapping, cell_type, degree):
    axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
    dimension = len(axes)
    if dimension not in (2, 3):
        raise ValueError("mapped_grid requires two or three coordinate axes.")
    if isinstance(degree, bool) or degree not in (1, 2):
        raise ValueError("geometry_degree must be 1 or 2.")
    degree = int(degree)
    for axis in axes:
        if (axis.ndim != 1 or axis.size < degree + 1
                or not np.all(np.isfinite(axis))
                or (axis.size - 1) % degree
                or not (np.all(np.diff(axis) > 0) or np.all(np.diff(axis) < 0))):
            raise ValueError("Axes must be finite, strictly monotone and contain degree*cells+1 nodes.")
    tensor_type = "quadrilateral" if dimension == 2 else "hexahedron"
    simplex_type = "triangle" if dimension == 2 else "tetrahedron"
    if cell_type not in (tensor_type, simplex_type):
        raise ValueError("cell_type does not match the number of axes.")
    if cell_type == simplex_type and degree != 1:
        raise ValueError("Simplex subdivision currently requires geometry_degree=1.")
    # Axis 0 varies fastest, matching the tensor-product/Basix convention.
    parameters = np.asarray([tuple(reversed(p)) for p in product(*reversed(axes))]).T
    points = parameters if mapping is None else np.asarray(mapping(parameters), dtype=float)
    if points.shape != parameters.shape or not np.all(np.isfinite(points)):
        raise ValueError("mapping must return finite coordinates with shape (dimension, points).")
    sizes = np.asarray([len(axis) for axis in axes], dtype=np.int64)
    strides = np.concatenate(([1], np.cumprod(sizes[:-1])))
    counts = (sizes - 1) // degree
    basis = basix.create_element(basix.ElementFamily.P, getattr(basix.CellType, tensor_type),
                                 degree, basix.LagrangeVariant.equispaced)
    offsets = np.rint(degree * np.asarray(basis.points)).astype(np.int64)
    cells = []
    for reversed_index in product(*(range(int(n)) for n in reversed(counts))):
        base = degree * np.asarray(tuple(reversed(reversed_index)), dtype=np.int64)
        nodes = (base + offsets) @ strides
        if cell_type == tensor_type:
            cells.append(nodes)
        elif dimension == 2:
            a, b, c, d = nodes
            cells.extend(((a, b, d), (a, d, c)))
        else:
            a, b, c, d, e, f, g, h = nodes
            cells.extend(((a, b, d, h), (a, d, c, h), (a, c, g, h),
                          (a, g, e, h), (a, e, f, h), (a, f, b, h)))
    return np.asarray(cells, dtype=np.int64), np.ascontiguousarray(points.T)


def mapped_grid(*, axes, mapping=None, cell_type="hexahedron", geometry_degree=1,
                comm=MPI.COMM_WORLD):
    """Map a structured parameter grid into a 2D/3D physical domain.

    ``axes`` specify geometry nodes (degree*cells+1 per axis), not only cell
    corners. ``mapping`` receives/returns arrays shaped (dimension, points).
    The case owns this physical map; this routine owns connectivity, quadratic
    node ordering and MPI distribution. Root rank supplies axes and mapping.
    Supports Q1/Q2 quads/hexes and fixed Q1 triangle/tetrahedron subdivisions.
    The caller remains responsible for a nonfolding physical map; this is not
    a mesh-quality certificate. No crack, material or boundary is inferred.
    """
    from . import from_arrays

    error = None
    topology = coordinates = None
    if comm.rank == 0:
        try:
            topology, coordinates = _grid_arrays(axes, mapping, cell_type, geometry_degree)
            descriptor = (str(cell_type), int(geometry_degree), coordinates.shape[1],
                          topology.shape[1])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error is not None:
        raise ValueError(f"mapped_grid: {error}")
    selected_type, degree, dimension, width = comm.bcast(
        descriptor if comm.rank == 0 else None, root=0)
    if comm.rank != 0:
        topology = np.empty((0, width), dtype=np.int64)
        coordinates = np.empty((0, dimension), dtype=float)
    return from_arrays(cells=topology, coordinates=coordinates,
                       cell_type=selected_type, geometry_degree=degree, comm=comm)
