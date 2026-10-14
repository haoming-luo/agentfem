# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit, unsmoothed cell-data transport for accepted result arrays."""
import numpy as np
from dolfinx import fem


def cell_field_from_values(domain, values, *, name, cell_keys=None):
    """Create a detached DG0 field from one scalar/vector/tensor per cell.

    Without keys, rows are in DOLFINx local cell order. Optional keys identify
    the mesh's original input cell indices (not external solver element labels)
    and allow arbitrary row order. All cells must be covered exactly once.
    This serial adapter neither averages quadrature points nor smooths across
    interfaces. Units and physical meaning remain the caller's result contract.
    """
    if domain.comm.size != 1:
        raise NotImplementedError("Cell-array result transport currently requires a serial mesh.")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Cell field name must be a nonempty string.")
    array = np.asarray(values)
    count = domain.topology.index_map(domain.topology.dim).size_local
    if (array.ndim not in (1, 2, 3) or len(array) != count
            or any(size == 0 for size in array.shape[1:])
            or np.iscomplexobj(array) or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()):
        raise ValueError("Expected finite real scalar/vector/tensor values for every cell.")
    if cell_keys is not None:
        keys = np.asarray(cell_keys)
        original = np.asarray(domain.topology.original_cell_index)[:count]
        if keys.shape != (count,) or keys.dtype.kind not in "iu":
            raise ValueError("Cell keys must be an integer vector covering every cell.")
        order = np.argsort(keys)
        if not np.array_equal(keys[order], np.sort(original)) or len(np.unique(keys)) != count:
            raise ValueError("Cell keys must match the original mesh cell identities exactly once.")
        array = array[order[np.searchsorted(keys[order], original)]]
    shape = array.shape[1:]
    element = ("DG", 0, shape) if shape else ("DG", 0)
    space = fem.functionspace(domain, element)
    dofs = np.asarray(space.dofmap.list)
    if dofs.shape != (count, 1) or space.dofmap.bs != int(np.prod(shape) if shape else 1):
        raise RuntimeError("DG0 cell transport requires one block dof per cell.")
    result = fem.Function(space, name=name)
    result.x.array.reshape(-1, space.dofmap.bs)[dofs[:, 0]] = array.reshape(count, -1)
    result.x.scatter_forward()
    return result
