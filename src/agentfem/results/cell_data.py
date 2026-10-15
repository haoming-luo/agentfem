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
    return _CellFieldTransport(domain, cell_keys).field(values, name=name)


class _CellFieldTransport:
    """One fixed-mesh/key map; fields remain detached snapshots, not a cache.

    Private to the result transport. A series checks keys at every station;
    spaces and the permutation are reused, never mutable field coefficients.
    """

    def __init__(self, domain, cell_keys=None):
        if domain.comm.size != 1:
            raise NotImplementedError("Cell-array result transport currently requires a serial mesh.")
        self.domain = domain
        self.count = domain.topology.index_map(domain.topology.dim).size_local
        self.original = np.array(domain.topology.original_cell_index[:self.count], copy=True)
        self.keys = None
        self.permutation = None
        self.spaces = {}
        if cell_keys is not None:
            keys = np.asarray(cell_keys)
            if keys.shape != (self.count,) or keys.dtype.kind not in "iu":
                raise ValueError("Cell keys must be an integer vector covering every cell.")
            order = np.argsort(keys)
            if (not np.array_equal(keys[order], np.sort(self.original))
                    or len(np.unique(keys)) != self.count):
                raise ValueError("Cell keys must match the original mesh cell identities exactly once.")
            self.keys = np.array(keys, copy=True)
            self.permutation = order[np.searchsorted(keys[order], self.original)]

    def matches(self, domain, keys):
        return (domain is self.domain and self.keys is not None
                and domain.topology.index_map(domain.topology.dim).size_local == self.count
                and np.array_equal(self.keys, keys)
                and np.array_equal(self.original, domain.topology.original_cell_index[:self.count]))

    def field(self, values, *, name):
        domain, count = self.domain, self.count
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Cell field name must be a nonempty string.")
        array = np.asarray(values)
        if (array.ndim not in (1, 2, 3) or len(array) != count
                or any(size == 0 for size in array.shape[1:])
                or np.iscomplexobj(array) or array.dtype.kind not in "fiu"
                or not np.isfinite(array).all()):
            raise ValueError("Expected finite real scalar/vector/tensor values for every cell.")
        shape = array.shape[1:]
        if shape not in self.spaces:
            element = ("DG", 0, shape) if shape else ("DG", 0)
            space = fem.functionspace(domain, element)
            dofs = np.asarray(space.dofmap.list)
            if dofs.shape != (count, 1) or space.dofmap.bs != int(np.prod(shape) if shape else 1):
                raise RuntimeError("DG0 cell transport requires one block dof per cell.")
            self.spaces[shape] = space, dofs[:, 0].copy()
        space, dofs = self.spaces[shape]
        if self.permutation is not None:
            array = array[self.permutation]
        result = fem.Function(space, name=name)
        result.x.array.reshape(-1, space.dofmap.bs)[dofs] = array.reshape(count, -1)
        result.x.scatter_forward()
        return result
