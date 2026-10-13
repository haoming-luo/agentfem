# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Instance-local swept interval proof; unresolved paths use core reference."""
import ctypes
from ._native_library import load_library
import numpy as np

def hierarchy(vertices):
    left, right, facet = ([], [], [])
    centers = vertices.mean(axis=1)

    def add(ids):
        node = len(facet)
        left.append(-1)
        right.append(-1)
        facet.append(-1)
        if len(ids) == 1:
            facet[node] = int(ids[0])
        else:
            axis = int(np.argmax(np.ptp(centers[ids], axis=0)))
            ids = ids[np.argsort(centers[ids, axis], kind='stable')]
            middle = len(ids) // 2
            left[node], right[node] = (add(ids[:middle]), add(ids[middle:]))
        return node
    add(np.arange(len(vertices)))
    return tuple((np.asarray(a, dtype=np.int64) for a in (left, right, facet)))

def make_swept():
    library = load_library()
    function = library.swept_candidates
    real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    integer = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    function.argtypes = [ctypes.c_int64] * 3 + [integer] * 3 + [real] * 6 + [integer] * 2
    function.restype = ctypes.c_int64
    cache = {}

    def separated(old, new, cells, start, end, reference):
        raw = [np.asarray(a) for a in (old, new, start, end)]
        indices = np.asarray(cells)
        if any((np.iscomplexobj(a) or not np.isfinite(a).all() for a in raw)) or raw[0].ndim != 2 or raw[0].shape[1:] != (3,) or (raw[1].shape != raw[0].shape) or (raw[2].ndim != 2) or (raw[2].shape[1:] != (3,)) or (raw[3].shape != raw[2].shape) or (indices.ndim != 2) or (indices.shape[1:] != (4,)) or (indices.dtype.kind not in 'iu') or (not len(indices)) or np.any(indices < 0) or np.any(indices >= len(raw[0])):
            return reference(old, new, cells, start, end)
        x0, x1 = (np.ascontiguousarray(a[indices], dtype=float) for a in raw[:2])
        a, b = (np.ascontiguousarray(v, dtype=float) for v in raw[2:])
        key = (indices.shape, indices.dtype.str, indices.tobytes())
        if key not in cache:
            cache.clear()
            cache[key] = hierarchy(x0)
        left, right, facet = cache[key]
        nn, capacity = (len(facet), max(64, len(a) * 32))
        pairs = np.empty((capacity, 2), dtype=np.int64)
        count = function(len(a), nn, capacity, left, right, facet, x0, x1, a, b, np.empty((nn, 3)), np.empty((nn, 3)), np.empty(nn, dtype=np.int64), pairs)
        if count < 0 or count > capacity:
            return reference(old, new, cells, start, end)
        result = np.ones(len(a), dtype=bool)
        pending = np.unique(pairs[:count, 0])
        if len(pending):
            if np.any(pending < 0) or np.any(pending >= len(a)):
                raise ValueError('Invalid native swept query identity')
            result[pending] = reference(old, new, cells, a[pending], b[pending])
        return result
    return separated
