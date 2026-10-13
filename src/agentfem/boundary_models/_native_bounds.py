# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Instance-local native convex enclosure traversal, no global installation."""
import ctypes
from ._native_library import load_library
import numpy as np

def make_bounds():
    library = load_library()
    function = library.distance_bounds
    real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    integer = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    function.argtypes = [ctypes.c_int64] * 4 + [real] * 3 + [ctypes.c_double] * 2 + [real] * 2 + [integer] * 3 + [real] * 5 + [integer, real, integer, real]
    function.restype = ctypes.c_int

    def bounds(surface, points, reference, *, refinement_distance=None):
        raw = np.asarray(points)
        targets = np.zeros(len(raw)) if raw.ndim and refinement_distance is None else np.asarray(refinement_distance)
        if raw.ndim != 2 or raw.shape[1:] != (3,) or targets.shape != (len(raw),) or any((np.iscomplexobj(v) or not np.isfinite(v).all() for v in (raw, targets))) or np.any(targets < 0):
            return reference(points, refinement_distance=refinement_distance)
        tree = surface.tree
        nn, nf = (len(tree._facet), len(surface.vertices))
        doubles = lambda *args: [np.ascontiguousarray(v, dtype=float) for v in args]
        integers = lambda *args: [np.ascontiguousarray(v, dtype=np.int64) for v in args]
        output = np.empty(len(raw))
        status = function(len(raw), nn, nf, tree._root, *doubles(raw, targets, surface.vertices), surface.padding, float(np.max(abs(surface.vertices))), *doubles(tree._lower, tree._upper), *integers(tree._left, tree._right, tree._facet), *doubles(surface._bound_lower, surface._bound_upper, surface._bound_normals, surface._slab_lower, surface._slab_upper), np.empty(nn, dtype=np.int64), np.empty(nn), np.empty(nf, dtype=np.int64), output)
        if status:
            return reference(points, refinement_distance=refinement_distance)
        return output
    return bounds
