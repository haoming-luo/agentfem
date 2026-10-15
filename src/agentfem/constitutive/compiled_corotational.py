# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Reviewed optional native polar-path response with per-point exact repair."""
import ctypes
import numpy as np
from .corotational import CorotationalIncrement, polar_strain_increment
from .native_build import NativeLibraryBundle


class CompiledCorotationalIncrement:
    """Bind an explicitly built polar_increment C ABI, keeping its library alive.

    The supplied bundle is trusted native code. Bundled polar_series/polar_wider
    sources use the same linear-right-stretch path as polar_strain_increment;
    only their fast-path admission radii differ. Rejected points take the
    reference path, not an approximate or zero increment. This neither chooses
    a VUMAT strain convention nor establishes whole-path stability.
    """
    def __init__(self, bundle, target):
        if not isinstance(bundle, NativeLibraryBundle):
            raise TypeError("Expected a reviewed NativeLibraryBundle")
        self._library = ctypes.CDLL(str(bundle.library(target)))
        self._function = self._library.polar_increment
        array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags="C_CONTIGUOUS")
        self._function.argtypes = [ctypes.c_int] + [array] * 5 + [
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")]
        self._function.restype = None

    def __call__(self, old, new):
        if np.iscomplexobj(old) or np.iscomplexobj(new) or np.ma.isMaskedArray(old) or np.ma.isMaskedArray(new):
            raise ValueError("Expected real unmasked deformation gradients")
        old, new = (np.ascontiguousarray(x, dtype=float) for x in (old, new))
        if old.ndim != 3 or old.shape[1:] != (3, 3) or new.shape != old.shape:
            raise ValueError("Expected matching (points,3,3) gradients")
        if len(old) > np.iinfo(np.int32).max or not np.isfinite(old).all() or not np.isfinite(new).all():
            raise ValueError("Nonfinite gradient or batch beyond compiled ABI")
        strain, r0, r1 = (np.empty_like(old) for _ in range(3))
        status = np.empty(len(old), dtype=np.int32)
        self._function(len(old), old, new, strain, r0, r1, status)
        rejected = status != 0
        if np.any(rejected):
            repair = polar_strain_increment(old[rejected], new[rejected])
            strain[rejected] = repair.strain_increment
            r0[rejected], r1[rejected] = repair.frame_old, repair.frame_new
        for array in (strain, r0, r1):
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite native corotational response")
            array.setflags(write=False)
        return CorotationalIncrement(strain, r0, r1)
