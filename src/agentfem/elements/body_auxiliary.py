# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Stateless generic kinematics/viscosity adapters for validated Hex8 batches."""
import ctypes
import numpy as np


def auxiliary_callbacks():
    from .. import _body_pipeline
    from ._hex_validity import _coordinate_edges
    from ._bulk_viscosity import BulkViscosityResponse
    library = ctypes.CDLL(_body_pipeline.__file__)
    real = np.ctypeslib.ndpointer(dtype=np.float64, flags=('C_CONTIGUOUS', 'ALIGNED'))
    integer = np.ctypeslib.ndpointer(dtype=np.int32, flags=('C_CONTIGUOUS', 'ALIGNED'))
    gradient = library.hex_gradient
    gradient.argtypes = [ctypes.c_int, real, real, real, real, integer, real, integer]
    gradient.restype = None
    bulk = library.hex_bulk_response
    bulk.argtypes = [ctypes.c_int] + [real]*15 + [integer]
    bulk.restype = None
    edges = np.ascontiguousarray(_coordinate_edges(), dtype=np.int32)

    def check_count(n):
        if n > np.iinfo(np.int32).max:
            raise ValueError('Hex8 batch exceeds native integer ABI')

    def checked(value, shape):
        a = np.asarray(value)
        # The execution boundary has already frozen and finite-checked these
        # arrays. Recheck layout, not every value for a second time per step.
        if a.shape != shape or np.iscomplexobj(a):
            raise ValueError('Native auxiliary array shape contract violated')
        return np.require(a, dtype=float, requirements=('C','A'))

    def kinematics(geometry, displacement):
        n = len(geometry.volume)
        check_count(n)
        f, flags = np.empty((n,3,3)), np.empty(n,dtype=np.int32)
        gradient(n, checked(geometry.coordinates,(n,8,3)), checked(displacement,(n,8,3)),
            checked(geometry.average_gradient,(n,8,3)),
            checked(geometry.reference_perturbation_radius,(n,)), edges, f, flags)
        return f, flags.astype(bool)

    def viscosity(geometry, f, velocity, coefficients, reference):
        n = len(geometry.volume)
        check_count(n)
        if geometry.lumped_mass is None:
            return reference()
        force = np.empty((n,8,3))
        power, rate, pressure, bound = (np.empty(n) for _ in range(4))
        status = np.empty(n,dtype=np.int32)
        bulk(n, checked(f,(n,3,3)), checked(geometry.average_gradient,(n,8,3)),
            checked(velocity,(n,8,3)), checked(geometry.lumped_mass,(n,8)),
            checked(geometry.volume,(n,)),
            *[checked(coefficients[name],(n,)) for name in
              ('density','wave_speed','characteristic_length','linear','quadratic')],
            force,power,rate,pressure,bound,status)
        if np.any(status):
            return reference()
        return BulkViscosityResponse(force,power,rate,pressure,bound)

    return kinematics, viscosity
