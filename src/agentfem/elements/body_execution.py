# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Built-in material-independent execution for the response-only Hex8 route."""
import numpy as np

from .explicit_execution import ExplicitHexBodyKernel, ExplicitHexBodyOutput
from ..constitutive._explicit_material import ExplicitMaterialBatchOutput


def explicit_hex_kernel(*, backend="auto"):
    """Use installed native arithmetic or the reference NumPy implementation.

    Material updates keep their ordinary array-batch interface and arbitrary
    state schema. This factory introduces no material law, thread pool, time
    step, inertia change or state owner. ``native`` fails if not installed;
    ``auto`` retains a deterministic reference fallback.
    """
    if backend not in ("auto", "native", "numpy"):
        raise ValueError("Body backend must be auto, native or numpy")
    native = None
    if backend != "numpy":
        try:
            from .. import _hex_native
            if hasattr(_hex_native, "body_response"):
                native = _hex_native
        except ImportError:
            pass
    if backend == "native" and native is None:
        raise RuntimeError("Native Hex8 body arithmetic is not installed")

    def evaluate(material, request, geometry, displacement):
        response = material.update_explicit_array_batch(request)
        if not isinstance(response, ExplicitMaterialBatchOutput):
            raise TypeError("Material must return ExplicitMaterialBatchOutput")
        n = request.point_count
        if response.point_count != n:
            raise ValueError("Material response point count differs from the request")
        if native is not None:
            raw = native.body_response(request.deformation_gradient_new,
                response.cauchy_stress, displacement, geometry.average_gradient,
                geometry.hourglass_modes, geometry.volume, geometry.hourglass_coefficient)
            piola, force, energy = (np.frombuffer(value, dtype=np.float64).reshape(shape)
                for value, shape in zip(raw, ((n, 3, 3), (n, 8, 3), (n,)), strict=True))
        else:
            from ..constitutive.material_driver import _cauchy_to_first_piola
            piola = _cauchy_to_first_piola(request.deformation_gradient_new,
                                          response.cauchy_stress)
            modes = geometry.hourglass_modes.swapaxes(1, 2) @ displacement
            force = geometry.volume[:, None, None] * (
                geometry.average_gradient @ piola.swapaxes(1, 2))
            force += geometry.hourglass_coefficient[:, None, None] * (
                geometry.hourglass_modes @ modes)
            energy = .5 * geometry.hourglass_coefficient * np.sum(modes**2, axis=(1, 2))
        return ExplicitHexBodyOutput(response, piola, force, energy)

    kinematics = viscosity = None
    if native is not None:
        try:
            from .body_auxiliary import auxiliary_callbacks
            kinematics, viscosity = auxiliary_callbacks()
        except ImportError:
            pass
    return ExplicitHexBodyKernel(evaluate, {
        "provider": "agentfem.explicit_hex_body", "revision": 1,
        "arithmetic": "native" if native is not None else "numpy",
        "material_protocol": "explicit_array_batch", "state_owner": "core",
        "native_auxiliary": kinematics is not None,
    }, prepare_kinematics=kinematics, evaluate_viscosity=viscosity)
