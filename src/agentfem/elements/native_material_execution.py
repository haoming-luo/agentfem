# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Trusted compiled material ABI, fused with built-in Hex8 arithmetic.

Provider authors supply a reentrant C function and exact Python fallback.
State width is arbitrary; ABI v1 carries stress, state and internal work only.
No pointers are serialized. Core retains increment checks and state ownership.
"""
import ctypes
import hashlib
from dataclasses import replace
import numpy as np
from .explicit_execution import ExplicitHexBodyKernel, ExplicitHexBodyOutput
from ..constitutive._explicit_material import ExplicitMaterialBatchOutput, ExplicitMaterialCutbackRequest


def native_explicit_hex_kernel(material, function, *, properties, density,
                               identity, fallback, validate=None, workers=1):
    """Bind trusted C material code, not a data-file address.

    ABI: void(int n, double* F_old, F_new, state_old, properties,
    double dt, double rho, double* stress, state_new, work, int* status).
    Arrays are contiguous row-major float64. Nonzero status requests repair.
    Every worker is joined before repair and before publishing immutable output.
    function must retain its CDLL owner for the entire kernel lifetime.
    """
    from .. import _body_pipeline
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 16:
        raise ValueError('Native material workers must be an integer in [1,16]')
    if not isinstance(function, ctypes._CFuncPtr):
        raise TypeError('A trusted ctypes C function is required')
    if not callable(fallback) or (validate is not None and not callable(validate)):
        raise TypeError('Material fallback and validator must be callable')
    if not np.isfinite(density) or density <= 0:
        raise ValueError('Positive finite density required')
    if np.iscomplexobj(properties):
        raise ValueError('Real properties required')
    parameters = np.asarray(properties, dtype=float)
    if parameters.ndim != 1 or not np.isfinite(parameters).all():
        raise ValueError('Finite one-dimensional properties required')
    parameters = np.frombuffer(parameters.tobytes(), dtype=float)
    width = material.state_schema.size
    if width < 1:
        raise ValueError('Native material ABI requires a nonempty state schema')
    address = ctypes.cast(function, ctypes.c_void_p).value

    def evaluate(selected, request, geometry, displacement):
        # Capturing function pins the executable, not just its numerical address.
        if function is None or selected is not material or request.state_schema != material.state_schema:
            raise ValueError('Native binding and material schema/owner differ')
        def repair(index_bytes):
            ids = np.frombuffer(index_bytes, dtype=np.int64)
            part = replace(request, deformation_gradient_old=request.deformation_gradient_old[ids],
                deformation_gradient_new=request.deformation_gradient_new[ids], state_old=request.state_old[ids])
            response = fallback(part)
            if not isinstance(response, ExplicitMaterialBatchOutput) or response.state_schema != request.state_schema:
                raise ValueError('Native material repair returned incompatible schema')
            if np.any(response.requested_time_scale < 1):
                raise ExplicitMaterialCutbackRequest('Native repair requests a smaller increment')
            if response.stored_energy_density is not None or response.dissipation_density_increment is not None:
                raise ValueError('Native material ABI v1 does not carry optional energy channels')
            return response.cauchy_stress, response.state_new, response.internal_work_density_increment
        if any(getattr(request, name) is not None for name in
               ('temperature', 'temperature_increment', 'field_variables')):
            raise ValueError('Native material ABI v1 does not carry extra fields')
        raw = _body_pipeline.run(address,
            (request.deformation_gradient_old, request.deformation_gradient_new,
             request.state_old, parameters, displacement, geometry.average_gradient,
             geometry.hourglass_modes, geometry.volume, geometry.hourglass_coefficient),
            request.time_increment, density, workers, width, len(parameters), repair)
        n = request.point_count
        shapes = ((n,3,3), (n,width), (n,), (n,3,3), (n,8,3), (n,))
        stress, state, work, piola, force, energy = (
            np.frombuffer(value, dtype=float).reshape(shape)
            for value, shape in zip(raw, shapes, strict=True))
        response = ExplicitMaterialBatchOutput(stress, state, request.state_schema, work)
        if validate is not None:
            validate(response)
        return ExplicitHexBodyOutput(response, piola, force, energy)

    from .body_auxiliary import auxiliary_callbacks
    kinematics, viscosity = auxiliary_callbacks()
    return ExplicitHexBodyKernel(evaluate, {
        'provider': 'agentfem.native_material_hex', 'abi': 1, 'revision': 1,
        'material': identity, 'state_width': width, 'property_count': len(parameters),
        'workers': workers, 'state_owner': 'core', 'density': float(density),
        'properties_sha256': hashlib.sha256(parameters.tobytes()).hexdigest()},
        prepare_kinematics=kinematics, evaluate_viscosity=viscosity)
