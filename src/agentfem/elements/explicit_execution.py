# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Instance-local trusted execution boundary for response-only Hex8 arithmetic.

Kernels return detached arrays. They do not receive quadrature Fields, PETSc
objects or a commit operation. This is an execution extension, not a new law,
an automatic stability certificate, or isolation from malicious Python code.
"""
from dataclasses import dataclass, field
import json
from typing import Callable, Mapping

import numpy as np

from ..constitutive._explicit_material import ExplicitMaterialBatchOutput
from ..constitutive.material_array_batch import _array


@dataclass(frozen=True)
class ExplicitHexGeometry:
    average_gradient: np.ndarray
    hourglass_modes: np.ndarray
    volume: np.ndarray
    hourglass_coefficient: np.ndarray
    coordinates: np.ndarray | None = None
    reference_perturbation_radius: np.ndarray | None = None
    lumped_mass: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.volume)
        for name, shape in (("average_gradient", (n, 8, 3)),
                            ("hourglass_modes", (n, 8, 4)),
                            ("volume", (n,)), ("hourglass_coefficient", (n,))):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if np.any(self.volume <= 0) or np.any(self.hourglass_coefficient < 0):
            raise ValueError("Invalid explicit Hex8 reference geometry")
        if (self.coordinates is None) != (self.reference_perturbation_radius is None):
            raise ValueError("Kinematic geometry requires coordinates and perturbation radii")
        if self.coordinates is not None:
            object.__setattr__(self, 'coordinates', _array(self.coordinates, (n,8,3), 'coordinates'))
            object.__setattr__(self, 'reference_perturbation_radius', _array(
                self.reference_perturbation_radius, (n,), 'reference_perturbation_radius'))
            if np.any(self.reference_perturbation_radius < 0):
                raise ValueError("Negative reference perturbation radius")
        if self.lumped_mass is not None:
            object.__setattr__(self, 'lumped_mass', _array(self.lumped_mass, (n,8), 'lumped_mass'))
            if np.any(self.lumped_mass <= 0):
                raise ValueError("Positive physical lumped masses required")


@dataclass(frozen=True)
class ExplicitHexBodyOutput:
    material_response: ExplicitMaterialBatchOutput
    first_piola_stress: np.ndarray
    internal_force: np.ndarray
    hourglass_energy: np.ndarray

    def __post_init__(self):
        if not isinstance(self.material_response, ExplicitMaterialBatchOutput):
            raise TypeError("Expected ExplicitMaterialBatchOutput")
        n = self.material_response.point_count
        for name, shape in (("first_piola_stress", (n, 3, 3)),
                            ("internal_force", (n, 8, 3)), ("hourglass_energy", (n,))):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if np.any(self.hourglass_energy < 0):
            raise ValueError("Hourglass stabilization energy must be nonnegative")
        records = []
        for name in ('first_piola_stress', 'internal_force', 'hourglass_energy'):
            value = getattr(self, name)
            owner = value
            while isinstance(owner, np.ndarray):
                owner = owner.base
            records.append((name, value, owner, value.shape, value.strides,
                            value.dtype.str, value.ctypes.data))
        object.__setattr__(self, '_transport_stamp', (n, tuple(records)))


def _owned_body_output(output, response):
    """Re-own unchanged immutable payload metadata, otherwise fully recheck.

    No accepted state is cached here. As for material response transport, this
    avoids rescanning constructor-validated byte storage and does not sandbox
    malicious Python callbacks.
    """
    stamp = getattr(output, '_transport_stamp', None)
    if type(output) is ExplicitHexBodyOutput and stamp is not None:
        count, entries = stamp
        unchanged = count == response.point_count
        for name, original, owner, shape, strides, dtype, pointer in entries:
            value = getattr(output, name)
            unchanged = unchanged and (value is original and isinstance(owner, bytes)
                and not value.flags.writeable and value.shape == shape
                and value.strides == strides and value.dtype.str == dtype
                and value.ctypes.data == pointer)
        if unchanged:
            result = object.__new__(ExplicitHexBodyOutput)
            object.__setattr__(result, 'material_response', response)
            for name, original, owner, shape, strides, dtype, pointer in entries:
                offset = pointer - np.frombuffer(owner, dtype=np.uint8).ctypes.data
                value = np.frombuffer(owner, dtype=float, count=original.size, offset=offset).reshape(shape)
                object.__setattr__(result, name, value)
            # This internal transfer is consumed immediately, not republished
            # as a callback-owned validated record.
            return result
    return ExplicitHexBodyOutput(response, output.first_piola_stress,
                                 output.internal_force, output.hourglass_energy)


@dataclass(frozen=True)
class ExplicitHexBodyKernel:
    """Trusted callback: ``evaluate(material, request, geometry, displacement)``.

    The output includes material response, work-conjugate Piola stress, total
    cell internal force (material plus hourglass), and hourglass energy. The
    callback must preserve the material's declared law and exact fallback rules.
    Identify its pinned implementation in ``identity``. Core checks transport
    and acceptance; scientific equivalence needs separate reference tests.
    """
    evaluate: Callable
    identity: Mapping
    prepare_kinematics: Callable | None = None
    evaluate_viscosity: Callable | None = None
    gather_cell_vectors: Callable | None = None
    _identity_json: str = field(init=False, repr=False)

    def __post_init__(self):
        if not callable(self.evaluate) or not isinstance(self.identity, Mapping):
            raise TypeError("Kernel requires a callback and identity mapping")
        encoded = json.dumps(dict(self.identity), sort_keys=True, allow_nan=False)
        if not self.identity:
            raise ValueError("Kernel requires a nonempty pinned identity")
        object.__setattr__(self, "_identity_json", encoded)
        if self.prepare_kinematics is not None and not callable(self.prepare_kinematics):
            raise TypeError("prepare_kinematics must be callable or None")
        if self.evaluate_viscosity is not None and not callable(self.evaluate_viscosity):
            raise TypeError("evaluate_viscosity must be callable or None")
        if self.gather_cell_vectors is not None and not callable(self.gather_cell_vectors):
            raise TypeError("gather_cell_vectors must be callable or None")

    def summary(self):
        return {"kind": "trusted_explicit_hex_body_kernel", "identity": json.loads(self._identity_json),
                "state_owner": "core", "scope": "serial_response_only_hex8",
                "kinematic_screen": self.prepare_kinematics is not None,
                "bulk_viscosity": self.evaluate_viscosity is not None,
                "cell_gather": self.gather_cell_vectors is not None}


def prepare_kernel_viscosity(kernel, geometry, gradient, velocity, options, reference):
    from types import MappingProxyType
    from ._bulk_viscosity import BulkViscosityResponse
    n = len(geometry.volume)
    coefficients = {}
    for name in ('density', 'wave_speed', 'characteristic_length', 'linear', 'quadratic'):
        raw = np.asarray(options[name])
        if raw.shape not in ((), (n,)):
            raise ValueError("Bulk coefficients require scalar or cellwise values")
        values = _array(np.broadcast_to(raw, (n,)), (n,), name)
        if np.any(values < 0 if name in ('linear', 'quadratic') else values <= 0):
            raise ValueError("Invalid bulk viscosity coefficient sign")
        coefficients[name] = values
    f = _array(gradient, (n,3,3), 'admitted_gradient')
    v = _array(velocity, (n,8,3), 'cell_velocity')
    response = kernel.evaluate_viscosity(geometry, f, v, MappingProxyType(coefficients), reference)
    if not isinstance(response, BulkViscosityResponse) or response.damping_row_majorant is not None:
        raise TypeError("Bulk kernel requires the bounded BulkViscosityResponse contract")
    result = BulkViscosityResponse(*(_array(getattr(response, name), shape, name)
        for name,shape in (('internal_force',(n,8,3)), ('artificial_power',(n,)),
                           ('volumetric_rate',(n,)), ('isotropic_stress',(n,)),
                           ('mass_scaled_damping_bound',(n,)))))
    if np.any(result.artificial_power < 0) or np.any(result.mass_scaled_damping_bound < 0):
        raise ValueError("Artificial damping power and bound must be nonnegative")
    return result


def prepare_kernel_kinematics(kernel, geometry, displacement):
    """Optional fused gradient/reference-ball screen with unchanged fallbacks.

The trusted callback returns F and a boolean mask certifying the same
reference-perturbation sufficient condition as the core. It cannot use a
positive mean determinant alone to certify the entire cell. Uncertified cells
retain affine/Bernstein checks; every mean F still needs positive determinant.
"""
    from ._hex_validity import admits_affine_perturbation, require_positive_hex_jacobian
    from .._orientation import positive_finite_determinants
    u = _array(displacement, geometry.coordinates.shape, 'cell_displacement')
    result = kernel.prepare_kinematics(geometry, u)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError("Kinematic kernel must return gradient and certified mask")
    f = _array(result[0], (len(u),3,3), 'deformation_gradient')
    admitted = np.asarray(result[1])
    if admitted.dtype.kind != 'b' or admitted.shape != (len(u),):
        raise ValueError("Kinematic kernel needs one boolean reference certificate per cell")
    admitted = admitted.copy()
    pending = ~admitted
    if np.any(pending):
        # Avoid full-mesh advanced-index copies before the cell-local screen.
        indices = np.flatnonzero(pending)
        for start in range(0, len(indices), 2048):
            selected = indices[start:start + 2048]
            admitted[selected] = admits_affine_perturbation(geometry.coordinates[selected],
                u[selected], f[selected], geometry.reference_perturbation_radius[selected])
    if np.any(~admitted):
        require_positive_hex_jacobian(geometry.coordinates[~admitted]+u[~admitted])
    if not positive_finite_determinants(f).all():
        raise ValueError("Mean deformation gradient must have positive determinant.")
    return f


def prepare_kernel_response(kernel, geometry, owner, material, displacement, gradient,
                            *, deformation_gradient_old, time, time_increment):
    from ..constitutive.material_array_batch import MaterialPointArrayBatchInput
    from ..constitutive._explicit_material import validated_explicit_material_update
    from ..constitutive._explicit_material_driver import ExplicitQuadratureTrial
    from ._finite_uniform_hex import FiniteUniformHexResponse

    request = MaterialPointArrayBatchInput(deformation_gradient_old, gradient,
        owner.response.committed_history(), material.state_schema, time, time_increment)
    u = _array(displacement, (request.point_count, 8, 3), "cell_displacement")
    outputs = []

    def evaluate(checked):
        output = kernel.evaluate(material, checked, geometry, u)
        if not isinstance(output, ExplicitHexBodyOutput):
            raise TypeError("Body kernel must return ExplicitHexBodyOutput")
        outputs.append(output)
        return output.material_response

    response = validated_explicit_material_update(material, request, response_evaluator=evaluate)
    if np.any(response.requested_time_scale < 1):
        from ..constitutive._explicit_material import ExplicitMaterialCutbackRequest
        raise ExplicitMaterialCutbackRequest("Material requested a smaller increment; trial rejected without state mutation.")
    output = outputs[0]
    # Repeat transport validation at the execution boundary, including mutated
    # NumPy shape metadata, without copying immutable byte-backed payloads.
    output = _owned_body_output(output, response)
    energy = None if response.stored_energy_density is None else response.stored_energy_density * geometry.volume
    return ExplicitQuadratureTrial(response, output.first_piola_stress), FiniteUniformHexResponse(
        output.internal_force, energy, output.hourglass_energy)
