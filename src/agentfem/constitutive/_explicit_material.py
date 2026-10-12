# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private response-only contract. No implicit tangent or stability is implied.

The input reuses MaterialPointArrayBatchInput. Material history can include
accepted stress/work when its algorithm needs them. Output work is an increment
per REFERENCE volume, never specific work or recoverable stored energy.
"""

from dataclasses import dataclass, replace

import numpy as np

from .material_array_batch import MaterialPointArrayBatchInput, _array
from .user_material import MaterialStateSchema


_RESPONSE_ARRAYS = (
    "cauchy_stress", "state_new", "internal_work_density_increment",
    "stored_energy_density", "dissipation_density_increment", "requested_time_scale",
)


def _symmetric_cauchy_stress(stress):
    """Check the three independent pairs of an already finite stress batch."""
    scale = np.maximum(np.max(np.abs(stress), axis=(1, 2)), np.finfo(float).tiny)
    tolerance = 1.0e-12 * scale
    return all(not np.any(np.abs(stress[:, i, j] - stress[:, j, i]) > tolerance)
               for i, j in ((0, 1), (0, 2), (1, 2)))


def _response_transport_stamp(response):
    """Remember immutable payloads and mutable NumPy metadata separately."""
    entries = []
    for name in _RESPONSE_ARRAYS:
        value = getattr(response, name)
        if value is None:
            entries.append((name, None, None, None, None))
            continue
        owner = value
        while isinstance(owner, np.ndarray):
            owner = owner.base
        if not isinstance(owner, bytes):  # only constructor-owned byte storage
            return None
        metadata = (value.shape, value.strides, value.dtype.str, value.nbytes)
        entries.append((name, value, owner, metadata, value.ctypes.data))
    return response.state_schema, tuple(entries)


def _owned_validated_response(response):
    """Copy transport ownership without rescanning unchanged immutable values.

    Subclasses and changed metadata/fields take the full constructor path.
    This is ordinary runtime validation, not isolation from malicious Python.
    """
    stamp = getattr(response, "_transport_stamp", None)
    if type(response) is not ExplicitMaterialBatchOutput or stamp is None:
        return replace(response)
    schema, entries = stamp
    if response.state_schema is not schema:
        return replace(response)
    for name, original, owner, metadata, pointer in entries:
        value = getattr(response, name)
        if value is not original:
            return replace(response)
        if value is not None and (
            value.flags.writeable
            or (value.shape, value.strides, value.dtype.str, value.nbytes) != metadata
            or value.ctypes.data != pointer
        ):
            return replace(response)
    result = object.__new__(ExplicitMaterialBatchOutput)
    object.__setattr__(result, "state_schema", schema)
    for name, original, owner, metadata, pointer in entries:
        if original is None:
            value = None
        else:
            offset = pointer - np.frombuffer(owner, dtype=np.uint8).ctypes.data
            value = np.frombuffer(owner, dtype=float, count=original.size,
                                  offset=offset).reshape(metadata[0])
        object.__setattr__(result, name, value)
    object.__setattr__(result, "_transport_stamp", _response_transport_stamp(result))
    return result


@dataclass(frozen=True)
class ExplicitMaterialIncrementLimit:
    """Provider-declared material-update limit in the request's time units.

    This is not a body/contact/damping spectral certificate. A stateless
    provider computes it from the request; history stays in state_old. The
    identity must identify the numerical policy, not the current array values.
    """

    maximum_time_increment: np.ndarray
    policy_identity: str

    def __post_init__(self):
        values = np.asarray(self.maximum_time_increment)
        if values.ndim != 1 or values.dtype.kind == "b":
            raise ValueError("Material increment limits require one value per point.")
        values = _array(values, values.shape, "maximum_time_increment")
        if np.any(values <= 0):
            raise ValueError("Material increment limits must be positive.")
        if (
            not isinstance(self.policy_identity, str)
            or not self.policy_identity.strip()
        ):
            raise ValueError("Material increment policy requires a nonempty identity.")
        object.__setattr__(self, "maximum_time_increment", values)


class ExplicitMaterialIncrementExceeded(ValueError):
    code = "explicit_material_increment_exceeded"

    def __init__(self, *, requested, allowed, point_index, policy_identity):
        self.requested_time_increment = float(requested)
        self.allowed_time_increment = float(allowed)
        self.point_index = int(point_index)
        self.policy_identity = policy_identity
        super().__init__(
            f"Material increment {requested} exceeds declared limit {allowed} "
            f"at point {point_index}; policy={policy_identity}."
        )


def explicit_increment_policy_identity(material):
    """Read the optional provider contract without executing a material update."""
    method = getattr(material, "explicit_increment_limit", None)
    identity = getattr(material, "explicit_increment_policy_identity", None)
    if method is None and identity is None:
        return None
    if not callable(method) or not isinstance(identity, str) or not identity.strip():
        raise ValueError(
            "Material increment query and stable policy identity are both required."
        )
    return identity


def _check_explicit_increment(material, request):
    identity = explicit_increment_policy_identity(material)
    if identity is None:
        return
    limit = material.explicit_increment_limit(request)
    if not isinstance(limit, ExplicitMaterialIncrementLimit):
        raise TypeError("Expected ExplicitMaterialIncrementLimit.")
    limit = replace(limit)
    if (
        limit.policy_identity != identity
        or limit.maximum_time_increment.shape != (request.point_count,)
        or explicit_increment_policy_identity(material) != identity
    ):
        raise ValueError("Material increment policy changed identity or point count.")
    if request.point_count:
        point = int(np.argmin(limit.maximum_time_increment))
        allowed = float(limit.maximum_time_increment[point])
        # Accepted times are accumulated and subtracted by the Procedure;
        # tolerate only floating-point reconstruction noise at equality.
        endpoint = float(request.time + request.time_increment)
        if not np.isfinite(endpoint):
            raise ValueError("Material increment endpoint must be finite.")
        clock_rounding = 2 * max(
            abs(np.spacing(float(request.time))),
            abs(np.spacing(endpoint)),
        )
        if clock_rounding > 1e-6 * min(request.time_increment, allowed):
            raise ValueError(
                "Absolute material time cannot resolve the increment limit; "
                "use a better-resolved time origin or units."
            )
        rounding = clock_rounding + 32 * np.finfo(float).eps * max(
            request.time_increment, allowed
        )
        if request.time_increment - allowed > rounding:
            raise ExplicitMaterialIncrementExceeded(
                requested=request.time_increment,
                allowed=allowed,
                point_index=point,
                policy_identity=identity,
            )


@dataclass(frozen=True)
class ExplicitMaterialBatchOutput:
    cauchy_stress: np.ndarray
    state_new: np.ndarray
    state_schema: MaterialStateSchema
    internal_work_density_increment: np.ndarray
    stored_energy_density: np.ndarray | None = None
    dissipation_density_increment: np.ndarray | None = None
    requested_time_scale: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("Expected MaterialStateSchema.")
        values = np.asarray(self.cauchy_stress)
        if values.ndim != 3:
            raise ValueError("Stress requires shape (points, 3, 3).")
        count = len(values)
        stress = _array(values, (count, 3, 3), "cauchy_stress")
        # Finite diagonal entries cancel identically; each off-diagonal pair
        # occurs twice in the full tensor difference. Check the three unique
        # pairs without allocating another nine-component field per point.
        if not _symmetric_cauchy_stress(stress):
            raise ValueError("Cauchy stress must be symmetric.")
        object.__setattr__(self, "cauchy_stress", stress)
        object.__setattr__(
            self,
            "state_new",
            _array(self.state_new, (count, self.state_schema.size), "state_new"),
        )
        object.__setattr__(
            self,
            "internal_work_density_increment",
            _array(
                self.internal_work_density_increment,
                (count,),
                "internal_work_density_increment",
            ),
        )
        for name in ("stored_energy_density", "dissipation_density_increment"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _array(value, (count,), name))
        scales = (
            np.ones(count)
            if self.requested_time_scale is None
            else self.requested_time_scale
        )
        scales = _array(scales, (count,), "requested_time_scale")
        if np.any(scales <= 0) or np.any(scales > 1):
            raise ValueError("Requested time scale must lie in (0, 1].")
        object.__setattr__(self, "requested_time_scale", scales)
        object.__setattr__(self, "_transport_stamp", _response_transport_stamp(self))

    @property
    def point_count(self):
        return len(self.cauchy_stress)

    def energy_contract(self):
        return {
            "volume_configuration": "reference",
            "internal_work": "increment",
            "stored_energy_available": self.stored_energy_density is not None,
            "dissipation_available": self.dissipation_density_increment is not None,
            "thermodynamic_validation": "not_established_by_response",
            "tangent_available": False,
            "explicit_stability": "requires_independent_procedure_contract",
        }


def validated_explicit_material_update(material, request):
    """Pure local provider evaluation: never accept state or fall back silently."""
    if not isinstance(request, MaterialPointArrayBatchInput):
        raise TypeError("Expected MaterialPointArrayBatchInput.")
    if request.state_schema != material.state_schema:
        raise ValueError("Explicit request state schema differs from material.")
    identity = explicit_increment_policy_identity(material)
    _check_explicit_increment(material, request)
    method = getattr(material, "update_explicit_array_batch", None)
    if not callable(method):
        raise TypeError("Material must implement update_explicit_array_batch.")
    response = method(request)
    if explicit_increment_policy_identity(material) != identity:
        raise ValueError(
            "Material increment policy changed during response evaluation."
        )
    if not isinstance(response, ExplicitMaterialBatchOutput):
        raise TypeError(
            "Expected ExplicitMaterialBatchOutput; no implicit response coercion."
        )
    response = _owned_validated_response(response)
    if (
        response.state_schema != request.state_schema
        or response.point_count != request.point_count
    ):
        raise ValueError("Explicit response changed point count or state identity.")
    return response
