# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional columnar transport of the existing finite-strain point contract.

No constitutive equations, state acceptance or framework objects live here.
Providers opting into ``update_array_batch`` avoid constructing one Python
input/output object per point. The scalar and ordered-object protocols remain
available. Optional channels are either defined for the whole batch or absent.
"""

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

from .user_material import (
    MaterialStateSchema,
    MaterialTangentConvention,
    UserMaterial,
    _STATE_NAME,
)


def _array(value, shape, label):
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real; complex values are not supported.")
    selected = np.asarray(value, dtype=float)
    if selected.shape != shape or not np.isfinite(selected).all():
        raise ValueError(f"{label} must be finite with shape {shape}.")
    selected = selected.copy()
    selected.setflags(write=False)
    return selected


@dataclass(frozen=True)
class MaterialPointArrayBatchInput:
    """Read-only finite-strain arrays sharing one schema, time and parameter set."""
    deformation_gradient_old: np.ndarray
    deformation_gradient_new: np.ndarray
    state_old: np.ndarray
    state_schema: MaterialStateSchema
    time: float
    time_increment: float
    properties: np.ndarray = field(default_factory=lambda: np.empty(0))
    temperature: np.ndarray | None = None
    temperature_increment: np.ndarray | None = None
    field_variables: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("state_schema must be a MaterialStateSchema.")
        values = np.asarray(self.deformation_gradient_new)
        if values.ndim != 3 or len(values) == 0:
            raise ValueError(
                "An array batch requires at least one deformation gradient."
            )
        count = len(values)
        for name in ("deformation_gradient_old", "deformation_gradient_new"):
            value = _array(getattr(self, name), (count, 3, 3), name)
            if np.any(np.linalg.det(value) <= 0):
                raise ValueError(f"{name} must have positive determinants.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "state_old",
            _array(self.state_old, (count, self.state_schema.size), "state_old"),
        )
        if (
            not np.isfinite(self.time)
            or not np.isfinite(self.time_increment)
            or self.time_increment <= 0
        ):
            raise ValueError("Material time must be finite and increment positive.")
        if np.iscomplexobj(self.properties):
            raise ValueError("properties must be real; complex values are not supported.")
        properties = np.asarray(self.properties, dtype=float).reshape(-1)
        object.__setattr__(
            self, "properties", _array(properties, properties.shape, "properties")
        )
        for name in ("temperature", "temperature_increment"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _array(value, (count,), name))
        if self.field_variables is not None:
            values = np.asarray(self.field_variables)
            if values.ndim != 2 or len(values) != count:
                raise ValueError("field_variables must have shape (points, fields).")
            object.__setattr__(
                self, "field_variables", _array(values, values.shape, "field_variables")
            )

    @property
    def point_count(self):
        return len(self.deformation_gradient_new)


@dataclass(frozen=True)
class MaterialPointArrayBatchOutput:
    """Stress, declared tangent and uncommitted state in matching point order.

    Optional energy channels use the same reference-volume convention as
    MaterialPointOutput. No clipping or thermodynamic approval is implied.
    """
    cauchy_stress: np.ndarray
    consistent_tangent: np.ndarray
    state_new: np.ndarray
    tangent_convention: MaterialTangentConvention
    state_schema: MaterialStateSchema
    strain_energy_density: np.ndarray | None = None
    dissipation_density_increment: np.ndarray | None = None
    suggested_time_scale: np.ndarray | None = None
    stored_energy_density_components: Mapping[str, np.ndarray] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not isinstance(self.state_schema, MaterialStateSchema):
            raise TypeError("state_schema must be a MaterialStateSchema.")
        if not isinstance(self.tangent_convention, MaterialTangentConvention):
            raise TypeError("tangent_convention must be a MaterialTangentConvention.")
        stress = np.asarray(self.cauchy_stress)
        if stress.ndim != 3:
            raise ValueError("An array response requires a batch of stress tensors.")
        count = len(stress)
        stress = _array(stress, (count, 3, 3), "cauchy_stress")
        scale = np.maximum(np.linalg.norm(stress, axis=(1, 2)), np.finfo(float).tiny)
        if np.any(
            np.max(np.abs(stress - stress.transpose(0, 2, 1)), axis=(1, 2))
            > 1e-10 * scale
        ):
            raise ValueError("cauchy_stress must be symmetric at every point.")
        object.__setattr__(self, "cauchy_stress", stress)
        object.__setattr__(
            self,
            "consistent_tangent",
            _array(
                self.consistent_tangent,
                (count, *self.tangent_convention.array_shape),
                "consistent_tangent",
            ),
        )
        object.__setattr__(
            self,
            "state_new",
            _array(self.state_new, (count, self.state_schema.size), "state_new"),
        )
        for name in ("strain_energy_density", "dissipation_density_increment"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _array(value, (count,), name))
        scales = (
            np.ones(count)
            if self.suggested_time_scale is None
            else self.suggested_time_scale
        )
        scales = _array(scales, (count,), "suggested_time_scale")
        if np.any(scales <= 0):
            raise ValueError("suggested_time_scale must be positive.")
        object.__setattr__(self, "suggested_time_scale", scales)
        components = {}
        for name, value in self.stored_energy_density_components.items():
            key = str(name).strip().upper()
            if not key or not _STATE_NAME.fullmatch(key) or key in components:
                raise ValueError(
                    "Stored-energy component names must be unique result-variable names."
                )
            components[key] = _array(value, (count,), key)
        if components:
            if self.strain_energy_density is None:
                raise ValueError(
                    "Stored-energy components require strain_energy_density."
                )
            energy = self.strain_energy_density
            tolerance = 2e-14 * np.maximum(1.0, np.abs(energy)) + 2e-12 * np.abs(energy)
            if np.any(np.abs(sum(components.values()) - energy) > tolerance):
                raise ValueError(
                    "Stored-energy components must sum to strain_energy_density at every point."
                )
        object.__setattr__(self, "stored_energy_density_components", components)

    @property
    def point_count(self):
        return len(self.cauchy_stress)


def validated_material_array_batch_update(
    material: UserMaterial, request: MaterialPointArrayBatchInput
) -> MaterialPointArrayBatchOutput:
    """Validate one columnar batch; failures never silently fall back."""
    if not isinstance(material, UserMaterial):
        raise TypeError(
            "Array providers must retain the ordinary UserMaterial contract."
        )
    if not isinstance(request, MaterialPointArrayBatchInput):
        raise TypeError("request must be a MaterialPointArrayBatchInput.")
    if request.state_schema != material.state_schema:
        raise ValueError("Array batch state schema does not match the material.")
    result = material.update_array_batch(request)
    if not isinstance(result, MaterialPointArrayBatchOutput):
        raise TypeError("update_array_batch must return MaterialPointArrayBatchOutput.")
    # Revalidate at the boundary even if provider code modified its result.
    result = replace(result)
    if result.point_count != request.point_count:
        raise ValueError("Array batch response count differs from request.")
    if result.state_schema != material.state_schema:
        raise ValueError("Array response changed the material state schema.")
    if result.tangent_convention != material.tangent_convention:
        raise ValueError("Array response changed the material tangent convention.")
    return result
